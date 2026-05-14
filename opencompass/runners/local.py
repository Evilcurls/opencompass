import os
import os.path as osp
import re
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock
from typing import Any, Dict, List, Tuple

import mmengine
import numpy as np
from mmengine.config import ConfigDict
from mmengine.device import is_npu_available
from tqdm import tqdm

from opencompass.registry import RUNNERS, TASKS
from opencompass.utils import get_logger, model_abbr_from_cfg

from .base import BaseRunner


class ProgressMonitor(threading.Thread):
    """Monitor subprocess log file and send Feishu progress updates.

    Parses tqdm-style progress lines from the inference log (e.g.
    " 50%|█████ | 25/49 [09:57<10:09, 25.41s/it]") and sends notifications
    at milestone percentages and periodic time intervals.
    """

    def __init__(self, out_path, lark_reporter, task_name,
                 check_interval=30, report_time_interval=300):
        super().__init__(daemon=True)
        self.out_path = out_path
        self.lark_reporter = lark_reporter
        self.task_name = task_name
        self.check_interval = check_interval
        self.report_time_interval = report_time_interval
        self._stop_event = threading.Event()
        self.reported_milestones = set()
        self.last_report_time = 0
        self.start_time = time.time()

    def run(self):
        # Wait for file to be created
        while not self._stop_event.is_set() and not os.path.exists(self.out_path):
            self._stop_event.wait(2)
        while not self._stop_event.is_set():
            self._check_progress()
            self._stop_event.wait(self.check_interval)
        # Final check after stop
        self._check_progress()

    def stop(self):
        self._stop_event.set()
        self.join(timeout=10)

    def _check_progress(self):
        if not self.lark_reporter or not os.path.exists(self.out_path):
            return
        try:
            with open(self.out_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return

        progress = self._parse_inference_progress(content)
        if progress is None:
            return

        current, total, speed = progress
        if total == 0:
            return
        pct = round(current / total * 100, 1)

        now = time.time()
        elapsed = now - self.start_time

        should_report = False

        # Milestone reporting (25%, 50%, 75%, 100%)
        milestone = int(pct // 25) * 25
        if milestone > 0 and milestone not in self.reported_milestones:
            self.reported_milestones.add(milestone)
            should_report = True

        # Time-based periodic reporting
        if (now - self.last_report_time >= self.report_time_interval
                and 0 < current < total):
            should_report = True

        if not should_report:
            return

        self.last_report_time = now

        elapsed_m = int(elapsed // 60)
        elapsed_s = int(elapsed % 60)
        eta_str = ''
        if speed and speed > 0 and current < total:
            remaining = (total - current) * speed
            eta_str = f', ETA ~{int(remaining // 60)}m{int(remaining % 60)}s'

        msg = (f'🔄 [{self.task_name}]\n'
               f'  Progress: {current}/{total} ({pct}%)\n'
               f'  Speed: {speed:.1f}s/sample\n'
               f'  Elapsed: {elapsed_m}m{elapsed_s}s{eta_str}')
        self.lark_reporter.post(msg)

    @staticmethod
    def _parse_inference_progress(content):
        """Parse inference tqdm progress from log content.

        Matches lines with 's/it' (seconds per item) which indicates actual
        model inference, not data loading/mapping (which uses 'it/s').
        Pattern: " 50%|█████ | 25/49 [09:57<10:09, 25.41s/it]"
        """
        pattern = r'(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[.*?([\d.]+)s/it\]'
        matches = list(re.finditer(pattern, content))
        if not matches:
            return None
        last = matches[-1]
        return int(last.group(2)), int(last.group(3)), float(last.group(4))


def get_command_template(gpu_ids: List[int]) -> str:
    """Format command template given available gpu ids."""
    if is_npu_available():
        tmpl = 'ASCEND_RT_VISIBLE_DEVICES=' + ','.join(str(i) for i in gpu_ids)
        tmpl += ' {task_cmd}'
    elif sys.platform == 'win32':  # Always return win32 for Windows
        # use command in Windows format
        tmpl = 'set CUDA_VISIBLE_DEVICES=' + ','.join(str(i) for i in gpu_ids)
        tmpl += ' & {task_cmd}'
    else:
        tmpl = 'CUDA_VISIBLE_DEVICES=' + ','.join(str(i) for i in gpu_ids)
        tmpl += ' {task_cmd}'
    return tmpl


@RUNNERS.register_module()
class LocalRunner(BaseRunner):
    """Local runner. Start tasks by local python.

    Args:
        task (ConfigDict): Task type config.
        max_num_workers (int): Max number of workers to run in parallel.
            Defaults to 16.
        max_workers_per_gpu (int): Max number of workers to run for one GPU.
            Defaults to 1.
        debug (bool): Whether to run in debug mode.
        lark_bot_url (str): Lark bot url.
    """

    def __init__(self,
                 task: ConfigDict,
                 max_num_workers: int = 16,
                 debug: bool = False,
                 max_workers_per_gpu: int = 1,
                 lark_bot_url: str = None,
                 keep_tmp_file: bool = False,
                 **kwargs):
        super().__init__(task=task, debug=debug, lark_bot_url=lark_bot_url)
        self.max_num_workers = max_num_workers
        self.max_workers_per_gpu = max_workers_per_gpu
        self.keep_tmp_file = keep_tmp_file
        logger = get_logger()
        for k, v in kwargs.items():
            logger.warning(f'Ignored argument in {self.__module__}: {k}={v}')

    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Launch multiple tasks.

        Args:
            tasks (list[dict]): A list of task configs, usually generated by
                Partitioner.

        Returns:
            list[tuple[str, int]]: A list of (task name, exit code).
        """

        status = []
        import torch

        if is_npu_available():
            visible_devices = 'ASCEND_RT_VISIBLE_DEVICES'
            device_nums = torch.npu.device_count()
        else:
            visible_devices = 'CUDA_VISIBLE_DEVICES'
            device_nums = torch.cuda.device_count()
        if visible_devices in os.environ:
            all_gpu_ids = [
                int(i)
                for i in re.findall(r'(?<!-)\d+', os.getenv(visible_devices))
            ]
        else:
            all_gpu_ids = list(range(device_nums))

        if self.debug:
            for task in tasks:
                task = TASKS.build(dict(cfg=task, type=self.task_cfg['type']))
                task_name = task.name
                num_gpus = task.num_gpus
                assert len(all_gpu_ids) >= num_gpus
                # get cmd
                mmengine.mkdir_or_exist('tmp/')
                import uuid
                uuid_str = str(uuid.uuid4())

                param_file = f'tmp/{uuid_str}_params.py'
                try:
                    task.cfg.dump(param_file)
                    # if use torchrun, restrict it behaves the same as non
                    # debug mode, otherwise, the torchrun will use all the
                    # available resources which might cause inconsistent
                    # behavior.
                    if len(all_gpu_ids) > num_gpus and num_gpus > 0:
                        get_logger().warning(f'Only use {num_gpus} GPUs for '
                                             f'total {len(all_gpu_ids)} '
                                             'available GPUs in debug mode.')
                    tmpl = get_command_template(all_gpu_ids[:num_gpus])
                    cmd = task.get_command(cfg_path=param_file, template=tmpl)
                    # run in subprocess if starts with torchrun etc.
                    if 'python3' in cmd or 'python ' in cmd:
                        # If it is an infer type task do not reload if
                        # the current model has already been loaded.
                        if 'infer' in self.task_cfg.type.lower():
                            # If a model instance already exists,
                            # do not reload it.
                            task.run(cur_model=getattr(self, 'cur_model',
                                                       None),
                                     cur_model_abbr=getattr(
                                         self, 'cur_model_abbr', None))
                            self.cur_model = task.model
                            self.cur_model_abbr = model_abbr_from_cfg(
                                task.model_cfg)
                        else:
                            task.run()
                    else:
                        tmp_logs = f'tmp/{os.getpid()}_debug.log'
                        get_logger().warning(
                            f'Debug mode, log will be saved to {tmp_logs}')
                        with open(tmp_logs, 'a') as log_file:
                            subprocess.run(cmd,
                                           shell=True,
                                           text=True,
                                           stdout=log_file,
                                           stderr=subprocess.STDOUT)
                finally:
                    if not self.keep_tmp_file:
                        os.remove(param_file)
                    else:
                        pass
                status.append((task_name, 0))
        else:
            if len(all_gpu_ids) > 0:
                gpus = np.zeros(max(all_gpu_ids) + 1, dtype=np.uint)
                gpus[all_gpu_ids] = self.max_workers_per_gpu
            else:
                gpus = np.array([], dtype=np.uint)

            pbar = tqdm(total=len(tasks))
            lock = Lock()

            def submit(task, index):
                task = TASKS.build(dict(cfg=task, type=self.task_cfg['type']))
                num_gpus = task.num_gpus
                assert len(gpus) >= num_gpus

                while True:
                    lock.acquire()
                    if sum(gpus > 0) >= num_gpus:
                        gpu_ids = np.where(gpus)[0][:num_gpus]
                        gpus[gpu_ids] -= 1
                        lock.release()
                        break
                    lock.release()
                    time.sleep(1)

                if num_gpus > 0:
                    tqdm.write(f'launch {task.name} on GPU ' +
                               ','.join(map(str, gpu_ids)))
                else:
                    tqdm.write(f'launch {task.name} on CPU ')

                res = self._launch(task, gpu_ids, index)
                pbar.update()

                with lock:
                    gpus[gpu_ids] += 1

                return res

            with ThreadPoolExecutor(
                    max_workers=self.max_num_workers) as executor:
                status = executor.map(submit, tasks, range(len(tasks)))

        return status

    def _launch(self, task, gpu_ids, index):
        """Launch a single task.

        Args:
            task (BaseTask): Task to launch.

        Returns:
            tuple[str, int]: Task name and exit code.
        """

        task_name = task.name

        pwd = os.getcwd()
        # Dump task config to file
        mmengine.mkdir_or_exist('tmp/')
        # Using uuid to avoid filename conflict
        import uuid
        uuid_str = str(uuid.uuid4())
        param_file = f'{pwd}/tmp/{uuid_str}_params.py'

        try:
            task.cfg.dump(param_file)
            tmpl = get_command_template(gpu_ids)
            get_cmd = partial(task.get_command,
                              cfg_path=param_file,
                              template=tmpl)
            cmd = get_cmd()

            logger = get_logger()
            logger.debug(f'Running command: {cmd}')

            # Run command
            out_path = task.get_log_path(file_extension='out')
            mmengine.mkdir_or_exist(osp.split(out_path)[0])
            stdout = open(out_path, 'w', encoding='utf-8')

            # Start progress monitor if lark_reporter is available
            monitor = None
            if self.lark_reporter:
                monitor = ProgressMonitor(
                    out_path=out_path,
                    lark_reporter=self.lark_reporter,
                    task_name=task_name,
                )
                monitor.start()

            result = subprocess.run(cmd,
                                    shell=True,
                                    text=True,
                                    stdout=stdout,
                                    stderr=stdout)

            # Stop progress monitor
            if monitor:
                stdout.flush()
                monitor.stop()

            if result.returncode != 0:
                logger.error(f'task {task_name} fail, see\n{out_path}')
        finally:
            # Clean up
            if not self.keep_tmp_file:
                os.remove(param_file)
            else:
                pass
        return task_name, result.returncode
