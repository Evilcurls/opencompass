#!/usr/bin/env python3
"""评测飞书通知工具

支持两种模式:
  start: 任务开始时发送通知（解析配置文件，提取模型/数据集/样本数/预估时间）
  end:   任务结束时发送结果报告（读取 summary CSV，生成表格 + BEST 标记）

用法:
    # 开始通知
    python tools/report_to_feishu.py --mode start --config eval_qwen35_2b_dpo.py

    # 结束报告
    python tools/report_to_feishu.py --mode end --auto-scan
    python tools/report_to_feishu.py --mode end --output-dirs ./outputs/default ./outputs/eval_qwen35_2b_dpo
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import requests

# ============ 飞书配置 ============
DEFAULT_LARK_URL = 'https://open.feishu.cn/open-apis/bot/v2/hook/1c8b31f3-73f1-480f-bf3a-e5095787c748'

# ============ 预估时间参数 ============
# 每个 sample 推理预估秒数（粗估，基于 HuggingFace + 单 GPU T4）
SECONDS_PER_SAMPLE = {
    'default': 8,
    'gen': 8,      # GenInferencer 一般较慢
    'ppl': 2,      # PPLInferencer 较快
}


# ============================================================
#  通用工具
# ============================================================

def send_to_feishu(lark_url, title, content):
    """发送飞书富文本消息"""
    msg = {
        'msg_type': 'post',
        'content': {
            'post': {
                'zh_cn': {
                    'title': title,
                    'content': content
                }
            }
        }
    }
    try:
        resp = requests.post(lark_url, json=msg, timeout=10)
        result = resp.json()
        if result.get('StatusCode') == 0:
            print(f'[INFO] Feishu message sent: {title}')
        else:
            print(f'[WARN] Feishu send failed: {result}')
        return result
    except Exception as e:
        print(f'[WARN] Feishu send error: {e}')
        return None


def send_text_to_feishu(lark_url, text):
    """发送飞书纯文本消息"""
    msg = {
        'msg_type': 'text',
        'content': {'text': text}
    }
    try:
        resp = requests.post(lark_url, json=msg, timeout=10)
        result = resp.json()
        if result.get('StatusCode') == 0:
            print(f'[INFO] Feishu text sent')
        else:
            print(f'[WARN] Feishu send failed: {result}')
        return result
    except Exception as e:
        print(f'[WARN] Feishu send error: {e}')
        return None


# ============================================================
#  MODE: START  —  解析配置，发送"开始"通知
# ============================================================

def parse_config_info(config_path):
    """解析 OpenCompass 配置文件，提取模型/数据集/样本数等信息

    Returns:
        dict with keys: models, datasets, total_samples, estimated_seconds
    """
    # 使用 mmengine Config 解析
    sys.path.insert(0, os.getcwd())
    from mmengine.config import Config
    cfg = Config.fromfile(config_path, format_python_code=False)

    # 提取模型信息
    models = []
    for m in cfg.get('models', []):
        models.append({
            'abbr': m.get('abbr', 'unknown'),
            'path': m.get('path', ''),
            'num_gpus': m.get('run_cfg', {}).get('num_gpus', 1),
            'max_out_len': m.get('max_out_len', 512),
            'batch_size': m.get('batch_size', 1),
        })

    # 提取数据集信息
    datasets = []
    total_samples = 0
    for ds in cfg.get('datasets', []):
        ds_info = {
            'abbr': ds.get('abbr', 'unknown'),
            'name': ds.get('name', ''),
            'path': ds.get('path', ''),
        }

        # 推断推理方式
        infer_cfg = ds.get('infer_cfg', {})
        inferencer = infer_cfg.get('inferencer', {})
        infer_type = 'gen'  # default
        if isinstance(inferencer, dict):
            infer_type_str = str(inferencer.get('type', ''))
            if 'PPL' in infer_type_str:
                infer_type = 'ppl'
        ds_info['infer_type'] = infer_type

        # 尝试计算样本数
        sample_count = _count_samples(ds, cfg)
        ds_info['sample_count'] = sample_count
        total_samples += sample_count

        # 提取评估指标
        eval_cfg = ds.get('eval_cfg', {})
        evaluator = eval_cfg.get('evaluator', {})
        ds_info['metric'] = _get_metric_name(evaluator)

        datasets.append(ds_info)

    # 预估总时间
    num_models = len(models) or 1
    estimated_seconds = total_samples * SECONDS_PER_SAMPLE.get('default', 8) * num_models

    return {
        'models': models,
        'datasets': datasets,
        'total_samples': total_samples,
        'estimated_seconds': estimated_seconds,
    }


def _count_samples(ds_cfg, full_cfg=None):
    """尝试统计数据集样本数"""
    ds_path = ds_cfg.get('path', '')
    ds_name = ds_cfg.get('name', '')
    reader_cfg = ds_cfg.get('reader_cfg', {})
    test_split = reader_cfg.get('test_split', 'val')

    # 方法1: 直接读 CSV 文件
    if ds_path and ds_name:
        csv_patterns = [
            os.path.join(ds_path, f'{ds_name}_{test_split}.csv'),
            os.path.join(ds_path, test_split, f'{ds_name}_{test_split}.csv'),
        ]
        for pattern in csv_patterns:
            if os.path.exists(pattern):
                try:
                    with open(pattern, 'r') as f:
                        return sum(1 for _ in csv.reader(f)) - 1  # 减去 header
                except Exception:
                    pass

    # 方法2: 在数据目录中搜索
    if ds_path and os.path.isdir(ds_path):
        # 搜索 val/ 或 test/ 子目录
        for split_dir in [test_split, 'val', 'test']:
            search_dir = os.path.join(ds_path, split_dir)
            if os.path.isdir(search_dir):
                # 用数据集 abbr 的最后部分匹配文件名
                abbr_suffix = ds_cfg.get('abbr', '').split('-')[-1] if '-' in ds_cfg.get('abbr', '') else ds_name
                if abbr_suffix:
                    for f in os.listdir(search_dir):
                        if abbr_suffix in f and f.endswith('.csv'):
                            try:
                                with open(os.path.join(search_dir, f), 'r') as fp:
                                    return sum(1 for _ in csv.reader(fp)) - 1
                            except Exception:
                                pass

    return -1  # 未知


def _get_metric_name(evaluator_cfg):
    """从 evaluator 配置中提取指标名称"""
    if isinstance(evaluator_cfg, dict):
        type_str = str(evaluator_cfg.get('type', ''))
        if 'Acc' in type_str:
            return 'accuracy'
        if 'Bleu' in type_str:
            return 'bleu'
        if 'Rouge' in type_str:
            return 'rouge'
    return 'accuracy'  # default


def format_duration(seconds):
    """将秒数格式化为可读的时间字符串"""
    if seconds < 0:
        return 'unknown'
    if seconds < 60:
        return f'{seconds}s'
    minutes = seconds // 60
    secs = seconds % 60
    if minutes < 60:
        return f'{minutes}m {secs}s'
    hours = minutes // 60
    mins = minutes % 60
    return f'{hours}h {mins}m'


def send_start_notification(lark_url, config_path):
    """解析配置并发送开始通知"""
    info = parse_config_info(config_path)

    content = []

    # 标题
    content.append([{'tag': 'text', 'text': '🚀 Evaluation Started\n'}])
    content.append([{'tag': 'text', 'text': '━━━━━━━━━━━━━━━━━━━━━━━━━━'}])

    # 模型信息
    content.append([{'tag': 'text', 'text': '\n📋 Models:\n'}])
    for m in info['models']:
        gpu_str = f" (x{m['num_gpus']} GPU)" if m['num_gpus'] > 1 else ''
        content.append([{'tag': 'text', 'text': f"  • {m['abbr']}{gpu_str}"}])

    # 数据集信息
    content.append([{'tag': 'text', 'text': '\n📊 Datasets:\n'}])
    for ds in info['datasets']:
        sample_str = f"{ds['sample_count']}" if ds['sample_count'] > 0 else '?'
        metric_str = ds.get('metric', 'accuracy')
        infer_str = ds.get('infer_type', 'gen')
        content.append([{
            'tag': 'text',
            'text': f"  • {ds['abbr']}  |  n={sample_str}  |  metric={metric_str}  |  mode={infer_str}"
        }])

    # 预估时间
    content.append([{'tag': 'text', 'text': '\n'}])
    content.append([{'tag': 'text', 'text': '━━━━━━━━━━━━━━━━━━━━━━━━━━'}])

    total_samples = info['total_samples']
    num_models = len(info['models']) or 1
    est_time = format_duration(info['estimated_seconds'])

    content.append([{
        'tag': 'text',
        'text': f'⏱ Total: {num_models} model(s) x {total_samples} samples = ~{est_time}'
    }])

    content.append([{'tag': 'text', 'text': '\n⏳ Running now...'}])

    send_to_feishu(lark_url, '🚀 Eval Started', content)

    # 打印到终端
    print(f'[INFO] Models: {[m["abbr"] for m in info["models"]]}')
    print(f'[INFO] Datasets: {[ds["abbr"] for ds in info["datasets"]]}')
    print(f'[INFO] Total samples: {total_samples}')
    print(f'[INFO] Estimated time: {est_time}')


# ============================================================
#  MODE: END  —  读取结果，发送结束报告
# ============================================================

def find_latest_summary(output_dir):
    """在 output_dir 下找到最新的 summary CSV"""
    csv_files = sorted(glob.glob(os.path.join(output_dir, '*/summary/summary_*.csv')))
    if not csv_files:
        return None
    return csv_files[-1]


def parse_summary_csv(csv_path):
    """解析 summary CSV，返回详细结果 + 元信息

    Returns:
        tuple: (all_results, meta_info)
            all_results: {model_name: {dataset: score}}
            meta_info: [{dataset, version, metric, mode}]
    """
    all_results = {}
    meta_info = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        return all_results, meta_info

    header = lines[0].strip().split(',')
    model_cols = header[4:]  # dataset, version, metric, mode 之后的列是模型名

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < 5:
            continue
        dataset = parts[0]
        version = parts[1]
        metric = parts[2]
        mode = parts[3]

        # 记录每个数据集的元信息
        meta_info.append({
            'dataset': dataset,
            'metric': metric,
            'mode': mode,
        })

        for i, model in enumerate(model_cols):
            if model not in all_results:
                all_results[model] = {}
            try:
                all_results[model][dataset] = float(parts[4 + i])
            except (ValueError, IndexError):
                pass

    return all_results, meta_info


def collect_results(output_dirs):
    """从多个 output 目录收集结果"""
    all_results = {}
    all_meta = {}
    for d in output_dirs:
        csv_path = find_latest_summary(d)
        if csv_path is None:
            print(f'[WARN] No summary found in {d}')
            continue
        print(f'[INFO] Found: {csv_path}')
        results, meta_info = parse_summary_csv(csv_path)
        for model, scores in results.items():
            if model in all_results:
                all_results[model].update(scores)
            else:
                all_results[model] = scores
        for m in meta_info:
            ds = m['dataset']
            if ds not in all_meta:
                all_meta[ds] = m

    return all_results, all_meta


def format_end_message(all_results, all_meta):
    """构建结束通知的飞书富文本消息"""
    models = list(all_results.keys())
    datasets = sorted(set(ds for scores in all_results.values() for ds in scores))

    if not models or not datasets:
        return []

    content = []

    # ── 标题 ──
    content.append([{'tag': 'text', 'text': '📊 Model Evaluation Report\n'}])
    content.append([{'tag': 'text', 'text': '━━━━━━━━━━━━━━━━━━━━━━━━━━'}])

    # ── 模型列表 ──
    content.append([{'tag': 'text', 'text': '\n📋 Models:\n'}])
    for m in models:
        content.append([{'tag': 'text', 'text': f'  • {m}'}])

    # ── 数据集信息 ──
    content.append([{'tag': 'text', 'text': '\n📊 Datasets:\n'}])
    for ds in datasets:
        meta = all_meta.get(ds, {})
        metric = meta.get('metric', 'accuracy')
        mode = meta.get('mode', '-')
        # 从结果中获取样本数（尝试读取原始结果文件）
        sample_str = _get_sample_count_from_results(models[0], ds) if models else '?'
        content.append([{
            'tag': 'text',
            'text': f"  • {ds}  |  metric={metric}  |  mode={mode}  |  n={sample_str}"
        }])

    # ── 分数表格 ──
    content.append([{'tag': 'text', 'text': '\n📈 Scores:\n'}])

    col1_w = max(len('Dataset'), max(len(ds.replace('ceval-', '')) for ds in datasets)) + 2
    model_widths = [max(len(m), 8) + 2 for m in models]

    header = f"{'Dataset'.ljust(col1_w)}"
    for m, w in zip(models, model_widths):
        header += m.ljust(w)
    content.append([{'tag': 'text', 'text': header}])

    sep_str = '-' * (col1_w + sum(model_widths))
    content.append([{'tag': 'text', 'text': sep_str}])

    for ds in datasets:
        row = ds.replace('ceval-', '').ljust(col1_w)
        for model, w in zip(models, model_widths):
            score = all_results[model].get(ds, 0)
            row += f'{score:.2f}'.ljust(w)
        content.append([{'tag': 'text', 'text': row}])

    # ── 分隔线 ──
    content.append([{'tag': 'text', 'text': '━━━━━━━━━━━━━━━━━━━━━━━━━━'}])

    # ── 最高分 ──
    for ds in datasets:
        scores = {m: all_results[m].get(ds, 0) for m in models}
        best = max(scores, key=scores.get)
        best_score = scores[best]
        ds_name = ds.replace('ceval-', '')
        content.append([{
            'tag': 'text',
            'text': f'🏆 {ds_name} Best: {best} ({best_score:.2f}%)'
        }])

    content.append([{'tag': 'text', 'text': '\n✅ Evaluation Complete!'}])

    return content


def _get_sample_count_from_results(model_name, dataset_name):
    """从 results JSON 文件中获取样本数"""
    search_dirs = ['./outputs']
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for run_dir in os.listdir(base):
            results_dir = os.path.join(base, run_dir)
            if not os.path.isdir(results_dir):
                continue
            for sub in os.listdir(results_dir):
                result_file = os.path.join(results_dir, sub, 'results', model_name, f'{dataset_name}.json')
                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            # OpenCompass 格式: details 中数字键即为样本索引
                            details = data.get('details', {})
                            if isinstance(details, dict):
                                n = sum(1 for k in details.keys() if k.isdigit())
                                if n > 0:
                                    return n
                            # 备选
                            for key in ['results', 'preds']:
                                if key in data and isinstance(data[key], list):
                                    return len(data[key])
                        elif isinstance(data, list):
                            return len(data)
                    except Exception:
                        pass
    return '?'


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='评测飞书通知工具')
    parser.add_argument('--mode', choices=['start', 'end'], required=True,
                        help='start: 发送开始通知; end: 发送结束报告')
    parser.add_argument('--config', type=str, default=None,
                        help='(start 模式) OpenCompass 配置文件路径')
    parser.add_argument('--output-dirs', nargs='+',
                        help='(end 模式) OpenCompass 输出目录列表')
    parser.add_argument('--auto-scan', action='store_true',
                        help='(end 模式) 自动扫描 outputs/ 目录')
    parser.add_argument('--lark-url', default=DEFAULT_LARK_URL,
                        help='飞书 webhook URL')
    args = parser.parse_args()

    if args.mode == 'start':
        if not args.config:
            print('[ERROR] --mode start requires --config <config_path>')
            sys.exit(1)
        send_start_notification(args.lark_url, args.config)

    elif args.mode == 'end':
        # 收集 output 目录
        if args.auto_scan:
            base = './outputs'
            if os.path.isdir(base):
                args.output_dirs = [os.path.join(base, d) for d in os.listdir(base)
                                   if os.path.isdir(os.path.join(base, d))]
                print(f'[INFO] Auto-scanned dirs: {args.output_dirs}')
            else:
                args.output_dirs = []

        if not args.output_dirs:
            print('[ERROR] No output directories. Use --output-dirs or --auto-scan')
            sys.exit(1)

        # 1. 收集结果
        all_results, all_meta = collect_results(args.output_dirs)
        if not all_results:
            print('[ERROR] No results found')
            sys.exit(1)

        print(f'[INFO] Results: {json.dumps(all_results, indent=2)}')

        # 2. 构建消息并发送
        content = format_end_message(all_results, all_meta)
        send_to_feishu(args.lark_url, '📊 Eval Complete', content)

        print('[DONE] End report sent to Feishu!')


if __name__ == '__main__':
    main()
