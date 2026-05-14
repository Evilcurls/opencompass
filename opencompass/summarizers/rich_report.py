"""Rich Report Summarizer - 在 DefaultSummarizer 基础上自动生成对比图表和飞书富文本通知。"""

import getpass
import os
import os.path as osp
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # 无头模式，服务器上不需要 GUI
import matplotlib.pyplot as plt
import numpy as np

from opencompass.summarizers.default import DefaultSummarizer
from opencompass.utils import LarkReporter


class RichReportSummarizer(DefaultSummarizer):
    """扩展 DefaultSummarizer，汇总完成后自动：
    1. 生成模型对比柱状图 (PNG)
    2. 发送飞书富文本消息（带格式化的分数表格、Best 标记）
    """

    def summarize(
        self,
        output_path: str = None,
        time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S'),
    ):
        # 1. 先调用父类完成默认汇总（生成 CSV/MD/TXT）
        super().summarize(output_path=output_path, time_str=time_str)

        # 2. 收集数据，生成图表和飞书消息
        try:
            raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
                self._pick_up_results()
            raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
                self._calculate_group_metrics(
                    raw_results, parsed_results, dataset_metrics, dataset_eval_mode)
        except Exception as e:
            self.logger.warning(f'Failed to pick up results for rich report: {e}')
            return

        # 3. 生成柱状图（允许失败，不影响飞书报告）
        chart_path = None
        try:
            chart_path = self._generate_chart(
                parsed_results, dataset_metrics, time_str)
        except Exception as e:
            self.logger.warning(f'Chart generation failed: {e}')

        # 4. 发送飞书富文本（允许失败，不影响主流程）
        if self.lark_reporter:
            try:
                self._send_feishu_report(
                    parsed_results, dataset_metrics, time_str, chart_path)
            except Exception as e:
                self.logger.warning(f'Feishu report failed: {e}')

    def _generate_chart(self, parsed_results, dataset_metrics, time_str):
        """生成模型对比柱状图，保存到 summary 目录。"""
        # 收集数据
        dataset_names = []
        model_scores = {}  # {model_abbr: [scores]}

        for dataset_abbr in dataset_metrics:
            for metric in dataset_metrics[dataset_abbr]:
                if metric not in ('accuracy', 'score', 'f1', 'exact_match'):
                    continue
                label = f'{dataset_abbr}\n({metric})'
                dataset_names.append(label)
                for model_abbr in self.model_abbrs:
                    if dataset_abbr in parsed_results.get(model_abbr, {}):
                        score = parsed_results[model_abbr][dataset_abbr].get(metric, 0)
                        model_scores.setdefault(model_abbr, []).append(score)
                    else:
                        model_scores.setdefault(model_abbr, []).append(0)

        if not dataset_names:
            self.logger.warning('No chartable metrics found, skip chart generation.')
            return None

        # 绘图
        fig, ax = plt.subplots(figsize=(max(8, len(dataset_names) * 3), 6))

        x = np.arange(len(dataset_names))
        n_models = len(self.model_abbrs)
        bar_width = 0.6 / max(n_models, 1)
        colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
                  '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

        for i, model_abbr in enumerate(self.model_abbrs):
            scores = model_scores.get(model_abbr, [])
            offset = (i - n_models / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, scores, bar_width * 0.9,
                          label=model_abbr,
                          color=colors[i % len(colors)],
                          edgecolor='white', linewidth=0.5)
            # 在柱子上方显示分数
            for bar, score in zip(bars, scores):
                if score > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'{score:.1f}', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')

        ax.set_xlabel('Dataset', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Model Evaluation Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, fontsize=9)
        ax.legend(fontsize=10, loc='upper right')
        ax.set_ylim(0, min(max(max(v) for v in model_scores.values()) * 1.15, 105) if model_scores else 100)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # 保存
        summary_dir = osp.join(self.work_dir, 'summary')
        os.makedirs(summary_dir, exist_ok=True)
        chart_path = osp.join(summary_dir, f'chart_{time_str}.png')
        fig.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f'Chart saved to {osp.abspath(chart_path)}')
        print(f'\n📊 Chart saved to: {osp.abspath(chart_path)}')
        return chart_path

    def _send_feishu_report(self, parsed_results, dataset_metrics,
                            time_str, chart_path):
        """发送飞书富文本消息（post 格式），包含格式化的分数表格和 Best 标记。"""
        content_lines = []

        # ── 标题 ──
        content_lines.append([{
            'tag': 'text',
            'text': 'Eval Complete\n'
        }])
        content_lines.append([{
            'tag': 'text',
            'text': '📊 Model Evaluation Report\n'
        }])
        content_lines.append([{
            'tag': 'text',
            'text': '━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        }])

        # ── 模型列表 ──
        content_lines.append([{
            'tag': 'text',
            'text': '\n📋 Models:\n'
        }])
        for model_abbr in self.model_abbrs:
            content_lines.append([{
                'tag': 'text',
                'text': f'  • {model_abbr}\n'
            }])

        # ── 数据集信息 + 分数表格 ──
        # 收集可展示的 dataset+metric 对
        chartable_metrics = ('accuracy', 'score', 'f1', 'exact_match')
        display_items = []  # [(dataset_abbr, metric)]
        for dataset_abbr in dataset_metrics:
            for metric in dataset_metrics[dataset_abbr]:
                if metric in chartable_metrics:
                    display_items.append((dataset_abbr, metric))

        if display_items:
            content_lines.append([{
                'tag': 'text',
                'text': '\n📊 Datasets:\n'
            }])
            for dataset_abbr, metric in display_items:
                ds_short = dataset_abbr.replace('ceval-', '')
                content_lines.append([{
                    'tag': 'text',
                    'text': f'  • {ds_short}  |  metric={metric}\n'
                }])

            # ── 分数表格 ──
            content_lines.append([{
                'tag': 'text',
                'text': '\n📈 Scores:\n'
            }])

            # 表头
            col1_w = max(len('Dataset'), max(len(ds.replace('ceval-', '')) for ds, _ in display_items)) + 2
            model_widths = [max(len(m), 8) + 2 for m in self.model_abbrs]

            header = 'Dataset'.ljust(col1_w)
            for m, w in zip(self.model_abbrs, model_widths):
                header += m.ljust(w)
            content_lines.append([{'tag': 'text', 'text': header + '\n'}])

            sep_str = '-' * (col1_w + sum(model_widths))
            content_lines.append([{'tag': 'text', 'text': sep_str + '\n'}])

            # 数据行
            for dataset_abbr, metric in display_items:
                ds_short = dataset_abbr.replace('ceval-', '')
                row = ds_short.ljust(col1_w)
                for model_abbr, w in zip(self.model_abbrs, model_widths):
                    score = parsed_results.get(model_abbr, {}).get(dataset_abbr, {}).get(metric, None)
                    if score is not None:
                        row += f'{score:.2f}'.ljust(w)
                    else:
                        row += '-'.ljust(w)
                content_lines.append([{'tag': 'text', 'text': row + '\n'}])

        # ── 分隔线 ──
        content_lines.append([{
            'tag': 'text',
            'text': '━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        }])

        # ── Best 标记 ──
        for dataset_abbr, metric in display_items:
            scores = {}
            for model_abbr in self.model_abbrs:
                s = parsed_results.get(model_abbr, {}).get(dataset_abbr, {}).get(metric, None)
                if s is not None:
                    scores[model_abbr] = s
            if scores:
                best = max(scores, key=scores.get)
                best_score = scores[best]
                ds_short = dataset_abbr.replace('ceval-', '')
                content_lines.append([{
                    'tag': 'text',
                    'text': f'🏆 {ds_short} Best: {best} ({best_score:.2f}%)\n'
                }])

        # ── 图表位置 ──
        if chart_path:
            content_lines.append([{
                'tag': 'text',
                'text': f'\n📊 Chart: {osp.abspath(chart_path)}\n'
            }])

        # ── 结果目录 ──
        content_lines.append([{
            'tag': 'text',
            'text': f'📁 Results: {osp.abspath(self.work_dir)}\n'
        }])

        content_lines.append([{
            'tag': 'text',
            'text': '\n✅ Evaluation Complete!'
        }])

        # 发送
        title = f'📊 Eval Complete - {time_str}'
        self.lark_reporter.post(content_lines, title=title)
        self.logger.info('Rich report sent to Feishu.')
