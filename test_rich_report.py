#!/usr/bin/env python3
"""Quick test for RichReportSummarizer with existing results."""

from opencompass.summarizers.rich_report import RichReportSummarizer
from mmengine import ConfigDict

cfg = ConfigDict(
    models=[dict(abbr='qwen3-4b-medicalgpt', path='/workspace/MedicalGPT/model/qwen')],
    datasets=[dict(type='CEvalDataset', abbr='ceval-physician')],
    work_dir='/workspace/opencompass/outputs/default/20260419_094244',
)

s = RichReportSummarizer(cfg)
raw, parsed, dm, dem = s._pick_up_results()
print('parsed_results:', parsed)
print('dataset_metrics:', dm)

# 测试生成图表
chart_path = s._generate_chart(parsed, dm, '20260424_test')
print(f'Chart path: {chart_path}')

# 测试飞书富文本构建
content_lines = s._send_feishu_report(parsed, dm, '20260424_test', chart_path)
