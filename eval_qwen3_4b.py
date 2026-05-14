from mmengine.config import read_base

# ==========================================
# 评估配置：Qwen3-4B 基座模型
# 用法: python run.py eval_qwen3_4b.py
# ==========================================
with read_base():
    from .opencompass.configs.models.qwen3.my_medicalgpt import models
    from .opencompass.configs.datasets.ceval.ceval_physician import datasets
    from .opencompass.configs.summarizers.rich_report import summarizer

# 输出目录
work_dir = './outputs/eval_qwen3_4b'

# 飞书通知（评估完成后自动发消息）
lark_bot_url = 'https://open.feishu.cn/open-apis/bot/v2/hook/1c8b31f3-73f1-480f-bf3a-e5095787c748'
