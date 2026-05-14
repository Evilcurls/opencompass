from mmengine.config import read_base

# ==========================================
# 评估配置：Qwen3.5-2B DPO 微调模型
# 用法: python run.py eval_qwen35_2b_dpo.py
# ==========================================
with read_base():
    from .opencompass.configs.models.qwen3_5.qwen3_5_models import qwen35_2b_dpo_models as models
    from .opencompass.configs.datasets.ceval.ceval_physician import datasets
    from .opencompass.configs.summarizers.rich_report import summarizer

# 输出目录
work_dir = './outputs/eval_qwen35_2b_dpo'

# 飞书通知
lark_bot_url = 'https://open.feishu.cn/open-apis/bot/v2/hook/1c8b31f3-73f1-480f-bf3a-e5095787c748'
