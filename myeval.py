from mmengine.config import read_base

# 1. 明确告诉框架，从哪几个文件里“吸取”模型和数据集的定义
with read_base():
    # 这里使用相对导入（注意点号个数，确保能找到对应的文件夹）
    # 如果你的 bot_cfg.py 在 configs/ 目录下：
    from .opencompass.configs.models.qwen3.my_medicalgpt import models
    from .opencompass.configs.datasets.ceval.ceval_physician import datasets

# 2. 飞书地址也稳稳地坐在这里
lark_bot_url = 'https://open.feishu.cn/open-apis/bot/v2/hook/1c8b31f3-73f1-480f-bf3a-e5095787c748'

# 3. 如果你想让飞书显示的标题好看点，甚至可以加个任务缩写
# abbr = 'medical_physician_task'