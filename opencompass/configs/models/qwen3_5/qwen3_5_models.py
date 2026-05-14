from opencompass.models import HuggingFaceCausalLM


# ==========================================
# 模型 1: Qwen3-4B 基座（本地）
# ==========================================
qwen3_4b_models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen3-4b',
        path='/workspace/MedicalGPT/model/qwen',
        tokenizer_path=None,
        max_seq_len=2048,
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]

# ==========================================
# 模型 2: Qwen3.5-2B 原始（HuggingFace 缓存）
# ==========================================
qwen35_2b_base_models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen35-2b-base',
        path='Qwen/Qwen3.5-2B',
        max_seq_len=2048,
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]

# ==========================================
# 模型 3: Qwen3.5-2B DPO 微调（本地合并后）
# ==========================================
qwen35_2b_dpo_models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen35-2b-dpo',
        path='/workspace/MedicalGPT/models/qwen-dpo-merged',
        max_seq_len=2048,
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
