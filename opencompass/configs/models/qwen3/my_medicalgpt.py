from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type='HuggingFaceCausalLM',
        abbr='qwen3-4b-medicalgpt',
        path='/workspace/MedicalGPT/model/qwen',  # 模型路径
        tokenizer_path=None,
        max_seq_len=2048,
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]