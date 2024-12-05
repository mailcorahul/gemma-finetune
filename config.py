import torch

TRAINING_CONFIGS = {
    "base_model_url": "google/gemma-2-27b-it",
    "adapter_model_url": "/dev2/hf/gemma-2-27b-it-cnn_dailymail-adapter-final",
    "dataset_name": "abisee/cnn_dailymail",
    "wandb_project_name": "gemma-2-27b-it on cnn_dailymail-adapter-final",
    "attn_implementation" : "eager",
    "torch_dtype" : torch.bfloat16,
    "num_dataset_samples": 1000,
    "num_epochs": 3,
    "eval_steps": 1000000,
    "dataset_shuffle_seed": 65,
    "test_size": 0.1,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "device_map": "auto",
    "eval_metric": "rouge",
    "resume": False,
    "pretrained_adapter_url": "gemma-2-9b-it-cnn_dailymail-adapter-v3/checkpoint-4500"
}


MERGE_ADAPTER_CONFIGS = {
    "base_model_url": "google/gemma-2-27b-it",
    "adapter_model_url": "/dev2/hf/gemma-2-27b-it-cnn_dailymail-adapter-final/checkpoint-676",
    "new_model_url": "/dev2/hf/gemma-2-27b-it-cnn_dailymail-finetuned-final",
    "device_map": "cpu",
    "torch_dtype" : torch.bfloat16,
    "attn_implementation" : "eager"
}

EVAL_CONFIGS = {
    "model_url": "google/gemma-2-27b-it",
    "dataset_name": "abisee/cnn_dailymail",
    "use_bnb_quant": True,
    "attn_implementation" : "eager",
    "torch_dtype" : torch.bfloat16,
    "num_dataset_samples": 100,
    "dataset_shuffle_seed": 65,
    "eval_batch_size": 1,
    "device_map": "auto",
    "device": "cuda",
    "eval_metric": "rouge"
}

MODEL_UPLOAD_CONFIG = {
    "model_url": "gemma-2-27b-it-cnn_dailymail-finetuned-final",
    # "model_url": "gemma-2-9b-it-cnn_dailymail-finetuned-final",
    "torch_dtype" : torch.bfloat16,
    "device_map": "auto"
}


OLLAMA_MODEL_CONFIG = {
    "model_name": "hf.co/raghul-asokan/gemma-2-9b-it-cnn_dailymail-finetuned-Q4_K_M-GGUF:latest"#"gemma2:latest"
}
