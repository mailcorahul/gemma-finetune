import torch

TRAINING_CONFIGS = {
    "base_model_url": "google/gemma-2-9b-it",
    "adapter_model_url": "gemma-2-9b-it-cnn_dailymail-adapter-v3",
    "dataset_name": "abisee/cnn_dailymail",
    "wandb_project_name": "finetune gemma-2-9b-it",
    "attn_implementation" : "eager",
    "torch_dtype" : torch.bfloat16,
    "num_dataset_samples": 100000,
    "num_epochs": 1,
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
    "base_model_url": "google/gemma-2-9b-it",
    "adapter_model_url": "gemma-2-9b-it-cnn_dailymail-adapter",
    "new_model_url": "gemma-2-9b-it-cnn_dailymail-finetuned",
    "device_map": "cpu",
    "torch_dtype" : torch.bfloat16,
    "attn_implementation" : "eager"
}

EVAL_CONFIGS = {
    "model_url": "google/gemma-2-9b-it",
    "dataset_name": "abisee/cnn_dailymail",
    "attn_implementation" : "eager",
    "torch_dtype" : torch.bfloat16,
    "num_dataset_samples": 10,
    "dataset_shuffle_seed": 65,
    "eval_batch_size": 1,
    "device_map": "auto",
    "device": "cuda",
    "eval_metric": "rouge"
}

MODEL_UPLOAD_CONFIG = {
    "model_url": "gemma-2-9b-it-cnn_dailymail-finetuned",
    "torch_dtype" : torch.bfloat16,
    "device_map": "auto"
}


OLLAMA_MODEL_CONFIG = {
    "model_name": "hf.co/raghul-asokan/gemma-2-9b-it-cnn_dailymail-finetuned-Q4_K_M-GGUF:latest"#"gemma2:latest"
}