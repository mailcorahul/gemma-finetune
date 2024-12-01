import torch

TRAINING_CONFIGS = {
    "base_model_url": "google/gemma-2-9b-it",
    "adapter_model_url": "gemma-2-9b-it-cnn_dailymail-adapter",
    "dataset_name": "abisee/cnn_dailymail",
    "wandb_project_name": "finetune gemma-2-9b-it",
    "attn_implementation" : "eager",
    "torch_dtype" : torch.bfloat16,
    "num_dataset_samples": 1000,
    "num_epochs": 1,
    "eval_steps": 5000,
    "dataset_shuffle_seed": 65,
    "test_size": 0.1,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "eval_steps": 5000,
    "device_map": "auto",
    "eval_metric": "rouge",
}


MERGE_ADAPTER_CONFIGS = {
    "base_model_url": "google/gemma-2-9b-it",
    "adapter_model_url": "gemma-2-9b-it-cnn_dailymail-adapter",
    "new_model_url": "gemma-2-9b-it-cnn_dailymail-finetuned",
    "device_map": "cpu",
    "torch_dtype" : torch.bfloat16,
    "attn_implementation" : "eager"
}