from transformers import (
    TrainingArguments,
    pipeline,
    logging,
)

import os, torch
import wandb
import numpy as np

from model import create_gemma_peft_model
from dataset import create_dataset

import evaluate
from trl import SFTTrainer, setup_chat_format
from config import TRAINING_CONFIGS

from huggingface_hub import login
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()

from secret_tokens import hf_token, wb_token
login(token = hf_token)

# wb_token = user_secrets.get_secret("wandb")

wandb.login(key=wb_token)
run = wandb.init(
    project=TRAINING_CONFIGS["wandb_project_name"],
    job_type="training",
    anonymous="allow"
)

rouge = evaluate.load(TRAINING_CONFIGS["eval_metric"])

def compute_metrics(eval_pred):

    # print(dir(eval_pred))

    # predictions, labels = eval_pred
    predictions = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids

    # print(predictions)
    # print(labels)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


if __name__ == "__main__":
    # create Gemma PEFT model for finetuning
    model, tokenizer, peft_config = create_gemma_peft_model(TRAINING_CONFIGS)

    if TRAINING_CONFIGS["resume"]:
        print("[/] loading adapter weights to resume finetuning...")
        model.load_adapter(TRAINING_CONFIGS["pretrained_adapter_url"], adapter_name="lora_adapter")

    # create dataset for finetuning
    dataset = create_dataset(TRAINING_CONFIGS, tokenizer)

    # get training params
    adapter_model_url = TRAINING_CONFIGS["adapter_model_url"]

    print("[/] finetuning...")
    # Setting Hyperparamter
    training_arguments = TrainingArguments(
        output_dir=adapter_model_url,
        per_device_train_batch_size=TRAINING_CONFIGS["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIGS["per_device_eval_batch_size"],
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=TRAINING_CONFIGS["num_epochs"],
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIGS["eval_steps"],
        logging_steps=1,
        save_steps=0.05,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        eval_accumulation_steps=1,
        report_to="wandb"
    )
    # Setting sft parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        max_seq_length= 512,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        # predict_with_generate=True,
        compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        packing=False,
    )

    model.config.use_cache = False
    trainer.train()

    wandb.finish()
    model.config.use_cache = True

    trainer.model.save_pretrained(adapter_model_url)
    # trainer.model.push_to_hub(adapter_model_url, use_temp_dir=False)
