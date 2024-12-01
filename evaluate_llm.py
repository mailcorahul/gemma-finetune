import torch
import os
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import numpy as np
import evaluate

from model import init_model
from dataset import create_dataset

from datasets import load_dataset
from evaluate import evaluator

from secret_tokens import hf_token
from tqdm import tqdm

from huggingface_hub import login
login(token = hf_token)

from config import EVAL_CONFIGS

def generate_batch_sized_chunks(list_of_elements, batch_size):
    """
    split the dataset into smaller batches that we can process simultaneously
    Yield successive batch-sized chunks from list_of_elements.
    """
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]


def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                            batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu",
                            column_text="article",
                            column_summary="highlights"):
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches), total=len(article_batches)):

        inputs = tokenizer(article_batch, truncation=True,
                        padding="max_length", return_tensors="pt")

        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        max_new_tokens=800)
                        # length_penalty=0.8, num_beams=8)#, max_length=128)

        # Finally, we decode the generated texts,
        # replace the  token, and add the decoded texts with the references to the metric.
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
            for s in summaries]

        print("GT")
        print(target_batch)
        print("\nPRED")
        print(decoded_summaries[0])
        # decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
        # print(decoded_summaries[0])
        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    #  Finally compute and return the ROUGE scores.
    score = metric.compute(use_aggregator=True)
    return score

def format_chat_template(row):
    # row_json = [{"role": "system", "content": row["instruction"]},
    row_json = [
            {"role": "user", "content": row["article"]},
            {"role": "assistant", "content": row["highlights"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


if __name__ == "__main__":

    base_model_url = EVAL_CONFIGS["model_url"]
    device = EVAL_CONFIGS["device"]

    # load Gemma finetuned model
    model, tokenizer = init_model(EVAL_CONFIGS)

    # load dataset for evaluation
    dataset = create_dataset(EVAL_CONFIGS, tokenizer, split_dataset=False)

    rouge = evaluate.load(EVAL_CONFIGS["eval_metric"])
    score = calculate_metric_on_test_ds(
        dataset,
        rouge,
        model,
        tokenizer,
        batch_size=EVAL_CONFIGS["eval_batch_size"],
        device=device
    )

    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    print(score)
    rouge_dict = dict((rn, score[rn]) for rn in rouge_names )
    print("[/] rouge metric", rouge_dict)


    # dump metrics json
    eval_metric_path = os.path.join(base_model_url, "eval_metrics.json")
    with open(eval_metric_path, "w") as f:
        json.dump(rouge_dict, f, indent=2)