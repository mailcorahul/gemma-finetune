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


def calculate_metric_on_test_ds(dataset, rouge_metric, bert_metric, model, tokenizer,
                            batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu",
                            column_text="article",
                            column_summary="highlights"):
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches), total=len(article_batches)):

        num_words = 50
        for idx, article in enumerate(article_batch):
            article_batch[idx] = "Generate a short summary for the following text:\n" + article

        inputs = tokenizer(article_batch, truncation=True,
                        padding="max_length", return_tensors="pt")
        input_size = inputs["input_ids"].size()[1]

        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        max_new_tokens=len(target_batch[0]))
                        # length_penalty=0.8, num_beams=8)#, max_length=128)

        # Finally, we decode the generated texts,
        # replace the  token, and add the decoded texts with the references to the metric.
        decoded_summaries = [tokenizer.decode(summary[input_size:], skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
            for summary in summaries]

        print("GT")
        print(target_batch)
        print("\nPRED")
        print(decoded_summaries[0])
        # decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
        # print(decoded_summaries[0])
        rouge_metric.add_batch(predictions=decoded_summaries, references=target_batch)
        bert_metric.add_batch(predictions=decoded_summaries, references=target_batch)

    #  Finally compute and return the ROUGE and BERT scores.
    rouge_score = rouge_metric.compute(use_aggregator=True)
    bert_score = bert_metric.compute(model_type="distilbert-base-uncased")
    return rouge_score, bert_score

def format_chat_template(row):
    # row_json = [{"role": "system", "content": row["instruction"]},
    words = row["highlights"].split(" ")
    num_words = 50#len(words)

    row_json = [
        {"role": "user", "content": f"Generate a summary(of not more than {num_words} words) for the following text:\n" + row["article"]},
        {"role": "assistant", "content": "Summary:\n" + row["highlights"]}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


if __name__ == "__main__":

    base_model_url = EVAL_CONFIGS["model_url"]
    device = EVAL_CONFIGS["device"]

    # load Gemma finetuned model
    model, tokenizer = init_model(EVAL_CONFIGS, use_bnb_quant=EVAL_CONFIGS["use_bnb_quant"])

    # load dataset for evaluation
    dataset = create_dataset(EVAL_CONFIGS, tokenizer, split_dataset=False)

    rouge = evaluate.load(EVAL_CONFIGS["eval_metric"])
    bertscore_metric = evaluate.load("bertscore")

    rouge_score, bert_score = calculate_metric_on_test_ds(
        dataset,
        rouge,
        bertscore_metric,
        model,
        tokenizer,
        batch_size=EVAL_CONFIGS["eval_batch_size"],
        device=device,
        column_text="article"
    )

    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    # print(rouge_score)
    rouge_dict = dict((rn, rouge_score[rn]) for rn in rouge_names )
    print("[/] rouge metric", rouge_dict)

    # print(bert_score)
    bert_mean_score = {}
    bert_mean_score["precision"] = float(np.mean(bert_score["precision"])) if len(bert_score["precision"]) > 0 else 0.
    bert_mean_score["recall"] = float(np.mean(bert_score["recall"])) if len(bert_score["recall"]) > 0 else 0.
    bert_mean_score["f1"] = float(np.mean(bert_score["f1"])) if len(bert_score["f1"]) > 0 else 0.

    print("[/] bert score", bert_mean_score)

    # dump metrics json
    os.makedirs(base_model_url, exist_ok=True)
    metrics = {
        "rouge": rouge_dict,
        "bertscore": bert_mean_score
    }

    eval_metric_path = os.path.join(base_model_url, "eval_metrics.json")
    with open(eval_metric_path, "w") as f:
        json.dump(metrics, f, indent=2)