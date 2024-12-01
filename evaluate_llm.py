import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import numpy as np
import evaluate

from datasets import load_dataset
from evaluate import evaluator

from secret_tokens import hf_token

from tqdm import tqdm


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

        print(target_batch)
        print(decoded_summaries[0])
        # decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
        # print(decoded_summaries[0])
        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    #  Finally compute and return the ROUGE scores.
    score = metric.compute(use_aggregator=True)
    return score


from huggingface_hub import login
#from kaggle_secrets import UserSecretsClient
#user_secrets = UserSecretsClient()

login(token = hf_token)

base_model_url = "Gemma-2-27b-it-cnn_dailymail-merged"
# base_model_url = "google/gemma-2-27b-it"

device = "auto"

if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"

attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

rouge = evaluate.load("rouge")
# task_evaluator = evaluator("text-classification")

# Load model
print(f"[/] model init, using {attn_implementation}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_url,
    quantization_config=bnb_config,
    device_map=device,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation
)

# pipe = pipeline(
#     "text-generation",
#     model=base_model_url,
#     quantization_config=bnb_config,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_url, trust_remote_code=True)

print("[/] model loaded")

# training_args = TrainingArguments(
#     "test_trainer",
#     remove_unused_columns=False
#     )

print("loading dataset")
#Importing the dataset
dataset_name = "abisee/cnn_dailymail"
dataset = load_dataset(dataset_name, "1.0.0", split="all")
dataset = dataset.shuffle(seed=65).select(range(100)) # Only use 1000 samples for quick demo

def format_chat_template(row):
    # row_json = [{"role": "system", "content": row["instruction"]},
    row_json = [
               {"role": "user", "content": row["article"]},
               {"role": "assistant", "content": row["highlights"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc= 4,
)


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     eval_dataset=dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )


# print("[/] evaluating...")
# trainer.evaluate()


score = calculate_metric_on_test_ds(
    dataset[0:10],
    rouge,
    model,
    tokenizer,
    batch_size=1,
    # device="cpu"
)

rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
print(score)
rouge_dict = dict((rn, score[rn]) for rn in rouge_names )
print("[/] rouge metric", rouge_dict)
