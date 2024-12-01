from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch
from trl import setup_chat_format

base_model_url = "google/gemma-2-27b-it"
new_model_url = "Gemma-2-27b-it-cnn_dailymail"
save_dir = "Gemma-2-27b-it-cnn_dailymail-merged"

print("[/] loading base model and tokenizer")
# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_url)

base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model_url,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    attn_implementation="eager"
)

print("[/] merging adapter weights with the base model...")
# base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
model = PeftModel.from_pretrained(base_model_reload, new_model_url)

model = model.merge_and_unload()

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
