from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch
from trl import setup_chat_format

from config import MERGE_ADAPTER_CONFIGS

base_model_url = MERGE_ADAPTER_CONFIGS["base_model_url"]
adapter_model_url = MERGE_ADAPTER_CONFIGS["adapter_model_url"]
save_dir = MERGE_ADAPTER_CONFIGS["new_model_url"]


print("[/] loading base model and tokenizer...")
# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_url)

base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model_url,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=MERGE_ADAPTER_CONFIGS["torch_dtype"],
    device_map=MERGE_ADAPTER_CONFIGS["device_map"],
    attn_implementation=MERGE_ADAPTER_CONFIGS["attn_implementation"]
)

print("[/] merging adapter weights with the base model...")
# base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
model = PeftModel.from_pretrained(base_model_reload, adapter_model_url)

model = model.merge_and_unload()

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
