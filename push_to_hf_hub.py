import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from huggingface_hub import login
#from kaggle_secrets import UserSecretsClient
#user_secrets = UserSecretsClient()
from secret_tokens import hf_token
login(token = hf_token)


base_model_url = "Gemma-2-9b-it-chat-doctor-merged"

print("[/] loading base model and tokenizer")
# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_url)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_url,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
    # attn_implementation=attn_implementation
)
print("[/] model loaded.")

print("[/] uploading model...")
# base_model.push_to_hub("Gemma-2-9b-it-chat-doctor-merged", use_temp_dir=False)
tokenizer.push_to_hub("Gemma-2-9b-it-chat-doctor-merged", use_temp_dir=False)