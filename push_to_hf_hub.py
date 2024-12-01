import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from huggingface_hub import login
#from kaggle_secrets import UserSecretsClient
#user_secrets = UserSecretsClient()
from secret_tokens import hf_token
login(token = hf_token)

from config import MODEL_UPLOAD_CONFIG


if __name__ == "__main__":

    model_url = MODEL_UPLOAD_CONFIG["model_url"]
    torch_dtype = MODEL_UPLOAD_CONFIG["torch_dtype"]
    device_map = MODEL_UPLOAD_CONFIG["device_map"]

    print("[/] loading base model and tokenizer")
    # Reload tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_url)

    model = AutoModelForCausalLM.from_pretrained(
        model_url,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        # attn_implementation=attn_implementation
    )
    print("[/] model loaded.")

    print("[/] uploading model...")
    model.push_to_hub(model_url, use_temp_dir=False)
    tokenizer.push_to_hub(model_url, use_temp_dir=False)