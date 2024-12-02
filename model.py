from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import torch

from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)

from trl import setup_chat_format

import bitsandbytes as bnb


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def create_gemma_peft_model(args):

    attn_implementation = args["attn_implementation"]
    base_model = args["base_model_url"]
    torch_dtype = args["torch_dtype"]
    device_map = args["device_map"]

    # if torch.cuda.get_device_capability()[0] >= 8:
    #     torch_dtype = torch.bfloat16
    #     attn_implementation = "flash_attention_2"
    # else:
    #     torch_dtype = torch.float16
    #     attn_implementation = "eager"

    # bitsandbytes model load config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print("[/] model init...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation
    )
    print("[/] model loaded!")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("[/] finding all linear modules...")
    modules = find_all_linear_names(model)

    print("[/] getting peft model...")
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules
    )
    # model, tokenizer = setup_chat_format(model, tokenizer)
    model = get_peft_model(model, peft_config)
    return model, tokenizer, peft_config

def init_model(args):

    attn_implementation = args["attn_implementation"]
    base_model_url = args["model_url"]
    torch_dtype = args["torch_dtype"]
    device = args["device_map"]


    # bitsandbytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"[/] model init, using {attn_implementation}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_url,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_url, trust_remote_code=True)
    print("[/] model loaded!")

    return model, tokenizer
