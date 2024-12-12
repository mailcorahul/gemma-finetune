# gemma-finetune
A framework to finetune, evaluate and deploy Gemma-2 9B and 27B models.

## Table of Contents:

- [Introduction](#introduction)
- [Installation](#installation)
- [Supervised Finetuning](#supervised-finetuning)
- [Gemma-2 Evaluation on Text Summarization](#gemma-2-evaluation-on-text-summarization)
- [Quantization, Deployment and Gradio Demo](#quantization-deployment-and-gradio-demo)


## Introduction

**gemma-finetune** framework supports:
1. Supervised Finetuning of **Gemma-2 9B and 27B LLMs** on text data(supports text summarization data).
2. Evaluation of LLMs on Summarization data using **ROUGE and BERTScore metrics.**
3. **GGUF Quantization + ollama** inference for efficient deployment of LLMs.
4. Demo with **Gradio**(with ollama backend)

### Features:
1. **Modular** and **config driven** training and evaluation repository.
2. Integration with **Weights & Biases** to log training metadata and evaluation metrics.
3. Supports **LoRA** and **Q-LoRA** techniques for finetuning of 9B and 27B models.
4. **torch.bfloat16** support for memory efficient loading and finetuning of models.
5. Supports **distributed model training** and evaluation using HF accelerate.


## Installation

1. Create virtualenv python enviroment

```
virtualenv -p python3.8 "path to gemma environment"
pip install -r requirements.txt
```


## Supervised Finetuning

1. In **config.py**, under **TRAINING_CONFIGS**, configure the base Gemma model to be finetuned, path to the new adapter model to be trained, wandb project to use and get going.

```
TRAINING_CONFIGS["base_model_url"] = "google/gemma-2-9b-it" # or "google/gemma-2-27b-it"
TRAINING_CONFIGS["adapter_model_url"] = "path to adapater model save path"
TRAINING_CONFIGS["dataset_name"] = "abisee/cnn_dailymail" # change it to HF dataset of your choice
TRAINING_CONFIGS["wandb_project_name"] = "wandb_project_name"
```
Please go through the complete config parameters and update as needed.

2. To start **Q-LoRA adapter** training, run
```
python train.py
```

3. Once the adapter model is trained, merge the base gemma weights with the adapter weights. Use **MERGE_ADAPTER_CONFIGS** in config.py and set ```"device_map": "CPU"``` to load the weights on CPU(incase of GPU memory limit).
```
python merge_and_unload.py
```


## Gemma-2 Evaluation on Text Summarization


1. Update **EVAL_CONFIGS** in config.py and run,
```
python evaluate_llm.py
```


## Quantization, Deployment and Gradio Demo

1. To quantize(GGUF quants) the finetuned Gemma model(or any model on HuggingFace), you can make use of the following repository:
https://huggingface.co/spaces/ggml-org/gguf-my-repo

Select the necessary model id and set Quantization method to **Q4_K_M**

2. Once the model is quantized with the above HF repo, get the model url on HF for running with ollama. For example, hf.co/{your username}/{hf model id}. You will get this URL by selecting "Use this model -> Ollama from Local Apps" dropdown in your model page.

```
OLLAMA_MODEL_CONFIG["model_name"] = "the above HF model url"
python demo.py
```




