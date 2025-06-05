# LLM Fine-Tuning

## Overview
This project focuses on fine-tuning the Qwen3 8B model for the ArgLLMSs++ project using Generative Reward-based Policy Optimization (GRPO): Fine-tuning with reinforcement learning techniques to optimize task-specific generation quality.

## Prerequisites
- Python 3.10
- PyTorch
- Transformers
- Datasets
- TRL (Transformer Reinforcement Learning)
- Wandb
- PEFT

## Installation
```bash
pip install torch transformers datasets wandb peft trl
```

## Usage

### Huggingface Login
```bash
huggingface-cli login
```

## Fine-Tuning 
### GRPO
To finetuning the model, use the bash script:

```bash
./finetune_grpo_qwen.sh -m train
```

### Script Parameters
- `-i, --model-id`: Base model or trained model ID
- `-d, --dataset`: Path to the dataset
- `-o, --output-dir`: Output directory for trained model
- `-l, --lr`: Learning rate
- `-e, --epochs`: Number of training epochs
- `-b, --batch-size`: Gradient accumulation steps

## Inference
Use the provided inference script for batch processing of input files and generating responses.
### Basic Inference
```bash
python inference_script.py <config_path> <label_folder> <prompt> [output_path]
```

### Parameters
1. `config_path`: Path to the YAML configuration file
2. `label_folder`: Folder containing input text files
3. `prompt`: Base prompt to use
4. `output_path` (optional): Path to save results JSON

### Example
```bash
python inference_script.py \
    config.yaml \
    /path/to/label/folder \
    "Extract information from the following passport:" \
    /path/to/output/results.json
```