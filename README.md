# LLM Fine-Tuning

## Overview
This project focuses on fine-tuning the Qwen2.5 model using two methods:

1. Supervised Fine-Tuning (SFT): Aimed at improving the model's specific task performance using labeled data.

2. Generative Reward-based Policy Optimization (GRPO): Fine-tuning with reinforcement learning techniques to optimize task-specific generation quality.
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

## Fine-Tuning Methods
### 1. Supervised Fine-Tuning (SFT) with LlamaFactory
#### Training Script
```bash
llamafactory-cli train Llama-factory-config/qwen2.5_lora_sft_train.yaml
```
#### Merge LoRA Script
```bash
llamafactory-cli train Llama-factory-config/qwen2.5_lora_sft_merge_lora.yaml
```
#### Inference Script
```bash
llamafactory-cli train Llama-factory-config/qwen2.5_lora_sft_inference.yaml
```

### 2. Fine-Tuning with GRPO
To finetuning the model, use the bash script with the `train` mode:

```bash
./finetune_grpo_qwen.sh -m train
```

#### Advanced Fine-Tuning Options
```bash
./finetune_grpo_qwen.sh \
    -i "Qwen/Qwen2.5-1.5B-Instruct" \
    -d "/path/to/dataset.jsonl" \
    -o "output_model_directory" \
    -l 1e-5 \
    -e 3 \
    -b 16 \
    -m transformers
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

## Configuration File
The configuration file (YAML) supports:
- `model_name_or_path`: Model path or HuggingFace ID
- `template`: Prompt template (llama3, llama2, mistral, chatml, qwen, etc.)
- `trust_remote_code`: Whether to trust remote code
- `generation_config`: Inference generation parameters

## Supported Prompt Templates
- LLaMA3
- LLaMA2
- Mistral
- ChatML
- Qwen
- DeepSeek
- Zephyr

## Notes
- Ensure you have a compatible GPU with sufficient memory
- The script uses Wandb for experiment tracking
- The model is pushed to HuggingFace Hub after training

## Troubleshooting
- Check your dataset path
- Ensure all required libraries are installed
- Verify GPU compatibility