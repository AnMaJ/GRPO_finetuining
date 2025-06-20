#!/usr/bin/env python3

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import argparse
import json
import re

import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from util import format_reward, accuracy_reward, get_dataset

def train_transformers_model(args):
    """Train the model using GRPO with transformers/PEFT method."""
    # Initialize wandb
    wandb.init(project="grpo")

    # Load dataset
    train_dataset, test_dataset = get_dataset(args)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Configure training arguments
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.batch_size,
        num_train_epochs=args.epochs,
        bf16=True,
        max_completion_length=2048,
        num_generations=4,
        max_prompt_length=2048,
        report_to=["tensorboard"],
        logging_steps=10,
        push_to_hub=True,
        save_strategy="steps",
        save_steps=10,
    )

    # Initialize and train GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.push_to_hub(dataset_name="passport_en_grpo")


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Passport Data Extraction Training and Inference")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-8B,
                        help="Base model ID to use for training or inference")
    parser.add_argument("--dataset", type=str, default="./passport_en_grpo.jsonl",
                        help="Path to the dataset")
    parser.add_argument("--output-dir", type=str, default="Qwen3-8B-GRPO",
                        help="Output directory for trained model")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size/gradient accumulation steps for training")
    parser.add_argument("--method", type=str, default="transformers",
                        help="Training method to use: 'transformers'")

    args = parser.parse_args()

    # Training the model
    train_transformers_model(args)

if __name__ == "__main__":
    main()