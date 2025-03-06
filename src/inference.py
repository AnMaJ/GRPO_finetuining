import os
import re
import sys
import json
import yaml
import torch
import logging
from typing import Dict, Any, Optional
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ModelInferenceHandler:
    """
    A comprehensive handler for model inference with support for various architectures
    """
    PROMPT_TEMPLATES = {
        # LLM Prompt Templates
        "llama3": lambda
            user_input: f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n",

        "llama2": lambda
            user_input: f"<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{user_input} [/INST]",

        "mistral": lambda user_input: f"<s>[INST] {user_input} [/INST]",

        "chatml": lambda
            user_input: f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n",

        "qwen": lambda
            user_input: f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n",

        "deepseek": lambda user_input: f"<|begin_of_text|>User: {user_input}\n\nAssistant:",

        "zephyr": lambda
            user_input: f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{user_input}</s>\n<|assistant|>\n"
    }

    def __init__(self, config_path: str):
        """
        Initialize the inference handler

        :param config_path: Path to the configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup model and tokenizer
        self.model, self.tokenizer, self.template = self._setup_model_and_tokenizer()

    def _setup_model_and_tokenizer(self):
        """
        Set up the model and tokenizer based on the configuration

        :return: Tuple of (model, tokenizer, template)
        """
        model_path = self.config.get("model_name_or_path")
        template = self.config.get("template", "llama3")
        trust_remote_code = self.config.get("trust_remote_code", True)

        logger.info(f"Loading model from: {model_path}")

        # Check for LoRA adapter
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            logger.info("LoRA adapter detected. Loading base model and adapter...")
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_path = peft_config.base_model_name_or_path

            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=trust_remote_code
            )

            # Load LoRA adapter
            model = PeftModel.from_pretrained(model, model_path)
        else:
            # Load full model directly
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=trust_remote_code
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path if os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else model_path,
            trust_remote_code=trust_remote_code
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer, template

    def generate_response(
        self,
        user_input: str,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response using the model

        :param user_input: Input prompt
        :param generation_config: Optional generation configuration
        :return: Generated response
        """
        # Select appropriate prompt template
        prompt_template_func = self.PROMPT_TEMPLATES.get(
            self.template,
            self.PROMPT_TEMPLATES["llama3"]
        )
        prompt = prompt_template_func(user_input)

        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Default generation config
        if generation_config is None:
            generation_config = {
                "max_new_tokens": 2048,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.1
            }

        # Set up streamer for real-time output
        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        logger.info("Generating response...")

        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(**generation_config),
                streamer=streamer
            )

        # Extract and return the generated response
        full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = full_response[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]

        return response

    def batch_process(
        self,
        label_folder: str,
        prompt: str,
        output_path: Optional[str] = None
    ) -> list:
        """
        Process multiple files in a batch
        :param label_folder: Folder containing input files
        :param prompt: Base prompt to use
        :param output_path: Optional path to save results
        :return: List of processing results
        """
        results = []
        logger.info(f"Processing files in: {label_folder}")
        for label_file in os.listdir(label_folder):
            if not label_file.endswith('.txt'):
                continue
            try:
                # Read input file
                with open(os.path.join(label_folder, label_file), 'r') as f:
                    data_text = f.read()
                # Combine prompt and file content
                user_input = f"{prompt}\n{data_text}"
                # Generate response
                response = self.generate_response(user_input)
                parts = re.split(r"<think>", response, maxsplit=1)
                if len(parts) < 2:
                    logger.warning(f"No <think> tag found. Raw response: {response}")
                    continue

                # Extract the content after </think> tag
                think_parts = re.split(r"</think>", parts[1], maxsplit=1)
                if len(think_parts) < 2:
                    # Try to find JSON in the thinking section
                    json_matches = re.findall(r"\{[\s\S]*?\}", parts[1].strip())
                    parsed_response = None

                    for json_str in reversed(json_matches):
                        try:
                            parsed_response = json.loads(json_str)
                            break  # Use the first valid JSON
                        except json.JSONDecodeError:
                            continue

                    if parsed_response is None:
                        logger.warning(f"JSON parsing failed. Raw response: {response}")
                        continue
                else:
                    # Process the content after </think>
                    after_think = think_parts[1].strip()
                    try:
                        parsed_response = json.loads(after_think)
                    except json.JSONDecodeError:
                        # Try to find a JSON object in the text
                        json_matches = re.findall(r"\{[\s\S]*?\}", after_think)
                        parsed_response = None

                        for json_str in json_matches:
                            try:
                                parsed_response = json.loads(json_str)
                                break
                            except json.JSONDecodeError:
                                continue

                        if parsed_response is None:
                            logger.warning(f"JSON parsing failed. Raw response: {response}")
                            continue

                # Prepare result
                result = {
                    "image_prefix": os.path.basename(label_file).replace('.txt', ''),
                    "assistant": parsed_response,
                }
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {label_file}: {e}")

        # Optionally save results
        if output_path:
            try:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)
                logger.info(f"Results saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save results: {e}")

        return results


def main():
    """Main function to run inference"""
    if len(sys.argv) < 4:
        print("Usage: python script.py <config_path> <label_folder> <prompt> [output_path]")
        sys.exit(1)

    config_path = sys.argv[1]
    label_folder = sys.argv[2]
    prompt = sys.argv[3]
    output_path = sys.argv[4] if len(
        sys.argv) > 4 else "./llm_base_results.json"

    # Initialize inference handler
    handler = ModelInferenceHandler(config_path)

    # Batch process files
    handler.batch_process(label_folder, prompt, output_path)


if __name__ == "__main__":
    main()