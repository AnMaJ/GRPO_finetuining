import json
import re

from datasets import load_dataset


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>\s*\{[\s\S]*?\}\s*</answer>\s*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    return [1.0 if re.match(pattern, content) else 0.0 for content in completion_contents]


def accuracy_reward(prompts, completions, **kwargs):
    """Reward function that checks if the model output matches the ground truth."""
    rewards = []

    conversations_list = kwargs.get("conversations", [])
    for completion, conversations in zip(completions, conversations_list):
        try:
            # Get ground truth from assistant response
            assistant_response = next(
                (msg["content"] for msg in conversations if msg["role"] == "assistant"),
                ""
            )

            # Get model output
            model_output = completion[0]["content"]
            json_match = re.search(r"</think>\s*<answer>\s*(\{[\s\S]*\})\s*</answer>\s*$", model_output, re.DOTALL)

            assistant_match = re.search(r"</think>\s*(\{[\s\S]*\})\s*$", assistant_response)

            if json_match:
                json_str = json_match.group(1)
            else:
                rewards.append(0.0)
                continue

            assistant_response = assistant_match.group(1)

            # Convert to JSON dictionaries for comparison
            parsed_solution = json.loads(assistant_response)
            parsed_output = json.loads(json_str)

            # Compare exactly (1.0) or calculate matching field ratio
            if parsed_solution == parsed_output:
                rewards.append(1.0)
            else:
                matching_keys = sum(
                    1 for key in parsed_solution if key in parsed_output and parsed_solution[key] == parsed_output[key])
                total_keys = len(parsed_solution)
                rewards.append(matching_keys / total_keys)
        except Exception:
            rewards.append(0.0)  # If JSON or format error, return 0

    return rewards

def make_conversation(example):
    """Prepare conversation data for training."""


    SYSTEM_PROMPT = """
            Respond in the following format:
            <think>
            Explain the steps taken to extract the requested information from the provided data. Mention any challenges, assumptions, or decisions made (e.g., date formatting, handling missing data). Ensure confidence in the extracted data and avoid guessing.
            </think>
            <answer>
            Provide the extracted information in JSON format based on the specified schema. Include only the fields listed in the schema and return null for any missing data.
            </answer>

            Task:
            Extract information from the provided document ONLY if you are confident. Do not guess or make up answers. Return the extracted data in JSON format according to the schema below. List the extracted information in the order it appears in the document.

            Output schema:
            {
                "id_number": "ID number of the person",
                "passport_number": "Passport number of the person",
                "full_name": "Full name of the person",
                "gender": "Gender of the person",
                "nationality": "Nationality of the person",
                "dob": "Date of birth of the person",
                "place": "Place of birth of the person",
                "place_issue": "Place of issue of the passport",
                "issue_date": "Date of issue of the passport",
                "expire_date": "Date of expiry of the passport",
                "mrz": "Machine Readable Zone"
            }
        """

    conversation = example["conversations"]

    user_content = next((msg["content"] for msg in conversation if msg["role"] == "user"), "")
    assistant_content = next((msg["content"] for msg in conversation if msg["role"] == "assistant"), "")

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response": assistant_content
    }

def get_dataset(args):
    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    train_dataset = train_dataset.map(make_conversation)
    test_dataset = test_dataset.map(make_conversation)
    return train_dataset, test_dataset
