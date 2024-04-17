import torch
import langcodes
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


# Chat settings
SOURCE_LANGUAGE = "English"
CHAT_HISTORY = []

# Limit GPU memory usage to 10 GB
torch.cuda.empty_cache()
MAX_MEMORY = 10
TOTAL_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1024**3
MEMORY_LIMIT_FRACTION = MAX_MEMORY / TOTAL_MEMORY
print(f"Total GPU memory: {TOTAL_MEMORY:.2f} GB")
print(f"New GPU memory limit: {MEMORY_LIMIT_FRACTION * TOTAL_MEMORY:.2f} GB")
torch.cuda.set_per_process_memory_fraction(MEMORY_LIMIT_FRACTION)

# Load main LLM
LLM_MODEL_ID = "openchat/openchat-3.5-0106"
ASSISTANT_DELIMITER = 'GPT4 Correct Assistant:'
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, load_in_4bit=True)
model.eval()

def generate_response(text, language, debug=False, **generation_kwargs):
    """Generate a response given a user input."""
    if not CHAT_HISTORY:
        language = langcodes.get(language).display_name()
        CHAT_HISTORY.append(
            {
                "role": "system",
                "content": (
                    f"You are a friendly teacher who aims to help your student a new language."
                    f" It is extremely important that you always answer in {language} and only {language}."#, then its translation in {SOURCE_LANGUAGE}."
                    f" Your answers should be short and clear (under 100 tokens)."
                )
            },
        )
    CHAT_HISTORY.append({"role": "user", "content": text})
    input_ids = tokenizer.apply_chat_template(CHAT_HISTORY, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if debug:
        print(tokenizer.apply_chat_template(CHAT_HISTORY, tokenize=False, add_generation_prompt=True))

    output = model.generate(
        input_ids,
        max_new_tokens=150,
        temperature=0.1,
        **generation_kwargs
    )
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    output = output.split(ASSISTANT_DELIMITER)[-1].strip()
    CHAT_HISTORY.append({"role": "assistant", "content": output})
    return output


if __name__ == "__main__":
    user_queries = [
        "Hello, how are you?",
        "Can you help me learn French?",
        "How can I ask for directions in French?",
    ]

    print("\nThe chatbot is ready to chat!\n--->\n")
    for user_query in user_queries:
        response = generate_response(user_query)
        print(f"[[ User ]]: {user_query}")
        print(f"[[ Assistant ]]: {response}\n")
