import torch
import langcodes
from transformers import AutoTokenizer, AutoModelForCausalLM

from constants import SYSTEM_PROMPT

import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


# Chat settings
MAX_NEW_TOKENS = 150
CHAT_HISTORY = []

# Limit LLM GPU memory usage to 10 GB
torch.cuda.empty_cache()
MAX_MEMORY = 10
TOTAL_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1024**3
MEMORY_LIMIT_FRACTION = MAX_MEMORY / TOTAL_MEMORY
print(f"Total GPU memory: {TOTAL_MEMORY:.2f} GB")
print(f"LLM GPU memory limit: {MEMORY_LIMIT_FRACTION * TOTAL_MEMORY:.2f} GB")
torch.cuda.set_per_process_memory_fraction(MEMORY_LIMIT_FRACTION)


# Load main LLM
LLM_MODEL_ID = "openchat/openchat-3.5-0106"
ASSISTANT_DELIMITER = 'GPT4 Correct Assistant:'
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, load_in_4bit=True)
model.eval()


def generate_response(text, language, system_prompt=None, keep_chat_history=False, debug=False, **generation_kwargs):
    """Generate a response given a user input."""
    global CHAT_HISTORY

    if not keep_chat_history:
        CHAT_HISTORY = []
    
    if not CHAT_HISTORY:
        language = langcodes.get(language).display_name()
        CHAT_HISTORY.append(
            {
                "role": "system",
                "content": system_prompt if system_prompt is not None else SYSTEM_PROMPT.format(language=language)
            },
        )

    CHAT_HISTORY.append({"role": "user", "content": '- ' + text})
    
    input_ids = tokenizer.apply_chat_template(CHAT_HISTORY, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if debug:
        print(tokenizer.apply_chat_template(CHAT_HISTORY, tokenize=False, add_generation_prompt=True))

    output = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
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
