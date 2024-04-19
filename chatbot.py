import torch
import langcodes
from transformers import AutoTokenizer, AutoModelForCausalLM

from constants import SYSTEM_PROMPT

import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


# Chat settings
MAX_NEW_TOKENS = 300
CHAT_HISTORY = []

# Limit LLM GPU memory usage to 10 GB
#torch.cuda.empty_cache()
#MAX_MEMORY = 10
#TOTAL_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1024**3
#MEMORY_LIMIT_FRACTION = MAX_MEMORY / TOTAL_MEMORY
#print(f"Total GPU memory: {TOTAL_MEMORY:.2f} GB")
#print(f"LLM GPU memory limit: {MEMORY_LIMIT_FRACTION * TOTAL_MEMORY:.2f} GB")
#torch.cuda.set_per_process_memory_fraction(MEMORY_LIMIT_FRACTION)

# Load main LLM
#LLM_MODEL = "OPENCHAT-3.5"
LLM_MODEL = "LLAMA-3"
if LLM_MODEL == "LLAMA-3":
    LLM_MODEL_ID = "meta-llama/Meta-LLAMA-3-8B-Instruct"
else:
    LLM_MODEL_ID = "openchat/openchat-3.5-0106"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, load_in_4bit=True)
model.eval()


def generate_response(text, language, system_prompt=None, keep_chat_history=False, debug=False, **generation_kwargs):
    """Generate a response given a user input."""
    torch.cuda.empty_cache()

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

    terminators = [tokenizer.eos_token_id]
    if LLM_MODEL == "LLAMA-3":
        terminators += [tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
        **generation_kwargs
    )
    response = outputs[0][input_ids.shape[-1]:]
    output = tokenizer.decode(response, skip_special_tokens=True)

    CHAT_HISTORY.append({"role": "assistant", "content": output})
    return output
