import argparse
import langcodes
import subprocess
from chatbot import generate_response
from text_to_speech import initialize_xtts_text_to_speech
from constants import SYSTEM_PROMPT

convert_text_to_speech = initialize_xtts_text_to_speech()

def play_audio():
    subprocess.run(
        ["play", "output.wav"],
        capture_output=True,
        text=True,
        check=True,
    )

def speak(text, language=None):
    convert_text_to_speech(text, language=language)
    play_audio()


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--language",
        type=str,
        help="The target language for the assistant to translate to.",
    )
    args = argument_parser.parse_args()
    language_fullname = langcodes.get(args.language).display_name()
    

    while True:
        user_input = input()
        response = generate_response(
            user_input,
            language=args.language,
            keep_chat_history=False,
            system_prompt=SYSTEM_PROMPT.format(language=language_fullname),
            debug=False,
        )
        sentences = (tuple(filter(lambda x: bool(x.strip()), response.split('\n'))))
        if len(sentences) == 1:
            sentences = (sentences[0], sentences[0])
        elif len(sentences) > 2:
            sentences = (sentences[0], sentences[1])
            print("Warning: The assistant's response contains more than 2 sentences. Only the first 2 sentences will be spoken.")
        else:
            print("Weird: The assistant's response contains 0 sentences.")

        print(f"[[ Assistant ]]: {response}\n")
        speak(sentences[0])#, language="en")
        speak(sentences[1])#, language=args.language)
