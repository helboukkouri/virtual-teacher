import argparse
import subprocess
from chatbot import generate_response
from text_to_speech import initialize_xtts_text_to_speech


def play_audio():
    subprocess.run(
        ["play", "output.wav"],
        capture_output=True,
        text=True,
        check=True,
    )


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="The language of the assistant",
    )
    args = argument_parser.parse_args()


    convert_text_to_speech = initialize_xtts_text_to_speech(language=args.language)
    def speak(text, language):
        convert_text_to_speech(text, language=language)
        play_audio()


    while True:
        user_input = input()
        response = generate_response(user_input, language=args.language)
        print(f"[[ Assistant ]]: {response}\n")
        speak(response, language=args.language)
