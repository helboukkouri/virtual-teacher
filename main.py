import re
import json
import argparse
import langcodes
import subprocess

from constants import SYSTEM_PROMPT, SUPPORTED_LANGUAGES


def play_audio():
    subprocess.run(
        ["play", "output.wav"],
        capture_output=True,
        text=True,
        check=True,
    )


# Thanks ChatGPT! :)
def find_json_objects(input_string):
    # Regular expression for matching JSON objects
    json_pattern = r'\{[^{}]*\}'
    json_objects = []

    while re.search(json_pattern, input_string):
        matches = re.finditer(json_pattern, input_string)
        for match in matches:
            substring = match.group()
            try:
                # Attempt to parse the substring as a JSON object
                json_obj = json.loads(substring)
                if isinstance(json_obj, dict):  # Ensure it's a dictionary
                    json_objects.append(json_obj)
                # Replace the parsed object with spaces to avoid re-matching
                input_string = input_string.replace(substring, ' ' * len(substring), 1)
            except json.JSONDecodeError:
                # Remove the problematic substring and continue
                input_string = input_string.replace(substring, ' ' * len(substring), 1)

    return json_objects


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--language",
        type=str,
        help="The target language for the assistant to translate to.",
    )
    argument_parser.add_argument(
        "--use-speech-recognition",
        action="store_true",
        help="Use speech recognition to transcribe user input.",
    )
    argument_parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List available languages (TTS limitation).",
    )
    args = argument_parser.parse_args()

    if args.list_languages:
        print("Supported languages:")
        for lang in SUPPORTED_LANGUAGES:
            print(f"- {lang}")
        exit(0)

    # Load this here to avoid loading models unless we plan to actually use them
    from chatbot import generate_response
    from text_to_speech import initialize_xtts_text_to_speech
    from speech_recognition import initialize_whisper_speech_recognition

    # Get the full name of the target language
    language_fullname = langcodes.get(args.language).display_name()

    # Initialize text-to-speech    
    convert_text_to_speech = initialize_xtts_text_to_speech()
    def speak(text, language=None):
        convert_text_to_speech(text, language=language)
        play_audio()

    if args.use_speech_recognition:
        # Initialize speech recognition
        transcribe_audio = initialize_whisper_speech_recognition()

    print()
    try:
        while True:
            if args.use_speech_recognition:
                user_input = transcribe_audio()
            else:
                user_input = input("Ask your teacher a question:\n")
            response = generate_response(
                user_input,
                language=args.language,
                keep_chat_history=False,
                system_prompt=SYSTEM_PROMPT.format(language=language_fullname),
                debug=False,
            )
            
            json_objects = find_json_objects(response)
            if not json_objects:
                print("The assistant's response does not contain valid JSONs.")
                print("Here is the response:")
                print(response)
                continue

            for obj in json_objects:                
                for language, text in obj.items():
                    if language not in SUPPORTED_LANGUAGES:
                        language = None
                    print(f"[[ Assistant ({language})]]: {text}\n")
                    speak(text, language=language)

    except KeyboardInterrupt:
        print("\nGoodbye!")
        exit(0)
