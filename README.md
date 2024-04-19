# Virtual Teacher

Attempt at creating a virtual assistant that can run locally and that could be used for learning various things, mostly languages as a first focus.

Currently the system uses:
- A large language model: `Llama-3 8B`
- A text-to-speech model: `xtts_v2`
- A speech-recognition model: `whisper large-v3`

## Installation

First, install all the dependencies by running the following commands:
```bash
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

Then, you can check the available languages by running:
```bash
python main.py --list-languages
```

Finally, choose a language (e.g. `fr` for French) and run the following command:
```bash
# If you do not want to use speech recognition
python main.py --language=fr

# If you want to use speech recognition
python main.py --language=fr --use-speech-recognition
```