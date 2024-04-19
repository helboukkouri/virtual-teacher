# Virtual Teacher

Attempt at creating a virtual assistant that can run locally and that could be used for learning various things, mostly languages as a first focus.

Currently the system uses:
- A large language model: `Llama-3 8B`
- A text-to-speech model: `xtts_v2`
- A speech-recognition model: `whisper large-v3`

## Installation

First, install all the dependencies by running the following commands:
```bash
conda create python=3.10 --name="virtual-teacher" -y
conda activate virtual-teacher
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

Then, you can check the available languages by running:
```bash
python main.py --list-languages
```

Finally, choose a language (e.g. `es` for Spanish) and run the following command:
```bash
# If you do not want to use speech recognition
python main.py --language=es

# If you want to use speech recognition
python main.py --language=es --use-speech-recognition
```

> Note: If you ever get an error when recording your voice, just try again. I've been experiencing some issues as well but it usually works the second time.
