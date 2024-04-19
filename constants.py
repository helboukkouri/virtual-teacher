import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.utils.manage import ModelManager
from TTS.utils.generic_utils import get_user_data_dir

TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
ModelManager().download_model(TTS_MODEL)

TTS_MODEL_PATH = os.path.join(get_user_data_dir("tts"), TTS_MODEL.replace("/", "--"))
TTS_MODEL_CONFIG = XttsConfig()
TTS_MODEL_CONFIG.load_json(os.path.join(TTS_MODEL_PATH, "config.json"))
SUPPORTED_LANGUAGES = TTS_MODEL_CONFIG.languages

SYSTEM_PROMPT = """
    You are a language teacher speaking both English and {language}.
    The student wants to learn {language}. Provide an English answer first, then a translation of the answer in {language}.
    
    Always generate your answers as a valid JSON with the LANGUAGE CODE as key the TEXT as value.
    The answers should be very short and simple.
"""
