import os
import re
import time
import subprocess

import torch
import torchaudio
from transformers import AutoTokenizer, AutoModel

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

OUTPUT_FILE = "output.wav"
MAX_CHARS = float("inf")


def initialize_mms_text_to_speech():
    model = AutoModel.from_pretrained("facebook/mms-tts-eng").to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    
    def convert_text_to_speech(text, language):
        if language != "en":
            raise Exception(f"Unsupported language for MMS text-to-speech: {language}")
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model(**inputs).waveform
            torchaudio.save(OUTPUT_FILE, output.cpu(), sample_rate=model.config.sampling_rate)

    return convert_text_to_speech


# TODO: refactor this
def initialize_xtts_text_to_speech(language):
    
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    ModelManager().download_model(model_name)
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
    
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=os.path.join(model_path, "model.pth"),
        vocab_path=os.path.join(model_path, "vocab.json"),
        speaker_file_path=os.path.join(model_path, "speakers_xtts.pth"),
        eval=True,
        use_deepspeed=False,
    )
    _ = model.cuda()
    model.eval()
    
    supported_languages = config.languages

    def convert_text_to_speech(
        text,
        language=language,
        use_mic=False,
        voice_cleanup=True,
        no_lang_auto_detect=False,
        debug=False
    ):
        prompt = text
        if language not in supported_languages:
            raise Exception(
                f"Language you put {language} in is not in is not in our Supported Languages, please choose from dropdown."
            )
    
        language_predicted = language # langid.classify(prompt)[0].strip()  # strip need as there is space at end!
    
        # tts expects chinese as zh-cn
        if language_predicted == "zh":
            # we use zh-cn
            language_predicted = "zh-cn"
    
        if debug:
            print(f"Detected language:{language_predicted}, Chosen language:{language}")
    
        # After text character length 15 trigger language detection
        if len(prompt) > 15:
            # allow any language for short text as some may be common
            # If user unchecks language autodetection it will not trigger
            # You may remove this completely for own use
            if language_predicted != language and not no_lang_auto_detect:
                # Please duplicate and remove this check if you really want this
                # Or auto-detector fails to identify language (which it can on pretty short text or mixed text)
                raise Exception(
                    "It looks like your text isn’t the language you chose , if you’re sure the text is the same language you chose, please check disable language auto-detection checkbox"
                )
    
        if use_mic == True:
            raise Exception(
                "Please record your voice with Microphone, or uncheck Use Microphone to use reference audios"
            )
            speaker_wav = "custom.wav"
        else:
            speaker_wav = "./female.wav"
        if debug:
            print("Using speaker file:", speaker_wav)
    
        # Apply all on demand
        lowpassfilter = denoise = trim = loudness = True
    
        lowpass_highpass = "lowpass=8000,highpass=75,"
        trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02"
        
        if voice_cleanup:
            try:
                out_filename = "processed_speaker.wav" #speaker_wav + str(uuid.uuid4()) + ".wav"  # ffmpeg to know output format
    
                # we will use newer ffmpeg as that has afftn denoise filter
                shell_command = f"ffmpeg -y -i {speaker_wav} -af {lowpass_highpass}{trim_silence} {out_filename}".split(
                    " "
                )
    
                command_result = subprocess.run(
                    [item for item in shell_command],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                speaker_wav = out_filename
                if debug:
                    print("Filtered microphone input")
            except subprocess.CalledProcessError:
                # There was an error - command exited with non-zero code
                print("Error: failed filtering, using original microphone input")
        else:
            speaker_wav = speaker_wav
    
        if len(prompt) < 2:
            raise Exception("Please give a longer prompt text")
        if len(prompt) > MAX_CHARS:
            raise Exception(
                "Text length limited to 200 characters for this demo, please try shorter text. You can clone this space and edit code for your own usage"
            )
    
        metrics_text = ""
        t_latent = time.time()
    
        # note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
        try:
            (
                gpt_cond_latent,
                speaker_embedding,
            ) = model.get_conditioning_latents(audio_path=speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60)
        except Exception as e:
            print("Speaker encoding error", str(e))
            raise Exception(
                "It appears something wrong with reference, did you unmute your microphone?"
            )
    
        latent_calculation_time = time.time() - t_latent
        metrics_text += f"Embedding calculation time: {latent_calculation_time:.2f} seconds\n"
    
        # temporary comma fix
        prompt= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2\2",prompt)
    
        
        wav_chunks = []
        ## Direct mode
        if debug:
            print("I: Generating new audio...")
        t0 = time.time()
        out = model.inference(
            prompt,
            language,
            gpt_cond_latent,
            speaker_embedding,
            repetition_penalty=5.0,
            temperature=0.75,
        )
        inference_time = time.time() - t0
        if debug:
            print(f"I: Time to generate audio: {round(inference_time*1000)} milliseconds")
        metrics_text += f"Time to generate audio: {round(inference_time*1000)} milliseconds\n"
        real_time_factor= (time.time() - t0) / out['wav'].shape[-1] * 24000
        if debug:
            print(f"Real-time factor (RTF): {real_time_factor}")
        metrics_text += f"Real-time factor (RTF): {real_time_factor:.2f}\n"
        
        torchaudio.save(OUTPUT_FILE, torch.tensor(out["wav"]).unsqueeze(0), sample_rate=24000)

    return convert_text_to_speech