import queue
import sys

import sounddevice as sd
import soundfile as sf

from faster_whisper import WhisperModel

USER_QUERY_AUDIO_FILE = "user_query.wav"

def record_audio():
    channels = 1    

    device = None
    device_info = sd.query_devices(device, 'input')
    samplerate = int(device_info['default_samplerate'])

    q = queue.Queue()
    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    try:
        input('Press a key to start recording..')
        # Make sure the file is opened before recording anything:
        with sf.SoundFile(USER_QUERY_AUDIO_FILE, mode='w', samplerate=samplerate, channels=channels) as file:
            with sd.InputStream(samplerate=samplerate, device=device, channels=channels, callback=callback):
                print("[[ Recording started ]]")
                while True:
                    file.write(q.get())

    except KeyboardInterrupt:
        pass


def initialize_whisper_speech_recognition():
    model_size = "large-v3"

    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe_audio():
        record_audio()
        segments, info = model.transcribe(USER_QUERY_AUDIO_FILE, beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        transcribed_text = "\n".join([segment.text for segment in segments])
        return transcribed_text

    return transcribe_audio


if __name__ == "__main__":
    record_audio()