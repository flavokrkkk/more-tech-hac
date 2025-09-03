import os
import uuid
import wave
import requests
import subprocess
import sounddevice as sd
import pyaudio
from get_token import get_iam_token
from dotenv import load_dotenv
import vosk
import queue
import json

load_dotenv()

FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
IAM_TOKEN = get_iam_token()

FFMPEG_PATH = os.path.join(os.path.dirname(__file__), "interview", "ffmpeg", "bin", "ffmpeg.exe")
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "interview", "vosk-model-small-ru-0.22")

# загрузка модели
if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found in path: {VOSK_MODEL_PATH}")
model = vosk.Model(VOSK_MODEL_PATH)


# TTS
def speak(text: str):
    print("Бот говорит:", text)

    tts_url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    headers = {"Authorization": f"Bearer {IAM_TOKEN}"}
    data = {
        "text": text,
        "lang": "ru-RU",
        "voice": "ermil",
        "folderId": FOLDER_ID,
    }

    response = requests.post(tts_url, headers=headers, data=data)

    if response.status_code == 200:
        ogg_file = f"output_{uuid.uuid4()}.ogg"
        wav_file = ogg_file.replace(".ogg", ".wav")

        with open(ogg_file, "wb") as f:
            f.write(response.content)

        subprocess.run([FFMPEG_PATH, "-y", "-i", ogg_file, wav_file],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with wave.open(wav_file, 'rb') as wf:
            pa = pyaudio.PyAudio()
            stream = pa.open(format=pa.get_format_from_width(wf.getsampwidth()),
                             channels=wf.getnchannels(),
                             rate=wf.getframerate(),
                             output=True)

            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)

            stream.stop_stream()
            stream.close()
            pa.terminate()

        os.remove(ogg_file)
        os.remove(wav_file)

    else:
        print("Ошибка синтеза речи:", response.text)


# STT
def listen(duration: int = 4) -> str:
    print("Говорите... (идёт запись)")

    samplerate = 16000
    q = queue.Queue()

    def callback(indata, frames, time, status):
        q.put(bytes(indata))

    try:
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            rec = vosk.KaldiRecognizer(model, samplerate)
            for _ in range(int(samplerate / 8000 * duration)):
                data = q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    print("Распознанный текст:", result.get("text", ""))
                    return result.get("text", "")
            result = json.loads(rec.FinalResult())
            print("Распознанный текст:", result.get("text", ""))
            return result.get("text", "")
    except Exception as e:
        print("Ошибка распознавания:", e)
        return ""
