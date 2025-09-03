import os
import uuid
import wave
import requests
import subprocess
import sounddevice as sd
from scipy.io.wavfile import write
import pyaudio
from get_token import get_iam_token
from dotenv import load_dotenv
import soundfile as sf

load_dotenv()

FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
IAM_TOKEN = get_iam_token()

FFMPEG_PATH = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffmpeg.exe")


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
def listen(duration: int = 3) -> str:

    print("Говорите... (идёт запись)")
    fs = 16000
    wav_raw = "speech_input_raw.wav"
    wav_final = "speech_input.wav"

    try:
        # Запись в raw wav
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        sf.write(wav_raw, recording, fs, format='WAV', subtype='PCM_16')

        # Преобразуем в строгий формат WAV через ffmpeg
        subprocess.run([FFMPEG_PATH, "-y", "-i", wav_raw, "-ar", "16000", "-ac", "1", "-f", "wav", wav_final],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"Файл переконвертирован: {wav_final}, размер: {os.path.getsize(wav_final)} байт")

        stt_url = f"https://stt.api.cloud.yandex.net/speech/v1/stt:recognize?lang=ru-RU&folderId={FOLDER_ID}"
        headers = {"Authorization": f"Bearer {IAM_TOKEN}"}

        with open(wav_final, "rb") as f:
            files = {"audio": ("speech_input.wav", f, "audio/x-wav")}
            response = requests.post(stt_url, headers=headers, files=files)

        if response.status_code == 200:
            result = response.json()
            text = result.get("result", "")
            print(f"Вы сказали: {text}")
            return text
        else:
            print("Ошибка распознавания:", response.text)
            return ""

    except Exception as e:
        print(f"Ошибка записи или распознавания: {e}")
        return ""

    finally:
        for f in [wav_raw, wav_final]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                print(f"Не удалось удалить файл {f}: {e}")