import json
from voice import run_voice_interview
from get_token import get_iam_token
from dotenv import load_dotenv
import os

load_dotenv()

FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
IAM_TOKEN = get_iam_token()

# Загружаем вакансию из файла
with open("vacancy.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

# Преобразуем под формат системы
vacancy_data = {
    "title": raw.get("post", "Не указано"),
    "description": f"Компания {raw['company']['name']} ищет специалиста в область {raw.get('post', '')}. Работа: {raw.get('region', '')}.",
    "requirements": raw["responsibilities"]["description"] + raw.get("tags", [])
}

# Запускаем голосовое интервью
run_voice_interview(vacancy_data)