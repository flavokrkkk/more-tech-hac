import os
import json
import requests
import jwt
from dotenv import load_dotenv
import time

load_dotenv()

FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
IAM_TOKEN = os.getenv("YANDEX_IAM_TOKEN")
API_KEY = os.getenv("YANDEX_API_KEY")
KEY_FILE = "authorized_key.json"
API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


def get_iam_token(key_file: str = KEY_FILE) -> str:
    """Получает IAM токен из сервисного аккаунта."""
    with open(key_file, "r") as f:
        key_data = json.load(f)

    service_account_id = key_data["service_account_id"]
    key_id = key_data["id"]
    private_key = key_data["private_key"]

    now = int(time.time())
    payload = {
        "aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens",
        "iss": service_account_id,
        "iat": now,
        "exp": now + 360
    }

    encoded_jwt = jwt.encode(
        payload,
        private_key,
        algorithm="PS256",
        headers={"kid": key_id}
    )

    response = requests.post(
        "https://iam.api.cloud.yandex.net/iam/v1/tokens",
        json={"jwt": encoded_jwt}
    )

    if response.status_code == 200:
        return response.json()["iamToken"]
    else:
        raise RuntimeError(f"IAM token error: {response.text}")


class YaGPTClient:
    """Клиент для работы с YaGPT API."""
    
    def __init__(self, folder_id: str = None, api_key: str = None, iam_token: str = None, model: str = "yandexgpt"):
        """
        Инициализация клиента YaGPT.
        
        Args:
            folder_id: ID папки в Yandex Cloud
            api_key: API ключ (опционально)
            iam_token: IAM токен (опционально)
            model: Название модели (по умолчанию yandexgpt)
        """
        self.folder_id = folder_id or FOLDER_ID
        self.model = model
        
        # Приоритет: 1) переданный IAM токен, 2) IAM токен из .env, 3) IAM токен из файла, 4) API ключ
        if iam_token:
            print("🔑 Используется переданный IAM токен")
            self.token = iam_token
            self.use_iam = True
        elif IAM_TOKEN:
            print("🔑 Используется IAM токен из .env")
            self.token = IAM_TOKEN
            self.use_iam = True
        elif os.path.exists(KEY_FILE):
            print("🔑 Используется IAM токен из authorized_key.json")
            self.token = get_iam_token()
            self.use_iam = True
        else:
            print("🔑 Используется API ключ")
            self.token = api_key or API_KEY
            self.use_iam = False
            
        if not self.token:
            raise ValueError("Не указан API ключ или IAM токен")
        if not self.folder_id:
            raise ValueError("Не указан folder_id")

    def generate_text(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Генерирует текст с помощью YaGPT.
        
        Args:
            prompt: Текст запроса
            temperature: Температура генерации (0.0 - 1.0)
            
        Returns:
            str: Сгенерированный текст
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}" if self.use_iam else f"Api-Key {self.token}",
            "x-folder-id": self.folder_id
        }
        
        data = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": "2000"
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }
        
        # print(f"🤖 Отправка запроса к YaGPT...")  # Отладка отключена
        
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            if "result" in result and "alternatives" in result["result"] and result["result"]["alternatives"]:
                return result["result"]["alternatives"][0]["message"]["text"]
            else:
                print(f"❌ Неожиданный формат ответа: {result}")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка API: {e}")
            return None
