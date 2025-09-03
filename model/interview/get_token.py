import time
import json
import jwt
import requests
from dotenv import load_dotenv
import os

load_dotenv()

FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
KEY_FILE = "authorized_key.json"


def get_iam_token():
    with open(KEY_FILE, "r") as f:
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