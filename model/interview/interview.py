from yandex_neural_api.client import YandexNeuralAPIClient
from get_token import get_iam_token
import os
from dotenv import load_dotenv

load_dotenv()

FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
IAM_TOKEN = get_iam_token()
client = YandexNeuralAPIClient(iam_token=IAM_TOKEN, folder_id=FOLDER_ID)


def get_first_question(vacancy_text: str) -> str:
    prompt = f"""
Ты — рекрутер. Придумай первый вопрос кандидату на основе вакансии.

Вакансия:
{vacancy_text}

Ответь только вопросом.
"""
    response = client.generate_text(prompt=prompt)
    return response.strip()


def get_next_question(prev_question: str, answer: str, vacancy_text: str) -> str:
    prompt = f"""
Ты — рекрутер. На основе ответа кандидата и описания вакансии, задай следующий логичный вопрос.

Вакансия:
{vacancy_text}

Вопрос:
{prev_question}

Ответ кандидата:
{answer}

Следующий вопрос:
"""
    response = client.generate_text(prompt=prompt)
    return response.strip()


def generate_interview_summary(dialogue, vacancy_text: str) -> str:
    messages = []
    for q, a in dialogue:
        messages.append(f"Вопрос: {q}\nОтвет: {a}")
    dialogue_text = "\n\n".join(messages)

    prompt = f"""
Ты — HR-специалист. Проанализируй интервью кандидата и оцени его соответствие вакансии.

Вакансия:
{vacancy_text}

Интервью:
{dialogue_text}

Ответь по пунктам:
1. Краткий вывод по опыту
2. Краткий вывод по навыкам
3. Краткий вывод по мотивации
4. Подходит ли кандидат (Да/Нет)?
5. Общая оценка от 0 до 100
"""

    return client.generate_text(prompt=prompt)