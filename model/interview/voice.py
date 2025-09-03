from interview import get_first_question, get_next_question, generate_interview_summary
from speech import speak, listen
from utils import format_vacancy_text
from get_token import get_iam_token
from dotenv import load_dotenv
import os

load_dotenv()

FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
IAM_TOKEN = get_iam_token()


def run_voice_interview(vacancy_data: dict):
    vacancy_text = format_vacancy_text(vacancy_data)
    dialogue = []
    max_questions = 15

    question = get_first_question(vacancy_text)

    while question and len(dialogue) < max_questions:
        speak(question)  # голосом задаём вопрос
        answer = listen()  # слушаем ответ

        dialogue.append((question, answer))
        question = get_next_question(question, answer, vacancy_text)

    speak('спасибо за уделённое время, мы завершаем интервью')
    summary = generate_interview_summary(dialogue, vacancy_text)
    print('\n===ОТЧЁТ===\n', summary)

    speak('вот краткий итог')
    speak(summary)
    return dialogue
