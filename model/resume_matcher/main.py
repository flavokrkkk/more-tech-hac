#!/usr/bin/env python3
"""
Запуск LLM-фильтратора вакансий через YaGPT 5 Pro API.

Использование (минимум):
  python main.py --vacancies vacancies.json --resume resume.pdf --threshold 0.4 --api-key YOUR_API_KEY --folder-id YOUR_FOLDER_ID

Опции:
  --model yandexgpt      Модель YaGPT (yandexgpt, yandexgpt-lite)
  --output matched_ids.txt  Куда сохранить ID подходящих вакансий
  --json                 Вывести подробный JSON-отчет в консоль
"""

import argparse
import json
import os
import sys
from typing import Optional, List

# Импорт модуля с обработкой путей
import sys
import os

# Добавляем текущую директорию в путь Python
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Используем оптимизированную YaGPT-реализацию
    from optimized_matcher import OptimizedYaGPTMatcher
    from dotenv import load_dotenv
    load_dotenv()  # Загружаем переменные окружения из .env файла
except ImportError as e:
    print("Не найден optimized_matcher.py с оптимизированной YaGPT-логикой или python-dotenv")
    print(f"Ошибка: {e}")
    sys.exit(1)


DEFAULT_VACANCIES = os.path.join(current_dir, "vacancies.json")
DEFAULT_RESUME = os.path.join(current_dir, "Резюме_Frontend_разработчик_Егор_Яровицын_от_25_06_2025_09_51.pdf")
DEFAULT_OUTPUT_PATH = os.path.join(current_dir, "matched_ids.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-фильтрация вакансий через YaGPT 5 Pro API",
    )
    parser.add_argument(
        "--vacancies",
        type=str,
        default=DEFAULT_VACANCIES,
        help="Путь к JSON файлу с вакансиями",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=DEFAULT_RESUME,
        help="Путь к файлу резюме (PDF/DOCX/TXT)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Порог отбора (0..1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yandexgpt-lite",
        help="Модель YaGPT (yandexgpt, yandexgpt-lite)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("YANDEX_API_KEY"),
        help="API ключ Yandex Cloud (или установите переменную YANDEX_API_KEY)",
    )
    parser.add_argument(
        "--folder-id",
        type=str,
        default=os.getenv("YANDEX_FOLDER_ID"),
        help="ID папки Yandex Cloud (или установите переменную YANDEX_FOLDER_ID)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Путь к txt файлу для записи отсортированных ID вакансий",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести подробный JSON-отчет",
    )
    parser.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Отключить предварительную фильтрацию (медленнее, но точнее)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Размер батча для обработки (по умолчанию: 5)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Количество потоков для параллельной обработки (по умолчанию: 2)",
    )
    return parser


def save_ids_to_txt(ids: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for vid in ids:
            f.write(vid + "\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    
    # Валидация входов
    if not os.path.exists(args.vacancies):
        print(f"Файл с вакансиями не найден: {args.vacancies}")
        return 1
    if not os.path.exists(args.resume):
        print(f"Файл резюме не найден: {args.resume}")
        return 1
    
    # Проверяем наличие API ключа и folder_id
    if not args.api_key:
        print("Не указан API ключ. Используйте --api-key или установите переменную YANDEX_API_KEY")
        return 1
    if not args.folder_id:
        print("Не указан folder_id. Используйте --folder-id или установите переменную YANDEX_FOLDER_ID")
        return 1

    print(f"Инициализация оптимизированного YaGPT матчера (модель: {args.model})...")
    matcher = OptimizedYaGPTMatcher(
        api_key=args.api_key,
        folder_id=args.folder_id,
        model_name=args.model,
        max_workers=args.max_workers
    )

    print("Запуск оптимизированного LLM-анализа (YaGPT 5 Pro)...")
    results = matcher.filter_vacancies(
        resume_path=args.resume,
        vacancies_path=args.vacancies,
        threshold=args.threshold,
        use_prefilter=not args.no_prefilter,
        batch_size=args.batch_size
    )

    # Сохраняем только ID по убыванию score
    ids_sorted = [v.id for (v, _s, _r) in results]
    try:
        save_ids_to_txt(ids_sorted, args.output)
        print(f"Сохранены ID подходящих вакансий (по убыванию) в: {args.output}")
    except Exception as e:
        print(f"Не удалось сохранить ID: {e}")

    if args.json:
        details = [
            {
                "id": v.id,
                "score": float(f"{s:.6f}"),
                "reason": r,
            }
            for (v, s, r) in results
        ]
        print(json.dumps({"results": details}, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
