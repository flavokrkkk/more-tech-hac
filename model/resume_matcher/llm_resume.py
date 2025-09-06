#!/usr/bin/env python3
"""
Минимальная LLM-реализация сопоставления резюме и вакансий через Ollama.

Требования: ollama, markitdown[all]

Принципы:
- Чтение резюме (PDF/DOCX/TXT) → текст
- Чтение вакансий из JSON → склейка полей в текст по каждой вакансии
- Для каждой вакансии вызываем модель Ollama с просьбой вернуть оценку [0..1]
- Возвращаем отсортированный список (по убыванию score)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ollama  # type: ignore

try:
    from markitdown import MarkItDown  # type: ignore
except Exception:  # pragma: no cover
    MarkItDown = None  # позволим работать хотя бы с .txt


@dataclass
class Vacancy:
    id: str
    raw: Dict[str, Any]
    text: str


class LLMResumeMatcher:
    def __init__(self, model_name: str = "llama3.1:8b", temperature: float = 0.1, max_ctx_tokens: int = 4096) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_ctx_tokens = max_ctx_tokens

    # ---------- Резюме ----------
    def read_resume_to_text(self, path: str) -> str:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл резюме не найден: {path}")

        # Если доступен markitdown, используем его для PDF/DOCX
        if MarkItDown is not None and (path.lower().endswith(".pdf") or path.lower().endswith(".docx")):
            md = MarkItDown()
            result = md.convert(path)
            content = result.text_content or result.md  # type: ignore[attr-defined]
            return content.strip()

        # Фоллбек: как обычный текст
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    # ---------- Вакансии ----------
    def load_vacancies_from_json(self, path: str) -> List[Vacancy]:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл вакансий не найден: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vacancies: List[Vacancy] = []
        if isinstance(data, dict):
            items = data.get("items") or data.get("vacancies") or []
        else:
            items = data

        for item in items:
            vid = str(item.get("id", ""))
            text = self._compose_vacancy_text(item)
            if not vid:
                # если нет id, пропускаем
                continue
            vacancies.append(Vacancy(id=vid, raw=item, text=text))
        return vacancies

    def _compose_vacancy_text(self, v: Dict[str, Any]) -> str:
        parts: List[str] = []
        def add(x: Optional[str]):
            if x:
                parts.append(str(x))

        add(v.get("post") or v.get("title"))
        company = v.get("company") or {}
        add(company.get("name") if isinstance(company, dict) else None)
        add(v.get("region") or v.get("location"))
        add(v.get("salary"))

        def join_desc(block: Optional[Dict[str, Any]], label: str) -> None:
            if not isinstance(block, dict):
                return
            desc = block.get("description")
            if isinstance(desc, list):
                add(label + ": " + "; ".join([str(x) for x in desc]))
            elif isinstance(desc, str):
                add(label + ": " + desc)

        join_desc(v.get("requirements"), "Требования")
        join_desc(v.get("responsibilities"), "Обязанности")

        tags = v.get("tags")
        if isinstance(tags, list) and tags:
            add("Тэги: " + ", ".join([str(x) for x in tags]))

        extra = v.get("description")
        if isinstance(extra, str):
            add(extra)

        return "\n".join(parts)

    # ---------- LLM вызовы ----------
    def _build_prompts(self, resume_text: str, vacancy_text: str) -> Tuple[str, str]:
        system_prompt = (
            "Ты помощник HR. Оцени степень соответствия резюме вакансии числом от 0 до 1. "
            "0 — совсем не подходит, 1 — идеально. Учитывай навыки, опыт, стек, образование. "
            "Верни строго JSON вида {\"score\": <float>, \"reason\": <string>} без лишнего текста."
        )
        user_prompt = (
            f"ВАКАНСИЯ:\n{vacancy_text}\n\n"
            f"РЕЗЮМЕ:\n{resume_text}\n\n"
            "Ответь JSON."
        )
        return user_prompt, system_prompt

    def score_with_llm(self, resume_text: str, vacancy_text: str) -> Tuple[float, str]:
        user_prompt, system_prompt = self._build_prompts(resume_text, vacancy_text)

        try:
            # Предпочтительно chat с system+user
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.max_ctx_tokens,
                },
            )

            content = response.get("message", {}).get("content", "").strip()
            data = None
            try:
                data = json.loads(content)
            except Exception:
                # попробуем вытащить JSON из текста
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(content[start : end + 1])

            if not isinstance(data, dict):
                raise ValueError("Модель вернула не-JSON ответ")

            score = float(data.get("score", 0.0))
            reason = str(data.get("reason", ""))
            # зажмём диапазон
            score = max(0.0, min(1.0, score))
            return score, reason

        except Exception as e:  # pragma: no cover
            # В случае ошибки возвращаем нейтральный балл
            return 0.0, f"LLM error: {e}"

    # ---------- Основной метод ----------
    def filter_vacancies(
        self,
        resume_path: str,
        vacancies_path: str,
        threshold: float = 0.3,
    ) -> List[Tuple[Vacancy, float, str]]:
        """Возвращает список (vacancy, score, reason) по убыванию score, где score >= threshold."""
        resume_text = self.read_resume_to_text(resume_path)
        vacancies = self.load_vacancies_from_json(vacancies_path)

        results: List[Tuple[Vacancy, float, str]] = []
        for v in vacancies:
            score, reason = self.score_with_llm(resume_text, v.text)
            if score >= threshold:
                results.append((v, score, reason))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


