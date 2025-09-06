#!/usr/bin/env python3
"""
Оптимизированный YaGPT-фильтратор вакансий с кэшированием и многопоточностью.

Улучшения:
1. Кэширование результатов анализа
2. Многопоточная обработка вакансий
3. Предварительная фильтрация по ключевым словам
4. Батчевая обработка
5. Улучшенные промпты для повышения качества
"""

import json
import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, asdict
from markitdown import MarkItDown
from yagpt_client import YaGPTClient
import re
from rapidfuzz import fuzz

@dataclass
class Vacancy:
    id: str
    title: str
    description: str
    requirements: str
    salary: str = ""
    location: str = ""
    company: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class AnalysisResult:
    score: float
    reasoning: str
    keywords_match: float
    title_match: float

class OptimizedYaGPTMatcher:
    """Оптимизированный класс для подбора вакансий с помощью YaGPT."""
    
    def __init__(self, api_key: str = None, folder_id: str = None, model_name: str = "yandexgpt", 
                 cache_file: str = "matcher_cache.pkl", max_workers: int = 3):
        """
        Инициализация оптимизированного матчера.
        
        Args:
            api_key: API ключ для Yandex Cloud
            folder_id: ID папки в Yandex Cloud
            model_name: Название модели
            cache_file: Файл для кэширования результатов
            max_workers: Количество потоков для параллельной обработки
        """
        self.client = YaGPTClient(folder_id=folder_id, api_key=api_key, model=model_name)
        self.markitdown = MarkItDown()
        self.cache_file = cache_file
        self.max_workers = max_workers
        self.cache = self._load_cache()
        
        # Ключевые слова для предварительной фильтрации
        self.tech_keywords = {
            'frontend': ['frontend', 'фронтенд', 'react', 'vue', 'angular', 'javascript', 'typescript', 'html', 'css'],
            'backend': ['backend', 'бэкенд', 'python', 'django', 'flask', 'fastapi', 'node.js', 'java', 'spring'],
            'mobile': ['mobile', 'мобильная', 'ios', 'android', 'react native', 'flutter', 'swift', 'kotlin'],
            'devops': ['devops', 'docker', 'kubernetes', 'ci/cd', 'aws', 'azure', 'terraform'],
            'data': ['data', 'данных', 'machine learning', 'ml', 'ai', 'python', 'pandas', 'numpy', 'sql'],
            'qa': ['qa', 'тестирование', 'тестировщик', 'автотесты', 'selenium', 'postman']
        }
    
    def _load_cache(self) -> Dict:
        """Возвращает пустой кэш (в памяти)."""
        return {}
    
    def _save_cache(self):
        """Ничего не делает (кэш только в памяти)."""
        return
    
    def _get_cache_key(self, resume_text: str, vacancy_id: str) -> str:
        """Создает ключ для кэширования."""
        content = f"{resume_text[:500]}_{vacancy_id}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_text_from_resume(self, resume_path: str) -> str:
        """Извлекает текст из резюме."""
        try:
            if not os.path.exists(resume_path):
                raise FileNotFoundError(f"Файл резюме не найден: {resume_path}")
                
            result = self.markitdown.convert(resume_path)
            text = result.text_content.strip()
            
            if not text:
                raise ValueError("Не удалось извлечь текст из резюме")
                
            return text
            
        except Exception as e:
            print(f"Ошибка при извлечении текста из резюме: {e}")
            raise
    
    def _load_vacancies(self, vacancies_path: str) -> List[Vacancy]:
        """Загружает вакансии из JSON файла."""
        try:
            with open(vacancies_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            vacancies = []
            for item in data:
                requirements = "; ".join(item.get("requirements", {}).get("description", []))
                description = "; ".join(item.get("responsibilities", {}).get("description", []))
                tags = item.get("tags", [])
                
                vacancy = Vacancy(
                    id=str(item.get("id", "")),
                    title=item.get("post", ""),
                    description=description,
                    requirements=requirements,
                    salary=item.get("salary", ""),
                    location=item.get("region", ""),
                    company=item.get("company", {}).get("name", ""),
                    tags=tags
                )
                vacancies.append(vacancy)
                
            print(f"Загружено {len(vacancies)} вакансий")
            return vacancies
            
        except Exception as e:
            print(f"Ошибка при загрузке вакансий: {e}")
            raise
    
    def _extract_skills_from_text(self, text: str) -> Set[str]:
        """Извлекает технические навыки из текста."""
        text_lower = text.lower()
        skills = set()
        
        for category, keywords in self.tech_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    skills.add(keyword)
        
        return skills
    
    def _calculate_keyword_match(self, resume_text: str, vacancy: Vacancy) -> float:
        """Вычисляет совпадение по ключевым словам."""
        resume_skills = self._extract_skills_from_text(resume_text)
        vacancy_text = f"{vacancy.title} {vacancy.description} {vacancy.requirements} {' '.join(vacancy.tags)}"
        vacancy_skills = self._extract_skills_from_text(vacancy_text)
        
        if not resume_skills or not vacancy_skills:
            return 0.0
        
        intersection = resume_skills.intersection(vacancy_skills)
        union = resume_skills.union(vacancy_skills)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_title_match(self, resume_text: str, vacancy_title: str) -> float:
        """Вычисляет совпадение с названием вакансии."""
        return fuzz.partial_ratio(resume_text.lower(), vacancy_title.lower()) / 100.0
    
    def _prefilter_vacancies(self, resume_text: str, vacancies: List[Vacancy], min_score: float = 0.1) -> List[Vacancy]:
        """Предварительная фильтрация вакансий по ключевым словам."""
        filtered = []
        
        for vacancy in vacancies:
            keyword_score = self._calculate_keyword_match(resume_text, vacancy)
            title_score = self._calculate_title_match(resume_text, vacancy.title)
            
            # Комбинированная оценка для предварительной фильтрации
            combined_score = keyword_score * 0.7 + title_score * 0.3
            
            if combined_score >= min_score:
                filtered.append(vacancy)
        
        print(f"Предварительная фильтрация: {len(filtered)} из {len(vacancies)} вакансий")
        return filtered
    
    def _create_enhanced_prompt(self, resume_text: str, vacancy: Vacancy) -> str:
        """Создает улучшенный промпт для анализа."""
        
        # Ограничиваем длину текстов
        resume_summary = resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text
        
        vacancy_info = f"""Название: {vacancy.title}
Компания: {vacancy.company}
Требования: {vacancy.requirements[:300]}...
Обязанности: {vacancy.description[:300]}...
Теги: {', '.join(vacancy.tags[:5])}"""
        
        prompt = f"""Ты эксперт-рекрутер. Оцени точно, насколько кандидат подходит для вакансии.

РЕЗЮМЕ КАНДИДАТА:
{resume_summary}

ВАКАНСИЯ:
{vacancy_info}

Критерии оценки:
1. Соответствие технических навыков (40%)
2. Релевантный опыт работы (30%)
3. Уровень сениорности (20%)
4. Дополнительные навыки (10%)

Формат ответа (строго соблюдай):
Оценка: [число от 0 до 100]
Анализ: [краткое обоснование в 1-2 предложениях]"""

        return prompt
    
    def _analyze_vacancy_match(self, resume_text: str, vacancy: Vacancy) -> AnalysisResult:
        """Анализирует соответствие вакансии резюме с помощью YaGPT."""
        
        # Проверяем кэш
        cache_key = self._get_cache_key(resume_text, vacancy.id)
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            return AnalysisResult(**cached_result)
        
        # Вычисляем метрики
        keyword_match = self._calculate_keyword_match(resume_text, vacancy)
        title_match = self._calculate_title_match(resume_text, vacancy.title)
        
        # Если низкое совпадение по ключевым словам, не тратим API запрос
        if keyword_match < 0.05 and title_match < 0.1:
            result = AnalysisResult(
                score=0.0,
                reasoning="Низкое совпадение по ключевым навыкам",
                keywords_match=keyword_match,
                title_match=title_match
            )
            self.cache[cache_key] = asdict(result)
            return result
        
        prompt = self._create_enhanced_prompt(resume_text, vacancy)
        response = self.client.generate_text(prompt, temperature=0.1)
        
        if not response:
            result = AnalysisResult(
                score=0.0,
                reasoning="Ошибка при получении ответа от API",
                keywords_match=keyword_match,
                title_match=title_match
            )
            self.cache[cache_key] = asdict(result)
            return result
        
        try:
            # Парсим ответ
            lines = response.strip().split('\n')
            score = 0.0
            reasoning = "Не удалось извлечь обоснование"
            
            for line in lines:
                line = line.strip()
                if line.startswith("Оценка:"):
                    score_text = re.search(r'(\d+)', line.replace("Оценка:", "").strip())
                    if score_text:
                        score = float(score_text.group(1)) / 100.0
                elif line.startswith("Анализ:"):
                    reasoning = line.replace("Анализ:", "").strip()
            
            # Применяем бонусы за совпадения
            if keyword_match > 0.3:
                score = min(1.0, score + 0.1)  # Бонус за хорошее совпадение навыков
            
            score = max(0.0, min(1.0, score))
            
            result = AnalysisResult(
                score=score,
                reasoning=reasoning,
                keywords_match=keyword_match,
                title_match=title_match
            )
            
            # Сохраняем в кэш
            self.cache[cache_key] = asdict(result)
            
            return result
            
        except Exception as e:
            print(f"Ошибка при парсинге ответа YaGPT: {e}")
            result = AnalysisResult(
                score=0.0,
                reasoning=f"Ошибка парсинга: {str(e)}",
                keywords_match=keyword_match,
                title_match=title_match
            )
            self.cache[cache_key] = asdict(result)
            return result
    
    def _process_vacancy_batch(self, resume_text: str, vacancies: List[Vacancy], threshold: float) -> List[Tuple[Vacancy, AnalysisResult]]:
        """Обрабатывает батч вакансий в отдельном потоке."""
        results = []
        
        for vacancy in vacancies:
            try:
                analysis = self._analyze_vacancy_match(resume_text, vacancy)
                
                if analysis.score >= threshold:
                    results.append((vacancy, analysis))
                    print(f"{vacancy.title} - {analysis.score:.3f}")
                else:
                    print(f"{vacancy.title} - {analysis.score:.3f}")
                
                # Задержка между запросами
                time.sleep(0.3)
                
            except Exception as e:
                print(f"Ошибка при анализе вакансии {vacancy.id}: {e}")
                continue
        
        return results
    
    def filter_vacancies(
        self, 
        resume_path: str, 
        vacancies_path: str, 
        threshold: float = 0.3,
        use_prefilter: bool = True,
        batch_size: int = 10
    ) -> List[Tuple[Vacancy, float, str]]:
        """
        Фильтрует вакансии по резюме с оптимизациями.
        
        Args:
            resume_path: Путь к файлу резюме
            vacancies_path: Путь к JSON файлу с вакансиями
            threshold: Минимальный порог соответствия (0-1)
            use_prefilter: Использовать предварительную фильтрацию
            batch_size: Размер батча для обработки
            
        Returns:
            Список кортежей (вакансия, оценка, обоснование), отсортированный по убыванию оценки
        """
        print("Извлечение текста из резюме...")
        resume_text = self._extract_text_from_resume(resume_path)
        
        print("Загрузка вакансий...")
        all_vacancies = self._load_vacancies(vacancies_path)
        
        # Предварительная фильтрация
        if use_prefilter:
            vacancies = self._prefilter_vacancies(resume_text, all_vacancies)
        else:
            vacancies = all_vacancies
        
        if not vacancies:
            print("Нет подходящих вакансий после предварительной фильтрации")
            return []
        
        print(f"Анализ {len(vacancies)} вакансий с помощью YaGPT (порог: {threshold})...")
        
        # Разбиваем на батчи для многопоточной обработки
        batches = [vacancies[i:i+batch_size] for i in range(0, len(vacancies), batch_size)]
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_vacancy_batch, resume_text, batch, threshold): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"Ошибка при обработке батча: {e}")
        
        # Сохраняем кэш
        self._save_cache()
        
        # Конвертируем результаты в нужный формат и сортируем
        final_results = [
            (vacancy, analysis.score, analysis.reasoning)
            for vacancy, analysis in all_results
        ]
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Найдено {len(final_results)} подходящих вакансий из {len(all_vacancies)}")
        
        return final_results
