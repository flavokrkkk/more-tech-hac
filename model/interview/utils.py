def format_vacancy_text(vacancy: dict) -> str:
    return f"""
Вакансия: {vacancy['title']}
Описание: {vacancy['description']}
Требования:
- {"\n- ".join(vacancy['requirements'])}
"""