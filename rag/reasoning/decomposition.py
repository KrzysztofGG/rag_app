import json
from ollama import Client
import re

def decompose_query(user_input: str, features: dict, ollama_model: str, ollama_client: Client) -> dict:    
    # Przypadki, które NIE wymagają dekompozycji
    if features["is_acronym"] or features["has_id"]:
        return {
            "main_question": user_input,
            "sub_questions": [],
            "decomposition_type": "factual"
        }
    
    if features["has_filter"]:
        return {
            "main_question": user_input,
            "sub_questions": [],
            "decomposition_type": "filter"
        }
    
    # Złożone pytania wymagające dekompozycji
    prompt = f"""Jesteś ekspertem od analizy zapytań. Twoim zadaniem jest rozłożyć pytanie na komponenty.

Pytanie: {user_input}

Zasady:
1. Jeśli pytanie jest proste i konkretne (np. "Co zawiera dokument X?", "Czy inflacja rośnie?"), zwróć je jako main_question bez sub_questions.
2. Jeśli pytanie jest złożone (np. "Jak poprawić pracę zespołową?"), rozbij je na 2-3 podzapytania.
3. Format odpowiedzi (JSON):
{{
  "main_question": "...",
  "sub_questions": ["...", "..."]
}}

NIE dodawaj komentarzy. Zwróć TYLKO JSON."""

    response = ollama_client.chat(
        model=ollama_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    
    try:
        content = response["message"]["content"]
        content = re.sub(r"```json\s*|\s*```", "", content).strip()
        
        result = json.loads(content)
        
        if "main_question" not in result:
            result["main_question"] = user_input
        if "sub_questions" not in result:
            result["sub_questions"] = []
        
        result["decomposition_type"] = "complex" if result["sub_questions"] else "simple"
        return result
        
    except Exception as e:
        print(f"Błąd dekompozycji: {e}")
        return {
            "main_question": user_input,
            "sub_questions": [],
            "decomposition_type": "error"
        }