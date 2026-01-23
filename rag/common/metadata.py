import regex
import ollama
from pydantic import BaseModel
from dateutil import parser

def match_regex_list(text: str, patterns: list[str]):
    results = []
    for pattern in patterns:
        res = regex.findall(pattern, text)
        if res:
            results += res
    return results

REGEX_STRICT = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{4}\.\d{2}.\d{2}\b",
    r"\b\d{4}\/\d{2}/\d{2}\b",
    r"\b\d{2}-\d{2}-\d{4}\b",
    r"\b\d{2}\.\d{2}.\d{4}\b",
    r"\b\d{2}\/\d{2}/\d{4}\b",
    r'\bw \d{4}\b',
    r'(?:o|O)d \d{4} do \d{4}\b',
    r'\d{4}-\d{4}',
]

def ner_find_dates(text: str, model):
    doc = model(text)
    return [value.text for value in doc.ents if value.label_ == "date"]

class DatesModel(BaseModel):
    dates: list[str]
    years: list[str]
    ranges: list[str]
    other: list[str]

def llm_find_dates(text: str, known_dates: list[str], model: str = 'gemma2:2b'):
    prompt_core = """
Wyodrębnij z poniższego tekstu tylko nietypowe daty i zakresy, których nie wykryły standardowe metody.
Oto daty już znalezione: {}

TEKST:
{}

Zwróć wynik w formacie JSON:
{{
  "dates": [],
  "years": [],
  "ranges": [],
  "other": []
}}
"""
    known_dates_str = ", ".join(known_dates)
    prompt = prompt_core.format(known_dates_str, text)
    res = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=DatesModel.model_json_schema()
    )

    dates = DatesModel.model_validate_json(res['message']['content'])
    return dates

def clean_dates(dates):
    cleaned = []
    for d in dates:
        if regex.search(r'\d{4}', d):  
            cleaned.append(d)
    return cleaned

def hybrid_date_extraction(text: str, model_ner):
    regex_res = match_regex_list(text, REGEX_STRICT)
    ner_res = ner_find_dates(text, model_ner)
    known = regex_res + ner_res

    llm_res = llm_find_dates(text, known)
    known += llm_res.dates + llm_res.years + llm_res.ranges + llm_res.other

    all_clean = clean_dates(known)

    return list(set(all_clean))


def extract_metadata_from_query(text: str, model_ner) -> dict:
    doc = model_ner(text)

    entities = sorted({
        ent.text for ent in doc.ents
        if ent.label_ in ("persName", "orgName")
    })

    places = sorted({
        ent.text for ent in doc.ents
        if ent.label_ in ("placeName", "geogName")
    })

    all_dates = hybrid_date_extraction(text, model_ner)
    years = set()

    for d in all_dates:
        try:
            years.add(parser.parse(d, fuzzy=True).year)
        except Exception:
            for fy in regex.findall(r"\b\d{4}\b", d):
                years.add(int(fy))

    years = sorted(years)

    return {
        "entities": entities,
        "places": places,
        "years": years
    }