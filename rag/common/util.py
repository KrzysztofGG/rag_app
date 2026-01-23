import re 
from typing import List
from sentence_transformers import SentenceTransformer

def extract_keywords_lemmatized(text: str, nlp):
    doc = nlp(text.lower())
    keywords = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and token.is_alpha
        and len(token.lemma_) > 2
    ]
    return list(dict.fromkeys(keywords))

def make_queries(text: str, nlp):
    '''
    Build queries for qdrant and es
    '''
    semantic_query = f"query: {text}"
    keywords = extract_keywords_lemmatized(text, nlp)
    keyword_query = " OR ".join(keywords)
    return semantic_query, keyword_query


ACRONYM_RE = re.compile(r"^[A-ZĄĆĘŁŃÓŚŻŹ]{2,}$")
ID_RE = re.compile(r"[A-Z]{1,5}[-_]?\d+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

FACTUAL_VERBS = {"być", "wynosić", "mieć", "było", "jest"}
FILTER_WORDS = {"autor", "dokumenty", "po", "przed", "od", "dotyczące"}
ABSTRACT_WORDS = {"czym", "co to", "jak", "dlaczego", "sens", "znaczenie"}

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def analyze_query(query: str, model_ner) -> dict:
    text = query.strip()
    text_lower = text.lower()
    tokens = TOKEN_RE.findall(text_lower)
    
    doc = model_ner(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    has_person = any(e[1] == "PERS" for e in entities)
    has_org = any(e[1] == "ORG" for e in entities)
    has_place = any(e[1] == "GPE" or e[1] == "LOC" for e in entities)
    has_date = any(e[1] == "DATE" for e in entities)

    features = {
        "has_number": any(t.isdigit() for t in tokens),
        "has_year": bool(YEAR_RE.search(text)) or has_date,
        "has_id": bool(ID_RE.search(text)),
        "is_acronym": bool(ACRONYM_RE.fullmatch(text)),
        "has_filter": any(t in FILTER_WORDS for t in tokens),
        "is_question": text.endswith("?"),
        "abstract": any(phrase in text_lower for phrase in ABSTRACT_WORDS),
        "token_len": len(tokens),
        
        "has_named_entity": len(entities) > 0,
        "has_specific_entity": has_person or has_org or has_place,
        "entities_list": entities
    }

    return features

def choose_weights(f: dict) -> dict:
    # f = analyze_query(query, model_ner)

    # 1. Twarde identyfikatory (kod, ID, akronim) -> ES dominuje
    if f["is_acronym"] or f["has_id"]:
        return {"es": 0.8, "qdrant": 0.2}

    # 2. Nazwy własne (NER) -> ES dominuje (szukamy konkretnych faktów o bycie)
    if f["has_specific_entity"]:
        es_weight = 0.7 if f["token_len"] > 4 else 0.6
        return {"es": es_weight, "qdrant": 1 - es_weight}

    # 3. Daty i liczby
    if f["has_year"] or f["has_number"]:
        return {"es": 0.65, "qdrant": 0.35}

    # 4. Zapytania o definicje i abstrakcje -> Qdrant dominuje
    if f["abstract"]:
        return {"es": 0.3, "qdrant": 0.7}

    # 5. Krótkie zapytania semantyczne
    if f["token_len"] <= 3 and not f["has_named_entity"]:
        return {"es": 0.3, "qdrant": 0.7}

    # 6. Domyślne dla naturalnego języka
    return {"es": 0.45, "qdrant": 0.55}

def embed(text: str, transformer_model: SentenceTransformer):
    return transformer_model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

def tokenize_regex(sentence: str) -> List:
    return re.findall(r"\w+|[^\w\s]", sentence)

def count_citations(answer: str) -> int:
    return len(re.findall(r"\[\d+\]", answer))