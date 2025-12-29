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

def analyze_query(query: str) -> dict:
    '''
    Analyze query for keywords and types
    '''
    text = query.strip()
    text_lower = text.lower()
    tokens = TOKEN_RE.findall(text_lower)

    features = {
        "has_number": any(t.isdigit() for t in tokens),
        "has_year": bool(YEAR_RE.search(text)),
        "has_id": bool(ID_RE.search(text)),
        "is_acronym": bool(ACRONYM_RE.fullmatch(text)),
        "has_filter": any(t in FILTER_WORDS for t in tokens),
        "is_question": text.endswith("?"),
        "abstract": any(phrase in text_lower for phrase in ABSTRACT_WORDS),
        "token_len": len(tokens),
    }

    return features

def choose_weights(f: dict) -> dict:
    '''
    Choose qdrant and es RRF weights based on query type
    '''

    # 1. Twarde przypadki (lookup)
    if f["is_acronym"] or f["has_id"]:
        return {"es": 0.9, "qdrant": 0.1}

    # 2. Fakty / liczby
    if f["has_number"] or f["has_year"]:
        return {"es": 0.8, "qdrant": 0.2}

    # 3. Filtry / metadata
    if f["has_filter"]:
        return {"es": 0.7, "qdrant": 0.3}

    # 4. Faktyczne pytania
    if f["is_question"] and f["has_number"]:
        return {"es": 0.75, "qdrant": 0.25}

    # 5. Abstrakcje / semantyka
    if f["abstract"] or f["token_len"] <= 3:
        return {"es": 0.4, "qdrant": 0.6}

    # 6. Domyślne
    return {"es": 0.55, "qdrant": 0.45}


def embed(text: str, transformer_model: SentenceTransformer):
    return transformer_model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

def tokenize_regex(sentence: str) -> List:
    return re.findall(r"\w+|[^\w\s]", sentence)

def count_citations(answer: str) -> int:
    return len(re.findall(r"\[\d+\]", answer))