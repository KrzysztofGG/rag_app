import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from common import tokenize_regex, embed

def filter_retrieved_with_stats(
        docs: List[str], 
        query: str, 
        query_vec: np.ndarray, 
        f: dict,  
        transformer_model: SentenceTransformer,
        min_tokens: int = 15, 
        max_docs: int = 5):
    query_tokens = {t.lower() for t in tokenize_regex(query) if len(t) > 2}

    stats = {
        "input_docs": len(docs),
        "kept_docs": 0,
        "rejected_short": 0,
        "rejected_overlap": 0,
        "overlaps": [],
    }

    docs_filtered = []
    doc_vec_cache = {}

    for text in docs:
        tokens = {t.lower() for t in tokenize_regex(text) if len(t) > 2}

        if len(tokens) < min_tokens:
            stats["rejected_short"] += 1
            continue

        overlap = len(set(tokens) & query_tokens) 
        stats["overlaps"].append(overlap)

        if f["is_acronym"] or f["has_id"] or f["has_number"] or f["has_year"] or f["has_filter"]:
            if overlap == 0:
                if text not in doc_vec_cache:
                    doc_vec_cache[text] = embed(text, transformer_model)

                sim = cosine_similarity(query_vec, doc_vec_cache[text])
                
                if sim < 0.55:
                    stats["rejected_overlap"] += 1
                    continue

        docs_filtered.append(text)
        stats["kept_docs"] += 1

    return docs_filtered[:max_docs], stats

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
