
from collections import defaultdict
from typing import List

def rrf_fusion_weighted(
        qdrant_ids: List[int], 
        es_ids: List[int], 
        qdrant_texts: List[str], 
        es_texts: List[str], 
        qdrant_weight: int = 1,
        es_weight: int = 1, 
        k: int = 3) -> List[tuple[str, float]]:
    
    scores = defaultdict(float)

    for rank, doc_id in enumerate(qdrant_ids, start=1):
        scores[doc_id] += qdrant_weight / rank

    for rank, doc_id in enumerate(es_ids, start=1):
        scores[doc_id] += es_weight / rank

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    fused_results = []
    for (doc_id, score) in fused:
        if doc_id in qdrant_ids:
            fused_results.append((qdrant_texts[qdrant_ids.index(doc_id)], score))
        elif doc_id in es_ids:
            fused_results.append((es_texts[es_ids.index(doc_id)], score))

    return fused_results