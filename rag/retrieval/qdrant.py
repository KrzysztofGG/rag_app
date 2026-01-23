from qdrant_client import QdrantClient
from typing import List, Dict
from qdrant_client.http import models as rest

def search_qdrant_enhanced(query_vector: List[float], qdrant_client: QdrantClient, collection_name: str, metadata: Dict):
    filter_conditions = []
    
    if metadata["years"]:
        filter_conditions.append(
            rest.FieldCondition(key="years", match=rest.MatchAny(any=metadata["years"]))
        )
    
    search_filter = None
    if filter_conditions:
        search_filter = rest.Filter(should=filter_conditions)

    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=search_filter,
        limit=20,
        with_payload=True
    ).points

    top_id = [hit.id for hit in response]
    top_text = [hit.payload.get("text", "") for hit in response]
    return top_id, top_text