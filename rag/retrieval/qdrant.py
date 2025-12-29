from qdrant_client import QdrantClient
from typing import List

def search_qdrant(query_vector, qdrant_client: QdrantClient, collection_name: str) -> tuple[List[int], List[str]]:
    result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=35
    ).points

    top_id = [hit.id for hit in result]
    top_text = [hit.payload.get("text", "") for hit in result]
    return top_id, top_text