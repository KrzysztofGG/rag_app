from elasticsearch import Elasticsearch
from typing import List

def search_es(es_query: str, es_client: Elasticsearch, index_name: str) -> tuple[List[int], List[str]]:
    response = es_client.search(
        index=index_name,
        query={
            "query_string": {
                "query": es_query
            }
        },
        size=35
    )
    
    hits = response["hits"]["hits"]
    top_id = [int(h["_id"]) for h in hits]
    top_text = [h["_source"]["text"] for h in hits]
    
    return top_id, top_text