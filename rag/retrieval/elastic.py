from elasticsearch import Elasticsearch
from typing import Dict


def search_es_enhanced(es_query: str, es_client: Elasticsearch, index_name: str, metadata: Dict):
    should_clauses = []
    
    for entity in metadata["entities"]:
        should_clauses.append({"term": {"entities": {"value": entity, "boost": 2.0}}})
    for place in metadata["places"]:
        should_clauses.append({"term": {"places": {"value": place, "boost": 1.5}}})
    for year in metadata["years"]:
        should_clauses.append({"term": {"years": {"value": year, "boost": 3.0}}})

    query_body = {
        "query": {
            "bool": {
                "must": [{"query_string": {"query": es_query}}],
                "should": should_clauses
            }
        }
    }

    response = es_client.search(index=index_name, query=query_body["query"], size=20)

    hits = response["hits"]["hits"]
    top_id = [int(h["_id"]) for h in hits]
    top_text = [h["_source"]["text"] for h in hits]
    
    return top_id, top_text
