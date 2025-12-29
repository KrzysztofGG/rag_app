from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams
import json

def create_es_index(index_name: str, es_client: Elasticsearch):
    if not es_client.indices.exists(index=index_name):
        index_body = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "pl_lemma": {
                            "tokenizer": "standard",
                            "filter": ["lowercase"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "domain": {"type": "keyword"},
                    "date": {"type": "date"},
                    "text": {"type": "text", "analyzer": "pl_lemma"},
                    "vector": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"}
                }
            }
        }
        es_client.indices.create(index=index_name, body=index_body)

def populate_index(data_file_path: str, index_name: str, es_client: Elasticsearch):
    if es_client.indices.exists(index=index_name):
        doc_count = es_client.count(index=index_name)['count']
        if doc_count > 0:
            print(f"Index '{index_name}' already has {doc_count} documents. Skipping insertion.")
            return
    else:
        print(f"Index '{index_name}' does not exist. Creating it first.")
        es_client.indices.create(index=index_name)

    actions = []
    with open(data_file_path, "r") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("{\"index\""):  # skip metadata lines
                continue
            doc = json.loads(line)
            if is_json_invalid(doc):
                print(f"Data row {i} invalid, skipping...")
                continue
            actions.append({
                "_index": index_name,
                "_id": doc["id"],
                "_source": doc
            })
    success, _ = bulk(es_client, actions)
    print(f"Inserted {success} documents into ES")

def create_qdrant_collection(collection_name: str, qdrant_client: QdrantClient):
    if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        ),

def populate_collection(data_file_path: str, collection_name: str, qdrant_client: QdrantClient):
    collection_stats = qdrant_client.get_collection(collection_name)
    num_points = collection_stats.points_count
    if num_points > 0:
        print(f"Collection '{collection_name}' already has {num_points} points. Skipping insertion.")
        return

    points = []
    with open(data_file_path, "r") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("{\"index\""):
                continue
            doc = json.loads(line)
            if is_json_invalid(doc):
                print(f"Data row {i} invalid, skipping...")
                continue
            points.append(PointStruct(
                id=int(doc["id"]),
                vector=doc["vector"],
                payload={k: v for k, v in doc.items() if k != "vector"}  # store text, date, etc.
            ))

    BATCH_SIZE = 500
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"Upserted points {i}-{i+len(batch)}")

def is_json_invalid(json_obj):
    obligatory_data_keys = ['id', 'text', 'vector']
    return any([key not in json_obj for key in obligatory_data_keys])