from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import regex
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm

from .metadata import hybrid_date_extraction
from dateutil import parser

def enrich_doc_if_missing(doc: dict, model_ner) -> dict:
    text = doc.get("text")
    if not text:
        return doc

    needs_entities = "entities" not in doc
    needs_years = "years" not in doc
    needs_places = "places" not in doc

    if not (needs_entities or needs_years or needs_places):
        return doc

    ner_res = model_ner(text)

    if needs_entities:
        doc["entities"] = sorted({
            ent.text for ent in ner_res.ents
            if ent.label_ in ("persName", "orgName")
        })

    if needs_places:
        doc["places"] = sorted({
            ent.text for ent in ner_res.ents
            if ent.label_ in ("placeName", "geogName")
        })

    if needs_years:
        all_dates = hybrid_date_extraction(text, model_ner)
        years = set()

        for d in all_dates:
            try:
                years.add(parser.parse(d, fuzzy=True).year)
            except Exception:
                for fy in regex.findall(r"\b\d{4}\b", d):
                    years.add(int(fy))

        doc["years"] = sorted(years)

    return doc

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
        print(f"ES index: {index_name} created")
    else:
        print(f"ES index: {index_name} already exists")

def populate_index(
    data_file_path: str,
    index_name: str,
    es_client: Elasticsearch,
    model_ner,
    max_docs: int = None,
    batch_size: int = 5,
    ner_enabled: bool = False
):
    if es_client.indices.exists(index=index_name):
        doc_count = es_client.count(index=index_name)['count']
        if doc_count > 0:
            print(f"Index '{index_name}' already has {doc_count} documents. Skipping insertion.")
            return
    else:
        print(f"Index '{index_name}' does not exist. Creating it first.")
        es_client.indices.create(index=index_name)

    actions = []
    total_inserted = 0

    with open(data_file_path, "r") as f:
        for i, line in tqdm(enumerate(f, start=1)):
            line = line.strip()

            if not line or line.startswith("{\"index\""):
                continue

            doc = json.loads(line)

            if is_json_invalid(doc):
                print(f"Data row {i} invalid, skipping...")
                continue

            if ner_enabled:
                doc = enrich_doc_if_missing(doc, model_ner)

            actions.append({
                "_index": index_name,
                "_id": doc["id"],
                "_source": doc
            })

            if len(actions) == batch_size:
                success, _ = bulk(es_client, actions)
                print(f"Added lines {i-len(actions)}-{i} to index: {index_name}")
                total_inserted += success
                actions.clear()

            if max_docs and total_inserted >= max_docs:
                break

    if actions:
        success, _ = bulk(es_client, actions)
        total_inserted += success

    print(f"Inserted {total_inserted} documents into ES")

def create_qdrant_collection(collection_name: str, qdrant_client: QdrantClient):
    if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"Qdrant collection: {collection_name} created")
    else:
        print(f"Qdrant collection: {collection_name} already exists")

def point_exists(qdrant_client: QdrantClient, collection_name: str, point_id: int) -> bool:
    res = qdrant_client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=False,
        with_vectors=False,
    )
    return len(res) > 0

def populate_collection(
    data_file_path: str,
    collection_name: str,
    qdrant_client: QdrantClient,
    model_ner,
    max_points: int = None,
    batch_size: int = 5,
    ner_enabled: bool = False
):
    collection_stats = qdrant_client.get_collection(collection_name)
    num_points = collection_stats.points_count

    if num_points > 0:
        print(f"Collection '{collection_name}' already has {num_points} points. Skipping insertion.")
        return

    points = []
    total_upserted = 0

    with open(data_file_path, "r") as f:
        for i, line in tqdm(enumerate(f, start=1)):
            line = line.strip()

            if not line or line.startswith("{\"index\""):
                continue

            doc = json.loads(line)

            if is_json_invalid(doc):
                print(f"Data row {i} invalid, skipping...")
                continue

            point_id = int(doc["id"])

            if point_exists(qdrant_client, collection_name, point_id):
                print(f"Point {point_id} already exists, skipping")
                continue
            
            if ner_enabled:
                doc = enrich_doc_if_missing(doc, model_ner)

            vector = doc.pop("vector")

            points.append(
                PointStruct(
                    id=int(doc["id"]),
                    vector=vector,
                    payload=doc
                )
            )

            if len(points) == batch_size:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"Added lines {i-len(points)}-{i} to collection: {collection_name}")
                total_upserted += len(points)
                points.clear()

            if max_points and total_upserted >= max_points:
                break

    if points:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        total_upserted += len(points)

    print(f"Upserted {total_upserted} points into '{collection_name}'")

def is_json_invalid(json_obj):
    obligatory_data_keys = ['id', 'text', 'vector']
    return any([key not in json_obj for key in obligatory_data_keys])