from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch

from rag import RAG
import config
from memory.unresolved_memory import UnresolvedQueriesMemory
from memory.change_detector import DocumentChangeDetector, match_query_with_new_docs

app = FastAPI()
memory = UnresolvedQueriesMemory(storage_path=config.UNRESOLVED_STORAGE_PATH)


FULL_DATA_PATH = os.path.join('data', config.DATA_FILE_NAME)
es_client = Elasticsearch(config.es_url)
qdrant_client = QdrantClient(config.qdrant_url)

rag = RAG(
    memory,
    config.PROMPT_CORES_LIST,
    config.OLLAMA_MODEL_NAME,
    FULL_DATA_PATH,
    config.TRANSFORMER_MODEL_NAME,
    config.SPACY_MODEL_NAME,
    config.QDRANT_INDEX_NAME,
    config.ES_INDEX_NAME,
    enable_decomposition=True,
    es_client=es_client,
    qdrant_client=qdrant_client,
    # es_url=config.es_url,
    # qdrant_url=config.qdrant_url,
    ollama_host=config.ollama_host
)

detector = DocumentChangeDetector(
    es_client=es_client,
    qdrant_client=qdrant_client,
    es_index=config.ES_INDEX_NAME,
    qdrant_collection=config.QDRANT_INDEX_NAME
)

class RagInfo(BaseModel):
    retry_strats: List[str] | None = config.RETRY_STRATEGIES_LIST_DEFAULT

@app.post("/ask")
async def run_rag(query: str, info: RagInfo):
    print(info.retry_strats)
    if info.retry_strats:
        retry_strategies = info.retry_strats
    else:
        retry_strategies = []
    res = rag.full_rag_process(query, retry_strategies)
    return {"model_answer": res}

@app.get("/pending")
async def get_pending_queries():
    queries = memory.get_pending_queries()
    return {'pending_queries': queries}

@app.get("/pending/{query_id}")
async def get_pending_query_by_id(query_id: int):
    match = next((q for q in memory.get_pending_queries() if q['id'] == query_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Query not found")
    
    return {"query": match}


@app.post("/retry_all")
async def retry_all_pending():
    new_docs = detector.get_new_documents()

    if not new_docs:
        return {
            "message": "No new documents to use",
            "retried_count": 0
        }
    
    pending = memory.get_pending_queries()

    if not pending:
        return {
            "message": "No pending queries to rerun",
        }

    results = []
    retried_count = 0

    for query_entry in pending:

        has_match, matched_doc_ids = match_query_with_new_docs(
            query_entry,
            new_docs
        )

        if not has_match:
            continue

        memory.increment_retry_count(query_entry['id'])

        # Retry RAG
        try:
            result = rag.rag_query_enhanced(
                query_entry['query']
            )

            if rag.evaluate_answer(result["answer"], result["stats"], result["chunks"]):
                memory.mark_as_resolved(
                    query_entry['id'],
                )
                status = "resolved"
            else:
                status = "still_pending"
            
            results.append({
                "query_id": query_entry['id'],
                "query": query_entry['query'],
                "status": status,
                "retry_count": query_entry.get("retry_count", 0) + 1,
                "matched_docs": matched_doc_ids
            })

            retried_count += 1
        
        except Exception as e:
            print(f"[ERR] Retry failed for query {query_entry['id']}: {e}")
            continue

@app.post("/retry")
async def retry_single_query(id: int):
    query_entry = memory.get_query_by_id(id)

    if not query_entry:
        raise HTTPException(status_code=404, detail=f"Query with id {id} not found in memory")
    
    if query_entry['status'] != 'pending':
        return {
            "message": f"Query with id {id} already resolved",
            "query": query_entry
        }
    
    memory.increment_retry_count(id)

    try:
        result = rag.rag_query_enhanced(
            query_entry['query']
        )

        if rag.evaluate_answer(result["answer"], result["stats"], result["chunks"]):
            memory.mark_as_resolved(
                query_entry['id'],
            )
            status = "resolved"
        else:
            status = "still_pending"

        return {
            "query_id": id,
            "query": query_entry['query'],
            "status": status,
            "retry_count": query_entry.get('retry_count', 0) + 1,
            "answer": result['answer'],
            "chunks_count": len(result['chunks'])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retry failed: {str(e)}")
    
@app.get("/stats")
async def get_stats():
    memory_stats = memory.get_statistics()

    return {
        "memory": memory_stats,
        "detector": {
            "initial_documents": len(detector.initial_doc_ids),
            "new_documents": len(detector.get_new_documents())
        }
    }