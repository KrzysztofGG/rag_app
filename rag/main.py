from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

from rag import RAG
import config
from memory.unresolved_memory import UnresolvedQueriesMemory

app = FastAPI()
memory = UnresolvedQueriesMemory(storage_path=config.UNRESOLVED_STORAGE_PATH)

FULL_DATA_PATH = os.path.join('data', config.DATA_FILE_NAME)

rag = RAG(
    memory,
    config.PROMPT_CORES_LIST,
    config.OLLAMA_MODEL_NAME,
    FULL_DATA_PATH,
    config.TRANSFORMER_MODEL_NAME,
    config.SPACY_MODEL_NAME,
    config.QDRANT_INDEX_NAME,
    config.ES_INDEX_NAME,
    es_url=config.es_url,
    qdrant_url=config.qdrant_url,
    ollama_host=config.ollama_host
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

