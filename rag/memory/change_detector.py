import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient

class DocumentChangeDetector:
    def __init__(
        self,
        es_client: Elasticsearch,
        qdrant_client: QdrantClient,
        es_index: str,
        qdrant_collection: str,
        state_path: str = "snapshots/initial_state.json"
    ):
        self.es = es_client
        self.qdrant = qdrant_client
        self.es_index = es_index
        self.qdrant_collection = qdrant_collection
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.initial_doc_ids: Set[int] = set()
        self._load_or_create_initial_state()
    
    def _load_or_create_initial_state(self):
        if self.state_path.exists():
            with open(self.state_path, 'r') as f:
                data = json.load(f)
                self.initial_doc_ids = set(data['doc_ids'])
                print(f"[INFO] Wczytano stan początkowy: {len(self.initial_doc_ids)} dokumentów")
        else:
            print("[INFO] Tworzenie stanu początkowego...")
            self.initial_doc_ids = self._get_all_doc_ids()
            self._save_state()
            print(f"[INFO] Zapisano stan początkowy: {len(self.initial_doc_ids)} dokumentów")

    def _save_state(self):
        with open(self.state_path, 'w') as f:
            json.dump({
                'doc_ids': list(self.initial_doc_ids),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

    def _get_all_doc_ids(self) -> Set[int]:
        doc_ids = set()
        
        query = {"query": {"match_all": {}}, "_source": False}
        
        response = self.es.search(
            index=self.es_index,
            body=query,
            scroll='2m',
            size=1000
        )
        
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        
        while hits:
            for hit in hits:
                doc_ids.add(int(hit['_id']))
            
            response = self.es.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
        
        self.es.clear_scroll(scroll_id=scroll_id)
        
        return doc_ids
    
    def get_new_documents(self) -> List[Dict]:
        current_ids = self._get_all_doc_ids()
        new_ids = current_ids - self.initial_doc_ids
        
        if not new_ids:
            print("[INFO] Brak nowych dokumentów")
            return []
        
        print(f"[INFO] Znaleziono {len(new_ids)} nowych dokumentów")
        
        new_docs = []
        
        for doc_id in new_ids:
            try:
                response = self.es.get(index=self.es_index, id=str(doc_id))
                source = response['_source']
                
                new_docs.append({
                    'id': doc_id,
                    'entities': source.get('entities', []),
                    'places': source.get('places', []),
                    'years': source.get('years', []),
                    'text': source.get('text', '')[:200] 
                })
            except Exception as e:
                print(f"[WARN] Błąd pobierania dokumentu {doc_id}: {e}")
                continue
        
        return new_docs
    def reset_initial_state(self):
        print("[INFO] Resetowanie stanu początkowego...")
        self.initial_doc_ids = self._get_all_doc_ids()
        self._save_state()
        print(f"[INFO] Nowy stan: {len(self.initial_doc_ids)} dokumentów")

def match_query_with_new_docs(
    query_metadata: Dict,
    new_documents: List[Dict]
) -> Tuple[bool, List[int]]:

    query_entities = set(query_metadata.get('entities_hint', []))
    query_places = set(query_metadata.get('places_hint', []))
    query_years = set(query_metadata.get('years_hint', []))
    
    matched_docs = []
    
    for doc in new_documents:
        doc_entities = set(doc.get('entities', []))
        doc_places = set(doc.get('places', []))
        doc_years = set(doc.get('years', []))
        
        has_entity_match = bool(query_entities & doc_entities)
        has_place_match = bool(query_places & doc_places)
        has_year_match = bool(query_years & doc_years)
        
        if has_entity_match or has_place_match or has_year_match:
            matched_docs.append(doc['id'])
    
    return len(matched_docs) > 0, matched_docs