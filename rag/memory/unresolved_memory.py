from pathlib import Path
import json
from typing import Dict, List
from datetime import datetime

class UnresolvedQueriesMemory:
    def __init__(self, storage_path: str = "unresolved_queries.json"):
        self.storage_path = Path(storage_path)
        self.queries = self._load_queries()
        self.next_id = max([q['id'] for q in self.queries], default=0) + 1

    def _load_queries(self) -> List[str]:
        print(f"[INFO] - Memory saving queries to {self.storage_path}")
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return []

    def _save_queries(self):

        with open(self.storage_path, 'w') as f:
            json.dump(self.queries, f, ensure_ascii=True, indent=2)

    def add_query(self, query: str, metadata: Dict):
        query_entry = {
            "id": self.next_id,
            "query": query,
            "entities_hint": metadata["entities"],
            "years_hint": metadata["years"],
            "places_hint": metadata["places"],
            "retry_count": 0,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }

        self.queries.append(query_entry)
        self._save_queries()

        query_id = self.next_id
        self.next_id += 1
        return query_id
    
    def get_pending_queries(self):
        return [q for q in self.queries if q["status"] == "pending"]
    
    def get_query_by_id(self, query_id: int):
        for query in self.queries:
            if query_id == query['id']:
                return query
        return None
    
    def increment_retry_count(self, query_id: int):
        for query in self.queries:
            if query_id == query['id']:
                query['retry_count'] = query.get('retry_count', 0) + 1
                self._save_queries()
                return True
        return False
    
    def mark_as_resolved(self, query_id: str):
        for query in self.queries:
            if query['id'] == query_id:
                query['status'] = "resolved"
                query['resolved_at'] = datetime.now().isoformat()
                self._save_queries()
                return True
        return False

    def get_statistics(self) -> Dict:
        pending = [q for q in self.queries if q['status'] == 'pending']                    
        resolved = [q for q in self.queries if q['status'] == 'resolved']

        return {
            "total": len(self.queries),
            "pending": len(pending),
            "resolved": len(resolved),
            "avg_retry_count": sum(q.get("retry_count", 0) for q in pending) / max(len(pending), 1)
        }
    
    def clear_resolved(self):
        self.queries = [q for q in self.queries if q["status"] == "pending"]
        self._save_queries()

    def should_save_as_unresolved(
            self, 
            answer: str, 
            used_chunks: List[str],
            stats: Dict,
            min_chunks = 1,
            min_citations = 1,
            ) -> bool:
        
        # 1. Check whether model didn't answer explicitly
        if "BRAK INFORMACJI" in answer.upper() or "BRAK ODPOWIEDZI" in answer.upper():
            return True # Model returned no answer based on context
        
        # 2. Count used chunks
        if len(used_chunks) < min_chunks:
            return True
        
        # 3. Count citations (if they are available in stats)
        citations = stats.get("citations", 0)
        if citations < min_citations:
            return True
        
        return False