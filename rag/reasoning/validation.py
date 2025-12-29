from difflib import SequenceMatcher
from typing import List, Dict
import re

class CitationValidator():

    def __init__(self, fuzzy_match_threshold: float = 0.75):
        self.fuzzy_threshold = fuzzy_match_threshold

    def extract_citations_with_numbers(self, answer: str) -> List[Dict]:
        citations = []
        pattern = r'\[(\d+)\]'
        matches = list(re.finditer(pattern, answer))

        for match in matches:
            citation_num = int(match.group(1))
            start = match.start()
            end = match.end()

            before_start = max(0, start - 200)
            before_text = answer[before_start:start].strip()

            after_end = min(end + 200, len(answer))
            after_text = answer[end:after_end].strip()

            if len(before_text) > len(after_text):
                sentences = re.split(r'[.?!]\s+', before_text)
                citation_text = sentences[-1] if sentences else before_text
            else:
                sentences = re.split(r'[.?!]\s+', after_text)
                citation_text = sentences[0] if sentences else after_text
            
            citations.append({
                "text": citation_text,
                "doc_number": citation_num
            })

            # Wyciągnij cytaty w cudzysłowach

            quoted_pattern = r'(?:\"([^"]+)\"\s*\[(\d+)\])|(?:\[(\d+)\]\s*\"([^"]+)\")'
            quoted_matches = re.finditer(quoted_pattern, answer)
            
            for match in quoted_matches:
                if match.group(1):  # "tekst" [num]
                    citation_text = match.group(1)
                    citation_num = int(match.group(2))
                else:  # [num] "tekst"
                    citation_text = match.group(4)
                    citation_num = int(match.group(3))
                
                if not any(c['text'] == citation_text for c in citations):
                    citations.append({
                        "text": citation_text,
                        "doc_number": citation_num,
                    })
        
        return citations
    
    def normalize_text(self, text: str) -> str:

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower().strip()

        return text
    
    def find_citation_in_doc(
            self,
            citation_text: str,
            document: str
    ) -> bool:
        
        # Strategy 1: Exact validation
        citation_norm = self.normalize_text(citation_text)
        doc_norm = self.normalize_text(document)
        if citation_norm in doc_norm:
            return True
        
        # Stratey 2: Fuzzy matching
        citation_len = len(citation_norm.split())
        doc_words = doc_norm.split()
        
        best_score = 0.0
        
        window_size = max(citation_len, 5)
        
        for i in range(len(doc_words) - window_size + 1):
            window = " ".join(doc_words[i:i + window_size])
            score = SequenceMatcher(None, citation_norm, window).ratio()
            
            if score > best_score:
                best_score = score
        
        found = best_score >= self.fuzzy_threshold
        return found

    
    def validate_answer(self, answer: str, retrieved_docs: List[str]) -> bool:
        citations = self.extract_citations_with_numbers(answer)

        if not citations:
            return False
        
        # Validate citations
        for citation in citations:
            doc_num = citation["doc_number"]
            citation_text = citation["text"]

            if doc_num < 1 or doc_num > len(retrieved_docs):
                return False # Invalid citation
            
            document = retrieved_docs[doc_num - 1]
            found = self.find_citation_in_doc(citation_text, document)

            if not found:
                return False
        
        return True
