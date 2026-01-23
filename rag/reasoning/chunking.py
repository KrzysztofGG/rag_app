from typing import List
from common import tokenize_regex

def chunk_document(text: str, max_tokens: int = 250, overlap: int = 30) ->List[str]:
    """Chunking z nakładaniem się - zachowuje więcej kontekstu"""
    tokens = tokenize_regex(text)
    chunks = []
    
    if len(tokens) <= max_tokens:
        return [text]
    
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        
        if end >= len(tokens):
            break
            
        start += (max_tokens - overlap)  
    
    return chunks