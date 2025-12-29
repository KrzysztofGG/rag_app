from typing import List

def chunk_document(text: str, nlp, max_tokens=200, overlap=30) -> List[str]:
    if not text:
        return []

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())

        if current_len + sent_len > max_tokens:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            if overlap > 0:
                overlap_tokens = 0
                overlap_chunk = []

                for s in reversed(current_chunk):
                    overlap_tokens += len(s.split())
                    overlap_chunk.insert(0, s)
                    if overlap_tokens >= overlap:
                        break

                current_chunk = overlap_chunk
                current_len = overlap_tokens
            else:
                current_chunk = []
                current_len = 0

        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks