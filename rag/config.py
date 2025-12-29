from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', 'gemma2:2b')
TRANSFORMER_MODEL_NAME = os.getenv('TRANSFORMER_MODEL_NAME', 'intfloat/multilingual-e5-small')
SPACY_MODEL_NAME = os.getenv('SPACY_MODEL_NAME', 'pl_core_news_sm')
QDRANT_INDEX_NAME = os.getenv('QDRANT_INDEX_NAME', 'culturax')
ES_INDEX_NAME = os.getenv('ES_INDEX_NAME', 'culturax')

RAG_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = RAG_DIR / "memory" / "unresolved_queries.json"

UNRESOLVED_STORAGE_PATH = Path(
    os.getenv("UNRESOLVED_STORAGE_PATH", DEFAULT_PATH)
).expanduser().resolve()

UNRESOLVED_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

DATA_FILE_NAME = os.getenv('DATA_FILE_NAME', 'culturax_vectors.ndjson')

es_url = os.getenv("ES_URL", "http://elasticsearch:9200")
qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")

PROMPT_CORES_LIST = [
    """Twoim zadaniem jest odpowiedzieć na pytanie WYŁĄCZNIE na podstawie fragmentów poniżej.

Zasady:
- Nie używaj wiedzy spoza fragmentów.
- Napisz odpowiedź i poprzyj ją cytatem w formie [numer_fragmentu] "cytat z fragmentu".
- Cały zwrócony tekst powinien mieć formę: ODPOWIEDź, [numer_fragmentu] "cytat z fragmentu
- Jeżeli nie wypiszesz żadnej odpowiedzi, zwróć dokładnie: "BRAK ODPOWIEDZI".
- Jeśli zwrócisz jakąkolwiek odpowiedź, albo cytat to NIE PISZ "BRAK ODPOWIEDZI".
""",
"""Twoim zadaniem jest odpowiedzieć na pytanie WYŁĄCZNIE na podstawie fragmentów poniżej.

Zasady:
- Nie używaj wiedzy spoza fragmentów.
- Każde zdanie odpowiedzi musi być poparte cytatem w formacie [numer_fragmentu] "cytat z fragmentu".
- Jeśli fragmenty nie zawierają odpowiedzi na pytanie, napisz dokładnie: "BRAK INFORMACJI".
""",
"Jesteś asystentem, który odpowiada na pytania wyłącznie na podstawie dostarczonych fragmentów."
]

RETRY_STRATEGIES_LIST_DEFAULT = ["change_interpretation", "modify_prompt", "save_to_memory"]