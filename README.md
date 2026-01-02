### Introduction
Repository contains definition of RAG process based on Qdrant and Elasticsearch retrieval. Data is Polish and comes from Culturax.

Process can be defined by these steps:
- Clarify query if it's ambiguous (run on one of possible interpretations)
- Decompose (add subquestions) if it's complicated
- Generate valid query forms for both engines (keywords division, embedding, ...)
- Retrieve docs from ES and Qdrant
- Do RRF fusion using weights obtained based on query features
- Collect documents for main and subquestion, chunk them and sort based on cosine similarity
- Filter invalid ones out (too short, not relevant)
- Using best chunks, build a prompt and ask model
- If model can't answer or answer is invalid, retry with one of the strategies
- If nothing else can be done and answer still invalid, return to unresolved memory
- If answer is evaluated to be valid, return it

### File structure
```
├── elasticsearch
│   └── Dockerfile                  # Builds elasticsearch image with morfologik
│
├── rag
│   ├── common                      # Entrypoint for the FastAPI application
│   │   ├── __init.py__
│   │   ├── data.py                 # Makes sure databases have data injected
│   │   └── util.py                 # Common util functions
│   │
│   ├── data                        # Contains ndjson file that populates database data
│   │
│   ├── memory
│   │   └── unresolved_memory.py    # Defines unresolved questions memory container
│   │
│   ├── reasoning
│   │   ├── __init.py__
│   │   ├── chunking.py             # Divides data from databases and splits them into chunks
│   │   ├── clarification.py        # System making sure that query is unambiguous
│   │   ├── decomposition.py        # Adds subquestions to complicated and ambiguous queries
│   │   ├── filtering.py            # Removes invalid documents retrieved from databases
│   │   ├── prompt.py               # Builds prompts for model
│   │   └── validation.py           # Makes sure model answer is valid
│   │
│   ├── retrieval
│   │   ├── __init.py__
│   │   ├── elastic.py              # Finds documents in ES index
│   │   ├── fusion.py               # Runs RRF to get best docs from both es and qdrant
│   │   └── qdrant.py               # Finds documents in qdrant collection
│   │
│   ├── config.py                   # Defined configuration
│   ├── main.py                     # FastAPI entrypoint (with endpoints definitions)
│   ├── rag.py                      # Defines a class running whole RAG logic
│   └── requirements.txt            # Python dependecies
│
├── docker-compose.yml               # Project build definition
└── README.md                        # Project documentation       
```

### HOW TO RUN
- `docker compose up -d`
- go to `localhost:8000/docs` in browser (to access swagger) or just curl to `localhost:8000`
- to check unresolved queries, use other endpoint or enter container using `docker exec -it $(docker ps | grep fastapi | awk '{ print $1 }') cat memory/unresolved_queries.json`

### ENCOUNTERED ERRORS
- Error response from daemon: failed to set up container networking: driver failed programming external connectivity on endpoint ollama (3383e7a3034f2b4748c23133ad13395472b812f9424860753529e1abae9ef5af): failed to bind host port for 0.0.0.0:11434:172.23.0.4:11434/tcp: address already in use \
FIX: `sudo systemctl stop ollama`

### EXAMPLE

REQUEST:
```bash
curl -X 'POST' \
  'http://localhost:8000/ask?query=pomys%C5%82%20na%20prezent' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "retry_strats": [
    "change_interpretation",
    "modify_prompt",
    "save_to_memory"
  ]
}'
```

RESPONSE_BODY:
```json
{
  "model_answer": {
    "original_query": "pomysł na prezent",
    "answer": "[1] SŁOIKI TEMATYCZNE Świetny pomysł na prezent, gdy dobrze znamy zainteresowania obdarowywanej osoby – w tym przypadku przyda się duży słoik, który pomieści jak najwięcej gadżetów! \n",
    "chunks": [
      "SŁOIKI TEMATYCZNE Świetny pomysł na prezent, gdy dobrze znamy zainteresowania obdarowywanej osoby – w tym przypadku przyda się duży słoik, który pomieści jak najwięcej gadżetów!",
      "Pobierz DAR Load More... Follow on Instagram Linki Polityka Prywatności Regulamin sprzedaży Powrót na górę ©2021 Logopestka.pl Blog logopedyczny Oparte na Anima & WordPress.",
      "Czytaj cały artykuł cookies a sklep internetowydane osobowedane osobowe w sklepie internetowymGIODOinstrukcja GIODOinstrukcja wypełniania formularza GIODOjak napisać politykę prywatnościobowiązek informacyjny w zakresie danychpolityka prywatnościpolityka prywatności dla sklepu internetowegopolityka prywatności w e-sklepieskąd wziąć politykę prywatnościsklep internetowy a polityka prywatnościwzór polityki prywatności Zapisz się do newslettera Regularnie otrzymuj powiadomienia o nowych materiałach.",
      "Sylwia Fallopia 20 lutego 2017 11:27 Uroczy szkrab, zdrówka mu życzę :) Świetne prezenty przygotowałaś :) Małgorzata Zoltek 20 lutego 2017 17:05 Piotruś zuch chłopak i jaki wesoły.",
      "Share on Facebook Share Share on TwitterTweet Share on Pinterest Share Dodaj komentarz Anuluj pisanie odpowiedzi Twój adres e-mail nie zostanie opublikowany.",
      "Dodaj komentarz Anuluj pisanie odpowiedzi Komentarz Nazwa * E-mail * Witryna internetowa Zapisz moje dane, adres e-mail i witrynę w przeglądarce aby wypełnić dane podczas pisania kolejnych komentarzy."
    ],
    "decomposition": {
      "main_question": "Pytanie o pomysł na prezent",
      "sub_questions": [],
      "decomposition_type": "simple"
    },
    "stats": {
      "tokens_used": 207,
      "input_docs": 15,
      "kept_docs": 7,
      "rejected_short": 8,
      "rejected_overlap": 0,
      "overlaps": [
        2,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "citations": 1
    },
    "clarification": {
      "needs_clarification": false,
      "original_query": "pomysł na prezent",
      "interpretations": [],
      "method": "heuristic_clear"
    }
  }
}
```

