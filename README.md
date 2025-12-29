### HOW TO RUN
- `docker compose up -d`
- go to `localhost:8000/docs` in browser (to access swagger) or just curl to `localhost:8000`
- to check unresolved queries, use other endpoint or enter container using `docker exec -it {rag_app-fastapi CONTAINER ID} bash` and `cat memory/unresolved_queries.json`

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
}```

