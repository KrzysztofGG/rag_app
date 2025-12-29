from typing import List, Dict
import re
from ollama import Client

from common.util import (
    TOKEN_RE,
    ID_RE,
    ACRONYM_RE,
    YEAR_RE,
)

def detect_ambiguity_hybrid(user_input: str) -> Dict:
    text_lower = user_input.lower()
    tokens = TOKEN_RE.findall(text_lower)
    
    # KROK 1: Szybkie heurystyki wykluczające (high precision)
    # Jeśli zapytanie ma te cechy, na pewno NIE jest niejednoznaczne
    if any([
        ID_RE.search(user_input),  # ma ID dokumentu
        ACRONYM_RE.fullmatch(user_input),  # samo akronim
        YEAR_RE.search(user_input) and len(tokens) <= 8,  # konkretna data + krótkie
        any(t.isdigit() for t in tokens) and len(tokens) <= 6,  # liczby + krótkie
    ]):
        return {
            "is_ambiguous": False,
            "confidence": 0.9,
            "reason": "zapytanie konkretne (ID/liczby/data)",
            "ambiguity_type": None,
            "method": "heuristic_exclude"
        }
    
    # KROK 2: Heurystyki wskazujące niejednoznaczność (high recall)
    ambiguity_signals = []
    
    # Wieloznaczne encje
    ambiguous_entities = {
        "pan": "PAN (instytucja) vs pan (osoba/grzecznościowe)",
        "rada": "która rada? (ministrów, nadzorcza, etc.)",
        "instytut": "który instytut?",
        "komisja": "która komisja?",
        "program": "jaki program? (komputerowy, polityczny, edukacyjny)",
        "organizacja": "która organizacja?"
    }
    
    for entity, desc in ambiguous_entities.items():
        if entity in text_lower and not any(x in text_lower for x in ["który", "jaki", "która", "jakie"]):
            ambiguity_signals.append(("entity", entity, desc))
    
    # Abstrakcyjne pojęcia bez kontekstu
    abstract_concepts = {
        "sens": "sens moralny/praktyczny/egzystencjalny?",
        "znaczenie": "znaczenie słowa/wydarzenia/symboliczne?",
        "odpowiedzialność": "moralna/prawna/społeczna/zawodowa?",
        "sukces": "sukces finansowy/osobisty/zawodowy?",
        "kryzys": "kryzys ekonomiczny/polityczny/osobisty/zdrowotny?",
        "efektywność": "efektywność czego dokładnie?",
        "rozwój": "rozwój osobisty/zawodowy/gospodarczy?",
        "zarządzanie": "zarządzanie czym? (ludźmi/projektem/firmą/czasem)"
    }
    
    for concept, desc in abstract_concepts.items():
        # Sprawdź czy jest kontekst
        has_context = any(ctx in text_lower for ctx in [
            "w kontekście", "w zakresie", "odnośnie", "dotycząc",
            "w przypadku", "dla", "przy"
        ])
        
        if concept in text_lower and not has_context:
            ambiguity_signals.append(("abstract", concept, desc))
    
    # Ogólne pytania bez zakresu
    if any(phrase in text_lower for phrase in ["jak zarządzać", "jak poprawić", "jak zwiększyć"]):
        # Sprawdź czy ma kontekst
        has_scope = any(s in text_lower for s in [
            "w firmie", "w zespole", "w projekcie", "w organizacji",
            "w przypadku", "dla", "przy"
        ])
        if not has_scope:
            ambiguity_signals.append(("scope", "brak zakresu", "nie określono kontekstu/zakresu"))
    
    # KROK 3: Decyzja
    if len(ambiguity_signals) == 0:
        return {
            "is_ambiguous": False,
            "confidence": 0.8,
            "reason": "zapytanie wydaje się konkretne",
            "ambiguity_type": None,
            "method": "heuristic_clear"
        }
    
    # Jest przynajmniej jeden sygnał niejednoznaczności
    primary_signal = ambiguity_signals[0]
    
    return {
        "is_ambiguous": True,
        "confidence": min(0.7 + len(ambiguity_signals) * 0.1, 0.95),
        "reason": f"wykryto niejednoznaczność: {primary_signal[2]}",
        "ambiguity_type": primary_signal[0],
        "signals": ambiguity_signals,
        "method": "heuristic_detect"
    }


def generate_clarification_question(user_input: str, ollama_model: str, ollama_client: Client) -> Dict:
    # KROK 1: Sprawdź czy jest niejednoznaczne
    ambiguity = detect_ambiguity_hybrid(user_input)
    
    if not ambiguity["is_ambiguous"]:
        return {
            "needs_clarification": False,
            "original_query": user_input,
            "interpretations": [],
            "method": ambiguity.get("method", "unknown")
        }
    
    # KROK 2: Przygotuj kontekst dla LLM na podstawie wykrytych sygnałów
    signals = ambiguity.get("signals", [])
    signal_desc = ""
    
    if signals:
        signal_type, signal_term, signal_explanation = signals[0]
        signal_desc = f"\n\nWykryto niejednoznaczność w terminie '{signal_term}': {signal_explanation}"
    
    # KROK 3: Uproszczony prompt z przykładami (few-shot)
    prompt = f"""Zapytanie użytkownika jest niejednoznaczne.
TWOJE ZADANIE:
Napisz 2-3 interpretacje W FORMIE ZDAŃ TWIERDZĄCYCH (nie pytań!).
Każda interpretacja powinna zaczynać się od "pytanie dotyczy" lub podobnego sformułowania.

PRZYKŁADY:

Zapytanie: "Co mówi PAN o kryzysie?"
Interpretacje:
pytanie dotyczy Polskiej Akademii Nauk (instytucja)
pytanie dotyczy wypowiedzi konkretnej osoby (pan jako osoba)

Zapytanie: "Jaki ma sens odpowiedzialność?"
Interpretacje:
pytanie dotyczy odpowiedzialności w kontekście moralnym
pytanie dotyczy odpowiedzialności w kontekście praktycznym (biznes, zarządzanie)
pytanie dotyczy odpowiedzialności w kontekście egzystencjalnym (filozofia życia)

ZAPYTANIE: "{user_input}"{signal_desc}

Napisz tylko interpretacje w formie zdań twierdzących, każda w nowej linii."""

    try:
        response = ollama_client.chat(
            model=ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "top_p": 0.9}
        )
        
        content = response["message"]["content"].strip()
        
        # KROK 4: Parsowanie odpowiedzi
        interpretations = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Usuń numerację jeśli jest (1., 2., -, *)
            line = re.sub(r'^[\d\-\*\.]+\s*', '', line)
            
            # Sprawdź czy to sensowna interpretacja
            if len(line) > 10 and not line.startswith('Interpretacje'):
                interpretations.append({
                    "label": f"Interpretacja {len(interpretations) + 1}",
                    "clarification": line
                })
        
        # KROK 5: Jeśli LLM zawiódł, użyj heurystyk
        if len(interpretations) == 0 and signals:
            # Wygeneruj interpretacje na podstawie wykrytych sygnałów
            for i, (sig_type, sig_term, sig_desc) in enumerate(signals[:3], 1):
                # Przekształć opis na zdanie twierdzące
                if sig_type == "entity":
                    # Dla encji: "PAN (instytucja) vs pan (osoba)" -> dwie interpretacje
                    parts = sig_desc.split(" vs ")
                    if len(parts) == 2:
                        interpretations.append({
                            "label": f"Interpretacja {len(interpretations) + 1}",
                            "clarification": f"pytanie dotyczy {parts[0].strip()}"
                        })
                        if len(interpretations) < 3:
                            interpretations.append({
                                "label": f"Interpretacja {len(interpretations) + 1}",
                                "clarification": f"pytanie dotyczy {parts[1].strip()}"
                            })
                    else:
                        interpretations.append({
                            "label": f"Interpretacja {len(interpretations) + 1}",
                            "clarification": f"pytanie dotyczy {sig_term}"
                        })
                elif sig_type == "abstract":
                    # Dla abstrakcyjnych pojęć: usuń pytajnik i przekształć
                    clean_desc = sig_desc.replace("?", "").strip()
                    # "sens moralny/praktyczny/egzystencjalny" -> rozdziel
                    if "/" in clean_desc:
                        variants = clean_desc.split("/")
                        for variant in variants[:2]:  # max 2 warianty
                            if len(interpretations) < 3:
                                interpretations.append({
                                    "label": f"Interpretacja {len(interpretations) + 1}",
                                    "clarification": f"pytanie dotyczy {sig_term} - {variant.strip()}"
                                })
                    else:
                        interpretations.append({
                            "label": f"Interpretacja {len(interpretations) + 1}",
                            "clarification": f"pytanie dotyczy {clean_desc}"
                        })
                else:
                    # Dla innych typów
                    interpretations.append({
                        "label": f"Interpretacja {len(interpretations) + 1}",
                        "clarification": f"pytanie dotyczy {sig_desc.replace('?', '').strip()}"
                    })
        
        # Minimum 2 interpretacje
        if len(interpretations) < 2:
            interpretations.append({
                "label": f"Interpretacja {len(interpretations) + 1}",
                "clarification": "pytanie wymaga doprecyzowania kontekstu"
            })
        
        return {
            "needs_clarification": True,
            "original_query": user_input,
            "interpretations": interpretations[:3],  # max 3
            "ambiguity_info": ambiguity,
            "method": "llm_with_fallback"
        }
        
    except Exception as e:
        print(f"Błąd generowania clarification: {e}")
        
        # OSTATECZNY FALLBACK: użyj tylko heurystyk
        if signals:
            interpretations = []
            for i, (sig_type, sig_term, sig_desc) in enumerate(signals[:3], 1):
                if sig_type == "entity" and " vs " in sig_desc:
                    parts = sig_desc.split(" vs ")
                    interpretations.append({
                        "label": f"Interpretacja {len(interpretations) + 1}",
                        "clarification": f"pytanie dotyczy {parts[0].strip()}"
                    })
                    if len(interpretations) < 3 and len(parts) > 1:
                        interpretations.append({
                            "label": f"Interpretacja {len(interpretations) + 1}",
                            "clarification": f"pytanie dotyczy {parts[1].strip()}"
                        })
                else:
                    clean_desc = sig_desc.replace("?", "").strip()
                    interpretations.append({
                        "label": f"Interpretacja {len(interpretations) + 1}",
                        "clarification": f"pytanie dotyczy {clean_desc}"
                    })
            
            return {
                "needs_clarification": True,
                "original_query": user_input,
                "interpretations": interpretations,
                "ambiguity_info": ambiguity,
                "method": "heuristic_fallback",
                "error": str(e)
            }
        
        # Jeśli wszystko zawiedzie
        return {
            "needs_clarification": False,
            "original_query": user_input,
            "interpretations": [],
            "method": "error",
            "error": str(e)
        }
    
def clarify_query(result: Dict, query: str, ollama_model: str, ollama_client: Client) -> tuple[List[str], bool]:
    clarification = generate_clarification_question(query, ollama_model, ollama_client)
    result["clarification"] = clarification
    
    if clarification["needs_clarification"]:
        print(f"\n[WARN] - WYKRYTO NIEJEDNOZNACZNOŚĆ")
        print(f"Zapytanie: {query}")
        print(f"Pewność: {clarification.get('confidence', 0):.2f}")
        print(f"Powód: {clarification.get('reason', 'brak')}")
        print(f"Możliwe interpretacje:")
        for i, interp in enumerate(clarification["interpretations"], 1):
            print(f"  {i}. {interp.get('clarification', 'brak')}")
        
        print(f'Dalszy proces dla interpretacji: {clarification["interpretations"][0]["clarification"]}')
        return [data['clarification'] for data in clarification["interpretations"]], True
        
    print(f"[INFO] Brak dwuznaczności, jedna interpretacja")
    return [], False