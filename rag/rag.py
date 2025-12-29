from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch
import spacy
from ollama import Client

from common import *

from reasoning.validation import CitationValidator
from reasoning.decomposition import decompose_query
from reasoning.chunking import chunk_document
from reasoning.filtering import filter_retrieved_with_stats
from reasoning.clarification import *
from reasoning.prompt import ask_model

from retrieval.elastic import search_es
from retrieval.qdrant import search_qdrant
from retrieval.fusion import rrf_fusion_weighted

from memory.unresolved_memory import UnresolvedQueriesMemory

class RAG:

    def __init__(
            self,
            memory: UnresolvedQueriesMemory, 
            prompt_core_list: List[str],
            ollama_model_name: str,
            data_source_path: str,
            transformer_model_name: str = "intfloat/multilingual-e5-small",
            spacy_model_name = "pl_core_news_sm",
            qdrant_collection_name: str = "culturax",
            es_index_name: str = "culturax",
            enable_decomposition: bool = True,
            es_url: str = "http://localhost:9200",
            qdrant_url: str = "http://localhost:6333",
            ollama_host: str = "http://ollama:11434"
            ):
        self.transformer_model = SentenceTransformer(transformer_model_name)
        self.memory = memory
        self.validator = CitationValidator()
        self.prompt_core_list = prompt_core_list
        self.es_index_name = es_index_name
        self.qdrant_collection_name = qdrant_collection_name
        self.enable_decomposition = enable_decomposition
        self.ollama_model_name = ollama_model_name

        self.es_client = Elasticsearch(es_url)
        self.qdrant_client = QdrantClient(qdrant_url)
        self.nlp = spacy.load(spacy_model_name)
        self.ollama_client = Client(ollama_host)

        self._initialize_engines(data_source_path)
        self._ensure_model_exists()

    def _initialize_engines(self, data_path):
        create_es_index(self.es_index_name, self.es_client)
        populate_index(data_path, self.es_index_name, self.es_client)
        create_qdrant_collection(self.qdrant_collection_name, self.qdrant_client)
        populate_collection(data_path, self.qdrant_collection_name, self.qdrant_client)
    
    def _ensure_model_exists(self):
        current_models = self.ollama_client.list()
        # Check if the model is already in the list of downloaded models
        if not any(m['model'].startswith(self.ollama_model_name) for m in current_models.get('models', [])):
            print(f"Downloading model {self.ollama_model_name}... this may take a while.")
            self.ollama_client.pull(self.ollama_model_name)

    def rag_query_enhanced(
        self,
        user_input: str,
        result: Dict,
        prompt_id: int,
        max_chunk_tokens=200,
        max_tokens_len=250,
    ) -> Dict:
        """
        Rozszerzona wersja RAG z dekompozycją i clarification.
        """
        features = analyze_query(user_input)

        # 2. Dekompozycja zapytania 
        if self.enable_decomposition:
            decomposition = decompose_query(user_input, features, self.ollama_model_name, self.ollama_client)
            result["decomposition"] = decomposition
            
            if len(decomposition['sub_questions']) > 0:
                print(f"\nDEKOMPOZYCJA ZAPYTANIA:")
                print(f"Główne pytanie: {decomposition['main_question']}")
                if decomposition["sub_questions"]:
                    print(f"Podzapytania ({len(decomposition['sub_questions'])}):")
                    for i, sq in enumerate(decomposition["sub_questions"], 1):
                        print(f"  {i}. {sq}")
            
        
        # 3. Logika RAG
        queries_to_process = [user_input]
        
        if self.enable_decomposition and result["decomposition"]["sub_questions"]:
            queries_to_process.extend(result["decomposition"]["sub_questions"])
        
        all_chunks_with_scores = []
        user_input_vec = None
        
        for i, query in enumerate(queries_to_process):
            qdrant_query, es_query = make_queries(query, self.nlp)
            vec = embed(qdrant_query, self.transformer_model)
            
            if i == 0:
                user_input_vec = vec

            ids_qdrant, texts_qdrant = search_qdrant(vec, self.qdrant_client, self.qdrant_collection_name)
            ids_es, texts_es = search_es(es_query, self.es_client, self.es_index_name)
            
            weights = choose_weights(features)
            
            fused_results = rrf_fusion_weighted(
                ids_qdrant,
                ids_es,
                texts_qdrant,
                texts_es,
                qdrant_weight=weights["qdrant"],
                es_weight=weights["es"],
                k=15
            )
            
            for text, score in fused_results:
                chunks = chunk_document(text, self.nlp, max_tokens=max_chunk_tokens)
                for chunk in chunks:
                    all_chunks_with_scores.append((chunk, score))


        best_chunk_scores = {}
        for chunk, score in all_chunks_with_scores:
            if chunk not in best_chunk_scores:
                best_chunk_scores[chunk] = score
            else:
                best_chunk_scores[chunk] = max(best_chunk_scores[chunk], score)

        all_chunks_with_scores = list(best_chunk_scores.items())
        all_chunks_with_scores.sort(key=lambda x: x[1], reverse=True)

        chunks_only = [chunk for chunk, _ in all_chunks_with_scores]

        # 5. Filtracja
        filtered_chunks, filter_stats = filter_retrieved_with_stats(
            chunks_only,
            user_input,
            user_input_vec,
            features,
            max_docs=10
        )
        
        # 6. Limit tokenów
        used_chunks = []
        used_len = 0

        for chunk in filtered_chunks:
            tokens = tokenize_regex(chunk)
            if used_len + len(tokens) <= max_tokens_len:
                used_chunks.append(chunk)
                used_len += len(tokens)
            else:
                break
        
        result["chunks"] = used_chunks
        result["stats"]["tokens_used"] = used_len
        result["stats"].update(filter_stats)
        
        print(f"\nUżyto {used_len} tokenów w {len(used_chunks)} chunkach")
        

        response = ask_model(used_chunks, self.prompt_core_list, prompt_id, user_input, self.ollama_model_name, self.ollama_client)

        result["answer"] = response["message"]["content"]
        result["stats"]["citations"] = count_citations(result["answer"])

        print(f'[INFO] model answer: {result["answer"]}')
        
        return result
    
    def full_rag_process(
            self,
            user_input: str,
            retry_strategies: List[str],
            max_chunk_tokens=200,
            max_tokens_len=250
        ) -> Dict:
            
            result = self.generate_result(user_input)
            interpretations, interpretation_req = clarify_query(result, user_input, self.ollama_model_name, self.ollama_client)
            interpretation_idx = 0
            if interpretation_req:
                final_user_input = user_input + ' ' + interpretations[interpretation_idx]
            else:
                final_user_input = user_input
            print(f"[INFO] RAG działa dla zapytania: {final_user_input}")
            
            prompt_core_idx = 0
            result = self.rag_query_enhanced(final_user_input, 
                                        result,
                                        prompt_core_idx,
                                        max_chunk_tokens,
                                        max_tokens_len)
            
            is_answer_valid = self.evaluate_answer(result["answer"], result["stats"], result["chunks"])

            while not is_answer_valid:
                if "modify_prompt" in retry_strategies:
                    prompt_core_idx += 1
                    if prompt_core_idx < len(self.prompt_core_list):
                        print(f"[INFO] Błąd, próba z promptem nr {prompt_core_idx+1}")
                        response = ask_model(result["chunks"], self.prompt_core_list, prompt_core_idx, 
                                             final_user_input, self.ollama_model_name, self.ollama_client)

                        new_answer = response["message"]["content"]
                        result["stats"]["citations"] = count_citations(new_answer)
                        is_answer_valid = self.evaluate_answer(new_answer, result["stats"], result["chunks"])
                        if is_answer_valid:
                            result["answer"] = new_answer
                    else:
                        print(f"[WARN] Brak innych promptów do wykorzystania")
                        retry_strategies.remove("modify_prompt")
                        continue
                if "change_interpretation" in retry_strategies:
                    if interpretation_idx + 1 < len(interpretations):
                        interpretation_idx += 1
                        final_user_input = user_input + ' ' + interpretations[interpretation_idx]
                        print(f"[INFO] Błąd, ponowna próba dla nowej interpretacji nr {interpretation_idx+1}: {final_user_input}")
                        result = self.rag_query_enhanced(final_user_input, 
                                        result,
                                        prompt_core_idx,
                                        max_chunk_tokens,
                                        max_tokens_len)
            
                        is_answer_valid = self.evaluate_answer(result["answer"], result["stats"], result["chunks"])
                        
                    else:
                        print(f"[WARN] Brak wielu interpretacji")
                        retry_strategies.remove("change_interpretation")
                        continue
                if "save_to_memory" in retry_strategies:
                    print(f"[INFO] Błąd w odpowiedzi, zapis pytania do pamięci")
                    self.memory.add_query(user_input)
                    return result
                else:
                    print(f"[WARN] Nieznana strategia rozwiązania błędu, zapis pytania do pamięci")
                    self.memory.add_query(user_input)
                    return result
            return result
        

    def generate_result(self, query: str):
        result = {
            "original_query": query,
            "answer": "",
            "chunks": [],
            "decomposition": None,
            "stats": {}
        }
        return result

    def evaluate_answer(self, model_answer, model_stats, chunks):
        if self.memory.should_save_as_unresolved(model_answer, chunks, model_stats):
            print("[WARN] Model nie był w stanie odpowiedzieć na podstawie podanych fragmentów")
            return False
        elif not self.validator.validate_answer(model_answer, chunks):
            print("[ERR] Model zwrócił błędne cytaty")
            return False
        return True
