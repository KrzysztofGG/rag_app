from typing import List
from ollama import Client


def build_prompt(chunks: List[str], prompt_core: str, question: str) -> str:
    context = "\n\n".join(
        [f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)]
    )

    prompt = f"{prompt_core}\nFragmenty:\n{context}\n\nPytanie:\n{question}"
    return prompt

def ask_model(chunks: List[str], 
                prompts_list: List[str],
                prompt_idx: int, 
                query: str, 
                ollama_model: str,
                ollama_client: Client):
    prompt_core = prompts_list[prompt_idx]
    prompt = build_prompt(chunks, prompt_core, query)

    model_resp = ollama_client.chat(
        model=ollama_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.6}
    )
    return model_resp