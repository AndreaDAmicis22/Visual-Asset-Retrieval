"""
rag.py — Graph RAG con Groq (Llama 3)
===================================
Combina:
  1. FAISS  → recupero semantico veloce
  2. NetworkX → contesto arricchito dal grafo
  3. Groq API → risposta ultra-rapida con Llama 3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dotenv import load_dotenv
from groq import Groq

from semantic_search.encoder import encode_text
from semantic_search.graph_query import serialize_subgraph
from semantic_search.search import search

load_dotenv()

if TYPE_CHECKING:
    import networkx as nx

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are an assistant specialized in the analysis of photographic archives.
You are part of an image retrieval system indexed with CLIP and a graph structure.
You will be provided with a structured context extracted from an image graph (COCO dataset with short English descriptions), which includes:
    - The most relevant images for the user's query
    - Their captions
    - Similar images linked within the graph
    - Entities (objects, subjects) present in each image
Respond in English in a precise and concise manner, based exclusively on the provided context.
If the context is insufficient to provide an answer, state so explicitly."""


def _build_context(
    G: nx.DiGraph,
    index,
    metadata: list[dict],
    model,
    processor,
    query: str,
    top_k: int = 3,
    graph_depth: int = 2,
) -> str:
    """
    Pipeline di recupero:
    1. FAISS → top_k immagini più simili alla query
    2. Per ognuna → serializza il sottografo dal grafo
    3. Concatena tutto come contesto testuale
    """
    query_vec = encode_text(model, processor, query)
    results = search(index, metadata, query_vec, top_k=top_k)

    if not results:
        return "Nessuna immagine rilevante trovata nell'archivio."

    context_parts = [f'Query: "{query}"\n', "Immagini rilevanti trovate:\n"]

    for i, result in enumerate(results, 1):
        fname = result["filename"]
        context_parts.append(f"--- Risultato #{i} (score FAISS: {result['score']:.3f}) ---")

        if fname in G:
            context_parts.append(serialize_subgraph(G, fname, depth=graph_depth))
        else:
            context_parts.append(f"[Immagine] {fname}\n  (non presente nel grafo)\n")

    return "\n".join(context_parts)


def graph_rag_query(
    query: str,
    G: nx.DiGraph,
    index,
    metadata: list[dict],
    model,
    processor,
    top_k: int = 3,
    graph_depth: int = 2,
) -> str:
    """
    Esegue una query Graph RAG completa e restituisce la risposta tramite Groq.
    """

    context = _build_context(G, index, metadata, model, processor, query, top_k, graph_depth)

    client = Groq()

    chat_completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context extracted from archive:\n\n{context}\n\nQuestion: {query}",
            },
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.15,
    )

    return chat_completion.choices[0].message.content


def print_rag_response(query: str, response: str):
    print(f"\n{'═' * 55}")
    print(f'  Query: "{query}"')
    print(f"{'═' * 55}")
    print(f"\n{response}\n")
    print("═" * 55 + "\n")
