"""
graph.py — Costruzione e persistenza del grafo NetworkX
========================================================
Nodi:
  - Image  : ogni immagine indicizzata
  - Entity : noun chunk estratto dalle didascalie

Archi:
  - SIMILAR_TO : top-k vicini per immagine (da FAISS)
  - CONTAINS   : immagine → entità estratta dalla didascalia
"""

import json
import re
from pathlib import Path

import networkx as nx
import numpy as np

from semantic_search.config import DEFAULT_MODEL, PROJECT_ROOT, get_index_paths

CAPTIONS_PATH = PROJECT_ROOT / "data" / "captions.json"
TOP_K_SIMILAR = 5  # vicini per arco SIMILAR_TO
MIN_ENTITY_LEN = 2  # scarta parole troppo corte
SIMILAR_TO = "SIMILAR_TO"
CONTAINS = "CONTAINS"


# ─────────────────────────────────────────────
# Estrazione entità dalle didascalie
# ─────────────────────────────────────────────

# Stopwords leggere — evita un modello NLP pesante
_STOPWORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "and",
    "or",
    "but",
    "not",
    "no",
    "its",
    "their",
    "some",
    "many",
    "few",
    "two",
    "three",
    "four",
    "five",
    "very",
    "just",
    "also",
    "as",
    "it",
    "he",
    "she",
    "they",
}


def _extract_entities(captions: list[str]) -> set[str]:
    """
    Estrae noun chunks semplici dalle didascalie:
    - tokenizza per spazio e punteggiatura
    - rimuove stopwords e parole corte
    - restituisce un set di token normalizzati
    """
    entities = set()
    for cap in captions:
        tokens = re.findall(r"[a-zA-Z]+", cap.lower())
        for tok in tokens:
            if tok not in _STOPWORDS and len(tok) >= MIN_ENTITY_LEN:
                entities.add(tok)
    return entities


# ─────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────


def build_graph(
    index,
    metadata: list[dict],
    model_name: str = DEFAULT_MODEL,
    top_k_similar: int = TOP_K_SIMILAR,
) -> nx.DiGraph:
    """
    Costruisce il grafo a partire da indice FAISS e metadati già caricati.
    Non ricalcola nessun embedding.
    """
    print("\n" + "═" * 55)
    print("  BUILD GRAPH")
    print("═" * 55)

    G = nx.DiGraph()

    # ── Carica didascalie ──────────────────────────────
    captions: dict[str, list[str]] = {}
    if CAPTIONS_PATH.exists():
        with open(CAPTIONS_PATH, encoding="utf-8") as f:
            captions = json.load(f)
        print(f"  Didascalie caricate: {len(captions)} immagini")
    else:
        print(f"  [WARN] {CAPTIONS_PATH} non trovato — nodi senza didascalie")

    # ── Nodi Image ─────────────────────────────────────
    print(f"  Aggiunta {len(metadata)} nodi Image...")
    for entry in metadata:
        fname = entry["filename"]
        G.add_node(
            fname,
            node_type="image",
            path=entry["path"],
            filename=fname,
            captions=captions.get(fname, []),
        )

    # ── Archi SIMILAR_TO (da FAISS kNN) ───────────────
    print(f"  Calcolo similarità top-{top_k_similar} per ogni immagine...")
    # Recupera tutti i vettori dall'indice FAISS
    all_vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
    for i in range(index.ntotal):
        all_vectors[i] = index.reconstruct(i)

    # kNN: top_k_similar + 1 perché il primo risultato è l'immagine stessa
    scores, indices = index.search(all_vectors, top_k_similar + 1)

    edge_count = 0
    for i, (row_scores, row_indices) in enumerate(zip(scores, indices, strict=False)):
        src = metadata[i]["filename"]
        for score, j in zip(row_scores[1:], row_indices[1:], strict=False):  # skip self (idx 0)
            if j < 0:
                continue
            dst = metadata[j]["filename"]
            G.add_edge(src, dst, relation=SIMILAR_TO, weight=float(score))
            edge_count += 1

    print(f"  Archi SIMILAR_TO aggiunti: {edge_count}")

    # ── Nodi Entity + archi CONTAINS ──────────────────
    print("  Estrazione entità dalle didascalie...")
    entity_count = 0
    contains_count = 0

    for entry in metadata:
        fname = entry["filename"]
        caps = captions.get(fname, [])
        if not caps:
            continue

        entities = _extract_entities(caps)
        for entity in entities:
            # Aggiungi nodo Entity se non esiste
            if not G.has_node(entity):
                G.add_node(entity, node_type="entity", label=entity)
                entity_count += 1
            # Arco Image → Entity
            if not G.has_edge(fname, entity):
                G.add_edge(fname, entity, relation=CONTAINS)
                contains_count += 1

    print(f"  Nodi Entity aggiunti: {entity_count}")
    print(f"  Archi CONTAINS aggiunti: {contains_count}")

    # ── Statistiche finali ─────────────────────────────
    n_images = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "image")
    n_entities = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "entity")
    print("\n  Grafo costruito:")
    print(f"    Nodi  : {G.number_of_nodes()} ({n_images} immagini, {n_entities} entità)")
    print(f"    Archi : {G.number_of_edges()}")
    print("═" * 55 + "\n")

    return G


# ─────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────


def get_graph_path(model_name: str = DEFAULT_MODEL) -> Path:
    index_path, _ = get_index_paths(model_name)
    return index_path.parent / "graph.json"


def save_graph(G: nx.DiGraph, model_name: str = DEFAULT_MODEL):
    path = get_graph_path(model_name)
    data = nx.node_link_data(G)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Grafo salvato: {path}")
    print(f"  ({G.number_of_nodes()} nodi, {G.number_of_edges()} archi)")


def load_graph(model_name: str = DEFAULT_MODEL) -> nx.DiGraph:
    path = get_graph_path(model_name)
    if not path.exists():
        msg = f"Grafo non trovato in {path}. Esegui prima: python main.py --build-graph"
        raise FileNotFoundError(msg)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    G = nx.node_link_graph(data)
    print(f"  Grafo caricato: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")
    return G
