"""
graph_query.py — Navigazione del grafo
=======================================
Funzioni per interrogare il grafo senza LLM:
  - immagini simili a una data immagine
  - immagini che condividono entità
  - sottografo attorno a un nodo
"""

from __future__ import annotations

import networkx as nx

SIMILAR_TO = "SIMILAR_TO"
CONTAINS = "CONTAINS"


def similar_images(G: nx.DiGraph, filename: str, top_k: int = 5) -> list[dict]:
    """
    Restituisce le top_k immagini più simili a `filename` navigando gli archi SIMILAR_TO.
    """
    if filename not in G:
        msg = f"Immagine non trovata nel grafo: {filename}"
        raise KeyError(msg)

    neighbors = [(dst, data["weight"]) for dst, data in G[filename].items() if data.get("relation") == SIMILAR_TO]
    neighbors.sort(key=lambda x: x[1], reverse=True)

    return [{**G.nodes[dst], "score": score} for dst, score in neighbors[:top_k]]


def images_by_entity(G: nx.DiGraph, entity: str) -> list[dict]:
    """
    Restituisce tutte le immagini che contengono una data entità.
    """
    entity = entity.lower()
    if entity not in G:
        return []

    # Archi entranti nel nodo entità provengono da immagini
    return [G.nodes[src] for src, _, data in G.in_edges(entity, data=True) if data.get("relation") == CONTAINS]


def shared_entities(G: nx.DiGraph, filename_a: str, filename_b: str) -> list[str]:
    """
    Restituisce le entità in comune tra due immagini.
    """

    def _entities(fname):
        return {dst for dst, data in G[fname].items() if data.get("relation") == CONTAINS}

    return sorted(_entities(filename_a) & _entities(filename_b))


def subgraph_around(G: nx.DiGraph, filename: str, depth: int = 2) -> nx.DiGraph:
    """
    Estrae il sottografo entro `depth` hop da `filename`.
    Utile per serializzarlo e passarlo al LLM.
    """
    nodes = nx.single_source_shortest_path_length(G, filename, cutoff=depth)
    return G.subgraph(nodes.keys()).copy()


def serialize_subgraph(G: nx.DiGraph, filename: str, depth: int = 2) -> str:
    """
    Serializza il sottografo in testo strutturato da passare al LLM come contesto.
    """
    sub = subgraph_around(G, filename, depth)
    lines = [f"Sottografo attorno a: {filename}\n"]

    image_nodes = [(n, d) for n, d in sub.nodes(data=True) if d.get("node_type") == "image"]
    for node, data in image_nodes:
        lines.append(f"[Immagine] {node}")
        caps = data.get("captions", [])
        if caps:
            lines.append(f"  Didascalie: {' | '.join(caps[:2])}")

        # Simili
        simili = [(dst, edata["weight"]) for dst, edata in sub[node].items() if edata.get("relation") == "SIMILAR_TO"]
        simili.sort(key=lambda x: x[1], reverse=True)
        for dst, w in simili[:3]:
            lines.append(f"  SIMILAR_TO {dst} (score: {w:.3f})")

        # Entità
        entities = [dst for dst, edata in sub[node].items() if edata.get("relation") == "CONTAINS"]
        if entities:
            lines.append(f"  CONTAINS: {', '.join(entities[:10])}")

        lines.append("")

    return "\n".join(lines)
