import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from PIL import Image

from semantic_search.config import DEFAULT_MODEL, get_index_paths
from semantic_search.encoder import encode_text
from semantic_search.graph import load_graph
from semantic_search.index import load_index
from semantic_search.model import load_model
from semantic_search.rag import graph_rag_query
from semantic_search.search import search


def visual_search(query: str, top_k: int = 6, cols: int = 3, model_name: str = DEFAULT_MODEL):
    """
    Esegue la ricerca semantica e mostra i risultati come griglia di immagini.

    Parametri
    ----------
    query  : testo della ricerca (italiano o inglese)
    top_k  : numero di risultati da mostrare (default 6)
    cols   : colonne della griglia (default 3)
    model_name : nome del modello HuggingFace (default: DEFAULT_MODEL)
    """
    # ── Ricerca ────────────────────────────────────────
    model, processor = load_model(model_name)
    index_path, meta_path = get_index_paths(model_name)
    index, metadata = load_index(index_path=index_path, meta_path=meta_path)
    query_vec = encode_text(model, processor, query)
    results = search(index, metadata, query_vec, top_k=top_k)

    if not results:
        print("Nessun risultato trovato.")
        return

    # ── Layout griglia ─────────────────────────────────
    rows = math.ceil(len(results) / cols)
    fig_w = cols * 4
    fig_h = rows * 3.8 + 0.6  # spazio extra per il titolo

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle(f'🔍  "{query}"', fontsize=14, fontweight="bold", color="white", y=1.01)

    # Normalizza axes in lista piatta anche con 1 riga
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # ── Colore score ───────────────────────────────────
    def score_color(s):
        if s >= 0.30:
            return "#4ade80"  # verde  — ottimo
        if s >= 0.25:
            return "#facc15"  # giallo — buono
        return "#f87171"  # rosso  — basso

    # ── Disegna ogni risultato ─────────────────────────
    for i, result in enumerate(results):
        row, col = divmod(i, cols)
        ax = axes[row][col]
        ax.set_facecolor("#1a1a1a")

        try:
            img = Image.open(result["path"]).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Errore:\n{e}", ha="center", va="center", color="red", transform=ax.transAxes)

        # Badge score colorato
        score = result["score"]
        color = score_color(score)
        ax.text(
            0.02,
            0.97,
            f"  #{i + 1}  score: {score:.3f}  ",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color="black",
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "alpha": 0.9, "edgecolor": "none"},
        )

        # Nome file come label sotto
        ax.set_xlabel(result["filename"], fontsize=8, color="#aaaaaa", labelpad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # ── Nascondi celle vuote ───────────────────────────
    total_cells = rows * cols
    for j in range(len(results), total_cells):
        row, col = divmod(j, cols)
        axes[row][col].set_visible(False)

    # ── Legenda score ──────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="#4ade80", label="≥ 0.30  ottimo"),
        mpatches.Patch(color="#facc15", label="≥ 0.25  buono"),
        mpatches.Patch(color="#f87171", label="< 0.25  basso"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=8,
        facecolor="#1a1a1a",
        edgecolor="#333333",
        labelcolor="white",
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    plt.show()
    print(f'  {len(results)} risultati per: "{query}"')


def visual_rag(query: str, top_k: int = 4, cols: int = 2, graph_depth: int = 2, model_name: str = DEFAULT_MODEL):
    """
    Esegue una query Graph RAG e mostra sia la risposta testuale dell'LLM
    che le immagini caricate nel contesto.
    """
    model, processor = load_model(model_name)
    index_path, meta_path = get_index_paths(model_name)
    index, metadata = load_index(index_path=index_path, meta_path=meta_path)

    G = load_graph()

    query_vec = encode_text(model, processor, query)
    results = search(index, metadata, query_vec, top_k=top_k)

    if not results:
        print("Nessun risultato trovato per il contesto.")
        return

    response = graph_rag_query(query, G, index, metadata, model, processor, top_k=top_k, graph_depth=graph_depth)

    print(f"\n{'═' * 60}")
    print(f' 🤖 RAG RESPONSE for: "{query}"')
    print(f"{'═' * 60}")
    print(f"\n{response}\n")
    print(f"{'─' * 60}")
    print("🖼️  Context Images Used:")

    rows = math.ceil(len(results) / cols)
    fig_w = cols * 4.5
    fig_h = rows * 4

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0f0f0f")

    axes = np.array(axes).flatten()

    for i, result in enumerate(results):
        ax = axes[i]
        ax.set_facecolor("#1a1a1a")

        try:
            img = Image.open(result["path"]).convert("RGB")
            ax.imshow(img)

            ax.text(
                0.02,
                0.96,
                f" #{i + 1} | {result['filename']}",
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                color="white",
                va="top",
                ha="left",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "#3b82f6", "alpha": 0.8, "edgecolor": "none"},
            )
        except Exception:
            ax.text(0.5, 0.5, f"Error loading\n{result['filename']}", ha="center", va="center", color="red")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    for j in range(len(results), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def visual_rag_html(query: str, top_k: int = 4, graph_depth: int = 2, model_name: str = DEFAULT_MODEL):
    """
    Esegue una query Graph RAG e alterna testo e immagini nel notebook.
    """
    # --- Recupero dati (Logica esistente) ---
    model, processor = load_model(model_name)
    index_path, meta_path = get_index_paths(model_name)
    index, metadata = load_index(index_path=index_path, meta_path=meta_path)
    G = load_graph()

    # Esecuzione query Groq
    response = graph_rag_query(query, G, index, metadata, model, processor, top_k=top_k, graph_depth=graph_depth)

    # Recupero risultati FAISS per le immagini
    query_vec = encode_text(model, processor, query)
    results = search(index, metadata, query_vec, top_k=top_k)

    # --- Visualizzazione ---

    # Titolo e Risposta dell'AI
    display(
        HTML(f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 5px solid #3b82f6; margin-bottom: 20px;">
            <h2 style="color: #3b82f6; margin-top: 0;">🔍 Query: {query}</h2>
            <div style="color: #e0e0e0; font-size: 1.1em; line-height: 1.6;">{response.replace("\n", "<br>")}</div>
        </div>
    """)
    )

    display(HTML("<h3 style='color: white; margin-left: 10px;'>📸 Immagini di contesto utilizzate:</h3>"))

    # Ciclo sulle immagini
    for i, res in enumerate(results):
        # IMPORTANTE: Se sei su Windows, i path devono avere gli slash giusti per l'HTML
        img_path = os.path.abspath(res["path"]).replace("\\", "/")

        # Se stai usando Jupyter locale, a volte serve il prefisso file:///
        img_url = f"file:///{img_path}"

        html_item = f"""
        <div style="display: flex; align-items: flex-start; background-color: #252525; padding: 15px; border-radius: 8px; margin: 10px; border: 1px solid #444;">
            <div style="flex: 0 0 250px;">
                <img src="{img_url}" style="width: 100%; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.5);">
            </div>
            <div style="flex: 1; margin-left: 20px; color: #ccc;">
                <b style="color: #3b82f6; font-size: 1.1em;">#{i + 1} - {res["filename"]}</b><br>
                <p style="margin-top: 5px;"><b>FAISS Score:</b> {res["score"]:.4f}</p>
                <p style="font-size: 0.9em; font-style: italic; color: #888;">Path: {res["path"]}</p>
            </div>
        </div>
        """
        display(HTML(html_item))


def print_results(results: list[dict], query: str):
    print(f"\n{'─' * 55}")
    print(f'  Query: "{query}"')
    print(f"{'─' * 55}")

    if not results:
        print("  Nessun risultato trovato.")
        return

    for i, r in enumerate(results, 1):
        bar_len = int(r["score"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"\n  #{i}  {r['filename']}")
        print(f"      Score: {r['score']:.4f}  [{bar}]")
        print(f"      Path:  {r['path']}")

    print(f"\n{'─' * 55}\n")
