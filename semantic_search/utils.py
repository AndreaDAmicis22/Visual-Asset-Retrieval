import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

from semantic_search.encoder import encode_text
from semantic_search.index import load_index
from semantic_search.model import load_model
from semantic_search.search import search


def visual_search(query: str, top_k: int = 6, cols: int = 3):
    """
    Esegue la ricerca semantica e mostra i risultati come griglia di immagini.

    Parametri
    ----------
    query  : testo della ricerca (italiano o inglese)
    top_k  : numero di risultati da mostrare (default 6)
    cols   : colonne della griglia (default 3)
    """
    # ── Ricerca ────────────────────────────────────────
    model, processor = load_model()
    index, metadata = load_index()
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
