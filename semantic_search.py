"""
Semantic Visual Asset Retrieval — Prototipo v1
==============================================
Ricerca semantica su archivio fotografico tramite CLIP + FAISS.

Installazione dipendenze:
    pip install torch torchvision transformers faiss-cpu Pillow requests tqdm

Utilizzo rapido:
    python semantic_search.py --demo          # scarica immagini demo e indicizza
    python semantic_search.py --query "scavo con tubazioni gas gialle"
    python semantic_search.py --query "operai al lavoro in trincea" --top-k 5
"""

import os
import json
import time
import argparse
import warnings
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# Configurazione
# ─────────────────────────────────────────────
DEFAULT_MODEL   = "openai/clip-vit-base-patch32"   # ~150 MB, ottimo su CPU
IMAGE_DIR       = Path("images")                    # cartella immagini da indicizzare
INDEX_PATH      = Path("faiss_index.bin")           # indice FAISS salvato
META_PATH       = Path("metadata.json")             # metadati immagini
EMBED_DIM       = 512                               # dimensione embedding CLIP ViT-B/32
TOP_K_DEFAULT   = 3                                 # risultati di default
SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────
# Caricamento modello
# ─────────────────────────────────────────────
def load_model(model_name: str = DEFAULT_MODEL):
    """Carica CLIP (vision + text encoder) su CPU."""
    from transformers import CLIPProcessor, CLIPModel

    print(f"[1/3] Caricamento modello: {model_name}")
    t0 = time.time()

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()  # disabilita dropout

    print(f"      Modello caricato in {time.time() - t0:.1f}s")
    return model, processor


# ─────────────────────────────────────────────
# Encoding immagini
# ─────────────────────────────────────────────
def encode_images(model, processor, image_paths: list[Path]) -> np.ndarray:
    """
    Genera embedding normalizzati per una lista di immagini.
    Restituisce array float32 di shape (N, EMBED_DIM).
    """
    embeddings = []

    for path in tqdm(image_paths, desc="Encoding immagini", unit="img"):
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model.vision_model(**inputs)
                # pooler_output è il vettore CLS dell'immagine
                feat = outputs.pooler_output
                # proiezione nello spazio condiviso testo-immagine
                feat = model.visual_projection(feat)
                feat = feat / feat.norm(dim=-1, keepdim=True)   # normalizzazione L2

            embeddings.append(feat.squeeze().numpy().astype(np.float32))

        except Exception as e:
            print(f"      [WARN] Errore su {path.name}: {e} — skip")

    return np.stack(embeddings) if embeddings else np.empty((0, EMBED_DIM), dtype=np.float32)


# ─────────────────────────────────────────────
# Encoding testo
# ─────────────────────────────────────────────
def encode_text(model, processor, query: str) -> np.ndarray:
    """
    Genera embedding normalizzato per una query testuale.
    Restituisce array float32 di shape (1, EMBED_DIM).
    """
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        feat = model.get_text_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.squeeze().numpy().astype(np.float32)


# ─────────────────────────────────────────────
# Costruzione indice FAISS
# ─────────────────────────────────────────────
def build_index(embeddings: np.ndarray):
    """
    Crea un indice FAISS IndexFlatIP (inner product = cosine su vettori L2-normalizzati).
    Per dataset > 100k immagini considera IndexIVFFlat per velocità.
    """
    import faiss

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    return index


def save_index(index, metadata: list[dict]):
    """Salva indice FAISS e metadati su disco."""
    import faiss

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"   Indice salvato: {INDEX_PATH} ({index.ntotal} vettori)")
    print(f"   Metadati salvati: {META_PATH}")


def load_index():
    """Carica indice FAISS e metadati da disco."""
    import faiss

    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Indice non trovato. Esegui prima: python semantic_search.py --demo"
        )

    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"   Indice caricato: {index.ntotal} immagini indicizzate")
    return index, metadata


# ─────────────────────────────────────────────
# Ricerca
# ─────────────────────────────────────────────
def search(index, metadata: list[dict], query_vec: np.ndarray, top_k: int = TOP_K_DEFAULT):
    """
    Cerca le top_k immagini più simili alla query.
    Restituisce lista di dict con path, score e metadati.
    """
    scores, indices = index.search(query_vec.reshape(1, -1), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:   # FAISS restituisce -1 se non ci sono abbastanza risultati
            continue
        entry = metadata[idx].copy()
        entry["score"] = float(score)
        results.append(entry)

    return results


# ─────────────────────────────────────────────
# Pipeline completa: indicizzazione
# ─────────────────────────────────────────────
def run_indexing(model, processor, image_dir: Path = IMAGE_DIR):
    """Scansiona la cartella immagini, genera embedding e salva l'indice."""
    image_paths = [
        p for p in sorted(image_dir.rglob("*"))
        if p.suffix.lower() in SUPPORTED_EXTS
    ]

    if not image_paths:
        raise ValueError(f"Nessuna immagine trovata in '{image_dir}'. Hai eseguito --demo?")

    print(f"[2/3] Trovate {len(image_paths)} immagini in '{image_dir}'")

    embeddings = encode_images(model, processor, image_paths)

    metadata = [
        {
            "path": str(p),
            "filename": p.name,
            "stem": p.stem,
        }
        for p in image_paths[:len(embeddings)]  # allineato a embed riusciti
    ]

    print("[3/3] Costruzione indice FAISS...")
    index = build_index(embeddings)
    save_index(index, metadata)
    return index, metadata


# ─────────────────────────────────────────────
# Dataset demo: scarica immagini di esempio
# ─────────────────────────────────────────────
DEMO_IMAGES = {
    # Immagini open-source da Unsplash (royalty-free)
    "construction_trench.jpg":  "https://images.unsplash.com/photo-1504307651254-35680f356dfd?w=640&q=80",
    "gas_pipeline_yellow.jpg":  "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=640&q=80",
    "workers_excavation.jpg":   "https://images.unsplash.com/photo-1504328345606-18bbc8c9d7d1?w=640&q=80",
    "road_repair.jpg":          "https://images.unsplash.com/photo-1621935010538-6bce8c00c74a?w=640&q=80",
    "pipe_installation.jpg":    "https://images.unsplash.com/photo-1504307651254-35680f356dfd?w=640&q=80",
    "aerial_construction.jpg":  "https://images.unsplash.com/photo-1531834685032-c34bf0d84c77?w=640&q=80",
    "safety_helmets.jpg":       "https://images.unsplash.com/photo-1542223616-9de9adb5e3e8?w=640&q=80",
    "urban_excavation.jpg":     "https://images.unsplash.com/photo-1590496793929-36417d3117de?w=640&q=80",
}


def download_demo_images():
    """Scarica immagini demo da Unsplash per testare il sistema."""
    import urllib.request

    IMAGE_DIR.mkdir(exist_ok=True)
    print(f"[DEMO] Download immagini di esempio in '{IMAGE_DIR}/'...")

    for filename, url in DEMO_IMAGES.items():
        dest = IMAGE_DIR / filename
        if dest.exists():
            print(f"       Già presente: {filename}")
            continue
        try:
            print(f"       Download: {filename}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(dest, "wb") as f:
                f.write(resp.read())
        except Exception as e:
            print(f"       [WARN] Impossibile scaricare {filename}: {e}")


# ─────────────────────────────────────────────
# Output risultati
# ─────────────────────────────────────────────
def print_results(results: list[dict], query: str):
    """Stampa i risultati in modo leggibile."""
    print(f"\n{'─' * 55}")
    print(f"  Query: \"{query}\"")
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


# ─────────────────────────────────────────────
# Valutazione qualitativa dell'encoder
# ─────────────────────────────────────────────
EVAL_QUERIES = [
    "scavo con tubazioni gas gialle a bassa profondità",
    "operai con elmetto in cantiere",
    "lavori stradali in area urbana",
    "installazione tubature sottoterra",
    "vista aerea di un cantiere",
    "safety helmets construction workers",
]


def run_evaluation(model, processor, index, metadata):
    """
    Esegue una batteria di query di valutazione e stampa i risultati.
    Utile per testare la qualità dell'encoder visivo.
    """
    print("\n" + "═" * 55)
    print("  VALUTAZIONE ENCODER — query multilingua")
    print("═" * 55)

    for query in EVAL_QUERIES:
        t0 = time.time()
        query_vec = encode_text(model, processor, query)
        results = search(index, metadata, query_vec, top_k=2)
        elapsed = time.time() - t0

        print(f"\n  Query: \"{query}\"")
        print(f"  Latenza: {elapsed*1000:.0f}ms")
        for i, r in enumerate(results, 1):
            print(f"    #{i} [{r['score']:.3f}] {r['filename']}")

    print("\n" + "═" * 55)
    print("  Suggerimento: score > 0.25 → buona corrispondenza semantica")
    print("                score > 0.30 → eccellente")
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Semantic Visual Asset Retrieval — CLIP + FAISS"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Scarica immagini demo e costruisce l'indice"
    )
    parser.add_argument(
        "--index", action="store_true",
        help="Ri-indicizza le immagini nella cartella 'images/'"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Query testuale per la ricerca (es: \"scavo con tubi gialli\")"
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K_DEFAULT,
        help=f"Numero di risultati (default: {TOP_K_DEFAULT})"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Esegui la batteria di valutazione dell'encoder"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Nome modello HuggingFace (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()

    # ── Modalità DEMO ──────────────────────────────────
    if args.demo:
        download_demo_images()
        model, processor = load_model(args.model)
        index, metadata = run_indexing(model, processor)
        print("\n  Indice pronto! Ora puoi cercare:")
        print('  python semantic_search.py --query "scavo con tubazioni gas gialle"')
        return

    # ── Solo indicizzazione ────────────────────────────
    if args.index:
        model, processor = load_model(args.model)
        run_indexing(model, processor)
        return

    # ── Query o valutazione: serve l'indice ───────────
    if args.query or args.eval:
        model, processor = load_model(args.model)
        index, metadata = load_index()

        if args.eval:
            run_evaluation(model, processor, index, metadata)

        if args.query:
            print(f"[2/3] Encoding query...")
            query_vec = encode_text(model, processor, args.query)
            print(f"[3/3] Ricerca top-{args.top_k}...")
            results = search(index, metadata, query_vec, top_k=args.top_k)
            print_results(results, args.query)
        return

    # ── Nessun argomento ──────────────────────────────
    parser.print_help()
    print("\n  Esempio rapido per iniziare:")
    print("    python semantic_search.py --demo")


if __name__ == "__main__":
    main()