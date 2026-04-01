import json
from pathlib import Path

import numpy as np

from semantic_search.config import EMBED_DIM, IMAGE_DIR, INDEX_PATH, META_PATH, SUPPORTED_EXTS
from semantic_search.encoder import encode_images


def build_index(embeddings: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    return index


def save_index(index, metadata: list[dict]):
    import faiss

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"   Indice salvato: {INDEX_PATH} ({index.ntotal} vettori)")


def load_index():
    import faiss

    if not INDEX_PATH.exists() or not META_PATH.exists():
        msg = "Indice non trovato. Esegui prima: python main.py --demo"
        raise FileNotFoundError(msg)

    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"   Indice caricato: {index.ntotal} immagini indicizzate")
    return index, metadata


def run_indexing(model, processor, image_dir: Path = IMAGE_DIR):
    image_paths = [p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in SUPPORTED_EXTS]

    if not image_paths:
        msg = f"Nessuna immagine trovata in '{image_dir}'."
        raise ValueError(msg)

    print(f"[2/3] Trovate {len(image_paths)} immagini in '{image_dir}'")
    embeddings = encode_images(model, processor, image_paths)

    metadata = [{"path": str(p), "filename": p.name, "stem": p.stem} for p in image_paths[: len(embeddings)]]

    print("[3/3] Costruzione indice FAISS...")
    index = build_index(embeddings)
    save_index(index, metadata)
    return index, metadata
