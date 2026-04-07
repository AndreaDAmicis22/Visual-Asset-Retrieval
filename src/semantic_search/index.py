import json
from pathlib import Path

import faiss
import numpy as np

from semantic_search.config import (
    DEFAULT_MODEL,
    EMBED_DIM,
    IMAGE_DIR,
    SUPPORTED_EXTS,
    get_index_paths,
)
from semantic_search.encoder import encode_images


def build_index(embeddings: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    return index


def save_index(index, metadata: list[dict], index_path: Path, meta_path: Path):
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_index(index_path: Path, meta_path: Path):
    if not index_path.exists() or not meta_path.exists():
        msg = f"Indice non trovato in {index_path.parent}. Esegui prima --index."
        raise FileNotFoundError(msg)
    index = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"   Indice caricato: {index.ntotal} immagini ({index_path.parent.name})")
    return index, metadata


def run_indexing(
    model, processor, image_dir: Path = IMAGE_DIR, index_path: Path | None = None, meta_path: Path | None = None
):
    if index_path is None or meta_path is None:
        index_path, meta_path = get_index_paths(DEFAULT_MODEL)
    image_paths = [p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in SUPPORTED_EXTS]

    if not image_paths:
        msg = f"Nessuna immagine trovata in '{image_dir}'."
        raise ValueError(msg)

    print(f"[2/3] Trovate {len(image_paths)} immagini in '{image_dir}'")
    embeddings = encode_images(model, processor, image_paths)

    metadata = [{"path": str(p), "filename": p.name, "stem": p.stem} for p in image_paths[: len(embeddings)]]

    print("[3/3] Costruzione indice FAISS...")
    index = build_index(embeddings)
    save_index(index, metadata, index_path, meta_path)
    return index, metadata
