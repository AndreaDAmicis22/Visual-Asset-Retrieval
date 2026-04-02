from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from semantic_search.config import EMBED_DIM


def encode_images(model, processor, image_paths: list[Path]) -> np.ndarray:
    embeddings = []

    for path in tqdm(image_paths, desc="Encoding immagini", unit="img"):
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                feat = model.vision_model(**inputs).pooler_output
                feat = model.visual_projection(feat)
                feat = feat / feat.norm(dim=-1, keepdim=True)

            embeddings.append(feat.squeeze().numpy().astype(np.float32))

        except Exception as e:
            print(f"      [WARN] Errore su {path.name}: {e} — skip")

    return np.stack(embeddings) if embeddings else np.empty((0, EMBED_DIM), dtype=np.float32)


def encode_text(model, processor, query: str) -> np.ndarray:
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        feat = model.text_model(**inputs).pooler_output
        feat = model.text_projection(feat)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.squeeze().numpy().astype(np.float32)
