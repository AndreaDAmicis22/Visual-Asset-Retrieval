from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# https://huggingface.co/models?search=clip
# https://huggingface.co/collections/google/siglip
# https://huggingface.co/collections/google/siglip2
# openai/clip-vit-base-patch32, openai/clip-vit-base-patch16, openai/clip-vit-large-patch14, openai/clip-vit-large-patch14-336
DEFAULT_MODEL = "openai/clip-vit-base-patch16"
IMAGE_DIR = PROJECT_ROOT / "images"


def get_index_paths(model_name: str = DEFAULT_MODEL):
    """Restituisce i path di indice e metadati per il modello dato."""
    # "openai/clip-vit-large-patch14" → "clip-vit-large-patch14"
    model_slug = model_name.rsplit("/", maxsplit=1)[-1]
    model_dir = PROJECT_ROOT / "data" / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "faiss_index.bin", model_dir / "metadata.json"


INDEX_PATH, META_PATH = get_index_paths(DEFAULT_MODEL)
# EMBED_DIM = 768
EMBED_DIM = 512
TOP_K_DEFAULT = 5
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
