from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# https://huggingface.co/models?search=clip
DEFAULT_MODEL = "openai/clip-vit-base-patch32"  # openai/clip-vit-base-patch16, openai/clip-vit-large-patch14, openai/clip-vit-large-patch14-336
IMAGE_DIR = PROJECT_ROOT / "images"
INDEX_PATH = PROJECT_ROOT / "data" / "faiss_index.bin"
META_PATH = PROJECT_ROOT / "data" / "metadata.json"
EMBED_DIM = 512
TOP_K_DEFAULT = 5
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
