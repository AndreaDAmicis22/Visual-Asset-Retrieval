from pathlib import Path

DEFAULT_MODEL = "openai/clip-vit-base-patch32"  # openai/clip-vit-base-patch16, openai/clip-vit-large-patch14, openai/clip-vit-large-patch14-336
IMAGE_DIR = Path("images")
INDEX_PATH = Path("data/faiss_index.bin")
META_PATH = Path("data/metadata.json")
EMBED_DIM = 512
TOP_K_DEFAULT = 6
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
