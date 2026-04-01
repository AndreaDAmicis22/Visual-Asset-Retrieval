from pathlib import Path

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
IMAGE_DIR = Path("images")
INDEX_PATH = Path("data/faiss_index.bin")
META_PATH = Path("data/metadata.json")
EMBED_DIM = 512
TOP_K_DEFAULT = 3
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
