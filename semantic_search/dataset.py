import json
import shutil  # Aggiunto per pulizia cartelle
import urllib.request
import zipfile
from pathlib import Path

from semantic_search.config import IMAGE_DIR

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOT_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ZIP = Path("data/val2017.zip")
ANNOT_ZIP = Path("data/annotations.zip")
ANNOT_FILE = Path("data/annotations/captions_val2017.json")


def _download(url: str, dest: Path, label: str):
    """Download con progress bar testuale."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"   Già presente: {dest.name}")
        return

    print(f"   Download {label}...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            mb = downloaded / 1_048_576
            print(f"\r   [{bar}] {pct:5.1f}%  {mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _extract(zip_path: Path, dest_dir: Path, label: str):
    print(f"   Estrazione {label}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    print(f"   Estratto in: {dest_dir}")


def _copy_to_image_dir(source_dir: Path, max_images: int = 5000):
    """Copia le immagini COCO. Svuota la cartella se il numero richiesto è cambiato."""
    if IMAGE_DIR.exists():
        # Controlliamo quante immagini ci sono già
        existing_files = list(IMAGE_DIR.glob("*.jpg"))
        if len(existing_files) != max_images:
            print(f"   Reset cartella '{IMAGE_DIR}' (da {len(existing_files)} a {max_images} immagini)...")
            shutil.rmtree(IMAGE_DIR)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(source_dir.glob("*.jpg"))[:max_images]

    print(f"   Copia {len(images)} immagini in '{IMAGE_DIR}'...")
    for img in images:
        dest = IMAGE_DIR / img.name
        if not dest.exists():
            dest.write_bytes(img.read_bytes())
    return len(images)


def _save_captions(max_images: int = 5000):
    """Salva captions.json sincronizzato esattamente con le immagini copiate."""
    if not ANNOT_FILE.exists():
        print("   ⚠️ File annotazioni non trovato!")
        return

    with open(ANNOT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Importante: usiamo lo stesso ordinamento di _copy_to_image_dir
    # COCO filenames sono come '000000000139.jpg'
    all_img_files = sorted([img["file_name"] for img in data["images"]])
    target_files = set(all_img_files[:max_images])

    # Mappa inversa per velocizzare il recupero
    id_to_file = {img["id"]: img["file_name"] for img in data["images"] if img["file_name"] in target_files}

    captions: dict[str, list[str]] = {}
    for ann in data["annotations"]:
        fname = id_to_file.get(ann["image_id"])
        if fname:
            captions.setdefault(fname, []).append(ann["caption"])

    out = Path("data/captions.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    print(f"   Didascalie salvate: {out} ({len(captions)} immagini)")


# --- Funzioni Pubbliche ---
def download_coco_resources():
    """Scarica ed estrae i file se non presenti."""
    print("\n" + "═" * 55 + "\n  STEP 1: DOWNLOAD RISORSE\n" + "═" * 55)
    _download(COCO_IMAGES_URL, COCO_ZIP, "immagini COCO (~1 GB)")
    if not Path("data/val2017").exists():
        _extract(COCO_ZIP, Path("data"), "immagini")

    _download(COCO_ANNOT_URL, ANNOT_ZIP, "annotazioni (~241 MB)")
    if not Path("data/annotations").exists():
        _extract(ANNOT_ZIP, Path("data"), "annotazioni")


def prepare_coco_dataset(max_images: int = 5000):
    """Prepara il sotto-insieme di immagini e metadati."""
    print("\n" + "═" * 55 + f"\n  STEP 2: PREPARAZIONE SAMPLE ({max_images})\n" + "═" * 55)
    coco_img_dir = Path("data/val2017")

    if not coco_img_dir.exists():
        print("❌ Errore: Dati raw non trovati. Esegui download_coco_resources().")
        return

    n = _copy_to_image_dir(coco_img_dir, max_images)
    _save_captions(max_images)

    print(f"\n✓ Dataset pronto con {n} immagini.")
    print("Esegui ora: python main.py --index")
