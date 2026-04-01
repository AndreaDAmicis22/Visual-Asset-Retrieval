"""
Dataset downloader — COCO 2017 Validation Set
5.000 immagini royalty-free, ~1 GB, nessuna API key richiesta.
"""

import json
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
    print(f"   URL: {url}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            mb = downloaded / 1_048_576
            print(f"\r   [{bar}] {pct:5.1f}%  {mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline dopo la progress bar


def _extract(zip_path: Path, dest_dir: Path, label: str):
    print(f"   Estrazione {label}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    print(f"   Estratto in: {dest_dir}")


def _copy_to_image_dir(source_dir: Path, max_images: int = 5000):
    """Copia le immagini COCO nella cartella IMAGE_DIR del progetto."""
    IMAGE_DIR.mkdir(exist_ok=True)
    images = sorted(source_dir.glob("*.jpg"))[:max_images]

    print(f"   Copia {len(images)} immagini in '{IMAGE_DIR}'...")
    for img in images:
        dest = IMAGE_DIR / img.name
        if not dest.exists():
            dest.write_bytes(img.read_bytes())

    print(f"   Fatto: {len(images)} immagini disponibili in '{IMAGE_DIR}/'")
    return len(images)


def _save_captions(max_images: int = 5000):
    """
    Salva un file captions.json con le didascalie COCO —
    utile per valutare la qualità della ricerca semantica.
    """
    if not ANNOT_FILE.exists():
        return

    with open(ANNOT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # mappa image_id → filename
    id_to_file = {img["id"]: img["file_name"] for img in data["images"][:max_images]}

    # raggruppa didascalie per immagine
    captions: dict[str, list[str]] = {}
    for ann in data["annotations"]:
        fname = id_to_file.get(ann["image_id"])
        if fname:
            captions.setdefault(fname, []).append(ann["caption"])

    out = Path("data/captions.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)

    print(f"   Didascalie salvate: {out} ({len(captions)} immagini)")


def download_coco_resources():
    """
    Gestisce solo il download e l'estrazione dei file raw.
    Da eseguire una sola volta.
    """
    print("\n" + "═" * 55)
    print("  DOWNLOAD COCO 2017 RESOURCES")
    print("═" * 55)

    # 1. Immagini
    _download(COCO_IMAGES_URL, COCO_ZIP, "immagini COCO val2017 (~1 GB)")
    coco_img_dir = Path("data/val2017")
    if not coco_img_dir.exists():
        _extract(COCO_ZIP, Path("data"), "immagini")

    # 2. Annotation
    _download(COCO_ANNOT_URL, ANNOT_ZIP, "annotazioni COCO (~241 MB)")
    annot_dir = Path("data/annotations")
    if not annot_dir.exists():
        _extract(ANNOT_ZIP, Path("data"), "annotazioni")

    print("\n✓ Risorse scaricate ed estratte in data/")


def prepare_coco_dataset(max_images: int = 5000):
    """
    Gestisce il campionamento, la copia e la generazione dei metadati.
    Può essere rieseguita per cambiare il numero di immagini.
    """
    print("\n" + "═" * 55)
    print("  PREPARAZIONE DATASET")
    print(f"  Target: {max_images} immagini → '{IMAGE_DIR}/'")
    print("═" * 55)

    coco_img_dir = Path("data/val2017")

    if not coco_img_dir.exists():
        print("❌ Errore: Cartella immagini non trovata. Esegui prima il download.")
        return

    # ── 1. Copia in images/ ───────────────────────────
    # Nota: _copy_to_image_dir dovrebbe gestire la pulizia
    # o l'aggiunta se vuoi cambiare il sample
    n = _copy_to_image_dir(coco_img_dir, max_images)

    # ── 2. Salva didascalie ───────────────────────────
    _save_captions(max_images)

    print("\n" + "═" * 55)
    print(f"  ✓ {n} immagini pronte in '{IMAGE_DIR}/'")
    print(f"  ✓ Didascalie sincronizzate per {n} immagini")
    print("\n  Prossimo passo (re-indicizzazione):")
    print("    python main.py --index")
    print("═" * 55 + "\n")


# def download_coco_val(max_images: int = 5000):
#     """
#     Pipeline completa:
#     1. Scarica COCO val2017 (~1 GB)
#     2. Estrae le immagini
#     3. Scarica le annotation con le didascalie
#     4. Copia tutto nella cartella images/ del progetto
#     """
#     print("\n" + "═" * 55)
#     print("  COCO 2017 Validation Set — downloader")
#     print(f"  Target: {max_images} immagini → '{IMAGE_DIR}/'")
#     print("═" * 55)

#     # ── 1. Immagini ──────────────────────────────────
#     _download(COCO_IMAGES_URL, COCO_ZIP, "immagini COCO val2017 (~1 GB)")
#     coco_img_dir = Path("data/val2017")
#     if not coco_img_dir.exists():
#         _extract(COCO_ZIP, Path("data"), "immagini")

#     # ── 2. Annotation (didascalie) ────────────────────
#     _download(COCO_ANNOT_URL, ANNOT_ZIP, "annotazioni COCO (~241 MB)")
#     annot_dir = Path("data/annotations")
#     if not annot_dir.exists():
#         _extract(ANNOT_ZIP, Path("data"), "annotazioni")

#     # ── 3. Copia in images/ ───────────────────────────
#     n = _copy_to_image_dir(coco_img_dir, max_images)

#     # ── 4. Salva didascalie ───────────────────────────
#     _save_captions(max_images)

#     print("\n" + "═" * 55)
#     print(f"  ✓ {n} immagini pronte in '{IMAGE_DIR}/'")
#     print("  ✓ Didascalie in 'data/captions.json' (utile per eval)")
#     print("\n  Prossimo passo:")
#     print("    python main.py --index")
#     print('    python main.py --query "a dog running on the beach"')
#     print("═" * 55 + "\n")
