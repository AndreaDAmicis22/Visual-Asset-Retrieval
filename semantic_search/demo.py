import urllib.request

from semantic_search.config import IMAGE_DIR

DEMO_IMAGES = {
    "construction_trench.jpg": "https://images.unsplash.com/photo-1504307651254-35680f356dfd?w=640&q=80",
    "gas_pipeline_yellow.jpg": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=640&q=80",
    "workers_excavation.jpg": "https://images.unsplash.com/photo-1504328345606-18bbc8c9d7d1?w=640&q=80",
    "road_repair.jpg": "https://images.unsplash.com/photo-1621935010538-6bce8c00c74a?w=640&q=80",
    "pipe_installation.jpg": "https://images.unsplash.com/photo-1504307651254-35680f356dfd?w=640&q=80",
    "aerial_construction.jpg": "https://images.unsplash.com/photo-1531834685032-c34bf0d84c77?w=640&q=80",
    "safety_helmets.jpg": "https://images.unsplash.com/photo-1542223616-9de9adb5e3e8?w=640&q=80",
    "urban_excavation.jpg": "https://images.unsplash.com/photo-1590496793929-36417d3117de?w=640&q=80",
}


def download_demo_images():
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
