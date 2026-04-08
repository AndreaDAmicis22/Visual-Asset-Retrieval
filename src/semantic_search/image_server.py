"""
image_server.py — Server HTTP leggero per servire immagini locali
=================================================================
Avvia un server FastAPI in background che espone le immagini
della cartella IMAGE_DIR tramite HTTP, così Pyvis può caricarle
nel browser senza problemi di CORS/sicurezza su file://.

Utilizzo standalone:
    poetry run python -m semantic_search.image_server

Utilizzo da codice (avvio in background):
    from semantic_search.image_server import start_server
    start_server()
"""

from __future__ import annotations

import socket
import threading
import time

import uvicorn

from semantic_search.config import IMAGE_DIR

HOST = "127.0.0.1"
PORT = 5050
BASE_URL = f"http://{HOST}:{PORT}"


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse

    app = FastAPI(title="Image Server", docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/images/{filename}")
    def serve_image(filename: str):
        path = IMAGE_DIR / filename
        if not path.exists() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            raise HTTPException(status_code=404, detail="Immagine non trovata")
        return FileResponse(path)

    @app.get("/health")
    def health():
        return {"status": "ok", "image_dir": str(IMAGE_DIR)}

    return app


def start_server(host: str = HOST, port: int = PORT) -> bool:
    """
    Avvia il server in un thread daemon in background.
    Restituisce True se avviato, False se era già in esecuzione.
    """

    # Controlla se il server è già in ascolto
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        already_running = s.connect_ex((host, port)) == 0

    if already_running:
        print(f"  [server] già in ascolto su {BASE_URL}")
        return False

    def _run():
        uvicorn.run(create_app(), host=host, port=port, log_level="error")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    for _ in range(20):
        time.sleep(0.2)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                print(f"  [server] in ascolto su {BASE_URL}")
                return True

    print("  [WARN] server non risponde — le immagini potrebbero non caricarsi")
    return False


if __name__ == "__main__":
    import uvicorn

    print(f"  Avvio server immagini su {BASE_URL}")
    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="info")
