# Semantic Visual Asset Retrieval

Ricerca semantica su archivio fotografico aziendale tramite **CLIP** (OpenAI) + **FAISS**.  
Permette di trovare immagini con query in linguaggio naturale — anche in italiano.

---

## Avvio rapido

```bash
# 2. Scarica immagini demo e costruisce l'indice (~2 min al primo avvio per il download del modello)
poetry run python semantic_search.py --demo

# 3. Cerca
poetry run python semantic_search.py --query "scavo con tubazioni gas gialle a bassa profondità"
poetry run python semantic_search.py --query "operai con elmetto in cantiere" --top-k 5

# 4. Valuta la qualità dell'encoder
poetry run python semantic_search.py --eval
```

---

## Usare le tue immagini

```bash
# 1. Copia le tue immagini nella cartella images/
cp /percorso/archivio/*.jpg images/

# 2. Indicizza
python semantic_search.py --index

# 3. Cerca
python semantic_search.py --query "la tua query"
```

---

## Architettura

```
Immagini → CLIP Vision Encoder → vettori 512-dim → FAISS IndexFlatIP
Query    → CLIP Text Encoder  → vettore 512-dim  ──┘
                                                     ↓
                                          Similarità coseno → top-k risultati
```

CLIP proietta immagini e testo nello **stesso spazio vettoriale** durante il pretraining
su 400M coppie immagine-testo. La similarità coseno tra il vettore di una query e
il vettore di un'immagine è quindi semanticamente significativa senza bisogno di etichette.

---

## Interpretare i risultati

| Score coseno | Interpretazione           |
|:---:|:---|
| > 0.30 | Corrispondenza eccellente |
| 0.25 – 0.30 | Buona corrispondenza |
| 0.20 – 0.25 | Corrispondenza parziale |
| < 0.20 | Bassa rilevanza |

---

## Modelli disponibili (tutti CPU-friendly)

| Modello | Dimensione | Accuratezza | Quando usarlo |
|:---|:---:|:---:|:---|
| `openai/clip-vit-base-patch32` | ~150 MB | ★★★☆☆ | **Default — inizia qui** |
| `openai/clip-vit-large-patch14` | ~900 MB | ★★★★★ | Fase 2, dopo validazione |
| `google/siglip-base-patch16-224` | ~400 MB | ★★★★☆ | Query brevi/ambigue |

Cambia modello con: `python semantic_search.py --demo --model google/siglip-base-patch16-224`

---

## Passi successivi

- [ ] Integrare metadati aziendali (data, luogo, ID intervento) nel JSON
- [ ] Aggiungere filtro per data/luogo prima della ricerca semantica  
- [ ] Esporre come API REST (FastAPI + uvicorn)
- [ ] Valutare SigLIP per query in italiano nativo
- [ ] Per dataset > 50k immagini: sostituire `IndexFlatIP` con `IndexIVFFlat`