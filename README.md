# Semantic Visual Asset Retrieval

Ricerca semantica su archivio fotografico aziendale tramite **CLIP** (OpenAI) + **FAISS**.  
Permette di trovare immagini con query in linguaggio naturale — anche in italiano.

---

## Struttura del progetto

```
visual/
├── data/                        # generato automaticamente, non in git
│   ├── faiss_index.bin
│   ├── metadata.json
│   └── annotations/             # annotation COCO (se scaricate)
├── images/                      # immagini da indicizzare, non in git
├── notebooks/
│   └── semantic_search_explorer.ipynb   # esplorazione visuale dei risultati
├── src/
│   └── semantic_search/         # package principale
│       ├── __init__.py
│       ├── config.py            # costanti e path
│       ├── model.py             # caricamento CLIP
│       ├── encoder.py           # encode immagini e testo
│       ├── index.py             # build/save/load indice FAISS
│       ├── search.py            # ricerca per similarità coseno
│       ├── demo.py              # download immagini demo (8 foto)
│       ├── dataset.py           # download COCO val2017 (5.000 foto)
│       ├── evaluation.py        # batteria di query di valutazione
│       └── utils.py             # output risultati su terminale
├── main.py                      # entry point CLI
├── pyproject.toml
├── ruff.toml
└── README.md
```

---

## Installazione

```bash
git clone <repo>
cd visual
poetry install
```

---

## Avvio rapido

```bash
# Scarica 8 immagini demo e costruisce l'indice
poetry run python main.py --demo

# Cerca
poetry run python main.py --query "scavo con tubazioni gas gialle a bassa profondità"
poetry run python main.py --query "operai con elmetto in cantiere" --top-k 5

# Valuta la qualità dell'encoder su una batteria di query
poetry run python main.py --eval
```

---

## Dataset COCO (5.000 immagini)

Per un test più realistico scarica il validation set di COCO 2017 (~1.2 GB):

```bash
# Scarica, estrae e indicizza automaticamente
poetry run python main.py --coco

# Oppure solo le prime 1.000 immagini
poetry run python main.py --coco --max-images 1000
```

Le immagini vengono salvate in `images/`, i file generati (indice, metadati, didascalie) in `data/`.

---

## Usare le tue immagini

```bash
# 1. Copia le tue immagini nella cartella images/
cp /percorso/archivio/*.jpg images/

# 2. Indicizza
poetry run python main.py --index

# 3. Cerca
poetry run python main.py --query "la tua query"
```

---

## Notebook

Il notebook `notebooks/semantic_search_explorer.ipynb` permette di eseguire ricerche
e visualizzare i risultati come griglia di immagini con score colorati direttamente in Jupyter.

```bash
poetry run jupyter notebook notebooks/semantic_search_explorer.ipynb
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
il vettore di un'immagine è semanticamente significativa senza bisogno di etichette.

---

## Interpretare i risultati

| Score coseno | Interpretazione |
|:---:|:---|
| > 0.30 | Corrispondenza eccellente |
| 0.25 – 0.30 | Buona corrispondenza |
| 0.20 – 0.25 | Corrispondenza parziale |
| < 0.20 | Bassa rilevanza |

[NB] La similarità coseno va da -1 a 1 in generale, ma tra vettori normalizzati (come quelli di CLIP) va da 0 a 1.
 CLIP non è addestrato per massimizzare la similarità assoluta tra testo e immagine ma è addestrato a distinguere coppie corrette da coppie sbagliate in un batch. Il risultato pratico è che anche una corrispondenza perfetta raramente si supera 0.35–0.40, perché i vettori di testo e immagine restano in zone dello spazio vettoriale relativamente distanti tra loro. I valori vicini a 1 ai ottengono solo confrontando immagine con immagine o testo con testo nello stesso encoder, non cross-modale.
---

## Passi successivi

- [ ] Integrare metadati aziendali (data, luogo, ID intervento) nel JSON
- [ ] Aggiungere filtro per data/luogo prima della ricerca semantica
- [ ] Esporre come API REST (FastAPI + uvicorn)
- [ ] Valutare SigLIP per query in italiano nativo
- [ ] Per dataset > 50k immagini: sostituire `IndexFlatIP` con `IndexIVFFlat`
