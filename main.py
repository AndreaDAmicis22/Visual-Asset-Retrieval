import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse

from src.semantic_search.config import DEFAULT_MODEL, TOP_K_DEFAULT
from src.semantic_search.dataset import download_coco_resources, prepare_coco_dataset
from src.semantic_search.demo import download_demo_images
from src.semantic_search.encoder import encode_text
from src.semantic_search.evaluation import run_evaluation
from src.semantic_search.index import load_index, run_indexing
from src.semantic_search.model import load_model
from src.semantic_search.search import search
from src.semantic_search.utils import print_results


def main():
    parser = argparse.ArgumentParser(description="Semantic Visual Asset Retrieval — CLIP + FAISS")
    parser.add_argument("--demo", action="store_true", help="Scarica immagini demo e costruisce l'indice")
    parser.add_argument("--index", action="store_true", help="Ri-indicizza le immagini nella cartella 'images/'")
    parser.add_argument("--query", type=str, default=None, help='Query testuale (es: "scavo con tubi gialli")')
    parser.add_argument(
        "--top-k", type=int, default=TOP_K_DEFAULT, help=f"Numero di risultati (default: {TOP_K_DEFAULT})"
    )
    parser.add_argument("--eval", action="store_true", help="Esegui la batteria di valutazione dell'encoder")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help=f"Modello HuggingFace (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--coco", action="store_true", help="Scarica COCO val2017 (5.000 immagini) (ZIP e Annotazioni)")
    parser.add_argument("--prepare", action="store_true", help="Prepara il sample di immagini da COCO")
    parser.add_argument("--max-images", type=int, default=5000, help="Numero di immagini per il sample (default: 5000)")
    args = parser.parse_args()

    if args.demo:
        download_demo_images()
        model, processor = load_model(args.model)
        run_indexing(model, processor)
        print("\n  Indice pronto! Ora puoi cercare:")
        print('  python main.py --query "scavo con tubazioni gas gialle"')
        return

    if args.index:
        model, processor = load_model(args.model)
        run_indexing(model, processor)
        return

    if args.query or args.eval:
        model, processor = load_model(args.model)
        index, metadata = load_index()

        if args.eval:
            run_evaluation(model, processor, index, metadata)

        if args.query:
            print("[2/3] Encoding query...")
            query_vec = encode_text(model, processor, args.query)
            print(f"[3/3] Ricerca top-{args.top_k}...")
            results = search(index, metadata, query_vec, top_k=args.top_k)
            print_results(results, args.query)
        return

    if args.coco:
        download_coco_resources()
        prepare_coco_dataset(max_images=args.max_images)
        model, processor = load_model(args.model)
        run_indexing(model, processor)
        return

    if args.prepare:
        prepare_coco_dataset(max_images=args.max_images)
        model, processor = load_model(args.model)
        run_indexing(model, processor)
        return

    parser.print_help()
    print("\n  Esempio rapido per iniziare:")
    print("    python main.py --demo")


if __name__ == "__main__":
    main()
