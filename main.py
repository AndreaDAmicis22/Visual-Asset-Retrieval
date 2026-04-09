import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import time

from semantic_search.config import DEFAULT_MODEL, TOP_K_DEFAULT, get_index_paths
from semantic_search.dataset import download_coco_resources, prepare_coco_dataset
from semantic_search.demo import download_demo_images
from semantic_search.encoder import encode_text
from semantic_search.evaluation import run_evaluation
from semantic_search.graph import build_graph, load_graph, save_graph
from semantic_search.graph_viz import viz_clusters, viz_subgraph
from semantic_search.index import load_index, run_indexing
from semantic_search.model import load_model
from semantic_search.rag import graph_rag_query, print_rag_response
from semantic_search.search import search
from semantic_search.utils import print_results


def main():
    parser = argparse.ArgumentParser(description="Semantic Visual Asset Retrieval — CLIP + FAISS")
    parser.add_argument("--demo", action="store_true", help="Scarica immagini demo e costruisce l'indice")
    parser.add_argument(
        "--index", action="store_true", help="Ri-indicizza le immagini nella cartella 'images/'"
    )  # poetry run python main.py --index
    parser.add_argument("--query", type=str, default=None, help='Query testuale (es: "scavo con tubi gialli")')
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT, help=f"Numero di risultati (default: {TOP_K_DEFAULT})")
    parser.add_argument("--eval", action="store_true", help="Esegui la batteria di valutazione dell'encoder")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Modello HuggingFace (default: {DEFAULT_MODEL})")
    parser.add_argument("--coco", action="store_true", help="Scarica COCO val2017 (5.000 immagini) (ZIP e Annotazioni)")
    parser.add_argument("--prepare", action="store_true", help="Prepara il sample di immagini da COCO")
    parser.add_argument("--max-images", type=int, default=5000, help="Numero di immagini per il sample (default: 5000)")
    parser.add_argument("--build-graph", action="store_true", help="Costruisce il grafo dall'indice esistente")
    parser.add_argument("--rag", type=str, default=None, help="Query Graph RAG in linguaggio naturale")
    parser.add_argument("--graph-depth", type=int, default=2, help="Profondità sottografo per RAG (default: 2)")
    parser.add_argument(
        "--viz-subgraph", type=str, default=None, help="Visualizza sottografo attorno a un filename (es: 000000001.jpg)"
    )
    parser.add_argument("--viz-clusters", action="store_true", help="Visualizza cluster di immagini simili")
    args = parser.parse_args()
    index_path, meta_path = get_index_paths(args.model)

    if args.demo:
        download_demo_images()
        model, processor = load_model(args.model)
        run_indexing(model, processor, index_path=index_path, meta_path=meta_path)
        print("\n  Indice pronto! Ora puoi cercare:")
        print('  python main.py --query "scavo con tubazioni gas gialle"')
        return

    if args.index:
        model, processor = load_model(args.model)
        run_indexing(model, processor, index_path=index_path, meta_path=meta_path)
        return

    if args.query or args.eval:
        model, processor = load_model(args.model)
        index, metadata = load_index(index_path=index_path, meta_path=meta_path)
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
        run_indexing(model, processor, index_path=index_path, meta_path=meta_path)
        return

    if args.prepare:
        prepare_coco_dataset(max_images=args.max_images)
        model, processor = load_model(args.model)
        run_indexing(model, processor, index_path=index_path, meta_path=meta_path)
        return

    if args.build_graph:
        model, processor = load_model(args.model)
        index, metadata = load_index(index_path=index_path, meta_path=meta_path)
        G = build_graph(index, metadata, model_name=args.model)
        save_graph(G, model_name=args.model)
        return

    if args.rag:
        model, processor = load_model(args.model)
        index, metadata = load_index(index_path=index_path, meta_path=meta_path)
        G = load_graph(model_name=args.model)
        response = graph_rag_query(
            args.rag, G, index, metadata, model, processor, top_k=args.top_k, graph_depth=args.graph_depth
        )
        print_rag_response(args.rag, response)
        return

    if args.viz_subgraph:
        G = load_graph(model_name=args.model)
        viz_subgraph(G, args.viz_subgraph, depth=args.graph_depth, model_name=args.model)
        print("  Server attivo — premi Ctrl+C per uscire")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Server fermato.")
        return

    if args.viz_clusters:
        G = load_graph(model_name=args.model)
        viz_clusters(G, model_name=args.model)
        print("  Server attivo — premi Ctrl+C per uscire")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Server fermato.")
        return

    parser.print_help()
    print("\n  Esempio rapido per iniziare:")
    print("    python main.py --demo")


if __name__ == "__main__":
    main()
