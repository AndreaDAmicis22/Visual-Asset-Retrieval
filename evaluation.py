import time

from encoder import encode_text
from search import search

EVAL_QUERIES = [
    "scavo con tubazioni gas gialle a bassa profondità",
    "operai con elmetto in cantiere",
    "lavori stradali in area urbana",
    "installazione tubature sottoterra",
    "vista aerea di un cantiere",
    "safety helmets construction workers",
]


def run_evaluation(model, processor, index, metadata):
    print("\n" + "═" * 55)
    print("  VALUTAZIONE ENCODER — query multilingua")
    print("═" * 55)

    for query in EVAL_QUERIES:
        t0 = time.time()
        query_vec = encode_text(model, processor, query)
        results = search(index, metadata, query_vec, top_k=2)
        elapsed = time.time() - t0

        print(f'\n  Query: "{query}"')
        print(f"  Latenza: {elapsed * 1000:.0f}ms")
        for i, r in enumerate(results, 1):
            print(f"    #{i} [{r['score']:.3f}] {r['filename']}")

    print("\n" + "═" * 55)
    print("  Suggerimento: score > 0.25 → buona corrispondenza semantica")
    print("                score > 0.30 → eccellente")
    print("═" * 55 + "\n")
