def print_results(results: list[dict], query: str):
    print(f"\n{'─' * 55}")
    print(f'  Query: "{query}"')
    print(f"{'─' * 55}")

    if not results:
        print("  Nessun risultato trovato.")
        return

    for i, r in enumerate(results, 1):
        bar_len = int(r["score"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"\n  #{i}  {r['filename']}")
        print(f"      Score: {r['score']:.4f}  [{bar}]")
        print(f"      Path:  {r['path']}")

    print(f"\n{'─' * 55}\n")
