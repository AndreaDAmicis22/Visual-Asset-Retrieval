import numpy as np

from semantic_search.config import TOP_K_DEFAULT


def search(index, metadata: list[dict], query_vec: np.ndarray, top_k: int = TOP_K_DEFAULT):
    scores, indices = index.search(query_vec.reshape(1, -1), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0], strict=False):
        if idx < 0:
            continue
        entry = metadata[idx].copy()
        entry["score"] = float(score)
        results.append(entry)

    return results
