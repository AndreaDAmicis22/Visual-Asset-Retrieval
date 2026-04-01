import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from semantic_search.config import DEFAULT_MODEL


def load_model(model_name: str = DEFAULT_MODEL):
    from transformers import CLIPModel, CLIPProcessor

    print(f"[1/3] Caricamento modello: {model_name}")
    t0 = time.time()
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    print(f"      Modello caricato in {time.time() - t0:.1f}s")
    return model, processor
