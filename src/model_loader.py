# cell 3
# src/model_loader.py

import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from typing import Tuple

from config.config import (
    EMBEDDING_MODEL,
    ZEROSHOT_MODEL,
    MAX_SEQ_LENGTH,
    TORCH_DTYPE,
)


# ------------------------------------------------------------------
# SBERT LOADER
# ------------------------------------------------------------------

def safe_load_sbert(
    model_name : str = EMBEDDING_MODEL,
    device     : str = "cuda",
) -> SentenceTransformer:
    print(f"Loading SBERT : {model_name} on {device}")
    t0 = time.time()

    try:
        model = SentenceTransformer(model_name, device=device)
        model.max_seq_length = MAX_SEQ_LENGTH

        # Test inference to catch CUDA errors early
        _ = model.encode(
            ["test sentence"],
            convert_to_numpy  = True,
            show_progress_bar = False,
        )

        # Detect actual device (model may have moved itself)
        actual_device = str(
            next(model._modules["0"].auto_model.parameters()).device
        )
        emb_dim = model.get_sentence_embedding_dimension()

        print(f"SBERT loaded  | device={actual_device} | "
              f"dim={emb_dim} | "
              f"time={time.time()-t0:.1f}s")
        return model

    except RuntimeError as e:
        if "CUDA" in str(e) or "kernel" in str(e).lower():
            print(f"  CUDA error on {device}: {e}")
            print("Falling back to CPU for SBERT...")
            model = SentenceTransformer(model_name, device="cpu")
            model.max_seq_length = MAX_SEQ_LENGTH
            print("SBERT loaded on CPU (fallback)")
            return model
        raise


# ------------------------------------------------------------------
# NLI MODEL LOADER
# ------------------------------------------------------------------

def safe_load_nli(
    model_name : str = ZEROSHOT_MODEL,
    device     : str = "cuda",
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, str]:
    print(f"Loading NLI   : {model_name} on {device}")
    t0 = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype = TORCH_DTYPE,   # float32 — P100 safe
        )
        model = model.to(device)
        model.eval()

        # Test forward pass to catch CUDA errors early
        with torch.no_grad():
            inputs = tokenizer(
                "test premise", "test hypothesis",
                return_tensors = "pt",
                max_length     = 64,
                truncation     = True,
                padding        = True,
            ).to(device)
            _ = model(**inputs)

        print(f"NLI loaded    | device={device} | "
              f"time={time.time()-t0:.1f}s")
        return tokenizer, model, device

    except RuntimeError as e:
        if "CUDA" in str(e) or "kernel" in str(e).lower():
            print(f"  CUDA error on {device}: {e}")
            print("Falling back to CPU for NLI model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model     = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype = TORCH_DTYPE,
            ).to("cpu")
            model.eval()
            print("NLI loaded on CPU (fallback)")
            return tokenizer, model, "cpu"
        raise