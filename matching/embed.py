"""MiniLM pair CLS embeddings — same recipe as the official baseline.

Product text and [CLS] pooling match `src/utils.py` from matching-baseline-lightweight.
Torch is imported only inside encoder helpers so unit tests stay torch-free.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

HIDDEN_SIZE = 384
MAX_SEQ_LENGTH = 256
DEFAULT_CE_DIR = Path(__file__).resolve().parents[1] / "models" / "cross-encoder-ms-marco-MiniLM-L12-v2"


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _attr_text(raw: Any) -> str:
    if raw is None:
        return ""
    try:
        if pd.isna(raw):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(raw, dict):
        attrs = raw
    else:
        try:
            attrs = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return ""
    if not isinstance(attrs, dict):
        return ""
    return " ".join(f"{k}: {v}" for k, v in attrs.items())


def product_text(name: Any, category: Any, attributes: Any) -> str:
    """Official pair-encoder string: Name / Category / Attributes."""
    return (
        f"Name: {_as_str(name)} "
        f"Category: {_as_str(category)} "
        f"Attributes: {_attr_text(attributes)}"
    )


def items_to_text(items: pd.DataFrame) -> dict[int, str]:
    return {
        int(row.id): product_text(row.name, row.category, row.attributes)
        for row in items.itertuples(index=False)
    }


def build_pair_texts(matches: pd.DataFrame, id_to_text: dict[int, str]) -> list[tuple[str, str]]:
    """One text pair per match row. Missing ids become empty strings — never drop rows."""
    out: list[tuple[str, str]] = []
    for id1, id2 in zip(matches["id1"].tolist(), matches["id2"].tolist()):
        out.append((id_to_text.get(int(id1), ""), id_to_text.get(int(id2), "")))
    return out


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_batch_size(device: str | None = None) -> int:
    device = device or best_device()
    if device == "cuda":
        return 512
    if device == "mps":
        return 64
    return 16


def load_encoder(model_path: str | Path, device: str | None = None):
    import torch
    from transformers import AutoModel, AutoTokenizer, logging as hf_logging

    device = device or best_device()
    path = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    if device == "cuda":
        dtype = torch.bfloat16
        attn = "sdpa"
    elif device == "mps":
        dtype = torch.float16
        attn = "sdpa"
    else:
        dtype = torch.float32
        attn = None

    hf_logging.set_verbosity_error()
    kwargs: dict[str, Any] = {"local_files_only": True}
    if attn is not None:
        kwargs["attn_implementation"] = attn
    try:
        model = AutoModel.from_pretrained(path, torch_dtype=dtype, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        try:
            model = AutoModel.from_pretrained(path, torch_dtype=dtype, **kwargs)
        except TypeError:
            model = AutoModel.from_pretrained(path, local_files_only=True)
    except Exception:
        kwargs.pop("attn_implementation", None)
        model = AutoModel.from_pretrained(path, torch_dtype=dtype, **kwargs)

    model = model.to(device).eval()
    print(f"[CE] device={device} dtype={dtype} path={path}")
    return model, tokenizer, device


def extract_cls(
    model,
    tokenizer,
    device: str,
    pairs: list[tuple[str, str]],
    batch_size: int = 64,
    max_length: int = MAX_SEQ_LENGTH,
    show_progress: bool = True,
    out_path: str | Path | None = None,
) -> np.ndarray:
    """CLS of the pair encoding [text1, text2]. Writes back to original pair order."""
    import torch

    n = len(pairs)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32, shape=(n, HIDDEN_SIZE)
        )
    else:
        result = np.empty((n, HIDDEN_SIZE), dtype=np.float32)

    if n == 0:
        return np.asarray(result)

    if n > batch_size:
        lengths = np.fromiter((len(a) + len(b) for a, b in pairs), dtype=np.int32, count=n)
        order = np.argsort(lengths)
        pairs_sorted = [pairs[i] for i in order]
    else:
        order = np.arange(n)
        pairs_sorted = pairs

    iterator = range(0, n, batch_size)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="CE CLS", total=(n + batch_size - 1) // batch_size)
        except ImportError:
            pass

    with torch.inference_mode():
        for start in iterator:
            batch = pairs_sorted[start : start + batch_size]
            features = tokenizer(
                text=[p[0] for p in batch],
                text_pair=[p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
            hidden = model(**features).last_hidden_state[:, 0, :]
            if hidden.dtype != torch.float32:
                hidden = hidden.float()
            chunk = hidden.cpu().numpy()
            orig = order[start : start + len(batch)]
            result[orig] = chunk
            if out_path is not None and start > 0 and start % (batch_size * 40) == 0:
                result.flush()

    if out_path is not None:
        result.flush()
        del result
        return np.load(out_path)
    return result
