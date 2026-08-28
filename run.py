#!/usr/bin/env python3
"""Competition entry: MiniLM pair CLS + identity features + sklearn.

  python -u run.py --items_path items.parquet --matches_path matches.parquet --output_path submit.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from matching.dataset import assemble_struct, concat_ce
from matching.embed import (
    DEFAULT_CE_DIR,
    best_device,
    build_pair_texts,
    default_batch_size,
    extract_cls,
    items_to_text,
    load_encoder,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "ce_identity.joblib"
DEFAULT_CE = ROOT / "models" / "cross-encoder-ms-marco-MiniLM-L12-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, default=None, help="test items data path")
    parser.add_argument("--matches_path", type=str, default=None, help="test matches data path")
    parser.add_argument(
        "--output_path",
        "--output-path",
        dest="output_path",
        type=str,
        default=None,
        help="output file",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--ce-path", default=str(DEFAULT_CE if DEFAULT_CE.exists() else DEFAULT_CE_DIR))
    args, _unknown = parser.parse_known_args()
    if not args.items_path or not args.matches_path or not args.output_path:
        parser.error("--items_path, --matches_path and --output_path are required")
    return args


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    items = pd.read_parquet(args.items_path)
    matches = pd.read_parquet(args.matches_path)

    struct, _ = assemble_struct(matches, items)
    id_to_text = items_to_text(items)
    pairs = build_pair_texts(matches, id_to_text)

    device = best_device()
    model, tokenizer, device = load_encoder(args.ce_path, device=device)
    emb = extract_cls(
        model,
        tokenizer,
        device,
        pairs,
        batch_size=default_batch_size(device),
        show_progress=True,
    )

    x = concat_ce(struct, emb)
    mapping = {c: i for i, c in enumerate(bundle["categories"])}
    x["category"] = x["category"].astype(str).map(mapping).fillna(-1).astype("int32")
    x = x.reindex(columns=bundle["feature_names"])

    scores = bundle["model"].predict_proba(x)[:, 1]
    out = pd.DataFrame({"id1": matches["id1"], "id2": matches["id2"], "predict": scores})
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)
    print(f"wrote {args.output_path} rows={len(out):,}")


if __name__ == "__main__":
    main()
