from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matching.dataset import assemble_struct, concat_ce
from matching.embed import (
    DEFAULT_CE_DIR,
    HIDDEN_SIZE,
    best_device,
    build_pair_texts,
    default_batch_size,
    extract_cls,
    items_to_text,
    load_encoder,
)
from matching.metric import macro_pr_auc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "ce_identity.joblib"
DEFAULT_CACHE = ROOT / "models" / "cache" / "human_ce_cls.npy"


def load_human(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.read_parquet(data_dir / "items_human.parquet")
    matches = pd.read_parquet(data_dir / "matches.parquet")
    return items, matches


def encode_category(frame: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    mapping = {c: i for i, c in enumerate(categories)}
    out = frame.copy()
    out["category"] = out["category"].astype(str).map(mapping).fillna(-1).astype(np.int32)
    return out


def split_xy(matches: pd.DataFrame, x: pd.DataFrame, seed: int = 42, val_frac: float = 0.2):
    y = matches["target"].astype(int).to_numpy()
    key = x["category"].astype(str).to_numpy() + "_" + y.astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(sss.split(x, key))
    return y, train_idx, val_idx


def report(y, scores, cats, title: str) -> float:
    from sklearn.metrics import average_precision_score

    micro = average_precision_score(y, scores)
    macro, per = macro_pr_auc(y, scores, cats)
    print(f"\n=== {title} ===")
    print(f"micro PR-AUC: {micro:.4f}")
    print(f"MACRO PR-AUC: {macro:.4f}")
    for cat, ap in sorted(per.items(), key=lambda kv: kv[1]):
        print(f"  {ap:.4f}  {cat}")
    return macro


def load_or_compute_embeddings(
    items: pd.DataFrame,
    matches: pd.DataFrame,
    model_path: Path,
    cache_path: Path,
    batch_size: int | None,
) -> np.ndarray:
    n = len(matches)
    if cache_path.exists():
        emb = np.load(cache_path, mmap_mode="r")
        if emb.shape == (n, HIDDEN_SIZE):
            print(f"loaded cache {cache_path} shape={emb.shape}")
            return np.asarray(emb)
        print(f"cache shape {emb.shape} != {(n, HIDDEN_SIZE)}, recomputing")

    device = best_device()
    bs = batch_size or default_batch_size(device)
    print(f"computing CLS embeddings n={n:,} batch={bs} device={device}")
    id_to_text = items_to_text(items)
    pairs = build_pair_texts(matches, id_to_text)
    model, tokenizer, device = load_encoder(model_path, device=device)
    emb = extract_cls(
        model,
        tokenizer,
        device,
        pairs,
        batch_size=bs,
        out_path=cache_path,
    )
    print(f"saved cache {cache_path} shape={emb.shape}")
    return emb


def make_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=0.3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
    )


def make_logreg() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=1234,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sklearn on MiniLM CLS + identity features")
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--ce-path", type=Path, default=DEFAULT_CE_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    items, matches = load_human(args.data_dir)
    print(f"items={len(items):,} matches={len(matches):,}")
    emb = load_or_compute_embeddings(
        items,
        matches,
        args.ce_path,
        args.cache,
        args.batch_size or None,
    )
    if args.cache_only:
        print("cache-only, skip training")
        return

    struct, _ = assemble_struct(matches, items)
    x = concat_ce(struct, emb)
    categories = sorted(x["category"].astype(str).unique())
    y, train_idx, val_idx = split_xy(matches, x)
    print(f"train={len(train_idx):,} val={len(val_idx):,} features={x.shape[1]}")

    x_enc = encode_category(x, categories)
    x_tr, x_va = x_enc.iloc[train_idx], x_enc.iloc[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]
    cat_names = x["category"].astype(str).to_numpy()

    candidates: dict[str, object] = {
        "hgb": make_hgb(),
        "logreg": make_logreg(),
    }
    scores_by_name: dict[str, tuple[float, np.ndarray, object]] = {}
    for name, model in candidates.items():
        print(f"\nfitting {name}...")
        model.fit(x_tr, y_tr)
        proba = model.predict_proba(x_va)[:, 1]
        macro = report(y_va, proba, cat_names[val_idx], f"val {name}")
        scores_by_name[name] = (macro, proba, model)

    winner_name = max(scores_by_name, key=lambda k: scores_by_name[k][0])
    winner_macro, _, winner = scores_by_name[winner_name]
    print(f"\nwinner={winner_name} macro={winner_macro:.4f}")

    if winner_name == "hgb":
        n_iter = int(getattr(winner, "n_iter_", 200) or 200)
        full = HistGradientBoostingClassifier(
            max_iter=n_iter,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=0.3,
            early_stopping=False,
            random_state=42,
        )
    else:
        full = make_logreg()
    print(f"refit {winner_name} on all human pairs")
    full.fit(x_enc, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": full,
            "feature_names": x.columns.tolist(),
            "categories": categories,
            "backend": f"ce_identity_{winner_name}",
            "hidden_size": HIDDEN_SIZE,
            "val_macro": float(winner_macro),
        },
        args.out,
        compress=3,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
