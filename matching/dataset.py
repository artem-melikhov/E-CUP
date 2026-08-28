from __future__ import annotations

import numpy as np
import pandas as pd

from matching.features import pair_features
from matching.fuzzy import jaro_winkler, ratio, token_sort_ratio
from matching.parse import profile_item
from matching.text import fit_tfidf, pairwise_cosine


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _attr_cell(value):
    if isinstance(value, dict):
        return value
    return _cell(value)


def restrict_items(matches: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    pair_ids = pd.unique(pd.concat([matches["id1"], matches["id2"]], ignore_index=True))
    items = items.loc[items["id"].isin(pair_ids)].drop_duplicates("id")
    missing = list(set(pair_ids) - set(items["id"].tolist()))
    if missing:
        items = pd.concat(
            [
                items,
                pd.DataFrame(
                    {
                        "id": missing,
                        "name": [""] * len(missing),
                        "attributes": ["{}"] * len(missing),
                        "category": [""] * len(missing),
                    }
                ),
            ],
            ignore_index=True,
        )
    return items


def build_profiles(items: pd.DataFrame) -> dict[int, object]:
    profiles = {}
    for row in items.itertuples(index=False):
        profiles[int(row.id)] = profile_item(
            int(row.id),
            _cell(row.name),
            _attr_cell(row.attributes),
            _cell(row.category),
        )
    return profiles


def struct_frame(matches: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    rows = []
    for row in matches.itertuples(index=False):
        rows.append(pair_features(profiles[int(row.id1)], profiles[int(row.id2)]))
    frame = pd.DataFrame(rows, index=matches.index)
    return frame


def text_frame(
    matches: pd.DataFrame,
    items: pd.DataFrame,
    profiles: dict,
    word_vec=None,
    char_vec=None,
) -> tuple[pd.DataFrame, object, object]:
    names = items["name"].fillna("").astype(str).tolist()
    pos_of = {int(i): p for p, i in enumerate(items["id"].tolist())}
    if word_vec is None or char_vec is None:
        word_vec, char_vec, xw, xc = fit_tfidf(names)
    else:
        xw = word_vec.transform(names)
        xc = char_vec.transform(names)

    i1 = np.fromiter((pos_of[int(x)] for x in matches["id1"]), dtype=np.int64, count=len(matches))
    i2 = np.fromiter((pos_of[int(x)] for x in matches["id2"]), dtype=np.int64, count=len(matches))

    n1 = [profiles[int(i)].name_norm for i in matches["id1"]]
    n2 = [profiles[int(i)].name_norm for i in matches["id2"]]
    raw1 = [profiles[int(i)].name for i in matches["id1"]]
    raw2 = [profiles[int(i)].name for i in matches["id2"]]

    frame = pd.DataFrame(
        {
            "cos_word": pairwise_cosine(xw, i1, i2),
            "cos_char": pairwise_cosine(xc, i1, i2),
            "jw": [jaro_winkler(a, b) for a, b in zip(raw1, raw2)],
            "jw_norm": [jaro_winkler(a, b) for a, b in zip(n1, n2)],
            "fuzz_ratio": [ratio(a, b) for a, b in zip(n1, n2)],
            "fuzz_token_sort": [token_sort_ratio(a, b) for a, b in zip(n1, n2)],
        },
        index=matches.index,
    )
    return frame, word_vec, char_vec


def assemble_struct(matches: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Identity / lexical-overlap features only — no TF-IDF, no fuzzy, no MiniLM."""
    items = restrict_items(matches, items)
    profiles = build_profiles(items)
    struct = struct_frame(matches, profiles)
    struct["category"] = [profiles[int(i)].category for i in matches["id1"]]
    return struct, profiles


def ce_feature_names(hidden: int = 384) -> list[str]:
    return [f"ce_{i}" for i in range(hidden)]


def concat_ce(struct: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    if embeddings.shape[0] != len(struct):
        raise ValueError(
            f"embedding rows {embeddings.shape[0]} != struct rows {len(struct)}"
        )
    ce = pd.DataFrame(
        embeddings.astype(np.float32, copy=False),
        index=struct.index,
        columns=ce_feature_names(embeddings.shape[1]),
    )
    return pd.concat([struct, ce], axis=1)


def assemble_features(
    matches: pd.DataFrame,
    items: pd.DataFrame,
    word_vec=None,
    char_vec=None,
) -> tuple[pd.DataFrame, object, object, dict]:
    items = restrict_items(matches, items)
    profiles = build_profiles(items)
    struct = struct_frame(matches, profiles)
    text, word_vec, char_vec = text_frame(matches, items, profiles, word_vec, char_vec)
    x = pd.concat([struct, text], axis=1)
    x["category"] = pd.Categorical(
        [profiles[int(i)].category for i in matches["id1"]],
        ordered=False,
    )
    return x, word_vec, char_vec, profiles
