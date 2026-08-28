from __future__ import annotations

from matching.parse import FASHION_CATEGORIES, JEWELRY_CATEGORY, PHARMA_CATEGORY, ItemProfile


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _tri_eq(x, y) -> tuple[int, int, int]:
    both = x is not None and y is not None and x != "" and y != ""
    if not both:
        return 0, 0, 1
    eq = int(x == y)
    return eq, int(not eq), 0


def pair_features(a: ItemProfile, b: ItemProfile) -> dict[str, float]:
    brand_eq, brand_neq, brand_missing = _tri_eq(a.brand, b.brand)
    color_eq, color_neq, color_missing = _tri_eq(a.color, b.color)
    type_eq, type_neq, type_missing = _tri_eq(a.item_type, b.item_type)
    size_eq, size_neq, size_missing = _tri_eq(a.size, b.size)

    art_inter = int(bool(a.articles & b.articles))
    art_both = int(bool(a.articles) and bool(b.articles))
    art_none = int(not a.articles and not b.articles)
    art_conflict = int(art_both and not art_inter)

    vol_inter = a.volumes & b.volumes
    vol_both = bool(a.volumes) and bool(b.volumes)
    vol_eq = int(vol_both and a.volumes == b.volumes)
    vol_conflict = int(vol_both and not vol_inter)

    common = set(a.identity) & set(b.identity)
    kv_eq = sum(1 for k in common if a.identity[k] == b.identity[k])
    kv_frac = kv_eq / max(min(len(a.identity), len(b.identity)), 1)

    is_fashion = int(a.category in FASHION_CATEGORIES)
    name_eq = int(a.name == b.name and a.name != "")
    name_eq_norm = int(a.name_norm == b.name_norm and a.name_norm != "")
    tok_min = min(a.n_tokens, b.n_tokens)
    tok_max = max(a.n_tokens, b.n_tokens, 1)

    return {
        "tok_jac": _jaccard(a.tokens, b.tokens),
        "alnum_jac": _jaccard(a.alnum, b.alnum),
        "alnum_inter": float(len(a.alnum & b.alnum)),
        "num_jac": _jaccard(a.numbers, b.numbers),
        "num_inter": float(len(a.numbers & b.numbers)),
        "num_conflict": float(int(bool(a.numbers | b.numbers) and not (a.numbers & b.numbers))),
        "len_ratio": min(len(a.name), len(b.name)) / max(len(a.name), len(b.name), 1),
        "ntok_min": float(tok_min),
        "ntok_ratio": tok_min / tok_max,
        "generic_name": float(int(tok_min <= 3)),
        "name_eq": float(name_eq),
        "name_eq_norm": float(name_eq_norm),
        "name_eq_fashion": float(name_eq and is_fashion),
        "brand_eq": float(brand_eq),
        "brand_neq": float(brand_neq),
        "brand_missing": float(brand_missing),
        "color_eq": float(color_eq),
        "color_neq": float(color_neq),
        "color_missing": float(color_missing),
        "color_neq_fashion": float(color_neq and is_fashion),
        "type_eq": float(type_eq),
        "type_neq": float(type_neq),
        "type_missing": float(type_missing),
        "size_eq": float(size_eq),
        "size_neq": float(size_neq),
        "size_missing": float(size_missing),
        "art_inter": float(art_inter),
        "art_jaccard": _jaccard(a.articles, b.articles),
        "art_both": float(art_both),
        "art_none": float(art_none),
        "art_conflict": float(art_conflict),
        "vol_eq": float(vol_eq),
        "vol_conflict": float(vol_conflict),
        "identity_kv_eq": float(kv_eq),
        "identity_kv_frac": float(kv_frac),
        "identity_keys_jac": _jaccard(set(a.identity), set(b.identity)),
        "is_fashion": float(is_fashion),
        "is_jewelry": float(a.category == JEWELRY_CATEGORY),
        "is_pharma": float(a.category == PHARMA_CATEGORY),
    }
