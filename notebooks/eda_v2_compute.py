#!/usr/bin/env python3
"""Deep EDA v2 for E-CUP product matching. Writes eda_v2_stats.json."""

from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "eda_v2_stats.json"
RNG = np.random.default_rng(42)

CYR = re.compile(r"[а-яёА-ЯЁ]")
LAT = re.compile(r"[a-zA-Z]")
DIGIT = re.compile(r"\d")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
WS = re.compile(r"\s+")
PUNCT = re.compile(r"[^\w\s]+", re.U)
SIZE_RE = re.compile(
    r"(?i)(?<!\d)(xxxs|xxs|xs|xxl|xxxl|xxxxl|\b[sml]\b|размер\s*\d{1,3}"
    r"|\b\d{2,3}\s*(?:р(?:азмер)?|eu|ru|us|uk)\b"
    r"|\b(?:eu|ru|us|uk)\s*\d{2,3}\b)"
)
VOLUME_RE = re.compile(
    r"(?i)(\d+(?:[.,]\d+)?\s*(?:мл|мл\.|л|л\.|г|гр|кг|шт|шт\.|таблеток|капсул))"
)
COLOR_WORDS = {
    "черный", "чёрный", "белый", "красный", "синий", "зеленый", "зелёный",
    "розовый", "серый", "бежевый", "коричневый", "голубой", "желтый", "жёлтый",
    "оранжевый", "фиолетовый", "золотой", "серебряный", "бордовый", "хаки",
    "прозрачный", "многоцветный", "black", "white", "red", "blue", "green",
    "pink", "grey", "gray", "beige", "brown", "yellow", "orange", "gold",
    "silver", "navy",
}
ARTICLE_KEY = re.compile(r"артикул|oem|оем|партномер|sku|штрих", re.I)
PACK_KEY = re.compile(r"упаковк|вес с упаков|габарит|длина упак|ширина упак|высота упак", re.I)
BRAND_KEYS = ("бренд", "бренд товара", "торговая марка", "марка")
COLOR_KEYS = ("цвет товара", "цвет", "название цвета", "расцветка")
SIZE_KEYS = ("размер", "размер товара", "российский размер", "размер производителя", "объем", "объём")
TYPE_KEYS = ("тип", "вид", "тип товара")

FASHION = {"Одежда", "Обувь", "Галантерея и аксессуары", "Ювелирные изделия"}


def log(msg: str) -> None:
    print(msg, flush=True)


def pct(n, d) -> float:
    return float(n) / float(d) if d else 0.0


def norm_text(s: str) -> str:
    s = str(s).lower().replace("ё", "е")
    s = PUNCT.sub(" ", s)
    return WS.sub(" ", s).strip()


def tokens(s: str) -> set[str]:
    return {t for t in norm_text(s).split() if t}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def script_mix(s: str) -> str:
    c, l, d = bool(CYR.search(s)), bool(LAT.search(s)), bool(DIGIT.search(s))
    if c and l:
        return "cyr_lat"
    if c:
        return "cyr"
    if l:
        return "lat"
    if d:
        return "digits"
    return "other"


def try_json(s):
    try:
        obj = json.loads(s) if s else {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return None


class UF:
    def __init__(self):
        self.p = {}
        self.r = {}

    def add(self, x):
        if x not in self.p:
            self.p[x] = x
            self.r[x] = 0

    def find(self, x):
        self.add(x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1

    def components(self) -> dict:
        g = collections.defaultdict(list)
        for x in self.p:
            g[self.find(x)].append(x)
        return g


def summarize(arr, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {}
    out = {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(len(a)),
    }
    for q in qs:
        out[f"p{int(q*100)}"] = float(np.quantile(a, q))
    return out


def pair_examples(df, n=6, cols=None):
    cols = cols or ["name_1", "name_2", "category_1", "target"]
    rows = []
    take = df.head(n) if len(df) <= n else df.sample(min(n, len(df)), random_state=42)
    for _, r in take.iterrows():
        row = {c: (None if pd.isna(r[c]) else r[c]) for c in cols if c in r.index}
        for c in ("id1", "id2", "jw", "tok_jac", "name_eq"):
            if c in r.index:
                v = r[c]
                row[c] = float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int, np.bool_)) else v)
        rows.append(row)
    return rows


def get_first(attr: dict, keys) -> str | None:
    for k in keys:
        if k in attr and str(attr[k]).strip():
            return str(attr[k]).strip().lower()
    return None


def get_articles(attr: dict) -> set[str]:
    arts = set()
    for k, v in attr.items():
        if ARTICLE_KEY.search(k):
            for x in re.split(r"[;,/|]+", str(v).lower()):
                x = x.strip()
                if len(x) >= 3:
                    arts.add(x)
    return arts


def strip_variant_tokens(s: str) -> str:
    t = norm_text(s)
    t = SIZE_RE.sub(" ", t)
    t = VOLUME_RE.sub(" ", t)
    toks = [w for w in t.split() if w not in COLOR_WORDS and w not in {"цвет", "размер"}]
    return " ".join(toks)


def main():
    t0 = time.time()
    stats: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    log("=== load human items + matches ===")
    items = pd.read_parquet(ROOT / "items_human.parquet")
    matches = pd.read_parquet(ROOT / "matches.parquet")
    log(f"items_human={len(items):,} matches={len(matches):,}")

    # ---------- quality ----------
    name_len = items["name"].str.len()
    attr_len = items["attributes"].str.len()
    empty_name = (items["name"].fillna("").str.strip() == "").sum()
    empty_attr = (items["attributes"].fillna("").str.strip().isin(["", "{}", "[]"])).sum()
    name_dup = items["name"].duplicated(keep=False).sum()
    uniq_names = items["name"].nunique()

    stats["human_items_quality"] = {
        "n": int(len(items)),
        "unique_ids": int(items["id"].nunique()),
        "empty_name": int(empty_name),
        "empty_or_emptyjson_attr": int(empty_attr),
        "unique_names": int(uniq_names),
        "items_with_duplicate_name": int(name_dup),
        "name_collision_rate": pct(len(items) - uniq_names, len(items)),
        "name_len": summarize(name_len),
        "attr_len": summarize(attr_len),
        "script_mix": items["name"].map(script_mix).value_counts().to_dict(),
        "name_has_digit": float(items["name"].str.contains(r"\d", regex=True).mean()),
    }

    # ---------- join pairs ----------
    log("=== join pairs ===")
    m = (
        matches.merge(items.add_suffix("_1"), left_on="id1", right_on="id_1", how="left")
        .merge(items.add_suffix("_2"), left_on="id2", right_on="id_2", how="left")
    )
    assert m["name_1"].notna().all() and m["name_2"].notna().all()
    stats["pairs_integrity"] = {
        "n": int(len(m)),
        "unique_pairs": int(m.drop_duplicates(["id1", "id2"]).shape[0]),
        "self_pairs": int((m["id1"] == m["id2"]).sum()),
        "undirected_dups": int(
            pd.MultiIndex.from_arrays(
                [np.minimum(m.id1, m.id2), np.maximum(m.id1, m.id2)]
            ).duplicated().sum()
        ),
        "category_mismatch": int((m["category_1"] != m["category_2"]).sum()),
        "pos_rate": float(m["target"].mean()),
        "n_pos": int((m["target"] == 1).sum()),
        "n_neg": int((m["target"] == 0).sum()),
        "unique_items_in_pairs": int(pd.concat([m.id1, m.id2]).nunique()),
        "items_human_not_in_pairs": int(
            len(set(items.id) - set(pd.concat([m.id1, m.id2])))
        ),
    }

    # ---------- text similarities ----------
    log("=== pairwise text features ===")
    n1 = m["name_1"].fillna("").astype(str).tolist()
    n2 = m["name_2"].fillna("").astype(str).tolist()
    nn1 = [norm_text(x) for x in n1]
    nn2 = [norm_text(x) for x in n2]
    tok1 = [tokens(x) for x in n1]
    tok2 = [tokens(x) for x in n2]

    m["name_eq"] = [a == b for a, b in zip(n1, n2)]
    m["name_eq_norm"] = [a == b and a != "" for a, b in zip(nn1, nn2)]
    m["tok_jac"] = [jaccard(a, b) for a, b in zip(tok1, tok2)]
    m["jw"] = [JaroWinkler.normalized_similarity(a, b) for a, b in zip(n1, n2)]
    m["fuzz_ts"] = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(n1, n2)]
    m["len_ratio"] = [min(len(a), len(b)) / max(len(a), len(b), 1) for a, b in zip(n1, n2)]

    nums1 = [set(NUM_RE.findall(a)) for a in n1]
    nums2 = [set(NUM_RE.findall(a)) for a in n2]
    m["num_inter"] = [len(a & b) for a, b in zip(nums1, nums2)]
    m["num_union"] = [len(a | b) for a, b in zip(nums1, nums2)]
    m["num_jaccard"] = [
        (len(a & b) / len(a | b) if (a or b) else 1.0) for a, b in zip(nums1, nums2)
    ]
    m["core_eq"] = [strip_variant_tokens(a) == strip_variant_tokens(b) and strip_variant_tokens(a) != "" for a, b in zip(n1, n2)]
    m["core_differs_only_variant"] = m["core_eq"] & ~m["name_eq_norm"]

    def split_by_target(col):
        return {
            "all": summarize(m[col]),
            "pos": summarize(m.loc[m.target == 1, col]),
            "neg": summarize(m.loc[m.target == 0, col]),
        }

    def rate_by_target(mask):
        mask = np.asarray(mask)
        pos = m["target"].values == 1
        return {
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "pos_rate_when": float(m.loc[mask, "target"].mean()) if mask.any() else None,
            "coverage_of_pos": float((mask & pos).sum() / pos.sum()),
            "coverage_of_neg": float((mask & ~pos).sum() / (~pos).sum()),
        }

    stats["similarity"] = {
        "jw": split_by_target("jw"),
        "tok_jac": split_by_target("tok_jac"),
        "fuzz_token_sort": split_by_target("fuzz_ts"),
        "len_ratio": split_by_target("len_ratio"),
        "num_jaccard": split_by_target("num_jaccard"),
        "exact_name": rate_by_target(m["name_eq"]),
        "exact_name_norm": rate_by_target(m["name_eq_norm"]),
        "core_name_eq_after_stripping_size_color_volume": rate_by_target(m["core_eq"]),
        "differs_only_by_size_color_volume": rate_by_target(m["core_differs_only_variant"]),
        "tok_jac_ge_0.9": rate_by_target(m["tok_jac"] >= 0.9),
        "tok_jac_le_0.2": rate_by_target(m["tok_jac"] <= 0.2),
        "jw_ge_0.9": rate_by_target(m["jw"] >= 0.9),
        "jw_le_0.5": rate_by_target(m["jw"] <= 0.5),
        "has_shared_number": rate_by_target(m["num_inter"] > 0),
        "numbers_conflict": rate_by_target((m["num_union"] > 0) & (m["num_inter"] == 0)),
    }

    # bins for canvas histograms
    def hist(col, bins):
        pos_h, _ = np.histogram(m.loc[m.target == 1, col], bins=bins)
        neg_h, _ = np.histogram(m.loc[m.target == 0, col], bins=bins)
        centers = ((bins[:-1] + bins[1:]) / 2).round(2).tolist()
        return {
            "centers": centers,
            "pos": pos_h.astype(int).tolist(),
            "neg": neg_h.astype(int).tolist(),
        }

    bins = np.linspace(0, 1, 21)
    stats["similarity_hists"] = {
        "jw": hist("jw", bins),
        "tok_jac": hist("tok_jac", bins),
        "fuzz_ts": hist("fuzz_ts", bins),
    }

    # ---------- category ----------
    log("=== per-category ----------")
    cat_rows = []
    for cat, g in m.groupby("category_1"):
        cat_rows.append({
            "category": cat,
            "n_pairs": int(len(g)),
            "pos_rate": float(g.target.mean()),
            "n_items": int(pd.concat([g.id1, g.id2]).nunique()),
            "exact_name_rate": float(g.name_eq.mean()),
            "exact_name_pos_rate": float(g.loc[g.name_eq, "target"].mean()) if g.name_eq.any() else None,
            "mean_jw_pos": float(g.loc[g.target == 1, "jw"].mean()),
            "mean_jw_neg": float(g.loc[g.target == 0, "jw"].mean()),
            "mean_tok_pos": float(g.loc[g.target == 1, "tok_jac"].mean()),
            "mean_tok_neg": float(g.loc[g.target == 0, "tok_jac"].mean()),
            "variant_only_n": int(g.core_differs_only_variant.sum()),
            "variant_only_pos_rate": float(g.loc[g.core_differs_only_variant, "target"].mean()) if g.core_differs_only_variant.any() else None,
            "hard_neg_n": int(((g.target == 0) & (g.tok_jac >= 0.8)).sum()),
            "hard_pos_n": int(((g.target == 1) & (g.tok_jac <= 0.3)).sum()),
            "fashion": cat in FASHION,
        })
    cat_df = pd.DataFrame(cat_rows).sort_values("pos_rate")
    stats["by_category"] = cat_df.to_dict(orient="records")

    # ---------- attributes ----------
    log("=== parse attributes ===")
    attr_cache = {}
    parse_fail = 0
    empty_dict = 0
    key_counter = collections.Counter()
    key_by_cat = collections.defaultdict(collections.Counter)
    nkeys = []
    for i, cat, raw in zip(items["id"], items["category"], items["attributes"]):
        obj = try_json(raw)
        if obj is None:
            parse_fail += 1
            obj = {}
        if not obj:
            empty_dict += 1
        attr_cache[i] = obj
        key_counter.update(obj.keys())
        key_by_cat[cat].update(obj.keys())
        nkeys.append(len(obj))

    stats["attributes_quality"] = {
        "parse_fail": int(parse_fail),
        "empty_dict": int(empty_dict),
        "n_keys_per_item": summarize(nkeys),
        "unique_keys": int(len(key_counter)),
        "top_keys": key_counter.most_common(40),
    }

    # alias / identity vs packaging
    identity_like = [k for k, _ in key_counter.most_common(200) if ARTICLE_KEY.search(k) or k in BRAND_KEYS + COLOR_KEYS + TYPE_KEYS + SIZE_KEYS]
    pack_like = [k for k, _ in key_counter.most_common(200) if PACK_KEY.search(k)]
    stats["attribute_key_groups"] = {
        "identity_like_top": [(k, key_counter[k]) for k in identity_like[:30]],
        "packaging_like": [(k, key_counter[k]) for k in pack_like],
        "brand_aliases": [(k, key_counter[k]) for k in key_counter if "бренд" in k or "торговая марка" in k],
        "article_aliases": [(k, key_counter[k]) for k in key_counter if ARTICLE_KEY.search(k)],
        "color_aliases": [(k, key_counter[k]) for k in key_counter if "цвет" in k],
        "oem_aliases": [(k, key_counter[k]) for k in key_counter if re.search(r"oem|оем", k, re.I)],
    }

    # coverage of key groups by category
    def cat_key_cov(keys):
        out = {}
        for cat, cnt in key_by_cat.items():
            n_items_cat = int((items.category == cat).sum())
            covered = sum(cnt[k] for k in keys)
            # covered is occurrence count not items; approx using any-of
            out[cat] = None
        return out

    brand_keys_present = [k for k in key_counter if k in BRAND_KEYS or "бренд" in k]
    art_keys_present = [k for k in key_counter if ARTICLE_KEY.search(k)]

    cov_rows = []
    for cat, cnt in key_by_cat.items():
        n_cat = int((items["category"] == cat).sum())
        # approximate item coverage: count of key occurrences / n is not exact if multiple keys
        # recompute properly from attr_cache for important groups
        cov_rows.append({"category": cat, "n_items": n_cat, "unique_keys": len(cnt), "top_key": cnt.most_common(1)[0] if cnt else None})
    # precise coverage
    log("=== attribute coverage by category ===")
    item_brand = {}
    item_color = {}
    item_arts = {}
    item_type = {}
    item_nkeys = {}
    has_pack = {}
    for i, obj in attr_cache.items():
        item_brand[i] = get_first(obj, BRAND_KEYS)
        item_color[i] = get_first(obj, COLOR_KEYS)
        item_arts[i] = get_articles(obj)
        item_type[i] = get_first(obj, TYPE_KEYS)
        item_nkeys[i] = len(obj)
        has_pack[i] = any(PACK_KEY.search(k) for k in obj)
    items["has_brand"] = items["id"].map(lambda i: item_brand[i] is not None)
    items["has_art"] = items["id"].map(lambda i: bool(item_arts[i]))
    items["has_color"] = items["id"].map(lambda i: item_color[i] is not None)
    items["has_type"] = items["id"].map(lambda i: item_type[i] is not None)
    items["has_pack"] = items["id"].map(lambda i: has_pack[i])
    cov_by_cat = (
        items.groupby("category")[["has_brand", "has_art", "has_color", "has_type", "has_pack"]]
        .mean()
        .reset_index()
    )
    stats["attr_coverage_by_category"] = cov_by_cat.to_dict(orient="records")
    stats["attr_coverage_overall"] = {
        "brand": float(items.has_brand.mean()),
        "article": float(items.has_art.mean()),
        "color": float(items.has_color.mean()),
        "type": float(items.has_type.mean()),
        "packaging": float(items.has_pack.mean()),
    }

    log("=== structural pair features from attributes ===")
    b1 = m["id1"].map(item_brand)
    b2 = m["id2"].map(item_brand)
    both_b = b1.notna() & b2.notna()
    m["brand_eq"] = both_b & (b1 == b2)
    m["brand_neq"] = both_b & (b1 != b2)
    m["brand_missing"] = ~both_b

    c1 = m["id1"].map(item_color)
    c2 = m["id2"].map(item_color)
    both_c = c1.notna() & c2.notna()
    m["color_eq"] = both_c & (c1 == c2)
    m["color_neq"] = both_c & (c1 != c2)

    a1 = m["id1"].map(item_arts)
    a2 = m["id2"].map(item_arts)
    m["art_inter"] = [bool(x & y) for x, y in zip(a1, a2)]
    m["art_both"] = [bool(x) and bool(y) for x, y in zip(a1, a2)]

    t1 = m["id1"].map(item_type)
    t2 = m["id2"].map(item_type)
    both_t = t1.notna() & t2.notna()
    m["type_eq"] = both_t & (t1 == t2)
    m["type_neq"] = both_t & (t1 != t2)

    # kv overlap
    kv_eq = []
    keys_jac = []
    pack_eq = []
    for i1, i2 in zip(m["id1"], m["id2"]):
        x, y = attr_cache[i1], attr_cache[i2]
        ks = set(x) | set(y)
        common = set(x) & set(y)
        eq = 0
        peq = 0
        pcommon = 0
        for k in common:
            same = str(x[k]).strip().lower() == str(y[k]).strip().lower()
            if same:
                eq += 1
            if PACK_KEY.search(k):
                pcommon += 1
                if same:
                    peq += 1
        kv_eq.append(eq)
        keys_jac.append(len(common) / len(ks) if ks else 1.0)
        pack_eq.append(peq / pcommon if pcommon else None)
    m["kv_eq"] = kv_eq
    m["keys_jac"] = keys_jac

    stats["struct_lifts"] = {
        "brand_eq": rate_by_target(m["brand_eq"]),
        "brand_neq": rate_by_target(m["brand_neq"]),
        "brand_missing": rate_by_target(m["brand_missing"]),
        "color_eq": rate_by_target(m["color_eq"]),
        "color_neq": rate_by_target(m["color_neq"]),
        "art_inter": rate_by_target(m["art_inter"]),
        "art_both_no_inter": rate_by_target(m["art_both"] & ~m["art_inter"]),
        "type_eq": rate_by_target(m["type_eq"]),
        "type_neq": rate_by_target(m["type_neq"]),
        "kv_eq_ge_3": rate_by_target(m["kv_eq"] >= 3),
        "keys_jac_ge_0.5": rate_by_target(np.array(keys_jac) >= 0.5),
    }

    # fashion-specific: color mismatch
    fashion_mask = m["category_1"].isin(FASHION)
    stats["fashion"] = {
        "n": int(fashion_mask.sum()),
        "pos_rate": float(m.loc[fashion_mask, "target"].mean()),
        "color_neq": rate_by_target(fashion_mask & m["color_neq"]),
        "color_eq": rate_by_target(fashion_mask & m["color_eq"]),
        "variant_only": rate_by_target(fashion_mask & m["core_differs_only_variant"]),
        "exact_name": rate_by_target(fashion_mask & m["name_eq"]),
    }

    # ---------- graph ----------
    log("=== pair graph ===")
    deg = collections.Counter(pd.concat([m.id1, m.id2], ignore_index=True))
    stats["graph"] = {
        "n_nodes": len(deg),
        "n_edges": int(len(m)),
        "degree": summarize(list(deg.values())),
        "n_degree_1": int(sum(1 for v in deg.values() if v == 1)),
        "n_degree_ge_5": int(sum(1 for v in deg.values() if v >= 5)),
        "max_degree": int(max(deg.values())),
    }
    uf_all = UF()
    uf_pos = UF()
    for r in m.itertuples(index=False):
        uf_all.union(r.id1, r.id2)
        if r.target == 1:
            uf_pos.union(r.id1, r.id2)
    comps_all = uf_all.components()
    comps_pos = uf_pos.components()
    sizes_all = [len(v) for v in comps_all.values()]
    sizes_pos = [len(v) for v in comps_pos.values() if len(v) > 1]
    stats["graph"]["connected_components"] = {
        "n": len(comps_all),
        "size": summarize(sizes_all),
        "n_size_ge_3": int(sum(1 for s in sizes_all if s >= 3)),
        "largest": int(max(sizes_all)),
    }
    stats["graph"]["positive_components"] = {
        "n_with_2plus": len(sizes_pos),
        "size": summarize(sizes_pos) if sizes_pos else {},
        "largest": int(max(sizes_pos) if sizes_pos else 0),
    }

    # transitivity: negative edge inside a positive component
    pos_root = {x: uf_pos.find(x) for x in uf_pos.p}
    pos_comp_size = collections.Counter()
    for x, r in pos_root.items():
        pos_comp_size[r] += 1
    mixed = 0
    for r in m.itertuples(index=False):
        if r.target != 0:
            continue
        if r.id1 in pos_root and r.id2 in pos_root and pos_root[r.id1] == pos_root[r.id2]:
            if pos_comp_size[pos_root[r.id1]] >= 2:
                mixed += 1
    stats["graph"]["negative_edges_inside_positive_component"] = int(mixed)

    # ---------- examples ----------
    log("=== examples ===")
    cols_ex = ["id1", "id2", "name_1", "name_2", "category_1", "target", "jw", "tok_jac"]
    stats["examples"] = {
        "easy_pos_high_sim": pair_examples(m[(m.target == 1) & (m.tok_jac >= 0.85)], 5, cols_ex),
        "hard_pos_low_sim": pair_examples(m[(m.target == 1) & (m.tok_jac <= 0.25)], 8, cols_ex),
        "hard_neg_high_sim": pair_examples(m[(m.target == 0) & (m.tok_jac >= 0.85)], 8, cols_ex),
        "exact_name_but_neg": pair_examples(m[(m.target == 0) & (m.name_eq)], 8, cols_ex),
        "exact_name_pos": pair_examples(m[(m.target == 1) & (m.name_eq)], 4, cols_ex),
        "variant_only": pair_examples(m[m.core_differs_only_variant], 8, cols_ex + ["core_differs_only_variant"]),
        "article_match": pair_examples(m[m.art_inter], 6, cols_ex),
        "article_conflict": pair_examples(m[m.art_both & ~m.art_inter], 6, cols_ex),
        "brand_conflict_but_pos": pair_examples(m[m.brand_neq & (m.target == 1)], 6, cols_ex),
        "color_conflict_fashion": pair_examples(m[fashion_mask & m.color_neq], 6, cols_ex),
    }

    # sample of raw attributes for a few categories
    sample_attr = {}
    for cat in ["Автотовары", "Одежда", "Ювелирные изделия", "Аптека", "Продукты питания", "Электроника"]:
        sub = items[items.category == cat].head(2)
        sample_attr[cat] = []
        for _, r in sub.iterrows():
            obj = attr_cache[r.id]
            sample_attr[cat].append({
                "name": r["name"],
                "n_keys": len(obj),
                "keys": list(obj.keys())[:25],
                "preview": {k: str(obj[k])[:80] for k in list(obj)[:8]},
            })
    stats["attribute_samples"] = sample_attr

    # ---------- LLM ----------
    log("=== LLM matches ===")
    llm = pd.read_parquet(ROOT / "matches_llm.parquet")
    votes = (llm["target"] * 9).round().astype(int)
    stats["llm"] = {
        "n": int(len(llm)),
        "unique_pairs": int(llm.drop_duplicates(["id1", "id2"]).shape[0]),
        "self_pairs": int((llm.id1 == llm.id2).sum()),
        "target_mean": float(llm.target.mean()),
        "vote_counts": votes.value_counts().sort_index().to_dict(),
        "share_0": float((llm.target == 0).mean()),
        "share_1": float((llm.target == 1).mean()),
        "share_grey_2_to_7": float(((votes >= 2) & (votes <= 7)).mean()),
        "unique_items": int(pd.concat([llm.id1, llm.id2]).nunique()),
    }
    keys_h = pd.DataFrame({
        "a": np.minimum(matches.id1.values, matches.id2.values),
        "b": np.maximum(matches.id1.values, matches.id2.values),
    })
    keys_l = pd.DataFrame({
        "a": np.minimum(llm.id1.values, llm.id2.values),
        "b": np.maximum(llm.id1.values, llm.id2.values),
    })
    overlap_und = pd.merge(keys_h.drop_duplicates(), keys_l.drop_duplicates(), on=["a", "b"], how="inner")
    stats["llm"]["pair_overlap_directed"] = 0  # computed earlier as 0, confirm:
    inter_dir = pd.merge(matches[["id1", "id2"]], llm[["id1", "id2"]], on=["id1", "id2"], how="inner")
    stats["llm"]["pair_overlap_directed"] = int(len(inter_dir))
    stats["llm"]["pair_overlap_undirected"] = int(len(overlap_und))

    h_items = set(items.id)
    llm_items_sample_check = pd.concat([llm.id1, llm.id2], ignore_index=True)
    # membership of human items in LLM item universe
    llm_item_set = set(llm_items_sample_check.unique())
    stats["llm"]["human_items_also_in_llm_pairs"] = int(len(h_items & llm_item_set))
    stats["llm"]["human_items_not_in_llm_pairs"] = int(len(h_items - llm_item_set))

    # LLM category via full items stream (id, category only)
    log("=== stream items.parquet for categories + quality ===")
    pf = pq.ParquetFile(ROOT / "items.parquet")
    cat_counts = collections.Counter()
    empty_name_full = 0
    empty_attr_full = 0
    name_len_sample = []
    uniq_name_hashes = set()
    n_full = 0
    id_to_cat = {}
    # We need categories for unique LLM items. 13M dict is heavy-ish but ok.
    # Use pandas categorical map via chunks into a Series then merge.
    cat_parts = []
    for b in pf.iter_batches(batch_size=1_000_000, columns=["id", "name", "attributes", "category"]):
        df = b.to_pandas()
        n_full += len(df)
        cat_counts.update(df["category"].value_counts().to_dict())
        empty_name_full += int((df["name"].fillna("").str.strip() == "").sum())
        empty_attr_full += int((df["attributes"].fillna("").str.strip().isin(["", "{}", "[]"])).sum())
        if len(name_len_sample) < 3_000_000:
            name_len_sample.append(df["name"].str.len())
        uniq_name_hashes.update(pd.util.hash_pandas_object(df["name"], index=False).tolist())
        cat_parts.append(df[["id", "category"]])
        del df
    items_cat = pd.concat(cat_parts, ignore_index=True)
    del cat_parts
    stats["full_items"] = {
        "n": int(n_full),
        "n_categories": len(cat_counts),
        "category_counts": dict(cat_counts.most_common()),
        "empty_name": int(empty_name_full),
        "empty_or_emptyjson_attr": int(empty_attr_full),
        "approx_unique_names": int(len(uniq_name_hashes)),
        "name_len_sample": summarize(pd.concat(name_len_sample)),
        "human_subset_confirmed": int(items["id"].isin(items_cat["id"]).sum()),
    }

    log("=== LLM by category ===")
    id2cat = items_cat.set_index("id")["category"]
    llm["cat1"] = llm["id1"].map(id2cat)
    llm["cat2"] = llm["id2"].map(id2cat)
    stats["llm"]["category_mismatch"] = int((llm["cat1"] != llm["cat2"]).sum())
    stats["llm"]["missing_cat"] = int(llm["cat1"].isna().sum() + llm["cat2"].isna().sum())
    llm_cat = (
        llm.dropna(subset=["cat1"])
        .groupby("cat1")
        .agg(n=("target", "size"), target_mean=("target", "mean"), share_1=("target", lambda s: (s == 1).mean()), share_0=("target", lambda s: (s == 0).mean()))
        .reset_index()
        .rename(columns={"cat1": "category"})
    )
    stats["llm_by_category"] = llm_cat.to_dict(orient="records")

    # LLM similarity on a sample joined to names from full items? names are large.
    # Use items_human names where possible, else skip.
    # Sample 200k LLM pairs and join names from items_cat? we don't have names in items_cat.
    # Sample from items_human overlap only.
    overlap_items = list(h_items & llm_item_set)
    stats["llm"]["n_overlap_items"] = len(overlap_items)

    # LLM pairs where BOTH items are in items_human — should be 0 pairs if disjoint pair sets,
    # but items can still co-occur if those items appear in other pairings
    llm_hh = llm[llm.id1.isin(h_items) & llm.id2.isin(h_items)]
    stats["llm"]["pairs_with_both_items_in_human_catalog"] = int(len(llm_hh))
    if len(llm_hh):
        stats["llm"]["hh_target_mean"] = float(llm_hh.target.mean())

    # sample LLM pairs: join names from a name lookup built while we still have items_cat only.
    # Reload names for sampled ids.
    log("=== LLM name-similarity sample ===")
    sample_llm = llm.sample(min(300_000, len(llm)), random_state=42)
    need_ids = pd.Index(pd.concat([sample_llm.id1, sample_llm.id2]).unique())
    # stream names for needed ids
    name_map = {}
    pf = pq.ParquetFile(ROOT / "items.parquet")
    need_set = set(need_ids)
    for b in pf.iter_batches(batch_size=1_000_000, columns=["id", "name"]):
        df = b.to_pandas()
        hit = df[df.id.isin(need_set)]
        if len(hit):
            name_map.update(zip(hit.id, hit.name))
        if len(name_map) >= len(need_set):
            break
        del df
    s1 = sample_llm.id1.map(name_map).fillna("")
    s2 = sample_llm.id2.map(name_map).fillna("")
    sample_llm = sample_llm.assign(name_1=s1.values, name_2=s2.values)
    sample_llm["tok_jac"] = [jaccard(tokens(a), tokens(b)) for a, b in zip(sample_llm.name_1, sample_llm.name_2)]
    sample_llm["name_eq"] = sample_llm.name_1.eq(sample_llm.name_2) & sample_llm.name_1.ne("")
    # mean tok_jac by vote
    by_vote = (
        sample_llm.assign(vote=(sample_llm.target * 9).round().astype(int))
        .groupby("vote")
        .agg(n=("tok_jac", "size"), mean_tok=("tok_jac", "mean"), exact=("name_eq", "mean"))
        .reset_index()
    )
    stats["llm_similarity_sample"] = {
        "n": int(len(sample_llm)),
        "names_found": int((sample_llm.name_1.ne("") & sample_llm.name_2.ne("")).sum()),
        "by_vote": by_vote.to_dict(orient="records"),
        "mean_tok_jac": float(sample_llm.tok_jac.mean()),
        "exact_name_rate": float(sample_llm.name_eq.mean()),
    }

    # compare human vs llm hardness
    stats["hardness_compare"] = {
        "human_mean_tok_jac": float(m.tok_jac.mean()),
        "human_mean_tok_jac_pos": float(m.loc[m.target == 1, "tok_jac"].mean()),
        "human_mean_tok_jac_neg": float(m.loc[m.target == 0, "tok_jac"].mean()),
        "llm_sample_mean_tok_jac": float(sample_llm.tok_jac.mean()),
        "llm_sample_mean_tok_sure_pos": float(sample_llm.loc[sample_llm.target == 1, "tok_jac"].mean()) if (sample_llm.target == 1).any() else None,
        "llm_sample_mean_tok_sure_neg": float(sample_llm.loc[sample_llm.target == 0, "tok_jac"].mean()) if (sample_llm.target == 0).any() else None,
    }

    # ---------- baseline leftover: name patterns ----------
    stats["name_patterns"] = {
        "human_pct_with_slash": float(items["name"].str.contains("/").mean()),
        "human_pct_with_comma": float(items["name"].str.contains(",").mean()),
        "human_pct_with_article_like_token": float(
            items["name"].str.contains(r"(?i)\b(?:арт|oem|артикул)\b", regex=True).mean()
        ),
        "median_tokens": float(items["name"].fillna("").map(lambda s: len(str(s).split())).median()),
    }

    stats["runtime_sec"] = round(time.time() - t0, 1)
    # numpy/pandas types → json
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if pd.isna(o) if not isinstance(o, (str, dict, list)) else False:
            return None
        return o

    OUT.write_text(json.dumps(conv(stats), ensure_ascii=False, indent=2))
    log(f"wrote {OUT} in {stats['runtime_sec']}s")


if __name__ == "__main__":
    main()
