from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

NOISE_TOKENS = frozenset({"nobrand", "noname", "безбренда"})
FASHION_CATEGORIES = frozenset(
    {"Одежда", "Обувь", "Галантерея и аксессуары"}
)
JEWELRY_CATEGORY = "Ювелирные изделия"
PHARMA_CATEGORY = "Аптека"

BRAND_KEYS = (
    "бренд",
    "бренд в одежде и обуви",
    "бренд товара",
    "торговая марка",
)
COLOR_KEYS = (
    "цвет товара",
    "цвет",
    "название цвета",
    "расцветка",
    "цвет производителя",
)
TYPE_KEYS = ("тип", "тип товара", "вид")
SIZE_KEYS = (
    "размер",
    "российский размер",
    "размер производителя",
    "размер товара",
)

_PACK_RE = re.compile(r"упаковк|габарит", re.I)
_DROP_IDENTITY = re.compile(
    r"валюта|рост модели|коллекция|страна-изготовитель|страна производства|"
    r"страна бренда|гарантийный срок|срок годности|уход за вещами",
    re.I,
)
_ARTICLE_KEY_RE = re.compile(
    r"(артикул|партномер|^oem$|^oem-|^оем-номер$|^оем номер$|^oem-номер$)",
    re.I,
)
_RELATED_ARTICLE_RE = re.compile(r"сопутствующ", re.I)
_VOLUME_RE = re.compile(
    r"(?i)(\d+(?:[.,]\d+)?)\s*(мл|мл\.|л|л\.|гр?\b|кг|шт\.?|таблеток|капсул)"
)
_COUNT_X_RE = re.compile(r"(?i)[xх×]\s*(\d+)\s*(шт)?")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_ALNUM_RE = re.compile(r"(?i)(?<![a-zа-я])[a-zа-я]*\d+[a-zа-я0-9\-/]*")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s\-/]+", re.U)
_NAME_ART_RE = re.compile(r"(?i)(?:арт\.?|артикул)\s*([a-zа-я0-9\-]+)")


@dataclass
class ParsedAttrs:
    brand: str | None = None
    color: str | None = None
    item_type: str | None = None
    size: str | None = None
    articles: set[str] = field(default_factory=set)
    identity: dict[str, str] = field(default_factory=dict)
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class ItemProfile:
    item_id: int
    name: str
    name_norm: str
    category: str
    tokens: set[str]
    alnum: set[str]
    numbers: set[str]
    volumes: set[tuple[float, str]]
    brand: str | None
    color: str | None
    item_type: str | None
    size: str | None
    articles: set[str]
    identity: dict[str, str]
    n_tokens: int


def _norm_val(s: str) -> str:
    return _WS_RE.sub(" ", str(s).strip().lower().replace("ё", "е"))


def normalize_name(name: str) -> str:
    s = str(name).lower().replace("ё", "е")
    s = s.replace("«", " ").replace("»", " ").replace('"', " ").replace("'", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    s = s.replace("no name", " ").replace("без бренда", " ")
    s = _WS_RE.sub(" ", s).strip()
    if " / " in s:
        left, right = s.split(" / ", 1)
        if 0 < len(left.split()) <= 4:
            s = right.strip()
    toks = [t for t in s.split() if t not in NOISE_TOKENS]
    return " ".join(toks)


def is_article_key(key: str) -> bool:
    k = key.strip().lower()
    if _RELATED_ARTICLE_RE.search(k):
        return False
    return bool(_ARTICLE_KEY_RE.search(k))


def is_identity_key(key: str) -> bool:
    k = key.strip().lower()
    if _PACK_RE.search(k) or _DROP_IDENTITY.search(k):
        return False
    return True


def extract_articles(attr: dict) -> set[str]:
    arts: set[str] = set()
    for k, v in attr.items():
        if not is_article_key(k):
            continue
        for part in re.split(r"[;,/|]+", str(v).lower()):
            part = part.strip().strip(".")
            if len(part) >= 3:
                arts.add(part)
    return arts


def extract_volumes(text: str) -> set[tuple[float, str]]:
    out: set[tuple[float, str]] = set()
    unit_map = {
        "мл.": "мл",
        "л.": "л",
        "гр": "г",
        "г": "г",
        "шт.": "шт",
        "шт": "шт",
        "таблеток": "шт",
        "капсул": "шт",
    }
    for match in _VOLUME_RE.finditer(text):
        raw_u = match.group(2).lower()
        unit = unit_map.get(raw_u, raw_u.replace(".", ""))
        try:
            val = float(match.group(1).replace(",", "."))
        except ValueError:
            continue
        out.add((val, unit))
    for match in _COUNT_X_RE.finditer(text):
        try:
            out.add((float(match.group(1)), "шт"))
        except ValueError:
            continue
    return out


def extract_numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUM_RE.findall(text)}


def extract_alnum(text: str) -> set[str]:
    return {t.lower().strip("-/") for t in _ALNUM_RE.findall(text) if len(t) >= 3}


def _first(attr: dict, keys: tuple[str, ...]) -> str | None:
    lowered = {_norm_val(k): v for k, v in attr.items()}
    for k in keys:
        v = lowered.get(_norm_val(k))
        if v is not None and str(v).strip():
            return _norm_val(str(v))
    return None


def parse_attributes(attr: dict) -> ParsedAttrs:
    parsed = ParsedAttrs(raw={_norm_val(k): _norm_val(str(v)) for k, v in attr.items()})
    parsed.brand = _first(attr, BRAND_KEYS)
    parsed.color = _first(attr, COLOR_KEYS)
    parsed.item_type = _first(attr, TYPE_KEYS)
    parsed.size = _first(attr, SIZE_KEYS)
    parsed.articles = extract_articles(attr)
    parsed.identity = {
        _norm_val(k): _norm_val(str(v))
        for k, v in attr.items()
        if is_identity_key(k) and str(v).strip()
    }
    return parsed


def loads_attr(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def profile_item(item_id: int, name: str, attributes: str, category: str) -> ItemProfile:
    name = name or ""
    parsed = parse_attributes(loads_attr(attributes))
    name_norm = normalize_name(name)
    articles = set(parsed.articles)
    for hit in _NAME_ART_RE.findall(name):
        if len(hit) >= 3:
            articles.add(hit.lower())
    tokens = {t for t in name_norm.split() if t}
    return ItemProfile(
        item_id=item_id,
        name=name,
        name_norm=name_norm,
        category=category or "",
        tokens=tokens,
        alnum=extract_alnum(name),
        numbers=extract_numbers(name),
        volumes=extract_volumes(name),
        brand=parsed.brand,
        color=parsed.color,
        item_type=parsed.item_type,
        size=parsed.size,
        articles=articles,
        identity=parsed.identity,
        n_tokens=len(tokens),
    )
