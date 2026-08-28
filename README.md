# E-CUP 2026 — матчинг товаров

Решение задачи 1 хакатона [E-CODE / E-CUP 2026](https://ods.ai): классификация заранее собранных пар карточек (retrieval уже сделан). Метрика — **macro PR-AUC**: `average_precision_score` по каждой из 20 категорий, затем среднее.

Текущий пайплайн сабмита: **MiniLM-L12 pair CLS** (как официальный бейзлайн) + **identity-фичи** (артикул, цвет, объём, размер) → sklearn. Лексический HGB на TF-IDF на паблике дал ~0.28 (почти случай) и в сабмит больше не идёт.

Чекер: Docker `odsai/ecup26-matching-baseline:1.0`, без интернета. Лимиты Check / Public / Private: 1 / 6 / 13 мин; H100 80GB, 20 CPU, 200GB RAM. Zip ≤ 5GB.

## Что где

| Путь | Зачем |
|---|---|
| `run.py` | Точка входа чекера. Пишет CSV `id1,id2,predict` |
| `metadata.json` | Образ и `python -u run.py` |
| `matching/` | Весь код модели |
| `matching/parse.py` | JSON-атрибуты, алиасы бренда/цвета/артикула/OEM |
| `matching/features.py` | Identity-фичи пары (~40 штук) |
| `matching/embed.py` | Текст карточки и CLS пары из MiniLM |
| `matching/dataset.py` | Сборка struct / concat с эмбеддингами; старый TF-IDF путь для тестов |
| `matching/train.py` | Кэш CLS на human-парах → HGB vs logreg → `models/ce_identity.joblib` |
| `matching/metric.py` | Macro PR-AUC как в соревновании |
| `matching/fuzzy.py`, `text.py` | Stdlib-fuzzy и TF-IDF; на инференсе CE-сабмита не используются |
| `tests/` | Юнит-тесты без torch (кроме опционального смоука энкодера) |
| `insights.md` | Выводы EDA: что считать одним SKU, почему лексика не переносится |
| `notebooks/` | `01`–`03` ранний EDA/бейзлайн, `04` + `eda_v2_*` — второй проход |
| `exmples_of_solution/` | Официальные примеры (опечатка в имени папки — так в исходниках) |
| `models/` | Веса и кэш, **не в git** |
| `*.parquet` | Данные соревнования в корне, **не в git** |

## Данные

Кладутся в корень репозитория (gitignore):

| Файл | Строк | Содержание |
|---|---|---|
| `items.parquet` | 13.4M | Все товары: `id, name, attributes, category` |
| `items_human.parquet` | 711k | Товары из ручной разметки |
| `matches.parquet` | 366k | Human-пары, `target ∈ {0,1}`, ~25.7% pos |
| `matches_llm.parquet` | 11.2M | LLM-пары, `target = k/9`. **Другой каталог**: пересечение id с human = 0 |

Категория внутри пары всегда совпадает. Тест ≈ human по объёму, товары новые.

## Модель

1. Текст карточки как у официального бейзлайна: `Name: … Category: … Attributes: …`
2. Cross-encoder MiniLM-L12: CLS пары `[text1, text2]` → 384 числа
3. Рядом identity: пересечение/конфликт артикулов, цвет в моде, объём/доза, числа в названии
4. Sklearn (HGB или `StandardScaler + LogReg`, кто лучше на val) на конкатенации

Веса MiniLM берутся из официального `matching-baseline-submit.zip` (локально уже лежат в `models/cross-encoder-ms-marco-MiniLM-L12-v2/`). Классификатор учится у нас на human.

## Локальный запуск

```bash
python3 -m pip install -r requirements.txt
# для обучения/инференса CE ещё:
python3 -m pip install torch transformers safetensors tqdm
# sklearn 1.9 — как в образе чекера, иначе joblib может не открыться

python3 -m pytest tests/ -q
```

Обучение (кэш CLS: `models/cache/human_ce_cls.npy`; на MPS сотни пар/сек, полный прогон human — десятки минут):

```bash
python3 -m matching.train --data-dir . --out models/ce_identity.joblib
# только эмбеддинги:
python3 -m matching.train --data-dir . --cache-only
```

Инференс — те же флаги, что у чекера:

```bash
python -u run.py \
  --items_path items_human.parquet \
  --matches_path matches.parquet \
  --output_path submit.csv
```

## Сабмит

В корне zip (без обёртки-папки):

- `metadata.json`, `run.py`, `matching/`
- `models/ce_identity.joblib`
- `models/cross-encoder-ms-marco-MiniLM-L12-v2/` — только `config.json`, `model.safetensors`, токенайзер (`tokenizer.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`)

Не класть в zip: кэш `*.npy`, `pytorch_model.bin`, onnx/openvino, `__pycache__`, `rapidfuzz`/`lightgbm`. В образе их нет; torch/transformers/sklearn — есть.

```bash
# пример сборки после обучения
TMP=$(mktemp -d)
cp metadata.json run.py "$TMP/"
cp -R matching "$TMP/"
mkdir -p "$TMP/models/cross-encoder-ms-marco-MiniLM-L12-v2"
cp models/ce_identity.joblib "$TMP/models/"
cp models/cross-encoder-ms-marco-MiniLM-L12-v2/{config.json,model.safetensors,tokenizer.json,vocab.txt,tokenizer_config.json,special_tokens_map.json} \
  "$TMP/models/cross-encoder-ms-marco-MiniLM-L12-v2/"
find "$TMP" -name __pycache__ -type d -exec rm -rf {} +
(cd "$TMP" && zip -r "$OLDPWD/matching-ce-v1.zip" .)
```

## Что не в git

- `*.parquet` — скачиваются отдельно
- `models/` — MiniLM, joblib, кэш эмбеддингов
- `matching-lgbm-v2.zip` — старый лексический сабмит
- `exmples_of_solution/matching-baseline-submit.zip` (~1.1GB, полные веса)

В репозитории остаётся лёгкий официальный пример `exmples_of_solution/matching-baseline-lightweight.zip` (CLI и logreg без MiniLM).
