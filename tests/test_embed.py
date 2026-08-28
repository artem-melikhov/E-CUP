import numpy as np

from matching.dataset import assemble_struct, concat_ce
from matching.embed import build_pair_texts, items_to_text, product_text

from tests.test_dataset import _catalog


def test_product_text_matches_official_shape():
    text = product_text(
        "чай мята",
        "Продукты питания",
        '{"бренд":"teatone","тип":"чай"}',
    )
    assert text.startswith("Name: чай мята Category: Продукты питания Attributes: ")
    assert "бренд: teatone" in text
    assert "тип: чай" in text


def test_product_text_accepts_dict_attributes():
    text = product_text("x", "Автотовары", {"oem-номер": "1", "цвет": "черный"})
    assert "oem-номер: 1" in text
    assert "цвет: черный" in text


def test_assemble_struct_has_identity_not_tfidf():
    items, matches = _catalog()
    x, profiles = assemble_struct(matches, items)
    assert len(x) == 2
    assert "tok_jac" in x.columns
    assert "art_inter" in x.columns
    assert "vol_conflict" in x.columns
    assert "category" in x.columns
    assert "cos_word" not in x.columns
    assert "jw" not in x.columns
    assert x.loc[1, "vol_conflict"] == 1
    assert set(profiles) == {1, 2, 3}


def test_concat_ce_appends_384_and_keeps_rows():
    items, matches = _catalog()
    struct, _ = assemble_struct(matches, items)
    emb = np.zeros((len(struct), 4), dtype=np.float32)
    emb[0, 0] = 1.0
    x = concat_ce(struct, emb)
    assert x.shape[0] == 2
    assert "ce_0" in x.columns
    assert x.loc[0, "ce_0"] == 1.0
    assert "tok_jac" in x.columns


def test_build_pair_texts_keeps_missing_ids():
    items, matches = _catalog()
    matches = matches.copy()
    matches.loc[0, "id2"] = 999
    texts = build_pair_texts(matches, items_to_text(items))
    assert len(texts) == 2
    assert texts[0][1] == ""
