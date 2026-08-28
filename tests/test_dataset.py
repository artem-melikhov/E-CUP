import pandas as pd

from matching.dataset import assemble_features


def _catalog():
    items = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": [
                "teatone / чай мята 100г",
                "чай мята teatone 100г",
                "чай мята teatone 250г",
            ],
            "attributes": [
                '{"бренд":"teatone","тип":"чай"}',
                '{"бренд":"teatone","тип":"чай"}',
                '{"бренд":"teatone","тип":"чай"}',
            ],
            "category": ["Продукты питания"] * 3,
        }
    )
    matches = pd.DataFrame({"id1": [1, 1], "id2": [2, 3], "target": [1.0, 0.0]})
    return items, matches


def test_assemble_features_on_tiny_catalog():
    items, matches = _catalog()
    x, _, _, _ = assemble_features(matches, items)
    assert len(x) == 2
    assert "tok_jac" in x.columns
    assert "cos_word" in x.columns
    assert "category" in x.columns
    assert x.loc[0, "vol_conflict"] == 0
    assert x.loc[1, "vol_conflict"] == 1
    assert x.loc[0, "tok_jac"] >= x.loc[1, "tok_jac"]


def test_assemble_features_handles_na_name_and_filters_unused_items():
    items, matches = _catalog()
    extra = items.iloc[[0]].copy()
    extra["id"] = 99
    extra["name"] = pd.NA
    items = pd.concat([items, extra], ignore_index=True)
    x, _, _, _ = assemble_features(matches, items)
    assert len(x) == 2


def test_assemble_features_keeps_rows_if_pair_id_missing():
    items, matches = _catalog()
    matches = matches.copy()
    matches.loc[0, "id2"] = 999
    x, _, _, _ = assemble_features(matches, items)
    assert len(x) == 2
