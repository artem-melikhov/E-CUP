from matching.features import pair_features
from matching.parse import profile_item


def _item(name, attrs, category="Автотовары", item_id=1):
    return profile_item(item_id, name, attrs, category)


def test_article_overlap_is_strong_positive_signal():
    a = _item("опора fenox", '{"oem-номер":"FEM0018; 111"}', item_id=1)
    b = _item("опора двигателя", '{"партномер":"fem0018"}', item_id=2)
    f = pair_features(a, b)
    assert f["art_inter"] == 1
    assert f["art_jaccard"] > 0


def test_article_conflict_when_both_present_and_disjoint():
    a = _item("кольцо diamant 51-310-01316-2", '{"артикул производителя":"51-310-01316-2"}', "Ювелирные изделия", 1)
    b = _item("кольцо diamant 51-310-01764-2", '{"артикул производителя":"51-310-01764-2"}', "Ювелирные изделия", 2)
    f = pair_features(a, b)
    assert f["art_inter"] == 0
    assert f["art_both"] == 1
    assert f["art_conflict"] == 1


def test_color_conflict_in_fashion():
    a = _item("футболка", '{"бренд в одежде и обуви":"acoola","цвет товара":"черный"}', "Одежда", 1)
    b = _item("футболка acoola", '{"бренд в одежде и обуви":"acoola","цвет товара":"белый"}', "Одежда", 2)
    f = pair_features(a, b)
    assert f["color_neq"] == 1
    assert f["is_fashion"] == 1
    assert f["color_neq_fashion"] == 1
    assert f["brand_eq"] == 1


def test_volume_conflict_on_dose():
    a = _item("смазка 5 мл", "{}", "Аптека", 1)
    b = _item("смазка 50 мл", "{}", "Аптека", 2)
    f = pair_features(a, b)
    assert f["vol_conflict"] == 1
    assert f["vol_eq"] == 0


def test_exact_name_in_fashion_flag():
    a = _item("кроссовки nike", "{}", "Обувь", 1)
    b = _item("кроссовки nike", "{}", "Обувь", 2)
    f = pair_features(a, b)
    assert f["name_eq"] == 1
    assert f["name_eq_fashion"] == 1


def test_brand_prefix_does_not_block_token_jaccard():
    a = _item("teatone / чай аромат мяты teatone черный", "{}", "Продукты питания", 1)
    b = _item("чай аромат мяты teatone черный", "{}", "Продукты питания", 2)
    f = pair_features(a, b)
    assert f["tok_jac"] == 1.0
