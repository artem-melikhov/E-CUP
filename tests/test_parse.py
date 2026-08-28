from matching.parse import (
    extract_alnum,
    extract_articles,
    extract_numbers,
    extract_volumes,
    is_article_key,
    is_identity_key,
    normalize_name,
    parse_attributes,
    profile_item,
)


def test_normalize_strips_brand_slash_prefix():
    assert normalize_name("teatone / чай аромат мяты teatone черный") == (
        "чай аромат мяты teatone черный"
    )


def test_normalize_strips_nobrand_and_quotes():
    out = normalize_name("nobrand калькулятор «ps-268a»")
    assert "nobrand" not in out
    assert "ps-268a" in out
    assert "«" not in out


def test_normalize_keeps_short_names_without_slash():
    assert normalize_name("кроссовки nike") == "кроссовки nike"


def test_article_key_matches_aliases_not_emkost():
    assert is_article_key("oem-номер")
    assert is_article_key("оем номер")
    assert is_article_key("партномер (артикул производителя)")
    assert is_article_key("артикул производителя")
    assert not is_article_key("энергоемкость")
    assert not is_article_key("высота проема")
    assert not is_article_key("артикул сопутствующего товара")


def test_extract_articles_splits_oem_list():
    arts = extract_articles(
        {
            "oem-номер": "3050086; a9064210012; 2e0615301",
            "партномер (артикул производителя)": "8500892sx",
        }
    )
    assert "3050086" in arts
    assert "a9064210012" in arts
    assert "8500892sx" in arts


def test_brand_alias_from_fashion_key():
    parsed = parse_attributes(
        {"бренд в одежде и обуви": "acoola", "цвет товара": "хаки"}
    )
    assert parsed.brand == "acoola"
    assert parsed.color == "хаки"


def test_identity_keys_drop_packaging_and_currency():
    assert not is_identity_key("длина упаковки")
    assert not is_identity_key("валюта")
    assert not is_identity_key("рост модели на фото")
    assert is_identity_key("проба")
    assert is_identity_key("вставка")
    assert is_identity_key("тип")


def test_volumes_distinguish_dose_and_count():
    v1 = extract_volumes("смазка 5 мл")
    v2 = extract_volumes("смазка 50 мл")
    v3 = extract_volumes("корм 85 г х 26 шт")
    assert v1 != v2
    assert (5.0, "мл") in v1
    assert (50.0, "мл") in v2
    assert (26.0, "шт") in v3 or any(x[1] == "шт" for x in v3)


def test_alnum_and_numbers_from_model_codes():
    name = "кольцо sokolov 110156 р. 18,5"
    assert "110156" in extract_alnum(name)
    nums = extract_numbers(name)
    assert "110156" in nums
    assert "18.5" in nums or "18,5" in nums


def test_profile_item_combines_name_and_json():
    p = profile_item(
        item_id=1,
        name="fenox / опора двигателя арт. fem0018",
        attributes='{"бренд":"fenox","oem-номер":"FEM0018; 123","тип":"опора"}',
        category="Автотовары",
    )
    assert p.brand == "fenox"
    assert "fem0018" in p.articles
    assert p.item_type == "опора"
    assert "nobrand" not in p.name_norm
    assert p.category == "Автотовары"
