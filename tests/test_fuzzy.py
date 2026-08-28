from matching.fuzzy import jaro_winkler, ratio, token_sort_ratio


def test_identical_strings_are_one():
    assert jaro_winkler("чай мята", "чай мята") == 1.0
    assert ratio("чай мята", "чай мята") == 1.0
    assert token_sort_ratio("мята чай", "чай мята") == 1.0


def test_empty_is_zero_unless_both_empty():
    assert jaro_winkler("", "a") == 0.0
    assert jaro_winkler("", "") == 1.0
    assert ratio("", "a") == 0.0


def test_prefix_boost_and_token_reorder():
    assert jaro_winkler("martha", "marhta") > 0.9
    assert token_sort_ratio("teatone чай мята", "чай мята teatone") > ratio(
        "teatone чай мята", "чай мята teatone"
    )
