import numpy as np
from sklearn.metrics import average_precision_score

from matching.metric import macro_pr_auc


def test_macro_pr_auc_is_unweighted_mean_of_per_category_ap():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.4])
    cats = np.array(["A", "A", "B", "B", "B", "B"])
    ap_a = average_precision_score(y[:2], s[:2])
    ap_b = average_precision_score(y[2:], s[2:])
    macro, per = macro_pr_auc(y, s, cats)
    assert abs(macro - (ap_a + ap_b) / 2) < 1e-12
    assert set(per) == {"A", "B"}


def test_macro_skips_category_without_both_classes():
    y = np.array([1, 1, 0, 1])
    s = np.array([0.9, 0.8, 0.1, 0.7])
    cats = np.array(["only_pos", "only_pos", "ok", "ok"])
    macro, per = macro_pr_auc(y, s, cats)
    assert "only_pos" not in per
    assert "ok" in per
    assert macro == per["ok"]
