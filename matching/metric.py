from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def macro_pr_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    categories: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Unweighted mean of per-category average_precision_score.

    Categories with only one class are skipped (sklearn cannot compute AP).
    """
    per: dict[str, float] = {}
    for cat in np.unique(categories):
        mask = categories == cat
        y = y_true[mask]
        if y.min() == y.max():
            continue
        per[str(cat)] = float(average_precision_score(y, scores[mask]))
    if not per:
        return float("nan"), per
    return float(np.mean(list(per.values()))), per
