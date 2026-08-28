from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(names: list[str]) -> tuple[TfidfVectorizer, TfidfVectorizer, sparse.spmatrix, sparse.spmatrix]:
    n = len(names)
    try:
        word = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=200_000, sublinear_tf=True
        )
        xw = word.fit_transform(names)
        xc = char.fit_transform(names)
        return word, char, xw, xc
    except ValueError:
        empty = sparse.csr_matrix((n, 1))
        dummy = TfidfVectorizer()
        return dummy, dummy, empty, empty


def pairwise_cosine(matrix: sparse.spmatrix, i1: np.ndarray, i2: np.ndarray) -> np.ndarray:
    a = matrix[i1]
    b = matrix[i2]
    dot = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    na = np.sqrt(np.asarray(a.multiply(a).sum(axis=1))).ravel()
    nb = np.sqrt(np.asarray(b.multiply(b).sum(axis=1))).ravel()
    return dot / np.maximum(na * nb, 1e-9)
