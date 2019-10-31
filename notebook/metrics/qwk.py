"""
Reference: https://www.kaggle.com/cpmpml/ultra-fast-qwk-calc-method
"""

import numpy as np
from numba import jit
from sklearn.metrics import confusion_matrix


def quad_kappa(act, pred, n: int = 4, hist_range: tuple = (0, 3)) -> float:
    O = confusion_matrix(act, pred)
    O = np.divide(O, np.sum(O))

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i][j] = ((i - j) ** 2) / ((n - 1) ** 2)

    act_hist = np.histogram(act, bins=n, range=hist_range)[0]
    prd_hist = np.histogram(pred, bins=n, range=hist_range)[0]

    E = np.outer(act_hist, prd_hist)
    E = np.divide(E, np.sum(E))

    num = np.sum(np.multiply(W, O))
    den = np.sum(np.multiply(W, E))

    return 1 - np.divide(num, den)


@jit
def qwk3(a1, a2, max_rat: int = 3) -> float:
    assert (len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1,))
    hist2 = np.zeros((max_rat + 1,))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e
