from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.metrics import cohen_kappa_score


class BaseOptimizedRounder(BaseEstimator):
    def __init__(self):
        self.coef_ = 0

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = np.array([0.5, 1.5, 2.5, 3.5])
        self.coef_ = minimize(loss_partial, initial_coef,
                              method='nelder-mead')

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif coef[0] <= pred < coef[1]:
                X_p[i] = 1
            elif coef[1] <= pred < coef[2]:
                X_p[i] = 2
            elif coef[2] <= pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif coef[0] <= pred < coef[1]:
                X_p[i] = 1
            elif coef[1] <= pred < coef[2]:
                X_p[i] = 2
            elif coef[2] <= pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


class OptimizedRounderV1(BaseOptimizedRounder):
    def __init__(self):
        super().__init__()


class OptimizedRounderV2(BaseOptimizedRounder):
    """
    Reference:
    https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """

    def __init__(self):
        super().__init__()

    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf],
                       labels=[0, 1, 2, 3, 4])
        return -cohen_kappa_score(y, preds, weights='quadratic')

    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf],
                       labels=[0, 1, 2, 3, 4])
        return preds
