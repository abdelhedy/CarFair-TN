"""Modèles maison compatibles scikit-learn."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class GroupMedianRegressor(BaseEstimator, RegressorMixin):
    """
    Baseline : prédit la médiane de la cible des annonces similaires du train,
    en élargissant le groupe tant qu'il compte moins de `min_count` annonces.

    Par défaut : marque + modèle + année → marque + modèle → marque + année → marque → globale.
    """

    def __init__(self, levels=(("marque", "modele", "annee"), ("marque", "modele"),
                               ("marque", "annee"), ("marque",)), min_count=3):
        self.levels = levels
        self.min_count = min_count

    def fit(self, X, y):
        data = X.assign(_y=np.asarray(y))
        self.tables_ = []
        for keys in self.levels:
            grouped = data.groupby(list(keys))["_y"]
            table = grouped.median()[grouped.size() >= self.min_count]
            self.tables_.append((list(keys), table.rename("_pred")))
        self.global_ = float(np.median(y))
        return self

    def predict(self, X):
        pred = pd.Series(np.nan, index=X.index)
        for keys, table in self.tables_:
            missing = pred.isna()
            if not missing.any():
                break
            pred[missing] = X.loc[missing, keys].join(table, on=keys)["_pred"]
        return pred.fillna(self.global_).to_numpy()
