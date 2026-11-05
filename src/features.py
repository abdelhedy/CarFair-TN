"""
Transformations qui APPRENNENT des statistiques sur les données.

Elles doivent être placées dans le pipeline de modélisation et fittées sur le
train uniquement (sinon fuite de données vers le test).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HierarchicalImputer(BaseEstimator, TransformerMixin):
    """
    Impute une colonne par la médiane (numérique) ou le mode (catégoriel)
    de groupes de plus en plus larges, appris sur le train.

    Exemple : HierarchicalImputer("kilometrage",
                                  levels=[["annee", "etat_du_vehicule"], ["annee"]],
                                  strategy="median")
    Repli final : médiane / mode global du train.
    """

    def __init__(self, column, levels, strategy="median", min_count=3):
        self.column = column
        self.levels = levels
        self.strategy = strategy
        self.min_count = min_count

    def _agg(self, s):
        if self.strategy == "median":
            return s.median()
        mode = s.mode()
        return mode.iloc[0] if len(mode) else np.nan

    def fit(self, X, y=None):
        known = X[X[self.column].notna()]
        self.tables_ = []
        for keys in self.levels:
            grouped = known.groupby(keys)[self.column]
            table = grouped.agg(self._agg)[grouped.size() >= self.min_count]
            self.tables_.append((list(keys), table.rename("_fill")))
        self.global_ = self._agg(known[self.column])
        return self

    def transform(self, X):
        X = X.copy()
        if self.strategy == "median":
            # Une colonne numérique entièrement vide (None) arrive en type object
            X[self.column] = pd.to_numeric(X[self.column], errors="coerce")
        for keys, table in self.tables_:
            missing = X[self.column].isna()
            if not missing.any():
                break
            fill = X.loc[missing, keys].join(table, on=keys)["_fill"]
            X.loc[missing, self.column] = fill
        X[self.column] = X[self.column].fillna(self.global_)
        return X


class CategoricalFiller(BaseEstimator, TransformerMixin):
    """Remplace les NaN catégoriels par une constante ('inconnu'). Sans apprentissage."""

    def __init__(self, columns, value="inconnu"):
        self.columns = columns
        self.value = value

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = X[self.columns].fillna(self.value)
        return X


class CategoryDtypeCaster(BaseEstimator, TransformerMixin):
    """
    Convertit des colonnes en type 'category' de pandas, avec les catégories
    apprises sur le train (pour XGBoost avec enable_categorical=True).
    Une catégorie absente du train devient NaN, traitée par XGBoost comme manquante.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.categories_ = {c: sorted(X[c].dropna().unique()) for c in self.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])
        # Les autres colonnes sont numériques : une valeur vide (None) arrive en type object
        for c in X.columns.difference(self.columns):
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X
