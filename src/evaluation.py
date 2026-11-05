"""
Métriques d'évaluation.

Le modèle est entraîné sur log1p(prix) ; les métriques présentées à un humain
sont calculées en dinars, après inversion par expm1.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error, r2_score


def to_tnd(y_log):
    return np.expm1(np.asarray(y_log, dtype=float))


def mae_tnd(y_true_log, y_pred_log):
    """Erreur absolue moyenne, en dinars."""
    return float(np.mean(np.abs(to_tnd(y_true_log) - to_tnd(y_pred_log))))


def median_ape(y_true_log, y_pred_log):
    """Erreur relative médiane (en %) : la moitié des estimations font mieux."""
    t, p = to_tnd(y_true_log), to_tnd(y_pred_log)
    return float(np.median(np.abs(p - t) / t) * 100)


def share_within(y_true_log, y_pred_log, pct):
    """Part des estimations à moins de `pct` % du prix réel."""
    t, p = to_tnd(y_true_log), to_tnd(y_pred_log)
    return float(np.mean(np.abs(p - t) / t <= pct / 100) * 100)


def rmse_log(y_true_log, y_pred_log):
    return float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))


def report(y_true_log, y_pred_log):
    """Tableau de métriques (log et dinars) pour un jeu de prédictions."""
    return {
        "RMSE (log)": rmse_log(y_true_log, y_pred_log),
        "R² (log)": float(r2_score(y_true_log, y_pred_log)),
        "MAE (TND)": mae_tnd(y_true_log, y_pred_log),
        "Erreur relative médiane (%)": median_ape(y_true_log, y_pred_log),
        "À ±10 % (%)": share_within(y_true_log, y_pred_log, 10),
        "À ±20 % (%)": share_within(y_true_log, y_pred_log, 20),
    }


# Scorers pour cross_validate (convention scikit-learn : plus grand = meilleur)
SCORERS = {
    "rmse_log": make_scorer(rmse_log, greater_is_better=False),
    "mae_tnd": make_scorer(mae_tnd, greater_is_better=False),
    "median_ape": make_scorer(median_ape, greater_is_better=False),
}


def cv_summary(cv_results):
    """Moyenne ± écart-type des scores de cross_validate, avec le bon signe."""
    rows = {}
    for name in SCORERS:
        scores = -np.asarray(cv_results[f"test_{name}"])
        rows[name] = (scores.mean(), scores.std())
    return pd.Series({f"{k} (moy.)": v[0] for k, v in rows.items()} |
                     {f"{k} (écart-type)": v[1] for k, v in rows.items()})
