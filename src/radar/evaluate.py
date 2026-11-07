"""Paso 6 – Evaluación: accuracy, macro-F1, top-3 y calibración (ECE), con su referencia mayoritaria."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


def ece(y_true, P, clases, n_bins: int = 10) -> float:
    """Expected Calibration Error: ¿cuando el modelo dice 0,9 acierta el 90 %?"""
    conf = P.max(1)
    acierto = clases[P.argmax(1)] == np.asarray(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for a, b in zip(bins[:-1], bins[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            total += m.mean() * abs(acierto[m].mean() - conf[m].mean())
    return float(total)


def metricas(modelo, X, y, nombre: str) -> dict:
    y = np.asarray(y)
    P = modelo.predict_proba(X)
    clases = modelo.classes_
    pred = clases[P.argmax(1)]
    conocida = np.isin(y, clases)
    mayoritaria = pd.Series(y).value_counts().index[0]
    return {
        "conjunto": nombre,
        "n": len(y),
        "clases": len(set(y)),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "top3": top_k_accuracy_score(y[conocida], P[conocida], k=3, labels=clases),
        "ece": ece(y, P, clases),
        "acc_mayoritaria": float((y == mayoritaria).mean()),
        "macroF1_mayoritaria": f1_score(y, np.full(len(y), mayoritaria), average="macro", zero_division=0),
    }
