"""Paso 5 – Representación y modelo base: TF-IDF (1-2 gramas) + regresión logística.

Es el modelo que va en la app: rápido, y su explicación es EXACTA (no aproximada),
porque el logit de cada familia es una suma de pesos por término.
"""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from .config import MODELS, SEED


def construir(alpha: float = 2e-6) -> Pipeline:
    """Regresión logística uno-contra-resto, entrenada por descenso de gradiente (log_loss).

    Sigue siendo un modelo lineal por familia (logit = suma de pesos por término),
    pero entrena en minutos con ~200 familias y 200k términos en un portátil.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)),
        ("clf", SGDClassifier(loss="log_loss", alpha=alpha, class_weight="balanced",
                              max_iter=30, tol=1e-4, n_jobs=-1, random_state=SEED)),
    ])


def top_k(modelo: Pipeline, textos, k: int = 3):
    """Devuelve (clases, probabilidades) de las k familias más probables por texto."""
    P = modelo.predict_proba(textos)
    idx = np.argsort(-P, axis=1)[:, :k]
    clases = modelo.classes_[idx]
    return clases, np.take_along_axis(P, idx, axis=1)


def guardar(modelo: Pipeline, nombre: str):
    joblib.dump(modelo, MODELS / f"{nombre}.joblib")


def cargar(nombre: str) -> Pipeline:
    return joblib.load(MODELS / f"{nombre}.joblib")
