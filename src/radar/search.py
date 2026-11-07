"""Paso 8 – Búsqueda de procesos publicados para una pyme (recuperación de información).

Puntaje híbrido por proceso:
    score = w_emb · coseno(embeddings) + w_tfidf · coseno(TF-IDF) + w_fam · P(familia del proceso | consulta)
- embeddings estáticos multilingües (model2vec, sin GPU) capturan sinónimos ("aires" ~ "climatización")
- TF-IDF premia coincidencias literales de términos técnicos
- el clasificador aporta la familia UNSPSC sugerida para la consulta
Los pesos se ajustan con las ~30 consultas de pyme juzgadas a mano (precisión@5).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

MODELO_EMB = "minishlab/potion-multilingual-128M"


def _encoder():
    from model2vec import StaticModel
    return StaticModel.from_pretrained(MODELO_EMB)


def _unit(E):
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)


class BuscadorProcesos:
    def __init__(self, procesos: pd.DataFrame, textos_preparados: list[str], textos_crudos: list[str],
                 embeddings: np.ndarray | None = None):
        self.p = procesos.reset_index(drop=True)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        self.Xt = self.tfidf.fit_transform(textos_preparados)
        self.enc = _encoder()
        self.Xe = _unit(embeddings if embeddings is not None else self.enc.encode(list(textos_crudos)))
        self.fam = self.p["fam"].fillna("").values

    def buscar(self, consulta_cruda: str, consulta_preparada: str, probs_familia: dict[str, float],
               k: int = 10, w_emb: float = 0.5, w_tfidf: float = 0.3, w_fam: float = 0.2) -> pd.DataFrame:
        s_emb = (self.Xe @ _unit(self.enc.encode([consulta_cruda])).T).ravel()
        s_tf = linear_kernel(self.tfidf.transform([consulta_preparada]), self.Xt).ravel()
        s_fam = np.array([probs_familia.get(f, 0.0) for f in self.fam])
        score = w_emb * s_emb + w_tfidf * s_tf + w_fam * s_fam
        idx = np.argsort(-score)[:k * 3]
        res = self.p.iloc[idx].assign(score=score[idx], sim_semantica=s_emb[idx],
                                      sim_lexica=s_tf[idx], p_familia=s_fam[idx])
        # varias entidades publican procesos con el mismo texto: se muestra uno por texto
        return res.drop_duplicates("texto").head(k)
