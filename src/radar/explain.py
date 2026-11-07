"""Paso 9 – Explicabilidad del modelo lineal (global, local, contrafactual y por ejemplos).

Para TF-IDF + regresión logística la contribución de cada término al logit de una
familia es exactamente  peso_tfidf(término) × coeficiente(familia, término).
No es una aproximación (como LIME o SHAP): es la cuenta que hace el modelo.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


def _partes(modelo):
    return modelo.named_steps["tfidf"], modelo.named_steps["clf"]


def global_por_familia(modelo, familia: str, k: int = 15) -> pd.DataFrame:
    """Términos que más empujan hacia (y en contra de) una familia."""
    vec, clf = _partes(modelo)
    i = list(clf.classes_).index(familia)
    coef = clf.coef_[i]
    voc = vec.get_feature_names_out()
    orden = np.argsort(coef)
    return pd.DataFrame({
        "a_favor": voc[orden[::-1][:k]], "peso_a_favor": coef[orden[::-1][:k]],
        "en_contra": voc[orden[:k]], "peso_en_contra": coef[orden[:k]],
    })


def local(modelo, texto_preparado: str, familia: str | None = None, k: int = 8) -> pd.DataFrame:
    """Contribución exacta de cada término del texto al logit de la familia (la predicha por defecto)."""
    vec, clf = _partes(modelo)
    x = vec.transform([texto_preparado])
    if familia is None:
        familia = clf.classes_[clf.predict_proba(x).argmax()]
    i = list(clf.classes_).index(familia)
    cols = x.indices
    contrib = x.data * clf.coef_[i, cols]
    voc = vec.get_feature_names_out()
    df = pd.DataFrame({"termino": voc[cols], "contribucion": contrib})
    return df.reindex(df.contribucion.abs().sort_values(ascending=False).index).head(k).reset_index(drop=True)


def contrafactual(modelo, texto_preparado: str, quitar: str) -> dict:
    """¿Qué pasa con la predicción si quitamos una palabra o frase?"""
    antes = modelo.predict_proba([texto_preparado])[0]
    despues_txt = " ".join(t for t in texto_preparado.split() if t not in set(quitar.split()))
    despues = modelo.predict_proba([despues_txt])[0]
    c = modelo.classes_
    return {"antes": (c[antes.argmax()], float(antes.max())),
            "despues": (c[despues.argmax()], float(despues.max())), "texto_sin": despues_txt}


def fidelidad(modelo, textos_preparados, k: int = 3) -> dict:
    """ERASER (DeYoung et al., 2020): comprehensiveness = caída de prob. al quitar los k términos
    más importantes; sufficiency = caída al dejar SOLO esos k términos. Más alta / más baja = mejor."""
    comp, suf = [], []
    for t in textos_preparados:
        P0 = modelo.predict_proba([t])[0]
        j = P0.argmax()
        top = set(local(modelo, t, modelo.classes_[j], k=k).termino.str.split().sum())
        sin = " ".join(w for w in t.split() if w not in top)
        solo = " ".join(w for w in t.split() if w in top)
        comp.append(P0[j] - modelo.predict_proba([sin])[0][j])
        suf.append(P0[j] - modelo.predict_proba([solo])[0][j])
    return {"comprehensiveness": float(np.mean(comp)), "sufficiency": float(np.mean(suf)), "n": len(comp)}


class Vecinos:
    """Explicación por ejemplos: los contratos reales más parecidos (similitud coseno TF-IDF)."""

    def __init__(self, modelo, textos_preparados, tabla: pd.DataFrame):
        self.vec = modelo.named_steps["tfidf"]
        self.X = self.vec.transform(textos_preparados)
        self.tabla = tabla.reset_index(drop=True)

    def buscar(self, texto_preparado: str, k: int = 3) -> pd.DataFrame:
        s = linear_kernel(self.vec.transform([texto_preparado]), self.X).ravel()
        idx = np.argsort(-s)[:k]
        return self.tabla.iloc[idx].assign(similitud=s[idx])
