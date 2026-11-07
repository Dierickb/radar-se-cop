"""Radar SECOP – demo en Streamlit.

Ejecutar desde la raíz del repo:  streamlit run app/app.py
Requiere el modelo (scripts/03) y el índice de procesos (scripts/07); con datos congelados, antes scripts/00.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from radar.config import DATA_PROC, REPORTS, UMBRAL_DISCREPANCIA
from radar.data import familias_unspsc
from radar.explain import local, Vecinos
from radar.model import cargar
from radar.search import BuscadorProcesos
from radar.text import preparar, limpiar

st.set_page_config(page_title="Radar SECOP", layout="wide")
UMBRAL_APP = 0.70   # umbral calibrado (ver reports/explicabilidad.json)


@st.cache_resource(show_spinner="Cargando modelo y catálogo…")
def recursos():
    modelo = cargar("familia_v1")
    nombres = familias_unspsc()
    c = pd.read_parquet(DATA_PROC / "contratos_prep.parquet",
                        columns=["id_contrato", "objeto_del_contrato", "nombre_entidad", "fam", "txt_lem"])
    gold = ROOT / "data" / "gold" / "gold_ids.csv"     # el gold set nunca se muestra como ejemplo
    if gold.exists():
        c = c[~c.id_contrato.isin(set(pd.read_csv(gold).id_contrato))]
    c = c.sample(min(60_000, len(c)), random_state=0)
    vecinos = Vecinos(modelo, c.txt_lem.tolist(), c[["objeto_del_contrato", "nombre_entidad", "fam"]])
    return modelo, nombres, vecinos


@st.cache_resource(show_spinner="Cargando procesos publicados…")
def buscador():
    p = pd.read_parquet(DATA_PROC / "procesos_idx.parquet")
    emb = np.load(DATA_PROC / "procesos_emb.npy")
    return BuscadorProcesos(p, p.txt_lem.tolist(), p.texto.tolist(), embeddings=emb)


def nombre(fam: str) -> str:
    return f"{fam} · {NOMBRES.get(fam, '¿?')}"


def grafico_explicacion(exp: pd.DataFrame):
    if exp.empty:
        st.info("El texto no tiene términos que el modelo conozca.")
        return
    d = exp.set_index("termino")["contribucion"].sort_values()
    st.bar_chart(d, horizontal=True, height=40 + 28 * len(d))
    st.caption("Barras positivas empujan HACIA la familia sugerida; negativas, en contra. "
               "Es la contribución exacta de cada término en el modelo lineal.")


MODELO, NOMBRES, VECINOS = recursos()

st.title("Radar SECOP")
st.write("Encuentre en qué categoría del Estado encaja su negocio y qué procesos de SECOP II le sirven.")
tab_pyme, tab_audit, tab_info = st.tabs(["🔎 Para pymes", "🧾 Revisar un contrato", "ℹ️ Cómo funciona"])

# ---------------------------------------------------------------- Pymes
with tab_pyme:
    q = st.text_area("Describa con sus palabras qué vende o qué servicio presta",
                     "Hacemos mantenimiento preventivo y correctivo de aires acondicionados para oficinas")
    if st.button("Buscar", type="primary") and q.strip():
        qp = preparar(q)
        P = MODELO.predict_proba([qp])[0]
        orden = np.argsort(-P)[:3]
        st.subheader("Categorías sugeridas (familia UNSPSC)")
        cols = st.columns(3)
        for c, i in zip(cols, orden):
            c.metric(nombre(MODELO.classes_[i]), f"{P[i]:.0%}")
        with st.expander("¿Por qué esta categoría?", expanded=True):
            grafico_explicacion(local(MODELO, qp, MODELO.classes_[orden[0]], k=8))
            st.write("**Contratos reales parecidos:**")
            v = VECINOS.buscar(qp, k=3)
            v["familia"] = v.fam.map(nombre)
            st.dataframe(v[["objeto_del_contrato", "nombre_entidad", "familia", "similitud"]],
                         hide_index=True, width="stretch")
        st.subheader("Procesos publicados que le pueden servir")
        try:
            B = buscador()
            probs = {MODELO.classes_[i]: float(P[i]) for i in orden}
            r = B.buscar(q, qp, probs, k=10)
            r["familia"] = r.fam.map(nombre) + np.where(r.familia_inferida, " (inferida)", "")
            r["precio_base"] = r.precio_base.map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
            st.dataframe(r[["texto", "entidad", "departamento_entidad", "precio_base", "familia",
                            "estado_del_procedimiento", "score", "urlproceso"]],
                         hide_index=True, width="stretch",
                         column_config={"urlproceso": st.column_config.LinkColumn("SECOP"),
                                        "score": st.column_config.ProgressColumn("ajuste", min_value=0, max_value=1)})
            st.caption("Ajuste = 0,5·similitud semántica + 0,3·similitud de términos + 0,2·probabilidad de la familia.")
        except FileNotFoundError:
            st.warning("Falta el índice de procesos: corra scripts/07_indexar_procesos.py")

# ---------------------------------------------------------------- Auditoría
with tab_audit:
    st.write("Pegue el objeto de un contrato y su código UNSPSC declarado. "
             "Si el modelo sugiere otra familia con confianza alta, se marca **para revisar** "
             "(una discrepancia no es un error ni una irregularidad: es un dato a verificar).")
    obj = st.text_area("Objeto del contrato",
                       "Prestación de servicios profesionales como abogado para el acompañamiento jurídico "
                       "de la personería municipal")
    cod = st.text_input("Código declarado (8 dígitos)", "80111600")
    if st.button("Revisar") and obj.strip():
        qp = preparar(obj)
        P = MODELO.predict_proba([qp])[0]
        i = P.argmax()
        pred, conf, decl = MODELO.classes_[i], float(P[i]), cod.strip()[:4]
        c1, c2 = st.columns(2)
        c1.metric("Familia declarada", nombre(decl))
        c2.metric("Familia sugerida por el modelo", nombre(pred), f"confianza {conf:.0%}")
        if pred != decl and conf >= UMBRAL_APP:
            st.error("Posible discrepancia: la descripción apunta a otra familia. Para revisar.")
        elif pred != decl:
            st.warning("El modelo sugiere otra familia, pero con confianza baja: no se marca.")
        else:
            st.success("La familia declarada coincide con la sugerida.")
        st.write("**Términos que explican la sugerencia**")
        grafico_explicacion(local(MODELO, qp, pred, k=8))
        st.write("**Texto que ve el modelo** (sin fórmulas de contratación, lematizado):")
        st.code(qp or "(vacío)")
        st.caption("Texto antes de lematizar: " + limpiar(obj))

# ---------------------------------------------------------------- Info
with tab_info:
    st.subheader("Métricas del modelo (acuerdo con el código declarado en SECOP)")
    try:
        st.dataframe(pd.read_csv(REPORTS / "metricas_familia.csv").round(3), hide_index=True)
    except FileNotFoundError:
        st.info("Corra scripts/03_entrenar_evaluar.py")
    st.markdown(
        "- Las etiquetas de SECOP tienen sesgo: la familia 8011 (recursos humanos) se usa como cajón de sastre. "
        "Por eso estas métricas miden **acuerdo con SECOP**, no exactitud; la evaluación final es contra un "
        "gold set anotado a mano.\n"
        "- Datos: SECOP II (datos.gov.co, CC BY-SA 4.0). Catálogo: UNSPSC v14 en español, Colombia Compra Eficiente.")
