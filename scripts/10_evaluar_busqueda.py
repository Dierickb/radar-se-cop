"""Evaluación de la búsqueda de procesos para pymes (recuperación de información).

Las consultas y los juicios de relevancia son HUMANOS (nunca generados por el LLM).

Paso 1 – armar el pool a juzgar:
    python scripts/10_evaluar_busqueda.py pool
  lee data/consultas/consultas_pyme.xlsx (o .csv; columnas: id, consulta, autor) — unas 30 descripciones escritas por
  personas como las escribiría una pyme — y junta los top-10 de cuatro sistemas (léxico, semántico,
  familia y el híbrido de la app). Escribe data/consultas/juicios.xlsx con la columna 'relevante' vacía.
  El pool va barajado y sin decir qué sistema trajo cada proceso (juicio a ciegas).

Paso 2 – después de marcar 'relevante' (1 = le sirve a esa pyme, 0 = no):
    python scripts/10_evaluar_busqueda.py evaluar
  calcula precisión@5, precisión@10 y nDCG@10 por sistema → reports/metricas_busqueda.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from radar.config import DATA_PROC, REPORTS, ROOT, SEED
from radar.model import cargar
from radar.search import BuscadorProcesos
from radar.text import preparar

C = ROOT / "data" / "consultas"
SISTEMAS = {"lexico": (0, 1, 0), "semantico": (1, 0, 0), "familia": (0, 0, 1), "hibrido_app": (.5, .3, .2)}


def rankings(k: int = 10) -> pd.DataFrame:
    xl = C / "consultas_pyme.xlsx"
    q = (pd.read_excel(xl, dtype=str) if xl.exists() else
         pd.read_csv(C / "consultas_pyme.csv", dtype=str, sep=None, engine="python", encoding="utf-8-sig"))
    q = q.dropna(subset=["consulta"])
    p = pd.read_parquet(DATA_PROC / "procesos_idx.parquet")
    B = BuscadorProcesos(p, p.txt_lem.tolist(), p.texto.tolist(), embeddings=np.load(DATA_PROC / "procesos_emb.npy"))
    m = cargar("familia_v1")
    filas = []
    for r in q.itertuples():
        qp = preparar(r.consulta)
        P = m.predict_proba([qp])[0]
        probs = {m.classes_[i]: float(P[i]) for i in np.argsort(-P)[:3]}
        for sis, (we, wt, wf) in SISTEMAS.items():
            res = B.buscar(r.consulta, qp, probs, k=k, w_emb=we, w_tfidf=wt, w_fam=wf)
            for pos, x in enumerate(res.itertuples(), 1):
                filas.append({"consulta_id": r.id, "consulta": r.consulta, "sistema": sis, "posicion": pos,
                              "id_del_proceso": x.id_del_proceso, "texto": x.texto, "entidad": x.entidad,
                              "url": x.urlproceso})
    return pd.DataFrame(filas)


def pool():
    R = rankings()
    R.to_csv(C / "rankings.csv", index=False)
    J = R.drop_duplicates(["consulta_id", "id_del_proceso"])[["consulta_id", "consulta", "id_del_proceso", "texto", "entidad", "url"]]
    J = J.sample(frac=1, random_state=SEED).sort_values("consulta_id", kind="stable").assign(relevante="")
    J.to_excel(C / "juicios.xlsx", index=False)
    print(f"{J.consulta_id.nunique()} consultas, {len(J)} pares a juzgar → {C / 'juicios.xlsx'}")


def evaluar():
    R = pd.read_csv(C / "rankings.csv")
    J = pd.read_excel(C / "juicios.xlsx")[["consulta_id", "id_del_proceso", "relevante"]]
    if J.relevante.isna().any():
        print(f"Aviso: {J.relevante.isna().sum()} pares sin juzgar se cuentan como no relevantes")
    R = R.merge(J, on=["consulta_id", "id_del_proceso"], how="left").fillna({"relevante": 0})
    R["relevante"] = R.relevante.astype(int)
    filas = []
    for (sis, qid), g in R.groupby(["sistema", "consulta_id"]):
        rel = g.sort_values("posicion").relevante.values
        ideal = np.sort(J[J.consulta_id == qid].relevante.fillna(0).astype(int).values)[::-1][:10]
        disc = 1 / np.log2(np.arange(2, 12))
        dcg, idcg = (rel[:10] * disc[:len(rel[:10])]).sum(), (ideal * disc[:len(ideal)]).sum()
        filas.append({"sistema": sis, "consulta_id": qid, "p@5": rel[:5].mean(), "p@10": rel[:10].mean(),
                      "ndcg@10": dcg / idcg if idcg else 0.0})
    D = pd.DataFrame(filas)
    res = D.groupby("sistema")[["p@5", "p@10", "ndcg@10"]].mean().round(3)
    res["n_consultas"] = D.groupby("sistema").size()
    res.to_csv(REPORTS / "metricas_busqueda.csv")
    print(res)


if __name__ == "__main__":
    {"pool": pool, "evaluar": evaluar}[sys.argv[1]]()
