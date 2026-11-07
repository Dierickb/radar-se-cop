"""Pasos 5-6: entrena y evalúa el clasificador de FAMILIA en dos variantes de texto.

A = texto normalizado (con fórmulas)     B = sin fórmulas + lematizado (el de la app)
Mismo split para ambas (agrupado por texto). Etiqueta = código declarado en SECOP:
estas métricas miden acuerdo con SECOP, NO con la verdad (eso lo hará el gold set).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json
import numpy as np
import pandas as pd

from radar.config import DATA_PROC, REPORTS, ROOT, SEED
from radar.labels import particion_por_texto, clases_validas, balancear, excluir_gold
from radar.model import construir, guardar
from radar.evaluate import metricas

df = pd.read_parquet(DATA_PROC / "contratos_prep.parquet")
clases = clases_validas(df, "fam")
df = df[df.fam.isin(clases)]
tr, te = particion_por_texto(df, "txt_norm")
n0 = len(tr)
tr = excluir_gold(tr, df, ROOT / "data" / "gold" / "gold_ids.csv")   # el gold set nunca se entrena
print(f"Excluidos del train por estar en el gold set (id o mismo texto): {n0 - len(tr)}")
tr_bal = balancear(tr, "fam")
te_bal = balancear(te, "fam", tope=300)
print(f"Familias: {len(clases)} | train: {len(tr):,} (balanceado {len(tr_bal):,}) | "
      f"test natural: {len(te):,} | test balanceado: {len(te_bal):,}")
pd.Series(te.id_contrato).to_csv(DATA_PROC / "test_ids.csv", index=False)

filas, modelos = [], {}
for var, col in [("A_normalizado", "txt_norm"), ("B_limpio_lematizado", "txt_lem")]:
    m = construir().fit(tr_bal[col], tr_bal.fam)
    modelos[var] = m
    for nombre, t in [("test_natural", te), ("test_balanceado", te_bal)]:
        r = metricas(m, t[col], t.fam, nombre)
        r["variante"] = var
        filas.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

res = pd.DataFrame(filas)[["variante", "conjunto", "n", "clases", "accuracy", "macro_f1", "top3", "ece",
                           "acc_mayoritaria", "macroF1_mayoritaria"]]
res.to_csv(REPORTS / "metricas_familia.csv", index=False)
guardar(modelos["B_limpio_lematizado"], "familia_v1")
guardar(modelos["A_normalizado"], "familia_v0_con_formulas")

# --- Auditoría del atajo: ¿qué tanto empujan las fórmulas hacia 8011? (solo variante A las ve)
mA = modelos["A_normalizado"]
vec, clf = mA.named_steps["tfidf"], mA.named_steps["clf"]
voc = pd.Series(vec.get_feature_names_out())
i8011 = list(clf.classes_).index("8011")
formulas = ["apoyo a", "la gestion", "sus propios", "propios medios", "plena autonomia",
            "servicios profesionales", "prestacion de", "prestar servicios", "de apoyo"]
pesos = {f: float(clf.coef_[i8011, voc[voc == f].index[0]]) for f in formulas if (voc == f).any()}
rank = pd.Series(clf.coef_[i8011], index=voc).rank(ascending=False)
audit = {"peso_8011_por_formula": pesos,
         "ranking_entre_terminos": {f: int(rank[f]) for f in pesos},
         "n_terminos": int(len(voc))}

# ¿A qué familia manda cada variante los contratos de oficios con familia propia según el Caso 10?
oficios = {"abogad": "8012", "contador": "8411", "ingenier": "8110", "medic": "8512",
           "enfermer": "8510", "docente": "8610", "conductor": "7811"}
tabla = []
for kw, fam_caso10 in oficios.items():
    sub = te[te.txt_norm.str.contains(kw)]
    if len(sub) < 20:
        continue
    fila = {"oficio": kw, "n_test": len(sub), "familia_caso10": fam_caso10,
            "SECOP_en_8011": (sub.fam == "8011").mean(),
            "SECOP_en_caso10": (sub.fam == fam_caso10).mean()}
    for var, col in [("A", "txt_norm"), ("B", "txt_lem")]:
        pred = modelos[f"{var}_normalizado" if var == "A" else f"{var}_limpio_lematizado"].predict(sub[col])
        fila[f"{var}_pred_8011"] = (pred == "8011").mean()
        fila[f"{var}_pred_caso10"] = (pred == fam_caso10).mean()
    tabla.append(fila)
oficio_df = pd.DataFrame(tabla)
oficio_df.to_csv(REPORTS / "auditoria_oficios.csv", index=False)
audit["oficios"] = oficio_df.round(3).to_dict(orient="records")
(REPORTS / "auditoria_atajo.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
print(json.dumps(audit["peso_8011_por_formula"], indent=1))
print(oficio_df.round(2).to_string(index=False))
