"""Variante C: mismo modelo (TF-IDF + logística, texto B) entrenado con datos corregidos por Claude.

  C1 = etiquetas de Claude en los contratos de entrenamiento re-etiquetados (data/llm/train_claude.jsonl)
  C2 = C1 + descripciones estilo pyme generadas por Claude (data/llm/aumentado.jsonl)

Reglas para no hacer trampa:
  - Solo se usa el re-etiquetado si Claude tuvo kappa ≥ 0,61 con el gold set (reports/gold_resumen.json);
    --forzar se salta esa compuerta solo para pruebas.
  - Se usan las etiquetas de Claude con confianza alta o media y familias válidas; los no clasificables salen.
  - El test es el mismo de siempre y NUNCA lleva datos del LLM. Contra SECOP se espera que C baje
    (corrige justo lo que SECOP etiqueta "mal"); la comparación que decide es contra el gold set
    (scripts/08_acuerdo_gold.py --modelos familia_v1 familia_v2_llm).
Salidas: models/familia_v2_llm.joblib, reports/metricas_llm.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from radar.config import DATA_PROC, REPORTS, ROOT
from radar.evaluate import metricas
from radar.labels import particion_por_texto, clases_validas, balancear, excluir_gold
from radar.model import construir, guardar
from radar.text import lematizar, limpiar

ap = argparse.ArgumentParser()
ap.add_argument("--forzar", action="store_true", help="ignora la compuerta de kappa (solo pruebas)")
ap.add_argument("--kappa-min", type=float, default=0.61)
a = ap.parse_args()

LLM = ROOT / "data" / "llm"
resumen = REPORTS / "gold_resumen.json"
k = json.loads(resumen.read_text()).get("claude_vs_gold", {}).get("kappa") if resumen.exists() else None
if not a.forzar and (k is None or k < a.kappa_min):
    sys.exit(f"Compuerta: kappa Claude–gold = {k}. Se necesita ≥ {a.kappa_min} para usar el re-etiquetado.")

# Mismo split que scripts/03 (misma semilla y agrupación por texto)
df = pd.read_parquet(DATA_PROC / "contratos_prep.parquet")
clases = clases_validas(df, "fam")
df = df[df.fam.isin(clases)]
tr, te = particion_por_texto(df, "txt_norm")
tr = excluir_gold(tr, df, ROOT / "data" / "gold" / "gold_ids.csv")
te_bal = balancear(te, "fam", tope=300)

# C1: reemplazar la etiqueta de SECOP por la de Claude donde la hay
r = pd.DataFrame([json.loads(l) for l in (LLM / "train_claude.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()])
r = r[r.familia_valida & ~r.no_clasificable & r.confianza.isin(["alta", "media"])].drop_duplicates("id")
assert not r.id.isin(set(te.id_contrato)).any(), "hay contratos de test re-etiquetados: no se permite"
nuevo = tr.id_contrato.map(dict(zip(r.id, r.familia)))
cambios = (nuevo.notna() & (nuevo != tr.fam)).sum()
tr_c1 = tr.assign(fam=nuevo.fillna(tr.fam))
tr_c1 = tr_c1[tr_c1.fam.isin(clases)]          # el modelo conserva el mismo conjunto de familias
print(f"Re-etiquetados usados: {len(r):,} | etiquetas cambiadas: {cambios:,} "
      f"({cambios / max(len(r), 1):.1%} de los re-etiquetados)")

variantes = {"B_secop": balancear(tr, "fam"), "C1_reetiquetado": balancear(tr_c1, "fam")}

# C2: + textos estilo pyme
aug_p = LLM / "aumentado.jsonl"
if aug_p.exists():
    g = pd.DataFrame([json.loads(l) for l in aug_p.read_text(encoding="utf-8").splitlines() if l.strip()])
    g = g[g.fam.isin(clases)]
    g["txt_lem"] = lematizar(g.texto.map(limpiar).tolist())
    variantes["C2_reetiquetado_aumentado"] = pd.concat([variantes["C1_reetiquetado"], g[["txt_lem", "fam"]]])
    print(f"Textos estilo pyme agregados: {len(g):,} en {g.fam.nunique()} familias")

filas, ultimo = [], None
for nombre, d in variantes.items():
    m = construir().fit(d.txt_lem, d.fam)
    ultimo = m
    for conj, t in [("test_natural", te), ("test_balanceado", te_bal)]:
        f = metricas(m, t.txt_lem, t.fam, conj)
        f["variante"] = nombre
        filas.append(f)
        print({x: (round(v, 3) if isinstance(v, float) else v) for x, v in f.items()})
pd.DataFrame(filas).to_csv(REPORTS / "metricas_llm.csv", index=False)
guardar(ultimo, "familia_v2_llm")
print("Recuerde: contra SECOP estas métricas miden acuerdo con etiquetas sesgadas; decide el gold set.")
