"""Paso 7: marca contratos de test donde el modelo contradice el código declarado con confianza alta.
Salida: reports/discrepancias.csv (insumo para la revisión manual de 100 casos)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json
import pandas as pd
from radar.config import DATA_PROC, REPORTS
from radar.data import familias_unspsc
from radar.model import cargar

UMBRAL = 0.70   # ver reports/explicabilidad.json -> calibracion (0,9 cubre solo ~1 %)
nombres = familias_unspsc()
m = cargar("familia_v1")
df = pd.read_parquet(DATA_PROC / "contratos_prep.parquet")
ids = set(pd.read_csv(DATA_PROC / "test_ids.csv").iloc[:, 0])
te = df[df.id_contrato.isin(ids) & df.fam.isin(m.classes_)].copy()
P = m.predict_proba(te.txt_lem)
te["pred"], te["conf"] = m.classes_[P.argmax(1)], P.max(1)
d = te[(te.pred != te.fam) & (te.conf >= UMBRAL)].copy()
d["nombre_declarado"] = d.fam.map(nombres); d["nombre_predicho"] = d.pred.map(nombres)
cols = ["id_contrato", "nombre_entidad", "objeto_del_contrato", "fam", "nombre_declarado", "pred", "nombre_predicho", "conf"]
d.sort_values("conf", ascending=False)[cols].to_csv(REPORTS / "discrepancias.csv", index=False)
pares = (d.groupby(["fam", "pred"]).size().sort_values(ascending=False).head(10)
         .reset_index(name="n"))
pares["de"] = pares.fam.map(nombres); pares["a"] = pares.pred.map(nombres)
res = {"umbral": UMBRAL, "test": len(te), "marcadas": len(d), "tasa": len(d) / len(te),
       "desde_8011": float((d.fam == "8011").mean()), "pares": pares.to_dict(orient="records")}
(REPORTS / "discrepancias_resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in res.items() if k != "pares"}, indent=1))
print(pares[["fam", "de", "pred", "a", "n"]].to_string(index=False))
