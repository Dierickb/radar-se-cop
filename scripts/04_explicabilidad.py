"""Paso 9: explicabilidad global, local, fidelidad (ERASER) y auditoría del atajo hacia 8011.

Salidas: reports/explicabilidad.json y reports/explicabilidad_global.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json
import numpy as np
import pandas as pd

from radar.config import DATA_PROC, REPORTS, SEED
from radar.data import familias_unspsc
from radar.explain import global_por_familia, local, fidelidad, contrafactual
from radar.model import cargar

nombres = familias_unspsc()
mB, mA = cargar("familia_v1"), cargar("familia_v0_con_formulas")
df = pd.read_parquet(DATA_PROC / "contratos_prep.parquet")
test_ids = set(pd.read_csv(DATA_PROC / "test_ids.csv").iloc[:, 0])
te = df[df.id_contrato.isin(test_ids) & df.fam.isin(mB.classes_)]

out = {}
# 1. Global: qué términos definen cada familia (8011 en A vs B muestra el atajo)
filas = []
for fam in ["8011", "8012", "8510", "8111", "8610", "9314"]:
    for var, m in [("A_con_formulas", mA), ("B_limpio", mB)]:
        g = global_por_familia(m, fam, k=10)
        filas.append({"familia": fam, "nombre": nombres.get(fam, ""), "variante": var,
                      "top_a_favor": ", ".join(g.a_favor)})
glob = pd.DataFrame(filas)
glob.to_csv(REPORTS / "explicabilidad_global.csv", index=False)
out["global"] = glob.to_dict(orient="records")

# 2. Local: ejemplos explicados (contribución exacta por término)
ejemplos = te.sample(5, random_state=SEED)
out["local"] = []
for _, r in ejemplos.iterrows():
    exp = local(mB, r.txt_lem, k=6)
    pred = mB.predict([r.txt_lem])[0]
    out["local"].append({"objeto": r.objeto_del_contrato[:200], "declarado": r.fam, "predicho": pred,
                         "nombre_predicho": nombres.get(pred, ""),
                         "terminos": exp.round(3).to_dict(orient="records")})

# 3. Contrafactual: quitar el oficio cambia la familia
cf_txt = "abogado acompanamiento juridico personeria municipal"
out["contrafactual"] = {"texto": cf_txt, **contrafactual(mB, cf_txt, quitar="abogado juridico")}

# 4. Fidelidad ERASER sobre 500 contratos de test
muestra = te.sample(500, random_state=SEED).txt_lem.tolist()
out["fidelidad_B"] = fidelidad(mB, muestra, k=3)
# referencia: quitar 3 términos al azar
rng = np.random.RandomState(SEED)
caidas = []
for t in muestra:
    w = t.split()
    P0 = mB.predict_proba([t])[0]; j = P0.argmax()
    quitar = set(rng.choice(w, size=min(3, len(w)), replace=False))
    caidas.append(P0[j] - mB.predict_proba([" ".join(x for x in w if x not in quitar)])[0][j])
out["fidelidad_B"]["comprehensiveness_azar"] = float(np.mean(caidas))

# 5. Calibración del umbral de discrepancia: cuando la confianza ≥ 0,9, ¿acierta con SECOP?
P = mB.predict_proba(te.txt_lem)
conf, pred = P.max(1), mB.classes_[P.argmax(1)]
for u in (0.5, 0.7, 0.9):
    m = conf >= u
    out.setdefault("calibracion", {})[str(u)] = {"cobertura": float(m.mean()),
                                                  "acuerdo_con_SECOP": float((pred[m] == te.fam.values[m]).mean())}

(REPORTS / "explicabilidad.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(glob.to_string(index=False))
print(json.dumps({k: out[k] for k in ("contrafactual", "fidelidad_B", "calibracion")}, indent=1, ensure_ascii=False))
print(json.dumps(out["local"][:2], indent=1, ensure_ascii=False))
