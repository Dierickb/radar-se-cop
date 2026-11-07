"""Sesgo por entidad: entrena con unas entidades y evalúa con OTRAS.
Si el desempeño cae mucho frente al split por texto, el modelo memoriza costumbres de cada entidad."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from radar.config import DATA_PROC, REPORTS, SEED
from radar.labels import clases_validas, balancear
from radar.model import construir
from radar.evaluate import metricas

df = pd.read_parquet(DATA_PROC / "contratos_prep.parquet")
df = df[df.fam.isin(clases_validas(df, "fam"))]
tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df, groups=df.nombre_entidad))
tr, te = df.iloc[tr_i], df.iloc[te_i]
te = te[te.fam.isin(set(tr.fam))]
m = construir().fit(balancear(tr, "fam").txt_lem, balancear(tr, "fam").fam)
res = [metricas(m, te.txt_lem, te.fam, "entidades_no_vistas_natural"),
       metricas(m, balancear(te, "fam", 300).txt_lem, balancear(te, "fam", 300).fam, "entidades_no_vistas_balanceado")]
print(f"entidades train {tr.nombre_entidad.nunique()} test {te.nombre_entidad.nunique()}")
for r in res: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
(REPORTS / "metricas_particion_entidad.json").write_text(json.dumps(res, indent=2))
