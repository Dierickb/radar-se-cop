"""Revisión humana de las discrepancias que marca la app (¿la alerta acierta?).

Paso 1:  python scripts/11_revisar_discrepancias.py muestra
  toma 100 contratos AL AZAR (semilla 42) de reports/discrepancias.csv (confianza >= 0,70) y crea
  reports/discrepancias_revision.xlsx con la columna 'revision' vacía. Al azar y no los 100 de mayor
  confianza, para que la cifra represente a todo lo que la app marca.
Paso 2:  llenar 'revision' con: modelo | secop | ninguna | no_clasificable
Paso 3:  python scripts/11_revisar_discrepancias.py evaluar   -> reports/discrepancias_revision_resumen.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from radar.config import REPORTS, SEED

XLSX = REPORTS / "discrepancias_revision.xlsx"
VALIDOS = {"modelo", "secop", "ninguna", "no_clasificable"}


def muestra(n: int = 100):
    d = pd.read_csv(REPORTS / "discrepancias.csv", dtype={"fam": str, "pred": str})
    m = d.sample(min(n, len(d)), random_state=SEED).sort_values("id_contrato")
    cols = ["id_contrato", "nombre_entidad", "objeto_del_contrato", "fam", "nombre_declarado", "pred",
            "nombre_predicho"]   # sin la confianza: no sesgar al revisor
    m[cols].assign(revision="", comentario="").to_excel(XLSX, index=False)
    print(f"{len(m)} contratos de {len(d)} -> {XLSX}")


def evaluar():
    r = pd.read_excel(XLSX, dtype=str)
    r["revision"] = r.revision.fillna("").str.strip().str.lower()
    malos = r[~r.revision.isin(VALIDOS)]
    if len(malos):
        sys.exit(f"{len(malos)} filas con 'revision' vacía o inválida (use {sorted(VALIDOS)}): "
                 f"{malos.id_contrato.head(5).tolist()}")
    p = r.revision.value_counts(normalize=True).round(3).to_dict()
    out = {"n": len(r), "proporciones": p, "precision_alerta": p.get("modelo", 0.0),
           "nota": "precision_alerta = fracción en que la familia sugerida por el modelo era la correcta"}
    (REPORTS / "discrepancias_revision_resumen.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"muestra": muestra, "evaluar": evaluar}[sys.argv[1]]()
