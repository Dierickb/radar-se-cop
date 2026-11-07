"""Pasos 2-4: limpia, lematiza y etiqueta los contratos descargados.

Entrada : data/raw/contratos.parquet   (scripts/01_descargar.py)
Salida  : data/processed/contratos_prep.parquet
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from radar.config import DATA_RAW, DATA_PROC
from radar.labels import derivar_niveles
from radar.text import normalizar, limpiar, lematizar

t0 = time.time()
df = pd.read_parquet(DATA_RAW / "contratos.parquet")
n0 = len(df)
df = derivar_niveles(df)
print(f"Contratos: {n0:,} | con código válido: {len(df):,} ({len(df)/n0:.1%})")

df["txt_norm"] = df.objeto_del_contrato.fillna("").map(normalizar)   # variante A
df = df[df.txt_norm.str.split().str.len() >= 3]
antes = len(df)
df = df.drop_duplicates(["txt_norm", "fam"])
print(f"Duplicados exactos (texto+familia) quitados: {antes-len(df):,} | quedan {len(df):,}")

limpio = df.objeto_del_contrato.map(limpiar)
df["txt_lem"] = lematizar(limpio.tolist())                           # variante B
df = df[df.txt_lem.str.len() > 0]
df.to_parquet(DATA_PROC / "contratos_prep.parquet")
print(f"Guardado {len(df):,} contratos en {time.time()-t0:.0f} s")
print(df[["objeto_del_contrato", "txt_norm", "txt_lem", "fam"]].head(3).to_string())
