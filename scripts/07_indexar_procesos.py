"""Paso 8 (índice): prepara los procesos publicados para la búsqueda.
- texto = nombre + descripción del procedimiento
- familia = la del código declarado; si viene UNSPECIFIED, la infiere el clasificador (marcada)
Salidas: data/processed/procesos_idx.parquet y procesos_emb.npy"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from radar.config import DATA_RAW, DATA_PROC
from radar.model import cargar
from radar.search import _encoder
from radar.text import limpiar, lematizar

p = pd.read_parquet(DATA_RAW / "procesos.parquet").drop_duplicates("id_del_proceso")
nom, desc = p.nombre_del_procedimiento.fillna(""), p.descripci_n_del_procedimiento.fillna("")
p["texto"] = pd.Series(np.where(nom.str.lower() == desc.str.lower(), desc, nom + ". " + desc), index=p.index).str.strip(". ")
p = p[p.texto.str.split().str.len() >= 3]
cod = p.codigo_principal_de_categoria.fillna("").str.extract(r"^V1\.(\d{8})$")[0]
p["fam"] = cod.str[:4]
p["txt_lem"] = lematizar(p.texto.map(limpiar).tolist())
m = cargar("familia_v1")
sin = p.fam.isna()
p.loc[sin, "fam"] = m.predict(p.loc[sin, "txt_lem"])
p["familia_inferida"] = sin
p["precio_base"] = pd.to_numeric(p.precio_base, errors="coerce")
p.to_parquet(DATA_PROC / "procesos_idx.parquet")
np.save(DATA_PROC / "procesos_emb.npy", _encoder().encode(p.texto.tolist()).astype("float32"))
print(f"Procesos indexados: {len(p):,} | familia inferida por el modelo: {sin.mean():.1%}")
