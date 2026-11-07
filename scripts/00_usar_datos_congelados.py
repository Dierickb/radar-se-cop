"""Paso 0 (opcional, recomendado para reproducir las cifras del documento).

SECOP cambia todos los días: volver a descargar con scripts/01 da otros contratos y otras cifras.
Este script pone en su lugar la foto exacta que se usó en el proyecto (descargada el 14-oct-2026):
  data/congelado/contratos_prep.parquet -> data/processed/   (salida del paso 02)
  data/congelado/procesos.parquet       -> data/raw/         (procesos publicados del 1-ago al 13-oct-2026)
  data/congelado/unspsc_v14.parquet     -> data/raw/         (catálogo v14 ya procesado)
Después se corre desde scripts/03 (los pasos 01 y 02 se saltan).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from radar.config import DATA_PROC, DATA_RAW, ROOT

SRC = ROOT / "data" / "congelado"
for nombre, destino in [("contratos_prep.parquet", DATA_PROC), ("procesos.parquet", DATA_RAW),
                        ("unspsc_v14.parquet", DATA_RAW)]:
    shutil.copy2(SRC / nombre, destino / nombre)
    print(f"{nombre} -> {destino.relative_to(ROOT)}")
print("Listo. Siga con: python scripts/03_entrenar_evaluar.py")
