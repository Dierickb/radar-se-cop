"""Paso 1: descarga contratos (entrenamiento), procesos publicados (búsqueda) y el catálogo UNSPSC.

Uso:  python scripts/01_descargar.py [--contratos 300000] [--desde 2025-01-01] [--procesos-desde 2026-08-01]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from radar.data import descargar_contratos, descargar_procesos, cargar_unspsc

ap = argparse.ArgumentParser()
ap.add_argument("--contratos", type=int, default=300_000)
ap.add_argument("--desde", default="2025-01-01")
ap.add_argument("--procesos-desde", default="2026-08-01")
a = ap.parse_args()

print("Catálogo UNSPSC v14:", len(cargar_unspsc()), "productos")
print("Contratos:", len(descargar_contratos(a.contratos, a.desde)))
print("Procesos publicados/abiertos:", len(descargar_procesos(a.procesos_desde)))
