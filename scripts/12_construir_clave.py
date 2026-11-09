"""Construye la clave del gold set: el código que declaró SECOP para cada uno de los 300 contratos.

Salida: data/gold/gold_set_CLAVE_no_abrir.xlsx con item_id, id_contrato, codigo_declarado, familia_declarada, estrato.
  - Toma el código de data/processed/contratos_prep.parquet y, para los contratos que no están ahí,
    lo consulta en SECOP II (datos.gov.co, API SODA, dataset jbjy-vk9h) por id_contrato.
  - estrato = "8011" si SECOP declaró la familia 8011 (el cajón de sastre del hallazgo central) y "otras" si no.
  - Comprueba que el objeto del contrato coincida con el de la plantilla del gold set.
Solo imprime conteos: no muestra ningún código, para no sesgar a quien anota.
No sobrescribe una clave existente salvo con --sobrescribir.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import requests

from radar.config import DATA_PROC, DS_CONTRATOS, ROOT, SODA_BASE
from radar.labels import derivar_niveles

G = ROOT / "data" / "gold"
CLAVE = G / "gold_set_CLAVE_no_abrir.xlsx"


def consultar_soda(ids: list[str]) -> pd.DataFrame:
    filas = []
    for i in range(0, len(ids), 50):
        lista = ",".join(f"'{x}'" for x in ids[i:i + 50])
        params = {"$select": "id_contrato,objeto_del_contrato,codigo_de_categoria_principal",
                  "$where": f"id_contrato in ({lista})", "$limit": 1000}
        for intento in range(3):
            try:
                r = requests.get(f"{SODA_BASE}/{DS_CONTRATOS}.json", params=params, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"  reintento {intento + 1}: {e}")
                time.sleep(5)
        else:
            sys.exit("No se pudo consultar SECOP II; intente más tarde.")
        filas.extend(r.json())
    return pd.DataFrame(filas, columns=["id_contrato", "objeto_del_contrato", "codigo_de_categoria_principal"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sobrescribir", action="store_true")
    args = ap.parse_args()
    if CLAVE.exists() and not args.sobrescribir:
        sys.exit(f"{CLAVE.name} ya existe; use --sobrescribir para rehacerla.")

    ids = pd.read_csv(G / "gold_ids.csv", dtype=str)
    plantilla = pd.read_excel(G / "gold_set_plantilla.xlsx", sheet_name="anotacion", dtype=str).set_index("item_id")
    cols = ["id_contrato", "objeto_del_contrato", "codigo_de_categoria_principal"]
    local = pd.read_parquet(DATA_PROC / "contratos_prep.parquet", columns=cols)
    local = local[local.id_contrato.isin(ids.id_contrato)].drop_duplicates("id_contrato")
    faltan = sorted(set(ids.id_contrato) - set(local.id_contrato))
    print(f"En los datos congelados: {len(local)} | a consultar en SECOP II: {len(faltan)}")
    remoto = consultar_soda(faltan) if faltan else pd.DataFrame(columns=cols)
    remoto = remoto.drop_duplicates("id_contrato")
    print(f"Encontrados en SECOP II: {len(remoto)} de {len(faltan)}")

    todo = pd.concat([local.assign(fuente="congelado"), remoto.assign(fuente="secop_api")])
    niveles = derivar_niveles(todo)[["id_contrato", "codigo", "fam", "fuente", "objeto_del_contrato"]]
    clave = ids.merge(niveles, on="id_contrato", how="left")
    sin_codigo = int(clave.fam.isna().sum())
    if sin_codigo:
        print(f"Aviso: {sin_codigo} contratos sin código válido (V1.XXXXXXXX); quedan fuera de las métricas contra SECOP.")

    # El objeto debe ser el mismo que vio quien anotó (control de que el cruce por id es correcto)
    norm = lambda s: " ".join(str(s).lower().split())
    obj_plantilla = clave.item_id.map(plantilla.objeto_del_contrato).map(norm)
    coincide = obj_plantilla == clave.objeto_del_contrato.map(norm)
    print(f"Objeto igual al de la plantilla: {int(coincide.sum())} de {int(clave.fam.notna().sum())} con código")

    clave["estrato"] = clave.fam.map(lambda f: "" if pd.isna(f) else ("8011" if f == "8011" else "otras"))
    out = clave.rename(columns={"codigo": "codigo_declarado", "fam": "familia_declarada"})
    out[["item_id", "id_contrato", "codigo_declarado", "familia_declarada", "estrato", "fuente"]].to_excel(CLAVE, index=False)
    print(f"Clave -> data/gold/{CLAVE.name} | por fuente: {out.fuente.value_counts().to_dict()} | "
          f"estratos: {out.estrato.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
