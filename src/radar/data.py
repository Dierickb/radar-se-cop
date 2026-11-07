"""Paso 1 – Adquisición: descarga de SECOP II y del catálogo UNSPSC."""
import time
import pandas as pd
import requests

from .config import SODA_BASE, DS_CONTRATOS, DS_PROCESOS, UNSPSC_XLS_URL, DATA_RAW

COLS_CONTRATOS = [
    "id_contrato", "nombre_entidad", "departamento", "sector", "orden",
    "descripcion_del_proceso", "objeto_del_contrato", "codigo_de_categoria_principal",
    "tipo_de_contrato", "modalidad_de_contratacion", "valor_del_contrato",
    "fecha_de_firma", "es_pyme",
]
COLS_PROCESOS = [
    "id_del_proceso", "entidad", "departamento_entidad", "ciudad_entidad",
    "nombre_del_procedimiento", "descripci_n_del_procedimiento", "precio_base",
    "modalidad_de_contratacion", "fecha_de_publicacion_del", "estado_del_procedimiento",
    "codigo_principal_de_categoria", "urlproceso",
]


def _paginar(dataset: str, select: list[str], where: str, n_max: int, paso: int = 50_000) -> pd.DataFrame:
    """Descarga paginada con la API SODA (sin registro)."""
    filas = []
    for offset in range(0, n_max, paso):
        params = {"$select": ",".join(select), "$where": where, "$order": ":id",
                  "$limit": min(paso, n_max - offset), "$offset": offset}
        for intento in range(3):
            try:
                r = requests.get(f"{SODA_BASE}/{dataset}.json", params=params, timeout=300)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"  reintento {intento + 1}: {e}")
                time.sleep(5)
        lote = r.json()
        filas.extend(lote)
        print(f"  {dataset}: {len(filas):,} filas")
        if len(lote) < params["$limit"]:
            break
    return pd.DataFrame(filas)


def descargar_contratos(n_max: int = 300_000, desde: str = "2025-01-01") -> pd.DataFrame:
    df = _paginar(DS_CONTRATOS, COLS_CONTRATOS, f"fecha_de_firma >= '{desde}T00:00:00'", n_max)
    df.to_parquet(DATA_RAW / "contratos.parquet")
    return df


def descargar_procesos(desde: str = "2026-08-01", n_max: int = 200_000) -> pd.DataFrame:
    where = (f"fecha_de_publicacion_del >= '{desde}T00:00:00' AND "
             "estado_del_procedimiento in ('Publicado','Abierto')")
    df = _paginar(DS_PROCESOS, COLS_PROCESOS, where, n_max)
    df.to_parquet(DATA_RAW / "procesos.parquet")
    return df


def cargar_unspsc() -> pd.DataFrame:
    """Catálogo oficial v14 en español: una fila por producto (49.022)."""
    destino = DATA_RAW / "unspsc_v14.parquet"
    if destino.exists():
        return pd.read_parquet(destino)
    xls = DATA_RAW / "clasificador_v14.xls"
    if not xls.exists():
        xls.write_bytes(requests.get(UNSPSC_XLS_URL, timeout=300).content)
    crudo = pd.read_excel(xls, header=None)
    fila = crudo.index[crudo.astype(str).apply(lambda r: r.str.contains("Código Segmento").any(), axis=1)][0]
    df = pd.read_excel(xls, header=fila)
    df.columns = ["seg", "nombre_seg", "fam", "nombre_fam", "cls", "nombre_cls", "prod", "nombre_prod"]
    for c in ("seg", "fam", "cls", "prod"):
        df[c] = df[c].astype("Int64").astype(str)
    df.to_parquet(destino)
    return df


def familias_unspsc() -> dict[str, str]:
    u = cargar_unspsc()
    return dict(zip(u["fam"], u["nombre_fam"].str.strip()))
