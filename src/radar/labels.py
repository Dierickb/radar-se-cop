"""Paso 4 – Etiquetas y balanceo a partir del código UNSPSC declarado (V1.XXXXXXXX)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import SEED, MIN_EJEMPLOS_CLASE, MAX_EJEMPLOS_CLASE

_COD_RX = r"^V1\.(\d{8})$"


def derivar_niveles(df: pd.DataFrame, col: str = "codigo_de_categoria_principal") -> pd.DataFrame:
    """Deja solo códigos válidos y agrega segmento (2), familia (4) y clase (6)."""
    cod = df[col].fillna("").str.extract(_COD_RX)[0]
    out = df.loc[cod.notna()].copy()
    out["codigo"] = cod.dropna()
    out["seg"], out["fam"], out["cls"] = out.codigo.str[:2], out.codigo.str[:4], out.codigo.str[:6]
    return out


def particion_por_texto(df: pd.DataFrame, col_texto: str, test_size: float = 0.2):
    """80/20 agrupado por texto: el mismo objeto nunca queda en train y test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr, te = next(gss.split(df, groups=df[col_texto]))
    return df.iloc[tr].copy(), df.iloc[te].copy()


def clases_validas(df: pd.DataFrame, nivel: str) -> list[str]:
    vc = df[nivel].value_counts()
    return sorted(vc[vc >= MIN_EJEMPLOS_CLASE].index)


def balancear(df: pd.DataFrame, nivel: str, tope: int = MAX_EJEMPLOS_CLASE) -> pd.DataFrame:
    """Tope de ejemplos por clase para que la familia 8011 no domine el entrenamiento."""
    d = df.sample(frac=1, random_state=SEED)
    return d[d.groupby(nivel).cumcount() < tope]


def excluir_gold(tr: pd.DataFrame, df: pd.DataFrame, ruta_ids) -> pd.DataFrame:
    """Saca del entrenamiento los contratos del gold set y cualquier contrato con el mismo texto,
    para que la evaluación contra el gold set no esté contaminada."""
    from pathlib import Path
    if not Path(ruta_ids).exists():
        return tr
    ids = set(pd.read_csv(ruta_ids).id_contrato)
    textos = set(df.loc[df.id_contrato.isin(ids), "txt_norm"])
    return tr[~tr.id_contrato.isin(ids) & ~tr.txt_norm.isin(textos)]
