"""Re-etiquetado de contratos con Claude Code siguiendo la guía de anotación (reglas R1–R8, Caso 10).

Dos usos, en este orden:
  1) Validación: los mismos 300 contratos del gold set, con la misma información que ven los humanos
     (objeto, entidad, tipo de contrato; SIN el código declarado). Luego scripts/08_acuerdo_gold.py
     mide el acuerdo Claude–humanos.
        python llm/reetiquetar.py --entrada data/gold/gold_set_plantilla.xlsx --salida data/llm/gold_claude.jsonl
  2) Escala (solo si el acuerdo del paso 1 es alto): contratos del conjunto de ENTRENAMIENTO
     (nunca de test ni del gold set, para no evaluar con etiquetas del LLM).
        python llm/reetiquetar.py --entrenamiento 5000 --salida data/llm/train_claude.jsonl

Es reanudable: si se corta (p. ej. por el límite de uso de la suscripción), al volver a correrlo
salta los ids ya etiquetados en el archivo de salida.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "src"))
sys.path.insert(0, str(AQUI))
from radar.config import DATA_PROC, ROOT, SEED          # noqa: E402
from radar.data import familias_unspsc                   # noqa: E402
from claude_cli import llamar, ErrorClaude               # noqa: E402


def sistema() -> str:
    fams = familias_unspsc()
    catalogo = "\n".join(f"{k} – {v}" for k, v in sorted(fams.items()))
    return (AQUI / "prompts" / "reetiquetar.md").read_text(encoding="utf-8") + "\n" + catalogo + "\n"


def cargar_entrada(a) -> pd.DataFrame:
    if a.entrenamiento:
        df = pd.read_parquet(DATA_PROC / "contratos_prep.parquet",
                             columns=["id_contrato", "objeto_del_contrato", "nombre_entidad", "tipo_de_contrato"])
        excluir = set(pd.read_csv(DATA_PROC / "test_ids.csv").iloc[:, 0])
        gold = ROOT / "data" / "gold" / "gold_ids.csv"
        if gold.exists():
            excluir |= set(pd.read_csv(gold).iloc[:, 0])
        df = df[~df.id_contrato.isin(excluir)].sample(a.entrenamiento, random_state=SEED)
        return df.rename(columns={"id_contrato": "id"})
    p = Path(a.entrada)
    if p.suffix == ".xlsx":
        df = pd.read_excel(p, sheet_name=0)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_parquet(p)
    col_id = "item_id" if "item_id" in df else "id_contrato"
    df = df.rename(columns={col_id: "id"})
    # Nunca se envía el código declarado, aunque venga en el archivo
    return df[["id", "objeto_del_contrato", "nombre_entidad", "tipo_de_contrato"]]


def mensaje(lote: pd.DataFrame) -> str:
    filas = [{"id": str(r.id), "objeto": str(r.objeto_del_contrato), "entidad": str(r.nombre_entidad),
              "tipo_de_contrato": str(r.tipo_de_contrato)} for r in lote.itertuples()]
    return ("Clasifica estos contratos. Devuelve un elemento en 'items' por contrato, con el mismo 'id'.\n\n"
            + json.dumps(filas, ensure_ascii=False, indent=1))


def validar(item: dict, objeto: str, fams: dict) -> dict:
    obj = objeto.lower()
    item["familia_valida"] = item["familia"] in fams or (item["no_clasificable"] and item["familia"] == "")
    item["segunda_valida"] = item["segunda_familia"] in fams or item["segunda_familia"] == ""
    pk = item["palabras_clave"]
    item["palabras_clave_literales"] = (sum(p.lower() in obj for p in pk) / len(pk)) if pk else None
    return item


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--entrada", help="xlsx/csv/parquet con item_id|id_contrato, objeto_del_contrato, nombre_entidad, tipo_de_contrato")
    g.add_argument("--entrenamiento", type=int, help="N contratos al azar del conjunto de entrenamiento")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default="sonnet", help="alias o id del modelo (se registra en la salida)")
    ap.add_argument("--lote", type=int, default=20, help="contratos por llamada")
    ap.add_argument("--limite", type=int, help="procesar solo los primeros N (piloto)")
    a = ap.parse_args()

    df = cargar_entrada(a)
    if a.limite:
        df = df.head(a.limite)
    salida = Path(a.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    hechos = set()
    if salida.exists():
        hechos = {json.loads(l)["id"] for l in salida.read_text(encoding="utf-8").splitlines() if l.strip()}
    pend = df[~df.id.astype(str).isin(hechos)]
    print(f"{len(df)} contratos | ya etiquetados {len(hechos)} | pendientes {len(pend)}")

    sis, fams = sistema(), familias_unspsc()
    esquema = json.loads((AQUI / "schemas" / "reetiquetar.json").read_text())
    huella = hashlib.sha256(sis.encode()).hexdigest()[:12]   # versión exacta del prompt usado
    objetos = dict(zip(df.id.astype(str), df.objeto_del_contrato.astype(str)))

    with salida.open("a", encoding="utf-8") as f:
        for i in range(0, len(pend), a.lote):
            lote = pend.iloc[i:i + a.lote]
            t0 = time.time()
            try:
                datos, meta = llamar(mensaje(lote), sis, esquema, a.modelo)
            except ErrorClaude as e:
                print(f"Se detiene en el lote {i // a.lote}: {e}. Vuelva a correr para reanudar.")
                break
            esperados = set(lote.id.astype(str))
            n_ok = 0
            for it in datos["items"]:
                if it["id"] not in esperados:
                    continue                      # id inventado o repetido: se descarta
                esperados.discard(it["id"])
                it = validar(it, objetos[it["id"]], fams)
                it.update(modelo=a.modelo, modelos_usados=meta.get("modelos_usados"), prompt_sha=huella)
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
                n_ok += 1
            f.flush()
            faltan = f" | sin respuesta: {sorted(esperados)}" if esperados else ""
            print(f"lote {i // a.lote + 1}: {n_ok}/{len(lote)} en {time.time() - t0:.0f}s{faltan}")


if __name__ == "__main__":
    main()
