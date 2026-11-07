"""Aumento de datos con Claude Code: descripciones de negocio "como las escribe una pyme" por familia UNSPSC.

Motivo: el modelo se entrena con objetos de contratos redactados por entidades públicas, pero en la app
la consulta la escribe una pyme. Estos textos se usan SOLO para entrenar (variante C); nunca para evaluar.

    python llm/aumentar.py --por-familia 40 --salida data/llm/aumentado.jsonl            # todas las familias del modelo
    python llm/aumentar.py --familias 7215 8111 --por-familia 40 --salida data/llm/aumentado.jsonl

Reanudable: cuenta lo que ya hay por familia en el archivo de salida y solo pide lo que falta.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "src"))
sys.path.insert(0, str(AQUI))
from radar.data import cargar_unspsc, familias_unspsc   # noqa: E402
from claude_cli import llamar, ErrorClaude              # noqa: E402


def mensaje(fam: str, nombre: str, clases: list[str], n: int, ya: list[str]) -> str:
    m = (f"Familia UNSPSC {fam} – {nombre}.\n"
         f"Clases que contiene (para que cubras su variedad):\n- " + "\n- ".join(clases) +
         f"\n\nEscribe {n} descripciones nuevas de pymes que venden bienes o servicios de esta familia.")
    if ya:
        m += "\nNo repitas ni parafrasees estas, que ya existen:\n- " + "\n- ".join(ya[-40:])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--familias", nargs="*", help="códigos de 4 dígitos; por defecto, las del modelo familia_v1")
    ap.add_argument("--por-familia", type=int, default=40)
    ap.add_argument("--lote", type=int, default=20, help="textos por llamada")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default="sonnet")
    a = ap.parse_args()

    nombres = familias_unspsc()
    cat = cargar_unspsc()
    if a.familias:
        fams = a.familias
    else:
        from radar.model import cargar
        fams = list(cargar("familia_v1").classes_)
    sis = (AQUI / "prompts" / "aumentar.md").read_text(encoding="utf-8")
    esquema = json.loads((AQUI / "schemas" / "aumentar.json").read_text())
    huella = hashlib.sha256(sis.encode()).hexdigest()[:12]

    salida = Path(a.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    previos = [json.loads(l) for l in salida.read_text(encoding="utf-8").splitlines() if l.strip()] if salida.exists() else []
    cuenta = Counter(p["fam"] for p in previos)
    vistos = {p["texto"].strip().lower() for p in previos}

    with salida.open("a", encoding="utf-8") as f:
        for fam in fams:
            clases = sorted(cat.loc[cat.fam == fam, "nombre_cls"].dropna().unique())
            intentos = 0
            while cuenta[fam] < a.por_familia and intentos < 4:
                intentos += 1
                n = min(a.lote, a.por_familia - cuenta[fam])
                ya = [p["texto"] for p in previos if p["fam"] == fam]
                try:
                    datos, _ = llamar(mensaje(fam, nombres.get(fam, ""), clases, n, ya), sis, esquema, a.modelo)
                except ErrorClaude as e:
                    print(f"Se detiene en {fam}: {e}. Vuelva a correr para reanudar.")
                    return
                nuevos = 0
                for t in datos["textos"]:
                    clave = t["texto"].strip().lower()
                    if clave in vistos:
                        continue
                    vistos.add(clave)
                    reg = {"fam": fam, "texto": t["texto"].strip(), "registro": t["registro"],
                           "modelo": a.modelo, "prompt_sha": huella}
                    f.write(json.dumps(reg, ensure_ascii=False) + "\n")
                    previos.append(reg)
                    cuenta[fam] += 1
                    nuevos += 1
                f.flush()
                print(f"{fam}: +{nuevos} (total {cuenta[fam]}/{a.por_familia})")


if __name__ == "__main__":
    main()
