"""Paso 3 (validación): todo lo que se mide contra el gold set humano.

Entradas en data/gold/:
  gold_set_anotador_David.xlsx, gold_set_anotador_Dierick.xlsx   (hoja 'anotacion')
  gold_set_CLAVE_no_abrir.xlsx                                     (código declarado en SECOP)
  gold_resuelto.xlsx (opcional: item_id, familia_gold, no_clasificable, disputado) tras discutir desacuerdos;
    se arma desde la plantilla que crea  python scripts/08_acuerdo_gold.py --desacuerdos
y, si existe, data/llm/gold_claude.jsonl (salida de llm/reetiquetar.py sobre los mismos 300).

Con UN solo anotador (solo existe gold_set_anotador_<nombre>.xlsx):
  python scripts/08_acuerdo_gold.py --repeticion    al terminar la primera pasada: crea
      gold_set_repeticion_<nombre>.xlsx con 60 ítems de G031–G300 al azar, barajados y sin las respuestas;
      se re-anota a ciegas (idealmente una semana después) y el kappa intra-anotador reemplaza al kappa
      entre anotadores. --desacuerdos lista las inconsistencias para resolverlas en gold_resuelto.xlsx.

Mide:
  1. Acuerdo entre anotadores (kappa de Cohen) en G031–G300, a nivel de familia y de segmento
     (con un anotador: kappa intra-anotador entre la primera pasada y la repetición).
  2. Tasa de error de la etiqueta de SECOP por estrato (¿qué tan sucia está 8011?).
  3. Acuerdo de Claude con el gold set (decide si el re-etiquetado a escala es confiable).
  4. Modelo vs gold: accuracy, macro-F1 y top-3, comparado con su acuerdo con SECOP en los mismos ítems.
  5. Plausibilidad de la explicación: solapamiento entre las palabras que resalta el modelo y las
     palabras_clave que marcaron los humanos (ERASER, token-F1), contra una referencia al azar.
Salidas: reports/gold_resumen.json y reports/gold_detalle.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from radar.config import ROOT, REPORTS, SEED
from radar.explain import local
from radar.model import cargar
from radar.text import preparar

G = ROOT / "data" / "gold"
SI = {"si", "sí", "x", "true", "1", "yes"}


def fam4(v) -> str:
    s = str(v).strip()
    return s[:4] if s[:4].isdigit() else ""


def leer_anotador(nombre: str, archivo: Path | None = None) -> pd.DataFrame:
    d = pd.read_excel(archivo or G / f"gold_set_anotador_{nombre}.xlsx", sheet_name="anotacion")
    d["fam"] = d.familia_anotada.map(fam4)
    d["fam2"] = d.segunda_familia.map(fam4)
    d["nc"] = d.no_clasificable.astype(str).str.strip().str.lower().isin(SI)
    d.loc[d.nc, "fam"] = "NC"
    return d.set_index("item_id")


def kappa(a: pd.Series, b: pd.Series) -> dict:
    m = (a != "") & (b != "")
    return {"kappa": float(cohen_kappa_score(a[m], b[m])), "acuerdo_bruto": float((a[m] == b[m]).mean()),
            "n": int(m.sum())}


ANOTADORES = ["David", "Dierick"]
N_REPETICION = 60   # ítems de G031-G300 que el anotador único vuelve a anotar a ciegas


def archivo_repeticion(nombre: str) -> Path:
    return G / f"gold_set_repeticion_{nombre}.xlsx"


def crear_repeticion(nombre: str, A: pd.DataFrame):
    """Planilla para re-anotar N_REPETICION ítems de la fase independiente, en otro orden y sin las respuestas."""
    destino = archivo_repeticion(nombre)
    if destino.exists():
        sys.exit(f"{destino.name} ya existe; no se sobrescribe (bórrela a mano si quiere rehacerla).")
    ind = A.index[~A.fase.astype(str).str.startswith("1")]
    faltan = int((A.loc[ind, "fam"] == "").sum())
    if faltan:
        sys.exit(f"Faltan {faltan} filas por anotar en G031-G300. Termine la primera pasada antes de la repetición.")
    muestra = pd.Series(ind).sample(min(N_REPETICION, len(ind)), random_state=SEED).tolist()   # ya viene barajada
    hoja = A.loc[muestra, ["objeto_del_contrato", "nombre_entidad", "tipo_de_contrato"]].copy()
    for c in ["familia_anotada", "segunda_familia", "confianza", "palabras_clave", "no_clasificable", "comentario"]:
        hoja[c] = ""
    familias = pd.read_excel(G / "gold_set_plantilla.xlsx", sheet_name="familias")
    with pd.ExcelWriter(destino) as w:
        hoja.reset_index().to_excel(w, sheet_name="anotacion", index=False)
        familias.to_excel(w, sheet_name="familias", index=False)
    print(f"{len(hoja)} ítems -> data/gold/{destino.name}. Anótelos sin abrir su planilla original "
          f"(idealmente una semana después de la primera pasada).")


def kappa_intra(nombre: str, A: pd.DataFrame) -> dict | None:
    """Acuerdo del anotador consigo mismo entre la primera pasada y la repetición."""
    if not archivo_repeticion(nombre).exists():
        return None
    R = leer_anotador(nombre, archivo_repeticion(nombre))
    R = R[R.fam != ""]
    if R.empty:
        return None
    a, b = A.loc[R.index, "fam"], R.fam
    return {"familia": kappa(a, b), "segmento": kappa(a.str[:2], b.str[:2]),
            "inconsistentes": a.index[a != b].tolist(), "_primera": a, "_repeticion": b}


def tokens(texto: str) -> set:
    return set(preparar(texto).split())


def main():
    global MODELOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["familia_v1"])
    ap.add_argument("--desacuerdos", action="store_true",
                    help="solo kappa + lista de desacuerdos y plantilla de gold_resuelto.xlsx (no abre la clave ni Claude)")
    ap.add_argument("--repeticion", action="store_true",
                    help="(un solo anotador) crea la planilla para re-anotar a ciegas 60 ítems de G031-G300")
    args = ap.parse_args()
    MODELOS = args.modelos
    out = {}
    presentes = [n for n in ANOTADORES if (G / f"gold_set_anotador_{n}.xlsx").exists()]
    if not presentes:
        sys.exit("No hay planillas gold_set_anotador_<nombre>.xlsx en data/gold/.")
    unico = presentes[0] if len(presentes) == 1 else None

    if unico:   # diseño con un solo anotador: consistencia intra-anotador en lugar de kappa entre anotadores
        A = leer_anotador(unico)
        if (A.fam == "").any():
            print(f"Aviso: {int((A.fam == '').sum())} filas sin familia ni 'no clasificable' ({unico})")
        if args.repeticion:
            crear_repeticion(unico, A)
            return
        intra = kappa_intra(unico, A)
        if args.desacuerdos:
            if intra is None:
                print("Un solo anotador: no hay kappa entre anotadores. Al terminar la primera pasada corra "
                      "--repeticion, re-anote esa planilla y vuelva a correr --desacuerdos.")
                return
            print("kappa intra-anotador familia:", intra["familia"])
            print("kappa intra-anotador segmento:", intra["segmento"])
            inc = intra["inconsistentes"]
            pd.DataFrame({"objeto_del_contrato": A.loc[inc, "objeto_del_contrato"],
                          "primera_pasada": A.loc[inc, "familia_anotada"],
                          "repeticion": intra["_repeticion"][inc]}).to_excel(G / "gold_desacuerdos.xlsx")
            dis = A.index.isin(inc)
            plantilla = pd.DataFrame({"familia_gold": A.fam.where(~dis & (A.fam != "NC"), ""),
                                      "no_clasificable": np.where(~dis & (A.fam == "NC"), "sí", ""),
                                      "disputado": "", "resuelto_en_discusion": np.where(dis, "pendiente", "")},
                                     index=A.index)
            plantilla.to_excel(G / "gold_resuelto_plantilla.xlsx")
            print(f"{len(inc)} inconsistencias -> data/gold/gold_desacuerdos.xlsx; "
                  f"plantilla -> data/gold/gold_resuelto_plantilla.xlsx (guárdela como gold_resuelto.xlsx)")
            return
    else:
        if args.repeticion:
            sys.exit("--repeticion es para el diseño de un solo anotador; con dos se usa --desacuerdos.")
        A, B = leer_anotador("David"), leer_anotador("Dierick")
        vacias = {n: int((d.fam == "").sum()) for n, d in (("David", A), ("Dierick", B))}
        if any(vacias.values()):
            print(f"Aviso: filas sin familia ni 'no clasificable' por anotador: {vacias}")

    if args.desacuerdos:
        ind = A.index[~A.fase.astype(str).str.startswith("1")]
        print("kappa familia (G031-G300):", kappa(A.loc[ind, "fam"], B.loc[ind, "fam"]))
        print("kappa segmento:", kappa(A.loc[ind, "fam"].str[:2], B.loc[ind, "fam"].str[:2]))
        dis = A.fam != B.fam
        tabla = pd.DataFrame({"objeto_del_contrato": A.objeto_del_contrato, "nombre_entidad": A.nombre_entidad,
                              "David": A.familia_anotada, "David_2": A.segunda_familia, "David_comentario": A.comentario,
                              "Dierick": B.familia_anotada, "Dierick_2": B.segunda_familia,
                              "Dierick_comentario": B.comentario})
        tabla[dis].to_excel(G / "gold_desacuerdos.xlsx")
        plantilla = pd.DataFrame({"familia_gold": A.fam.where(~dis & (A.fam != "NC"), ""),
                                  "no_clasificable": np.where(~dis & (A.fam == "NC"), "sí", ""),
                                  "disputado": "", "resuelto_en_discusion": np.where(dis, "pendiente", "")})
        plantilla.index.name = "item_id"
        plantilla.to_excel(G / "gold_resuelto_plantilla.xlsx")
        print(f"{int(dis.sum())} desacuerdos -> data/gold/gold_desacuerdos.xlsx; "
              f"plantilla -> data/gold/gold_resuelto_plantilla.xlsx (guárdela como gold_resuelto.xlsx)")
        return
    clave = pd.read_excel(G / "gold_set_CLAVE_no_abrir.xlsx").set_index("item_id")
    clave["fam_secop"] = clave.familia_declarada.astype(str).str.zfill(4)

    # 1. Acuerdo: entre anotadores (fase independiente) o, con uno solo, del anotador consigo mismo
    if unico:
        out["diseno"] = f"un anotador ({unico}); consistencia por re-anotación ciega de {N_REPETICION} ítems"
        out["kappa_intra_anotador"] = None if intra is None else {
            "familia": intra["familia"], "segmento": intra["segmento"], "n_inconsistentes": len(intra["inconsistentes"])}
        out["no_clasificable_por_anotador"] = {unico: float(A.nc.mean())}
    else:
        ind = A.index[~A.fase.astype(str).str.startswith("1")]
        out["kappa_familia"] = kappa(A.loc[ind, "fam"], B.loc[ind, "fam"])
        out["kappa_segmento"] = kappa(A.loc[ind, "fam"].str[:2], B.loc[ind, "fam"].str[:2])
        out["no_clasificable_por_anotador"] = {"David": float(A.nc.mean()), "Dierick": float(B.nc.mean())}

    # Etiqueta gold: la resuelta en la discusión si existe; si no, solo donde ambos coinciden
    res = G / "gold_resuelto.xlsx" if (G / "gold_resuelto.xlsx").exists() else G / "gold_resuelto.csv"
    if res.exists():
        if res.suffix == ".xlsx":
            r = pd.read_excel(res, dtype=str).fillna("").set_index("item_id")
        else:  # acepta , o ; (Excel en español)
            r = pd.read_csv(res, dtype=str, sep=None, engine="python", encoding="utf-8-sig").fillna("").set_index("item_id")
        gold = r.familia_gold.map(fam4)
        gold[r.get("no_clasificable", "").astype(str).str.lower().isin(SI)] = "NC"
        gold[r.get("disputado", "").astype(str).str.lower().isin(SI)] = ""
        if "resuelto_en_discusion" in r:
            disp = r.get("disputado", pd.Series("", index=r.index)).astype(str).str.strip().str.lower().isin(SI)
            pend = r.index[(r.resuelto_en_discusion.astype(str).str.strip().str.lower() == "pendiente") & ~disp].tolist()
            if pend:
                sys.exit(f"{len(pend)} filas siguen en 'pendiente' en {res.name}: {pend[:10]}. Resuélvalas o márquelas disputado.")
        out["fuente_gold"] = res.name
    elif unico:
        gold = A.fam
        out["fuente_gold"] = f"primera pasada de {unico}"
    else:
        gold = A.fam.where(A.fam == B.fam, "")
        out["fuente_gold"] = "coincidencias entre anotadores (desacuerdos excluidos)"
    if unico:
        det = pd.DataFrame({"gold": gold, unico.lower(): A.fam, "objeto": A.objeto_del_contrato,
                            "palabras_clave": A.palabras_clave.fillna("").astype(str)})
    else:
        det = pd.DataFrame({"gold": gold, "david": A.fam, "dierick": B.fam,
                            "objeto": A.objeto_del_contrato, "palabras_clave": A.palabras_clave.fillna("").astype(str)
                            + ";" + B.palabras_clave.fillna("").astype(str)})
    det = det.join(clave[["estrato", "fam_secop"]])
    det = det[det.gold != ""]
    out["n_gold"] = int(len(det))
    out["n_no_clasificable"] = int((det.gold == "NC").sum())
    ev = det[det.gold != "NC"].copy()

    # 2. Error de la etiqueta SECOP por estrato
    ev["secop_ok"] = ev.fam_secop == ev.gold
    out["error_secop"] = {"global": float(1 - ev.secop_ok.mean()),
                          "por_estrato": (1 - ev.groupby("estrato").secop_ok.mean()).round(3).to_dict()}
    tr = ev[~ev.secop_ok].groupby(["fam_secop", "gold"]).size().sort_values(ascending=False).head(10)
    out["error_secop"]["transiciones_top"] = {f"{a}->{b}": int(n) for (a, b), n in tr.items()}

    # 3. Claude vs gold
    cj = ROOT / "data" / "llm" / "gold_claude.jsonl"
    if cj.exists():
        c = pd.DataFrame([json.loads(l) for l in cj.read_text(encoding="utf-8").splitlines() if l.strip()])
        c = c.set_index("id")
        c["fam_c"] = np.where(c.no_clasificable, "NC", c.familia)
        det = det.join(c[["fam_c", "confianza"]].rename(columns={"confianza": "conf_claude"}))
        m = det.fam_c.notna()
        out["claude_vs_gold"] = {**kappa(det.loc[m, "gold"], det.loc[m, "fam_c"]),
                                 "por_confianza": det[m].assign(ok=det.gold == det.fam_c)
                                 .groupby("conf_claude").ok.agg(["mean", "size"]).round(3).to_dict("index"),
                                 "vs_secop_acuerdo": float((det.loc[m, "fam_c"] == det.loc[m, "fam_secop"]).mean())}
        ev = ev.join(c[["fam_c"]])

    # 4. Modelos vs gold (y vs SECOP en los mismos ítems)
    textos = [preparar(t) for t in ev.objeto]
    out["modelos_vs_gold"] = {}
    for nombre_m in MODELOS:
        mod = cargar(nombre_m)
        P = mod.predict_proba(textos)
        top = mod.classes_[np.argsort(-P, 1)[:, :3]]
        pred = top[:, 0]
        out["modelos_vs_gold"][nombre_m] = {
            "accuracy": float((pred == ev.gold).mean()),
            "macro_f1": float(f1_score(ev.gold, pred, average="macro", zero_division=0)),
            "top3": float(np.mean([g in t for g, t in zip(ev.gold, top)])),
            "accuracy_vs_secop_mismos_items": float((pred == ev.fam_secop).mean()),
            "gold_fuera_de_las_familias_del_modelo": float(1 - ev.gold.isin(mod.classes_).mean()),
            "n": int(len(ev))}
        if nombre_m == MODELOS[0]:
            mod0, ev["pred"] = mod, pred
    out["secop_vs_gold_accuracy"] = float(ev.secop_ok.mean())
    if "fam_c" in ev:
        m = ev.fam_c.notna()
        out["claude_vs_gold_accuracy_mismos_items"] = float((ev.fam_c[m] == ev.gold[m]).mean())

    # 5. Plausibilidad: palabras resaltadas por el modelo vs palabras_clave humanas
    rng = np.random.RandomState(SEED)
    f1s, f1_azar = [], []
    for t, pk, fam in zip(textos, ev.palabras_clave, ev.pred):
        humano = set().union(*[tokens(p) for p in pk.split(";") if p.strip()]) if pk.strip(";") else set()
        if not humano or not t:
            continue
        exp = local(mod0, t, fam, k=10)
        modelo = set(" ".join(exp[exp.contribucion > 0].head(5).termino).split())
        palabras = t.split()
        azar = set(rng.choice(palabras, size=min(len(modelo) or 1, len(palabras)), replace=False))
        for conj, lista in ((modelo, f1s), (azar, f1_azar)):
            inter = len(conj & humano)
            lista.append(0.0 if inter == 0 else 2 * inter / (len(conj) + len(humano)))
    out["plausibilidad_token_f1"] = {"modelo": float(np.mean(f1s)), "azar": float(np.mean(f1_azar)), "n": len(f1s)}

    det.to_csv(REPORTS / "gold_detalle.csv")
    (REPORTS / "gold_resumen.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
