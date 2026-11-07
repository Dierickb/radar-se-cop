# %% [markdown]
# # Radar SECOP – BETO afinado para familia UNSPSC (Colab, GPU)
#
# Compara un transformer en español contra el modelo lineal de la app, **con exactamente los mismos datos**:
# `data/beto/split_familia.parquet` congela el entrenamiento balanceado (52.555 contratos) y el test natural
# (36.844; 9.802 marcados para el test balanceado) que usa `scripts/03_entrenar_evaluar.py`.
# El gold set ya está excluido del entrenamiento.
#
# - Modelo: `dccuchile/bert-base-spanish-wwm-uncased` (BETO).
# - Texto: el objeto del contrato sin fórmulas de contratación (sin lematizar: BERT tiene su propio tokenizador).
# - Métricas: las mismas funciones del repo (`radar.evaluate.metricas`).
#
# Colab: Entorno de ejecución → Cambiar tipo → GPU T4. Tiempo estimado (sin medir): 30–60 minutos.

# %%
import os, subprocess, sys
REPO = "https://github.com/davidm094/radar-secop.git"
if not os.path.exists("radar-secop") and "google.colab" in sys.modules:
    subprocess.run(["git", "clone", "--depth", "1", REPO], check=True)
RAIZ = "radar-secop" if os.path.exists("radar-secop") else ".."
sys.path.insert(0, os.path.join(RAIZ, "src"))

# %%
import json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from radar.evaluate import metricas

SMOKE = os.environ.get("RADAR_SMOKE") == "1"   # prueba rápida en CPU, no para reportar
MODELO = "dccuchile/bert-base-spanish-wwm-uncased"
MAX_LEN, BATCH, EPOCAS, LR, SEMILLA = 128, 32, 3, 3e-5, 42
if SMOKE:
    MAX_LEN, BATCH, EPOCAS = 32, 8, 1
torch.manual_seed(SEMILLA); np.random.seed(SEMILLA)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("dispositivo:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "sin GPU")

# %%
d = pd.read_parquet(os.path.join(RAIZ, "data", "beto", "split_familia.parquet"))
tr, te = d[d.split == "train"], d[d.split == "test_natural"]
if SMOKE:
    tr, te = tr.sample(200, random_state=0), te.sample(200, random_state=0)
clases = np.array(sorted(d.fam.unique()))          # las 175 familias, mismo orden siempre
idx = {c: i for i, c in enumerate(clases)}
print(len(tr), "train |", len(te), "test natural |", int(te.en_test_bal.sum()), "test balanceado |", len(clases), "familias")

# %%
tok = AutoTokenizer.from_pretrained(MODELO)
modelo = AutoModelForSequenceClassification.from_pretrained(MODELO, num_labels=len(clases)).to(dev)

def lotes(textos, etiquetas=None, barajar=False):
    datos = list(range(len(textos)))
    def juntar(ids):
        enc = tok([textos[i] for i in ids], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        if etiquetas is not None:
            enc["labels"] = torch.tensor([etiquetas[i] for i in ids])
        return enc
    return DataLoader(datos, batch_size=BATCH, shuffle=barajar, collate_fn=juntar)

X_tr, y_tr = tr.texto_beto.tolist(), [idx[f] for f in tr.fam]
opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=0.01)
pasos = EPOCAS * len(lotes(X_tr, y_tr))
sched = get_linear_schedule_with_warmup(opt, int(0.1 * pasos), pasos)
usar_amp = dev == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)

# %%
t0 = time.time()
for ep in range(EPOCAS):
    modelo.train(); perdida = []
    for b in lotes(X_tr, y_tr, barajar=True):
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usar_amp):
            out = modelo(**b)
        scaler.scale(out.loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        perdida.append(out.loss.item())
    print(f"época {ep + 1}: pérdida {np.mean(perdida):.3f} ({(time.time() - t0) / 60:.1f} min)")

# %%
class Envoltura:
    """Expone predict_proba y classes_ para reutilizar radar.evaluate.metricas tal cual."""
    classes_ = clases
    @torch.no_grad()
    def predict_proba(self, textos):
        modelo.eval(); P = []
        for b in lotes(list(textos)):
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usar_amp):
                P.append(torch.softmax(modelo(**b).logits.float(), -1).cpu().numpy())
        return np.vstack(P)

m = Envoltura()
te_bal = te[te.en_test_bal]
filas = [metricas(m, te.texto_beto, te.fam, "test_natural")]
if len(te_bal):
    filas.append(metricas(m, te_bal.texto_beto, te_bal.fam, "test_balanceado"))
res = pd.DataFrame(filas).assign(variante="BETO_sin_formulas", modelo=MODELO, max_len=MAX_LEN, epocas=EPOCAS, batch=BATCH,
                                 gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                                 minutos_total=round((time.time() - t0) / 60, 1))
print(res.round(3).to_string(index=False))

# %% [markdown]
# Comparar con `reports/metricas_familia.csv` (modelo lineal, variante B): en test balanceado macro-F1 0,407 y
# top-3 0,759. Recuerde que contra SECOP se mide acuerdo con etiquetas sesgadas; el gold set decide.

# %%
if not SMOKE:
    os.makedirs(os.path.join(RAIZ, "reports"), exist_ok=True)
    res.to_csv(os.path.join(RAIZ, "reports", "metricas_beto.csv"), index=False)
    modelo.save_pretrained("beto_familia"); tok.save_pretrained("beto_familia")
    if "google.colab" in sys.modules:
        from google.colab import files
        files.download(os.path.join(RAIZ, "reports", "metricas_beto.csv"))
