# Uso de Claude en Radar SECOP

Claude (Anthropic) se usa **solo para preparar datos de entrenamiento**, nunca dentro de la app ni para evaluar.
Se corre con **Claude Code en modo no interactivo** (`claude -p`), que entra en la suscripción Pro/Max
del usuario y no requiere API de pago. El límite de uso se comparte con claude.ai, así que los scripts se
pueden cortar y retomar.

| Uso | Script | Entrada | Salida | Se usa para |
|---|---|---|---|---|
| Re-etiquetado (validación) | `reetiquetar.py` | los 300 del gold set (objeto, entidad, tipo; **sin** código declarado) | `data/llm/gold_claude.jsonl` | medir el acuerdo Claude–humanos (`scripts/08_acuerdo_gold.py`) |
| Re-etiquetado (escala) | `reetiquetar.py --entrenamiento N` | N contratos del **train** (nunca test ni gold) | `data/llm/train_claude.jsonl` | variante C1 (`scripts/09_entrenar_con_llm.py`), **solo si kappa ≥ 0,61** |
| Aumento estilo pyme | `aumentar.py` | familias del modelo | `data/llm/aumentado.jsonl` | variante C2 (solo entrenamiento) |

## Requisitos

1. Instalar Claude Code y entrar una vez con la cuenta de la suscripción: `npm install -g @anthropic-ai/claude-code` y luego `claude`.
2. **No usar `--bare`**: ese modo ignora el login de la suscripción y pide `ANTHROPIC_API_KEY` (cobro por API).
3. Haber corrido `scripts/01` a `03` (catálogo, contratos preparados, modelo y `test_ids.csv`).

## Cómo correr

```bash
# 0) piloto barato: 20 contratos, revisar a mano la salida
python llm/reetiquetar.py --entrada data/gold/gold_set_plantilla.xlsx --salida data/llm/piloto.jsonl --limite 20

# 1) los 300 del gold set (después de que los humanos terminen, o en paralelo: Claude no ve sus anotaciones)
python llm/reetiquetar.py --entrada data/gold/gold_set_plantilla.xlsx --salida data/llm/gold_claude.jsonl
python scripts/08_acuerdo_gold.py

# 2) solo si kappa Claude–gold ≥ 0,61
python llm/reetiquetar.py --entrenamiento 5000 --salida data/llm/train_claude.jsonl
python llm/aumentar.py --por-familia 40 --salida data/llm/aumentado.jsonl
python scripts/09_entrenar_con_llm.py
python scripts/08_acuerdo_gold.py --modelos familia_v1 familia_v2_llm
```

Cada llamada ejecuta:

```
claude -p --output-format json --json-schema <schemas/*.json> --system-prompt <prompts/*.md + catálogo>
       --tools "" --no-session-persistence --model sonnet      (el texto de entrada va por stdin)
```

- `--json-schema` obliga a responder con la estructura de `schemas/`; el script lee el campo `structured_output`.
- `--tools ""` deshabilita las herramientas: Claude solo lee y responde.
- El prompt de re-etiquetado es la **misma guía de anotación** que siguen los humanos (reglas R1–R8, Caso 10 de
  Colombia Compra Eficiente) más el catálogo completo de 420 familias v14 en español.
- Cada fila guardada registra el modelo y `prompt_sha` (huella del prompt exacto) para poder reportarlo.
- Controles automáticos: la familia debe existir en el catálogo, los ids deben coincidir con los enviados y se mide
  qué fracción de `palabras_clave` aparece literal en el objeto.

## Pruebas sin gastar cuota

`tests/fake_claude.py` imita la salida de `claude -p`:

```bash
RADAR_CLAUDE_CMD="python tests/fake_claude.py" python llm/reetiquetar.py --entrada data/gold/gold_set_plantilla.xlsx --salida /tmp/prueba.jsonl --limite 25
```

## Qué reportar en el documento

Modelo y fecha de las corridas, `prompt_sha`, número de contratos, kappa Claude–gold (global y por confianza),
porcentaje de etiquetas de SECOP que Claude cambió y, en la variante C, métricas contra el gold set
(no contra SECOP, cuyas etiquetas son justamente las que se corrigen).
