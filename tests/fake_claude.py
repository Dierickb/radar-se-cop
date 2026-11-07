"""Simulador de `claude -p` para probar los scripts de llm/ sin gastar cuota.

Uso:  RADAR_CLAUDE_CMD="python tests/fake_claude.py" python llm/reetiquetar.py ...
Imita el sobre JSON de `--output-format json` con `structured_output`.
"""
import json
import random
import re
import sys

args = sys.argv[1:]
esquema = json.loads(args[args.index("--json-schema") + 1])
assert args[args.index("--tools") + 1] == "", "las herramientas deben ir deshabilitadas"
assert "--bare" not in args, "--bare no usa la suscripción"
entrada = sys.stdin.read()

if "items" in esquema["properties"]:
    filas = json.loads(entrada[entrada.index("["):])
    salida = {"items": [{"id": f["id"], "familia": "8012", "segunda_familia": "", "confianza": "alta",
                         "no_clasificable": False, "palabras_clave": f["objeto"].split()[:2], "regla": "R2"}
                        for f in filas]}
else:
    n = int(re.search(r"Escribe (\d+)", entrada).group(1))
    salida = {"textos": [{"texto": f"texto simulado {random.random():.8f}", "registro": "breve"} for _ in range(n)]}

print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "",
                  "structured_output": salida, "session_id": "fake", "total_cost_usd": 0.0,
                  "duration_ms": 1, "num_turns": 1}))
