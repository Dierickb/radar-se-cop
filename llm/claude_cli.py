"""Llamada mínima a Claude Code en modo no interactivo (`claude -p`), usando la suscripción del usuario.

- Requiere haber iniciado sesión una vez con `claude` (cuenta Pro/Max). NO usar `--bare`: ese modo
  ignora el login de la suscripción y exige ANTHROPIC_API_KEY (cobro por API).
- Sin herramientas (`--tools ""`), sin guardar sesiones y con salida validada por JSON Schema.
- La variable de entorno RADAR_CLAUDE_CMD permite sustituir el ejecutable (p. ej. por un simulador
  en las pruebas, para no gastar cuota).
"""
import json
import os
import shlex
import subprocess
import time


class ErrorClaude(RuntimeError):
    pass


def llamar(prompt: str, sistema: str, esquema: dict, modelo: str, timeout: int = 600,
           reintentos: int = 3) -> tuple[dict, dict]:
    """Devuelve (salida_estructurada, metadatos). El prompt va por stdin."""
    cmd = shlex.split(os.environ.get("RADAR_CLAUDE_CMD", "claude")) + [
        "-p",
        "--output-format", "json",
        "--json-schema", json.dumps(esquema, ensure_ascii=False),
        "--system-prompt", sistema,
        "--tools", "",
        "--no-session-persistence",
        "--model", modelo,
    ]
    ultimo = None
    for intento in range(1, reintentos + 1):
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            ultimo = "timeout"
        else:
            if r.returncode == 0:
                try:
                    env = json.loads(r.stdout)
                except json.JSONDecodeError:
                    ultimo = f"salida no JSON: {r.stdout[:300]}"
                else:
                    if env.get("is_error"):
                        ultimo = f"error de Claude: {env.get('result', '')[:300]}"
                    else:
                        datos = env.get("structured_output")
                        if datos is None:          # respaldo: el texto del resultado es el JSON
                            try:
                                datos = json.loads(env.get("result", ""))
                            except json.JSONDecodeError:
                                datos = None
                        if datos is not None:
                            meta = {k: env.get(k) for k in ("session_id", "total_cost_usd", "duration_ms",
                                                            "num_turns", "usage")}
                            meta["modelos_usados"] = sorted((env.get("modelUsage") or {}).keys())
                            return datos, meta
                        ultimo = "sin structured_output"
            else:
                ultimo = f"código {r.returncode}: {(r.stderr or r.stdout)[:300]}"
        # Límite de uso de la suscripción u otro fallo transitorio: esperar y reintentar
        espera = 60 * intento
        print(f"  intento {intento} falló ({ultimo}); reintento en {espera}s")
        time.sleep(espera)
    raise ErrorClaude(ultimo)
