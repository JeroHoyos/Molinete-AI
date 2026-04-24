#!/usr/bin/env python
"""
web/runner.py — Ejecutor de ejemplos en subproceso.

El servidor lo lanza con: python runner.py <id>  (cwd=examples/)
stdout/stderr se capturan y retransmiten al frontend via WebSocket.
stdin se conecta al pipe del servidor para que los input() funcionen.
"""

import sys
import os
from pathlib import Path

# Mismo fix que molineteai.py: evitar que "import molineteai" resuelva
# al archivo .py del proyecto en lugar del módulo Rust compilado.
_examples = str(Path(__file__).parent.parent / "ejemplos")
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(_examples)]
sys.path.append(_examples)

EJEMPLOS = {
    "1":  ("modulos.tokenizadores",   "run_01_tokenizadores"),
    "2":  ("modulos.tensores",        "run_02_tensores"),
    "3":  ("modulos.arquitectura",    "run_03_arquitectura"),
    "4":  ("modulos.infraestructura", "run_04_infraestructura"),
    "5":  ("modulos.entrenamiento",   "run_05_diminuto"),
    "6":  ("modulos.entrenamiento",   "run_06_pequeno"),
    "7":  ("modulos.entrenamiento",   "run_07_mediano"),
    "8":  ("modulos.entrenamiento",   "run_08_gpt2"),
    "9":  ("modulos.entrenamiento",   "run_entrenar_presets"),
    "10": ("modulos.chat",            "run_chat"),
    "11": ("modulos.datos",           "run_descargar_datos"),
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python runner.py <id>", flush=True)
        sys.exit(1)

    eid = sys.argv[1].strip()
    if eid not in EJEMPLOS:
        ids_disp = ", ".join(sorted(EJEMPLOS, key=int))
        print(f"Ejemplo '{eid}' no encontrado. Disponibles: {ids_disp}", flush=True)
        sys.exit(1)

    mod_path, func_name = EJEMPLOS[eid]

    try:
        mod = __import__(mod_path, fromlist=[mod_path.split(".")[-1]])
        fn  = getattr(mod, func_name)
    except ImportError as exc:
        print(f"❌ Error al importar {mod_path}: {exc}", flush=True)
        print("   ¿Está molineteai compilado? Ejecuta: maturin develop --release", flush=True)
        sys.exit(1)
    except AttributeError:
        print(f"❌ Función '{func_name}' no encontrada en {mod_path}", flush=True)
        sys.exit(1)

    try:
        fn()
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido.", flush=True)
    except EOFError:
        print("\n⚠️  Conexión cerrada.", flush=True)
    except Exception as exc:
        print(f"\n❌ Error inesperado: {exc}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
