#!/usr/bin/env python
"""
web/runner.py — Ejecutor de ejemplos en subproceso.

El servidor lo lanza con: python web/runner.py <id>  (cwd=raíz del repo)
stdout/stderr se capturan y retransmiten al frontend via WebSocket.
stdin se conecta al pipe del servidor para que los input() funcionen.
"""

import sys

EJEMPLOS = {
    "1":  ("modules.tokenizadores",   "run_01_tokenizadores"),
    "2":  ("modules.tensores",        "run_02_tensores"),
    "3":  ("modules.arquitectura",    "run_03_arquitectura"),
    "4":  ("modules.infraestructura", "run_04_infraestructura"),
    "5":  ("modules.entrenamiento",   "run_05_diminuto"),
    "6":  ("modules.entrenamiento",   "run_06_pequeno"),
    "7":  ("modules.entrenamiento",   "run_07_mediano"),
    "8":  ("modules.entrenamiento",   "run_08_gpt2"),
    "9":  ("modules.entrenamiento",   "run_entrenar_presets"),
    "10": ("modules.chat",            "run_chat"),
    "11": ("modules.datos",           "run_descargar_datos"),
    "12": ("modules.comparar",        "run_comparar"),
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
