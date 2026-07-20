#!/usr/bin/env python3
"""Normaliza las rutas de slides/Presentacion.json para Linux/Mac.

La presentacion se compila en Windows y el JSON queda con rutas de tipo
slides\\files\\... que Linux no resuelve. Este script cambia \\ por /,
que funciona en todos los sistemas. Es idempotente e inofensivo en Windows.

Uso, desde esta carpeta:

    python3 normalizar_rutas.py

Solo usa la libreria estandar, no necesita instalar nada.
"""

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "slides" / "Presentacion.json"


def main() -> None:
    if not CONFIG.exists():
        print(f"No existe {CONFIG}. Compila primero la presentacion.")
        return

    datos = json.loads(CONFIG.read_text(encoding="utf-8"))
    cambiadas = 0
    for slide in datos.get("slides", []):
        for clave in ("file", "rev_file", "src"):
            valor = slide.get(clave)
            if isinstance(valor, str) and "\\" in valor:
                slide[clave] = valor.replace("\\", "/")
                cambiadas += 1

    if cambiadas:
        CONFIG.write_text(json.dumps(datos, indent=2), encoding="utf-8")
    print(f"{cambiadas} rutas normalizadas en {CONFIG.name}")


if __name__ == "__main__":
    main()
