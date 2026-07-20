#!/usr/bin/env python3
"""Descomprime presentacion_portable.zip y deja la carpeta portable/ lista.

Uso, desde la carpeta donde esten este script y el zip (en el otro PC
basta tener Python, no hay que instalar nada):

    python3 descomprimir.py

En Windows: py descomprimir.py

Si ya existe una carpeta portable/ de una compilacion anterior, la
reemplaza entera para no mezclar archivos de versiones distintas. Al
terminar borra el zip, que ya no hace falta.

En Linux/Mac normaliza ademas las rutas del JSON, porque se compila en
Windows con rutas de tipo slides\\files\\... que esos sistemas no
resuelven. En Windows no hace falta y no se toca nada.
"""

import json
import os
import shutil
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ZIP = AQUI / "presentacion_portable.zip"
CONFIG = AQUI / "portable" / "slides" / "Presentacion.json"


def normalizar_rutas() -> None:
    """Cambia \\ por / en las rutas del JSON. Idempotente e inofensivo."""
    if not CONFIG.exists():
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


def main() -> None:
    if not ZIP.exists():
        print(f"No se encuentra {ZIP.name} junto a este script.")
        return

    portable_previo = AQUI / "portable"
    if portable_previo.exists():
        shutil.rmtree(portable_previo)

    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(AQUI)

    ZIP.unlink()

    if os.name != "nt":
        normalizar_rutas()

    print(f"Listo: presentacion descomprimida en {AQUI / 'portable'}")
    print(f"Se borro {ZIP.name}, ya no hace falta.")
    print("Para presentar, entra a portable/ y sigue su README.md.")


if __name__ == "__main__":
    main()
