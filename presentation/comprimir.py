"""Empaqueta la carpeta portable/ en presentacion_portable.zip.

Uso, desde la carpeta presentation:

    uv run python comprimir.py

El zip queda junto a este script. Para llevar la presentacion a otro PC
basta copiar el zip (y descomprimir.py si el otro PC no tiene con que
descomprimir). Los videos .mp4 se guardan sin recomprimir porque ya son
un formato comprimido; el resto de archivos va con deflate.
"""

import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PORTABLE = AQUI / "portable"
ZIP = AQUI / "presentacion_portable.zip"


def main() -> None:
    if not (PORTABLE / "slides" / "Presentacion.json").exists():
        print("No hay presentacion compilada en portable/.")
        print("Ejecuta antes: uv run python compilar.py")
        return

    archivos = [a for a in sorted(PORTABLE.rglob("*")) if a.is_file()]
    with zipfile.ZipFile(ZIP, "w") as zf:
        for archivo in archivos:
            metodo = zipfile.ZIP_STORED if archivo.suffix == ".mp4" else zipfile.ZIP_DEFLATED
            zf.write(archivo, archivo.relative_to(AQUI), compress_type=metodo)

    peso_mb = ZIP.stat().st_size / 1024 / 1024
    print(f"Creado {ZIP.name} ({len(archivos)} archivos, {peso_mb:.0f} MB)")


if __name__ == "__main__":
    main()
