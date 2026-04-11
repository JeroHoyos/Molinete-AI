"""
modulos/infraestructura.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
Ejemplo 04 — Infraestructura de Entrenamiento.

Demuestra data loaders, división train/val y logging CSV
sin realizar un entrenamiento real.

Exporta:
    run_04_infraestructura()
"""

import csv
import math
import os
import time

from modulos.ui    import titulo, barra_progreso
from modulos.datos import verificar_corpus


def _verificar_molineteai() -> bool:
    try:
        import molineteai  # noqa: F401
        return True
    except ImportError:
        print("\n⚠️  El módulo 'molineteai' no está instalado.")
        print("   Compílalo con:  maturin develop --release\n")
        return False


def run_04_infraestructura():
    titulo("04 — Infraestructura de Entrenamiento")
    if not _verificar_molineteai() or not verificar_corpus():
        return

    import molineteai

    with open("shakespeare.txt", encoding="utf-8") as f:
        texto = f.read(100_000)
    print(f"Corpus (100K chars): {len(texto):,} caracteres")

    print("\nEntrenando tokenizador (vocab=512)...")
    barra_progreso("Tokenizador", segundos=0.3)
    tok = molineteai.TokenizadorBPE(512)
    tok.entrenar(texto, 512)
    print(f"✓ Vocabulario: {tok.tam_vocabulario()} tokens")

    todos = tok.codificar(texto)
    train_tok, val_tok = molineteai.dividir_entrenamiento_validacion(todos, 0.1)
    print(f"\nDivisión:")
    print(f"  Total:         {len(todos):,} tokens")
    print(f"  Entrenamiento: {len(train_tok):,} (90%)")
    print(f"  Validación:    {len(val_tok):,} (10%)")

    # Data loader manual
    long_sec, tam_lote = 64, 4
    pos = 0
    lotes_demo = []
    for _ in range(3):
        if pos + tam_lote * (long_sec + 1) >= len(todos):
            break
        entrada = [
            todos[pos + i * long_sec: pos + i * long_sec + long_sec]
            for i in range(tam_lote)
        ]
        pos += tam_lote * long_sec
        lotes_demo.append(entrada)
    print(f"\nData loader (long_sec={long_sec}, lote={tam_lote}):")
    for i, lote in enumerate(lotes_demo):
        print(f"  Lote {i+1}: {len(lote)} seqs × {len(lote[0])} tokens")

    perdida_ref = math.log(tok.tam_vocabulario())
    print(f"\nPérdida aleatoria (vocab={tok.tam_vocabulario()}): {perdida_ref:.4f}")
    print(f"Perplejidad de referencia:                        {math.exp(perdida_ref):.2f}")

    marca = int(time.time())
    dir_s = f"data/infra_{marca}"
    os.makedirs(dir_s, exist_ok=True)
    ruta_log = f"{dir_s}/simulacion.csv"

    columnas = [
        "paso", "segundos", "tasa_aprendizaje",
        "perdida_entren", "perdida_val",
        "perp_entren", "perp_val", "muestra",
    ]
    with open(ruta_log, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        print(f"\nSimulando 10 pasos de entrenamiento:")
        muestras = ["To be, or not to be", None, None, "ROMEO.", None, None, None, None, None, None]
        for paso in range(10):
            pe = perdida_ref * (1.0 - paso * 0.05)
            pv = perdida_ref * (1.0 - paso * 0.04)
            w.writerow({
                "paso": paso * 10,
                "segundos": f"{paso*2:.1f}",
                "tasa_aprendizaje": "0.001000",
                "perdida_entren": f"{pe:.4f}",
                "perdida_val": f"{pv:.4f}",
                "perp_entren": f"{math.exp(pe):.2f}",
                "perp_val": f"{math.exp(pv):.2f}",
                "muestra": muestras[paso] or "",
            })
            print(f"  Paso {paso*10:3d} | entren={pe:.4f} | val={pv:.4f} | perp={math.exp(pv):.2f}")

    print(f"\n✅ Log guardado en: {ruta_log}")
