"""
modulos/tokenizadores.py
━━━━━━━━━━━━━━━━━━━━━━━━
Ejemplo 01 — Tokenizadores BPE.

Entrena tokenizadores con varios tamaños de vocabulario,
analiza la compresión y guarda los resultados en disco.

Exporta:
    run_01_tokenizadores()
"""

import os
import time

from modulos.ui    import titulo, pedir_input, barra_progreso, SEPARADOR
from modulos.datos import verificar_corpus


def _verificar_molineteai() -> bool:
    try:
        import molineteai  # noqa: F401
        return True
    except ImportError:
        print("\n⚠️  El módulo 'molineteai' no está instalado.")
        print("   Compílalo con:  maturin develop --release\n")
        return False


def run_01_tokenizadores():
    titulo("01 — Tokenizadores BPE")
    if not _verificar_molineteai() or not verificar_corpus():
        return

    import molineteai

    TAMANOS = [256, 512, 1024, 1536]
    custom = pedir_input(
        f"Tamaños de vocabulario a probar {TAMANOS} "
        "(Enter para usar los de arriba, o escribe los tuyos separados por coma): "
    )
    if custom:
        try:
            TAMANOS = [int(x.strip()) for x in custom.split(",")]
        except ValueError:
            print("Entrada inválida, usando valores por defecto.")

    print("\nCargando corpus...")
    with open("shakespeare.txt", encoding="utf-8") as f:
        texto = f.read()
    print(f"Corpus: {len(texto):,} bytes ({len(texto)/1e6:.2f} MB)\n")

    marca = int(time.time())
    dir_s = f"data/tokenizadores_{marca}"
    os.makedirs(dir_s, exist_ok=True)

    resumen = []
    for tam in TAMANOS:
        print(f"{SEPARADOR}\nVocabulario = {tam}\n{SEPARADOR}")
        barra_progreso(f"Entrenando vocab={tam}", segundos=0.5)

        t0 = time.time()
        tok = molineteai.TokenizadorBPE(tam)
        tok.entrenar(texto, tam)
        t_e = time.time() - t0

        t0 = time.time()
        ids = tok.codificar(texto)
        t_c = time.time() - t0

        ratio = len(texto) / len(ids)
        print(f"  Tiempo entrenamiento: {t_e:.2f}s | Codificación: {t_c:.2f}s")
        print(f"  Tokens: {len(ids):,} | Compresión: {ratio:.2f}x")

        prueba = "To be, or not to be, that is the question"
        ok = tok.decodificar(tok.codificar(prueba)) == prueba
        print(f"  Ciclo codif→decodif: {'✓ PASADA' if ok else '✗ FALLIDA'}")

        if tam > 256:
            tok.analizar_vocabulario(texto)

        ruta = f"{dir_s}/tokenizador_{tam}.json"
        tok.guardar(ruta)
        print(f"  Guardado: {ruta}")
        resumen.append((tam, t_e, len(ids), ratio))

    print(f"\n{SEPARADOR}")
    print(f"{'Vocab':<8} {'Entren(s)':>10} {'Tokens':>12} {'Compresión':>12}")
    print(SEPARADOR)
    for tam, t_e, n_tok, ratio in resumen:
        print(f"{tam:<8} {t_e:>10.2f} {n_tok:>12,} {ratio:>11.2f}x")
    print(f"\n✅ Tokenizadores guardados en: {dir_s}/")
