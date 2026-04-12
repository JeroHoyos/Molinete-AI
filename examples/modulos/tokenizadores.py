"""
modulos/tokenizadores.py
━━━━━━━━━━━━━━━━━━━━━━━━
Ejemplo 01 — Tokenizadores BPE.

Entrena tokenizadores con varios tamaños de vocabulario,
analiza la compresión y guarda los resultados en disco.
Toda la lógica de tokenización se delega al módulo Rust (molineteai).

El corpus central es cervantes.txt. Si no existe, el usuario
puede indicar cualquier .txt propio.

Exporta:
    run_01_tokenizadores()
"""

import os
import time

from modulos.ui    import titulo, pedir_input, barra_progreso, SEPARADOR
from modulos.datos import elegir_corpus, verificar_corpus, es_corpus_cervantes


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
    if not _verificar_molineteai():
        return

    import molineteai

    ruta_corpus = elegir_corpus()
    if not ruta_corpus or not verificar_corpus(ruta_corpus):
        return

    # Frase de verificación: Cervantes o primeras palabras del corpus custom
    frase_prueba = (
        "En un lugar de la Mancha, de cuyo nombre no quiero acordarme"
        if es_corpus_cervantes(ruta_corpus) else
        None   # se determina tras cargar el texto
    )

    # Tamaños de vocabulario
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

    print(f"\nCargando corpus '{ruta_corpus}'...")
    with open(ruta_corpus, encoding="utf-8") as f:
        texto = f.read()
    print(f"Corpus: {len(texto):,} bytes ({len(texto)/1e6:.2f} MB)\n")

    # Para corpus custom: usar las primeras 10 palabras como frase de prueba
    if frase_prueba is None:
        palabras = texto.split()[:10]
        frase_prueba = " ".join(palabras)

    marca = int(time.time())
    dir_s = f"data/tokenizadores_{marca}"
    os.makedirs(dir_s, exist_ok=True)

    resumen = []
    for tam in TAMANOS:
        print(f"{SEPARADOR}\nVocabulario = {tam}\n{SEPARADOR}")
        barra_progreso(f"Entrenando vocab={tam}", segundos=0.5)

        # Tokenizador BPE (Rust)
        t0 = time.time()
        tok = molineteai.TokenizadorBPE(tam)
        tok.entrenar(texto, tam)
        t_e = time.time() - t0

        # Codificar corpus (Rust)
        t0 = time.time()
        ids = tok.codificar(texto)
        t_c = time.time() - t0

        ratio = len(texto) / len(ids)
        print(f"  Tiempo entrenamiento: {t_e:.2f}s | Codificación: {t_c:.2f}s")
        print(f"  Tokens: {len(ids):,} | Compresión: {ratio:.2f}x")

        # Verificar ciclo codif→decodif (Rust)
        ok = tok.decodificar(tok.codificar(frase_prueba)) == frase_prueba
        print(f"  Ciclo codif→decodif: {'✓ PASADA' if ok else '✗ FALLIDA'}")

        # Análisis de vocabulario (Rust)
        if tam > 256:
            tok.analizar_vocabulario(texto)

        # Guardar tokenizador (Rust)
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
