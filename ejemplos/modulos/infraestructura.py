"""
modulos/infraestructura.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
Ejemplo 04 — Infraestructura de Entrenamiento.

Demuestra cómo el módulo Rust gestiona la división de datos,
el tokenizador BPE y las estadísticas del vocabulario,
sin realizar un entrenamiento real.

El corpus central es cervantes.txt. Si no existe, el usuario
puede indicar cualquier .txt propio.

Exporta:
    run_04_infraestructura()
"""

import math

from modulos.ui    import titulo, barra_progreso
from modulos.datos import elegir_corpus, verificar_corpus, es_corpus_cervantes


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
    if not _verificar_molineteai():
        return

    import molineteai

    ruta_corpus = elegir_corpus()
    if not ruta_corpus or not verificar_corpus(ruta_corpus):
        return

    with open(ruta_corpus, encoding="utf-8") as f:
        texto = f.read(100_000)
    print(f"Corpus (100K chars): {len(texto):,} caracteres")

    # ── 1. Tokenizador BPE (Rust) ─────────────────────────────────────────────
    print("\nEntrenando tokenizador BPE (vocab=512)...")
    barra_progreso("Tokenizador", segundos=0.3)
    tok = molineteai.TokenizadorBPE(512)
    tok.entrenar(texto, 512)
    print(f"✓ Vocabulario: {tok.tam_vocabulario()} tokens")

    stats = tok.estadisticas()
    print(f"  tokens_base={stats['tokens_base']}, fusiones={stats['num_fusiones']}")

    # ── 2. División train/val (Rust) ──────────────────────────────────────────
    todos = tok.codificar(texto)
    tokens_train, tokens_val = molineteai.dividir_entrenamiento_validacion(todos, 0.1)
    print(f"\nDivisión entrenamiento/validación:")
    print(f"  Total:         {len(todos):,} tokens")
    print(f"  Entrenamiento: {len(tokens_train):,} (90%)")
    print(f"  Validación:    {len(tokens_val):,} (10%)")

    # ── 3. Ciclo codif→decodif (Rust) ─────────────────────────────────────────
    frase = (
        "En un lugar de la Mancha"
        if es_corpus_cervantes(ruta_corpus) else
        texto.split()[0] + " " + texto.split()[1]   # primeras dos palabras del corpus custom
    )
    ok = tok.decodificar(tok.codificar(frase)) == frase
    print(f"\nCiclo codif→decodif '{frase}': {'✓ PASADA' if ok else '✗ FALLIDA'}")

    # ── 4. Pérdida de referencia ──────────────────────────────────────────────
    perdida_ref = math.log(tok.tam_vocabulario())
    print(f"\nPérdida aleatoria (vocab={tok.tam_vocabulario()}): {perdida_ref:.4f}")
    print(f"Perplejidad de referencia:                        {math.exp(perdida_ref):.2f}")
    print("(Un modelo bien entrenado debería superar esto)")

    # ── 5. Estimación de parámetros (Rust) ────────────────────────────────────
    print("\nEstimación de parámetros con este vocabulario:")
    print(f"  {'Config':<12} {'Params':>12} {'Mem MB':>8}")
    print("  " + "─" * 35)
    for nombre, cfg_fn in [
        ("Diminuto", molineteai.Config.diminuta),
        ("Pequeño",  molineteai.Config.pequena),
        ("Mediano",  molineteai.Config.mediana),
    ]:
        cfg = cfg_fn(tok.tam_vocabulario())
        n = molineteai.contar_parametros_config(cfg)
        print(f"  {nombre:<12} {n:>12,} {n*4/1e6:>7.1f}")

    print(f"\n✅ Infraestructura verificada. Corpus listo para entrenamiento.")
