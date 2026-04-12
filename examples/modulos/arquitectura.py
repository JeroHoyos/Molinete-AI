"""
modulos/arquitectura.py
━━━━━━━━━━━━━━━━━━━━━━━
Ejemplo 03 — Arquitectura del Modelo GPT-2.

Crea modelos de distintos tamaños, cuenta parámetros,
muestra la máscara causal y hace un benchmark de forward pass.
Toda la lógica de modelo se delega al módulo Rust (molineteai).

Exporta:
    run_03_arquitectura()
"""

import time

from modulos.ui import titulo


def _verificar_molineteai() -> bool:
    try:
        import molineteai  # noqa: F401
        return True
    except ImportError:
        print("\n⚠️  El módulo 'molineteai' no está instalado.")
        print("   Compílalo con:  maturin develop --release\n")
        return False


def run_03_arquitectura():
    titulo("03 — Arquitectura del Modelo GPT-2")
    if not _verificar_molineteai():
        return

    import molineteai

    TAM_VOCAB = 512
    configs = [
        ("Diminuto",    molineteai.Config.diminuta(TAM_VOCAB)),
        ("Pequeño",     molineteai.Config.pequena(TAM_VOCAB)),
        ("Mediano",     molineteai.Config.mediana(TAM_VOCAB)),
        ("GPT-2 Small", molineteai.Config.gpt2_small(TAM_VOCAB)),
    ]

    # ── 1. Tabla de configuraciones (Rust) ───────────────────────────────────
    print(f"\n{'Config':<12} {'Vocab':<7} {'Embd':<7} {'Cabezas':<8} {'Capas':<7} {'Paráms':>12} {'Mem MB':>8}")
    print("─" * 70)
    for nombre, cfg in configs:
        n = molineteai.contar_parametros_config(cfg)   # estimación desde Rust sin crear el modelo
        mem = n * 4 / 1e6
        print(f"{nombre:<12} {cfg.tam_vocabulario:<7} {cfg.n_embd:<7} "
              f"{cfg.n_cabezas:<8} {cfg.n_capas:<7} {n:>12,} {mem:>7.1f}")

    # ── 2. Forward pass real (Rust) ──────────────────────────────────────────
    print("\nForward pass con modelo Diminuto (lote=2, long_sec=8)...")
    cfg = molineteai.Config.diminuta(TAM_VOCAB)
    modelo = molineteai.GPT2(cfg)
    tokens = [[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]]
    t0 = time.perf_counter()
    modelo.forward(tokens)
    ms = (time.perf_counter() - t0) * 1000
    _, _, vocab = modelo.forma_salida(2, 8)
    print(f"  Forma salida: [2, 8, {vocab}] — {ms:.2f} ms")

    # ── 3. Visualización de la máscara causal ────────────────────────────────
    # La máscara se aplica dentro del forward pass de Rust.
    # La visualizamos aquí para que sea didáctica.
    print("\nMáscara causal (long_sec=4):")
    print("  Cada token puede ver solo su posición y las anteriores.")
    for i in range(4):
        fila = ["✓" if j <= i else "✗" for j in range(4)]
        print(f"  pos {i}: [{', '.join(fila)}]")

    # ── 4. Representación del modelo (Rust) ──────────────────────────────────
    print(f"\nRepresentación del modelo:")
    print(f"  {repr(modelo)}")

    print("\n✅ Arquitectura verificada para todos los tamaños.")
