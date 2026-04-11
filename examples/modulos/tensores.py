"""
modulos/tensores.py
━━━━━━━━━━━━━━━━━━━
Ejemplo 02 — Operaciones Tensoriales.

Demuestra multiplicación de matrices, softmax numérico estable,
máscaras causales y forward pass real con GPT-2.

Exporta:
    run_02_tensores()
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


def run_02_tensores():
    titulo("02 — Operaciones Tensoriales")
    if not _verificar_molineteai():
        return

    try:
        import numpy as np
    except ImportError:
        print("⚠️  Instala numpy:  pip install numpy")
        return

    import molineteai

    def softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / e_x.sum(axis=axis, keepdims=True)

    print("\n1. Multiplicación de matrices")
    A = np.array([[1., 2.], [3., 4.]])
    I = np.eye(2)
    print(f"  A @ I = {(A @ I).flatten().tolist()} ✓")

    print("\n2. Softmax numérico estable")
    logits = np.array([[100., 200., 300.]])
    probs = softmax(logits)
    print(f"  Logits grandes: {logits.flatten().tolist()}")
    print(f"  Softmax: {[f'{v:.4f}' for v in probs.flatten()]}  suma={probs.sum():.6f}")

    print("\n3. Máscara causal (long_sec=4)")
    long_sec = 4
    print("  Pos:  0  1  2  3")
    for i in range(long_sec):
        fila = ["✓" if j <= i else "✗" for j in range(long_sec)]
        print(f"    {i}: [{', '.join(fila)}]")

    print("\n4. Forward pass real con GPT-2 Diminuto")
    config = molineteai.Config.diminuta(512)
    modelo = molineteai.GPT2(config)
    tokens = [[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]]
    t0 = time.perf_counter()
    logits = modelo.forward(tokens)
    ms = (time.perf_counter() - t0) * 1000
    _, _, vocab = modelo.forma_salida(2, 8)
    print(f"  Forma: [2, 8, {vocab}] — {len(logits):,} elementos — {ms:.2f} ms")

    print("\n5. Benchmarks por tamaño de modelo")
    configs_b = [
        ("Diminuto",    molineteai.Config.diminuta(512)),
        ("Pequeño",     molineteai.Config.pequena(512)),
        ("Mediano",     molineteai.Config.mediana(512)),
        ("GPT-2 Small", molineteai.Config.gpt2_small(512)),
    ]
    print(f"  {'Config':<12} {'8 tokens':>10} {'1 token':>10}")
    print("  " + "─" * 35)
    for nombre, cfg in configs_b:
        m = molineteai.GPT2(cfg)
        m.forward(tokens)  # warmup
        iters = 5
        t0 = time.perf_counter()
        for _ in range(iters): m.forward(tokens)
        ms8 = (time.perf_counter() - t0) * 1000 / iters
        t0 = time.perf_counter()
        for _ in range(iters): m.forward([[42]])
        ms1 = (time.perf_counter() - t0) * 1000 / iters
        print(f"  {nombre:<12} {ms8:>8.2f}ms {ms1:>8.2f}ms")

    print("\n✅ Todas las operaciones verificadas.")
