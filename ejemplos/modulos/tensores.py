"""
modulos/tensores.py
━━━━━━━━━━━━━━━━━━━
Ejemplo 02 — Operaciones Tensoriales.

Espejo en Python del ejemplo Rust 02_tensor_operations.rs.
Demuestra todas las operaciones de molineteai.Tensor:
creación, matmul, ops elemento a elemento, escalares,
broadcasting, softmax, reshape, transpose, mean/var y masked_fill.

Exporta:
    run_02_tensores()
"""

import time

from modulos.ui import titulo, SEPARADOR


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

    import molineteai
    T = molineteai.Tensor   # alias corto

    # ── 1. Creación de tensores ──────────────────────────────────────────────
    print(f"\n{SEPARADOR}")
    print("1. Creación de Tensores")
    print(SEPARADOR)

    t = T([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    print(f"\nTensor 2×3:")
    print(f"  forma: {t.forma}")
    print(f"  datos: {t.datos}")

    ceros = T.ceros([3, 4])
    print(f"\nTensor de ceros 3×4:")
    print(f"  forma: {ceros.forma}")
    print(f"  suma:  {sum(ceros.datos)}")

    rango = T.arange(0, 10)
    print(f"\nTensor rango [0, 10):")
    print(f"  forma: {rango.forma}")
    print(f"  datos: {rango.datos}")

    # ── 2. Multiplicación de matrices ────────────────────────────────────────
    print(f"\n{SEPARADOR}")
    print("2. Multiplicación de Matrices")
    print(SEPARADOR)

    a = T([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = T([1.0, 0.0, 0.0, 1.0], [2, 2])   # identidad

    print(f"\nMatrices pequeñas (2×2) — secuencial:")
    print(f"  A: {a.datos}")
    print(f"  B (identidad): {b.datos}")

    t0 = time.perf_counter()
    c = a.matmul(b)
    us = (time.perf_counter() - t0) * 1e6
    print(f"  A @ B: {c.datos}")
    print(f"  Tiempo: {us:.2f} µs")

    grande_a = T([1.0] * (64 * 64), [64, 64])
    grande_b = T([1.0] * (64 * 64), [64, 64])

    print(f"\nMatrices grandes (64×64) — paralelo con bloqueo de caché:")
    t0 = time.perf_counter()
    grande_c = grande_a.matmul(grande_b)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  forma resultado: {grande_c.forma}")
    print(f"  primer elemento (debe ser 64.0): {grande_c.datos[0]:.1f}")
    print(f"  Tiempo: {ms:.2f} ms")

    # ── 3. Operaciones elemento a elemento ───────────────────────────────────
    print(f"\n{SEPARADOR}")
    print("3. Operaciones Elemento a Elemento")
    print(SEPARADOR)

    x = T([1.0, 2.0, 3.0, 4.0], [2, 2])
    y = T([1.0, 1.0, 1.0, 1.0], [2, 2])

    print(f"\nX: {x.datos}")
    print(f"Y: {y.datos}")
    print(f"X + Y: {x.add(y).datos}")
    print(f"X * Y: {x.mul(y).datos}")
    print(f"X - Y: {x.sub(y).datos}")
    print(f"X / Y: {x.div(y).datos}")

    # ── 4. Operaciones escalares ─────────────────────────────────────────────
    print(f"\n{SEPARADOR}")
    print("4. Operaciones Escalares")
    print(SEPARADOR)

    x = T([1.0, 2.0, 3.0, 4.0], [2, 2])
    print(f"\nX: {x.datos}")
    print(f"X * 2:   {x.mul_scalar(2.0).datos}")
    print(f"X + 10:  {x.add_scalar(10.0).datos}")
    print(f"X / 2:   {x.div_scalar(2.0).datos}")
    print(f"sqrt(X): {[round(v, 4) for v in x.sqrt().datos]}")

    # ── 5. Broadcasting ───────────────────────────────────────────────────────
    print(f"\n{SEPARADOR}")
    print("5. Broadcasting")
    print(SEPARADOR)

    matriz = T([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    sesgo  = T([0.1, 0.2, 0.3], [3])

    print(f"\nMatriz [2, 3]: {matriz.datos}")
    print(f"Sesgo  [3]:    {sesgo.datos}")
    resultado = matriz.add(sesgo)
    print(f"Matriz + Sesgo (broadcast): {[round(v, 2) for v in resultado.datos]}")

    # ── 6. Softmax ────────────────────────────────────────────────────────────
    print(f"\n{SEPARADOR}")
    print("6. Softmax (Estabilidad Numérica)")
    print(SEPARADOR)

    logits = T([1.0, 2.0, 3.0, 4.0], [1, 4])
    probs  = logits.softmax(-1)
    print(f"\nLogits: {logits.datos}")
    print(f"Softmax (eje=-1): {[round(v, 4) for v in probs.datos]}")
    print(f"Suma de probabilidades: {sum(probs.datos):.6f} (debe ser 1.0)")

    logits_grandes = T([100.0, 200.0, 300.0], [1, 3])
    probs_estable  = logits_grandes.softmax(-1)
    print(f"\nLogits grandes: {logits_grandes.datos}")
    print(f"Softmax (estable): {probs_estable.datos}")
    print(f"Suma: {sum(probs_estable.datos):.6f} (¡sin overflow!)")

    # ── 7. Reshape ────────────────────────────────────────────────────────────
    print(f"\n{SEPARADOR}")
    print("7. Reshape")
    print(SEPARADOR)

    original = T([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    remodelo = original.reshape([3, 2])
    aplanado = remodelo.reshape([6])

    print(f"\nOriginal [2, 3]:  {original.datos}")
    print(f"Reshape  [3, 2]:  {remodelo.datos}  forma: {remodelo.forma}")
    print(f"Aplanado [6]:     {aplanado.datos}")

    # ── 8. Transpose ──────────────────────────────────────────────────────────
    print(f"\n{SEPARADOR}")
    print("8. Transpose")
    print(SEPARADOR)

    matriz = T([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    trans  = matriz.transpose(0, 1)

    print(f"\nOriginal [2, 3]:")
    print(f"  Fila 0: {matriz.datos[0:3]}")
    print(f"  Fila 1: {matriz.datos[3:6]}")
    print(f"Transpuesto [3, 2] (filas ↔ columnas):")
    print(f"  forma: {trans.forma}")
    for i in range(3):
        print(f"  Fila {i}: {trans.datos[i*2:(i+1)*2]}")

    # ── 9. Operaciones estadísticas ───────────────────────────────────────────
    print(f"\n{SEPARADOR}")
    print("9. Operaciones Estadísticas")
    print(SEPARADOR)

    datos = T([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3])
    print(f"\nDatos [3, 3]: {datos.datos}")

    medias_fila = datos.mean(-1, False)
    print(f"Medias por fila: {medias_fila.datos}")

    datos_3d = T([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3])
    print(f"\nDatos 3D [1, 2, 3]: {datos_3d.datos}")

    medias = datos_3d.mean(-1, True)
    print(f"Media último eje (mantener_dim=True):")
    print(f"  forma: {medias.forma}  valores: {medias.datos}")

    varianzas = datos_3d.var(-1, True)
    print(f"Varianza último eje (mantener_dim=True):")
    print(f"  forma: {varianzas.forma}  valores: {[round(v, 4) for v in varianzas.datos]}")

    # ── 10. Masked fill (máscara causal) ─────────────────────────────────────
    print(f"\n{SEPARADOR}")
    print("10. Masked Fill (Máscara Causal)")
    print(SEPARADOR)

    scores  = T([1.0, 2.0, 3.0, 4.0], [2, 2])
    mascara = T([0.0, 1.0,   # pos 0 no puede ver la pos 1
                 0.0, 0.0],  # pos 1 puede ver todo
                [2, 2])
    enmascarado = scores.masked_fill(mascara, float("-inf"))

    print(f"\nScores de atención [2, 2]: {scores.datos}")
    print(f"Máscara causal:            {mascara.datos}")
    print(f"Scores enmascarados:       {enmascarado.datos}")
    print(f"  (posiciones futuras → -inf)")

    # ── Resumen ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Resumen")
    print(f"{'='*70}")
    print("\n✓ Creación:          new, ceros, arange")
    print("✓ Álgebra lineal:    matmul (secuencial + paralelo cache-blocked + SIMD)")
    print("✓ Elemento a elem:   add, sub, mul, div, sqrt")
    print("✓ Escalares:         add_scalar, mul_scalar, div_scalar")
    print("✓ Broadcasting:      última dimensión y por lotes")
    print("✓ Activaciones:      softmax numéricamente estable")
    print("✓ Forma:             reshape, transpose")
    print("✓ Estadísticas:      mean, var")
    print("✓ Enmascaramiento:   masked_fill (máscara causal para atención)\n")
