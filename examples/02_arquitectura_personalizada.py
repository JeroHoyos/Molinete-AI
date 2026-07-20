"""
Ejemplo 2 — Inventar tu propia arquitectura.

En vez de un preset (diminuta, pequena...), aquí el modelo se arma
pieza a pieza con Config(...). Tú decides el tamaño de todo.

Cómo ejecutarlo (desde la raíz del repositorio):
    uv run python examples/02_arquitectura_personalizada.py

Necesitas cervantes.txt en la raíz (se descarga desde la interfaz web).
"""

import molineteai

# ── 1. Corpus y tokenizador (igual que en el ejemplo 1) ──────────
texto = open("cervantes.txt", encoding="utf-8").read()[:200_000]

tok = molineteai.TokenizadorBPE(512)
tok.entrenar(texto, 512)

# ── 2. La arquitectura, a tu gusto ───────────────────────────────
# Única regla: n_embd debe ser divisible por n_cabezas.
# Aquí: 96 / 4 = 24 dimensiones para cada cabeza de atención.
config = molineteai.Config(
    tam_vocabulario=tok.tam_vocabulario(),
    n_embd=96,        # ancho del modelo (dimensión de los embeddings)
    n_capas=3,        # cuántos bloques Transformer se apilan
    n_cabezas=4,      # cabezas de atención por bloque
    tam_bloque=96,    # cuántos tokens ve hacia atrás (contexto)
    tasa_dropout=0.1,
)

modelo = molineteai.GPT2Entrenable(config)
print(f"Tu modelo tiene {molineteai.contar_parametros_config(config):,} parámetros")

# ── 3. Entrenar ──────────────────────────────────────────────────
modelo.entrenar(tok, texto, pasos=800, tasa_aprendizaje=1e-3,
                dir_salida="data/ejemplo_personalizado")

# ── 4. Generar con dos temperaturas ──────────────────────────────
# Temperatura baja = conservador y repetitivo. Alta = creativo y caótico.
for temperatura in (0.5, 1.2):
    ids = modelo.generar(tok.codificar("Dulcinea"), 50, temperatura)
    print(f"\nCon temperatura {temperatura}:")
    print(tok.decodificar(ids))
