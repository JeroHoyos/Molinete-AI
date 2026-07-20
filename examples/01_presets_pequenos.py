"""
Ejemplo 1 — Entrenar los dos modelos más pequeños.

Pasos que verás abajo, siempre los mismos:
  1. Leer el corpus (un texto plano).
  2. Entrenar un tokenizador BPE (convierte texto en números).
  3. Crear el modelo con un preset.
  4. Entrenarlo.
  5. Pedirle que continúe una frase.

Cómo ejecutarlo (desde la raíz del repositorio):
    uv run python examples/01_presets_pequenos.py

Necesitas cervantes.txt en la raíz (se descarga desde la interfaz web).
"""

import molineteai

# ── 1. Leer el corpus ────────────────────────────────────────────
# Con 200.000 caracteres basta para probar (el archivo entero tarda más).
texto = open("cervantes.txt", encoding="utf-8").read()[:200_000]

frase = "En un lugar de la Mancha"


# ═════════════════════════════════════════════════════════════════
#  MODELO A — GPT-2 50K (el más pequeño de todos)
# ═════════════════════════════════════════════════════════════════

# ── 2. Tokenizador con 512 tokens de vocabulario ─────────────────
tok = molineteai.TokenizadorBPE(512)
tok.entrenar(texto, 512)

# ── 3. Crear el modelo con el preset "diminuta" ──────────────────
config = molineteai.Config.diminuta(tok.tam_vocabulario())
modelo = molineteai.GPT2Entrenable(config)

# ── 4. Entrenar (600 pasos: rápido, solo para ver cómo funciona) ─
modelo.entrenar(tok, texto, pasos=600, tasa_aprendizaje=3e-3,
                dir_salida="data/ejemplo_50k")

# ── 5. Generar texto ─────────────────────────────────────────────
ids = modelo.generar(tok.codificar(frase), 60, 0.8)
print("\nGPT-2 50K dice:")
print(tok.decodificar(ids))


# ═════════════════════════════════════════════════════════════════
#  MODELO B — GPT-2 200K (un poco más grande)
# ═════════════════════════════════════════════════════════════════

# Mismos pasos: solo cambian el vocabulario y el preset.
tok = molineteai.TokenizadorBPE(1024)
tok.entrenar(texto, 1024)

config = molineteai.Config.pequena(tok.tam_vocabulario())
modelo = molineteai.GPT2Entrenable(config)

modelo.entrenar(tok, texto, pasos=600, tasa_aprendizaje=2e-3,
                dir_salida="data/ejemplo_200k")

ids = modelo.generar(tok.codificar(frase), 60, 0.8)
print("\nGPT-2 200K dice:")
print(tok.decodificar(ids))
