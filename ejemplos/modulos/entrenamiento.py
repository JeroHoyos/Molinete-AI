"""
modulos/entrenamiento.py
━━━━━━━━━━━━━━━━━━━━━━━━
Ejemplos 05-09 — Entrenamiento de modelos GPT-2.

El corpus central es cervantes.txt. Si no existe, el usuario puede
indicar cualquier .txt propio. Todo el bucle de entrenamiento,
tokenización y generación se ejecuta en Rust (molineteai).

Exporta:
    run_05_diminuto()
    run_06_pequeno()
    run_07_mediano()
    run_08_gpt2()
    run_entrenar_presets()
"""

import os
import time

from modulos.ui    import titulo, pedir_input, barra_progreso, SEPARADOR
from modulos.datos import elegir_corpus, verificar_corpus, es_corpus_cervantes

# Prompts de generación según el corpus
PROMPTS_CERVANTES = [
    ("En un lugar de la Mancha", 0.8),
    ("Dulcinea del Toboso",      0.8),
    ("En un lugar de la Mancha", 0.3),
    ("Sancho Panza respondió",   1.2),
]
PROMPTS_CUSTOM = [
    ("El",    0.8),
    ("En",    0.8),
    ("El",    0.3),
    ("Una",   1.2),
]


def _verificar_molineteai() -> bool:
    try:
        import molineteai  # noqa: F401
        return True
    except ImportError:
        print("\n⚠️  El módulo 'molineteai' no está instalado.")
        print("   Compílalo con:  maturin develop --release\n")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Helper compartido para opciones 05-08
# ─────────────────────────────────────────────────────────────────────────────

def _entrenar_modelo(
    nombre_modelo: str,
    config_fn,
    vocab: int,
    max_chars: int | None,
    pasos: int,
    lr: float,
    paciencia: int,
):
    """
    Entrena un modelo GPT-2 usando el bucle de entrenamiento de Rust.

    Todo sucede en molineteai (Rust):
      - Tokenización BPE
      - División train/val
      - Warmup de LR + decaimiento coseno
      - Recorte de gradientes
      - Early stopping
      - Checkpoints y logging CSV
      - Generación de texto
    """
    if not _verificar_molineteai():
        return

    import molineteai

    ruta_corpus = elegir_corpus()
    if not ruta_corpus or not verificar_corpus(ruta_corpus):
        return

    es_cervantes = es_corpus_cervantes(ruta_corpus)

    marca = int(time.time())
    dir_s = f"data/{nombre_modelo}_{marca}"
    os.makedirs(dir_s, exist_ok=True)
    print(f"\nDirectorio de salida: {dir_s}/\n")

    # ── Cargar corpus ────────────────────────────────────────────────────────
    with open(ruta_corpus, encoding="utf-8") as f:
        texto_completo = f.read()
    texto = texto_completo[:max_chars] if max_chars else texto_completo
    print(f"Corpus: {len(texto):,} bytes ({len(texto)/1e6:.2f} MB)")

    # ── Tokenizador BPE (Rust) ────────────────────────────────────────────────
    print(f"\nEntrenando BPE (vocab={vocab})...")
    barra_progreso("Tokenizador", segundos=0.5)
    tok = molineteai.TokenizadorBPE(vocab)
    tok.entrenar(texto, vocab)
    print(f"✓ Vocabulario: {tok.tam_vocabulario()} tokens")
    tok.analizar_vocabulario(texto)
    tok.guardar(f"{dir_s}/tokenizador.json")

    ids = tok.codificar(texto)
    print(f"Tokens: {len(ids):,}  (compresión {len(texto)/len(ids):.2f}x)")

    # ── Modelo (Rust) ─────────────────────────────────────────────────────────
    cfg = config_fn(tok.tam_vocabulario())
    modelo = molineteai.GPT2Entrenable(cfg)
    n = molineteai.contar_parametros_config(cfg)
    print(f"\nModelo: {repr(modelo)}")
    print(f"Parámetros: {n:,} ({n/1e6:.2f}M) — Memoria estimada: ~{n*4/1e6:.0f} MB")

    # ── Entrenamiento (Rust) ──────────────────────────────────────────────────
    print(f"\nEntrenando {pasos:,} pasos (LR={lr}, paciencia={paciencia})...")
    modelo.entrenar(
        tok, texto,
        pasos=pasos,
        tasa_aprendizaje=lr,
        long_secuencia=cfg.tam_bloque,
        dir_salida=dir_s,
        paciencia=paciencia,
        fraccion_calentamiento=0.1,
        norma_recorte=1.0,
        fraccion_validacion=0.1,
        decaimiento_peso=0.01,
    )

    # ── Generación (Rust) ─────────────────────────────────────────────────────
    print(f"\n{'─'*70}\nGeneración de texto\n{'─'*70}")
    prompts = PROMPTS_CERVANTES if es_cervantes else PROMPTS_CUSTOM
    for prompt, temp in prompts:
        ids_p = tok.codificar(prompt)
        ids_g = modelo.generar(ids_p, 80, temp)
        print(f"\n── \"{prompt}\" (temp={temp}) ──")
        print(tok.decodificar(ids_g))

    print(f"\n✅ Archivos en {dir_s}/")
    print("  ├── tokenizador.json")
    print("  ├── punto_control_mejor.bin")
    print("  └── registro_entrenamiento.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Opciones individuales de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def run_05_diminuto():
    titulo("05 — GPT-2 Diminuto")
    print("~50K parámetros | Tiempo estimado: 3-8 minutos\n")
    import molineteai
    _entrenar_modelo(
        "diminuto", molineteai.Config.diminuta,
        vocab=512, max_chars=200_000,
        pasos=3000, lr=3e-3, paciencia=3000,
    )


def run_06_pequeno():
    titulo("06 — GPT-2 Pequeño")
    print("~200K parámetros | Tiempo estimado: 10-20 minutos\n")
    import molineteai
    _entrenar_modelo(
        "pequeno", molineteai.Config.pequena,
        vocab=1024, max_chars=500_000,
        pasos=100_000, lr=2e-3, paciencia=5000,
    )


def run_07_mediano():
    titulo("07 — GPT-2 Mediano")
    print("~4M parámetros | Tiempo estimado: 1-3 horas\n")
    print("💡 Tip: ejecuta en segundo plano con:  nohup python molineteai.py > log.txt &\n")
    import molineteai
    _entrenar_modelo(
        "mediano", molineteai.Config.mediana,
        vocab=1536, max_chars=None,
        pasos=100_000, lr=3e-4, paciencia=5000,
    )


def run_08_gpt2():
    titulo("08 — GPT-2 Small Completo")
    print("~163M parámetros | Tiempo estimado: 4-12 horas\n")
    print("⚠️  Requiere mucha RAM. Considera usar nohup para ejecución en segundo plano.\n")
    confirmar = pedir_input("¿Continuar? (s/n): ", "n")
    if confirmar.lower() != "s":
        print("Cancelado.")
        return
    import molineteai
    _entrenar_modelo(
        "gpt2_small", molineteai.Config.gpt2_small,
        vocab=20534, max_chars=None,
        pasos=200_000, lr=1e-4, paciencia=10000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entrenador con presets (opción 09)
# ─────────────────────────────────────────────────────────────────────────────

#  (embd, capas, cab, ctx, vocab, lr, pasos, paciencia, descripción)
PRESETS_INFO = {
    "pocket-bard":   (256, 6, 8,  448, 8192, 3e-4, 50000, 5000, "~9M params — mejor config desde cero"),
    "cyclops":       ( 64, 2, 1,   64, 8192, 1e-3, 30000, 5000, "Una cabeza de alta resolución"),
    "spider":        ( 64, 2, 8,   64, 8192, 1e-3, 30000, 5000, "Ocho cabezas de baja resolución"),
    "wide":          (256, 4, 4, 1024, 8192, 3e-4, 30000, 5000, "Ancho y superficial"),
    "narrow":        (128, 6, 1, 1024, 8192, 3e-4, 30000, 5000, "Estrecho y profundo"),
    "short-context": (256, 4, 4,  128, 8192, 3e-4, 30000, 5000, "Contexto corto (128 tokens)"),
    "long-context":  (256, 4, 4, 1024, 8192, 3e-4, 30000, 5000, "Contexto largo (1024 tokens)"),
}


def run_entrenar_presets():
    titulo("09 — Entrenador con Presets")
    if not _verificar_molineteai():
        return

    import molineteai

    print("Presets disponibles:\n")
    print(f"  {'Preset':<16} {'Embd':>5} {'Capas':>6} {'Cab':>4} {'Ctx':>5} {'Vocab':>6}   Descripción")
    print("  " + "─" * 75)
    for nombre, (embd, capas, cab, ctx, vocab, lr, pasos, pac, desc) in PRESETS_INFO.items():
        print(f"  {nombre:<16} {embd:>5} {capas:>6} {cab:>4} {ctx:>5} {vocab:>6}   {desc}")

    print()
    preset = pedir_input("Nombre del preset (o Enter para cancelar): ")
    if not preset or preset not in PRESETS_INFO:
        if preset:
            print(f"Preset '{preset}' no encontrado.")
        return

    embd, capas, cab, ctx, vocab, lr, pasos, pac, _ = PRESETS_INFO[preset]

    print(f"\nParámetros del preset '{preset}':")
    print(f"  embd={embd}, capas={capas}, cabezas={cab}, contexto={ctx}, vocab={vocab}")
    print(f"  lr={lr}, pasos={pasos}, paciencia={pac}")

    custom_pasos = pedir_input(f"\nPasos (Enter para usar {pasos}): ")
    if custom_pasos:
        try: pasos = int(custom_pasos)
        except ValueError: pass

    custom_lr = pedir_input(f"Learning rate (Enter para usar {lr}): ")
    if custom_lr:
        try: lr = float(custom_lr)
        except ValueError: pass

    ruta_corpus = elegir_corpus()
    if not ruta_corpus or not verificar_corpus(ruta_corpus):
        return

    es_cervantes = es_corpus_cervantes(ruta_corpus)

    if embd % cab != 0:
        print(f"⚠️  n_embd ({embd}) no es divisible por n_cabezas ({cab})")
        return

    marca = int(time.time())
    dir_s = f"data/{preset.replace('-', '_')}_{marca}"
    os.makedirs(dir_s, exist_ok=True)
    print(f"\nDirectorio de salida: {dir_s}/\n")

    with open(ruta_corpus, encoding="utf-8") as f:
        texto = f.read()
    print(f"Corpus: {len(texto)/1e6:.2f} MB")

    # Tokenizador BPE (Rust)
    barra_progreso(f"Tokenizador (vocab={vocab})", segundos=0.5)
    tok = molineteai.TokenizadorBPE(vocab)
    tok.entrenar(texto, vocab)
    print(f"✓ {tok.tam_vocabulario()} tokens")
    tok.guardar(f"{dir_s}/tokenizador.json")

    # Modelo (Rust)
    cfg = molineteai.Config(
        tam_vocabulario=tok.tam_vocabulario(),
        n_embd=embd, n_capas=capas, n_cabezas=cab,
        tam_bloque=ctx, tasa_dropout=0.1,
    )
    modelo = molineteai.GPT2Entrenable(cfg)
    n = molineteai.contar_parametros_config(cfg)
    print(f"Parámetros: {n:,} ({n/1e6:.2f}M)")

    # Entrenamiento (Rust)
    modelo.entrenar(
        tok, texto,
        pasos=pasos, tasa_aprendizaje=lr,
        long_secuencia=ctx, dir_salida=dir_s, paciencia=pac,
        fraccion_calentamiento=0.1, norma_recorte=1.0,
        fraccion_validacion=0.1, decaimiento_peso=0.01,
    )

    # Generación (Rust)
    print(f"\n{'─'*70}\nGeneración\n{'─'*70}")
    prompts = PROMPTS_CERVANTES[:2] if es_cervantes else PROMPTS_CUSTOM[:2]
    for prompt, temp in prompts:
        ids_g = modelo.generar(tok.codificar(prompt), 60, temp)
        print(f"\n── \"{prompt}\" (t={temp}) ──\n{tok.decodificar(ids_g)[:150]}")

    print(f"\n✅ Archivos en {dir_s}/")
