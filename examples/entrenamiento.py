"""
modulos/entrenamiento.py
━━━━━━━━━━━━━━━━━━━━━━━━
Ejemplos 05-09 — Entrenamiento de modelos GPT-2.

Contiene el helper compartido _entrenar_modelo() y las funciones
individuales para cada tamaño de modelo y el entrenador con presets.

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
from modulos.datos import verificar_corpus


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

def _entrenar_modelo(nombre_modelo: str, config_fn, vocab: int,
                     max_chars: int | None, pasos: int,
                     lr: float, paciencia: int):
    """Helper compartido para los ejemplos de entrenamiento 05-08."""
    if not _verificar_molineteai() or not verificar_corpus():
        return

    import molineteai

    marca = int(time.time())
    dir_s = f"data/shakespeare_{nombre_modelo}_{marca}"
    os.makedirs(dir_s, exist_ok=True)
    print(f"\nDirectorio de salida: {dir_s}/\n")

    with open("shakespeare.txt", encoding="utf-8") as f:
        texto_completo = f.read()

    texto = texto_completo[:max_chars] if max_chars else texto_completo
    print(f"Corpus: {len(texto):,} bytes ({len(texto)/1e6:.2f} MB)")

    print(f"\nEntrenando BPE (vocab={vocab})...")
    barra_progreso("Tokenizador", segundos=0.5)
    tok = molineteai.TokenizadorBPE(vocab)
    tok.entrenar(texto, vocab)
    print(f"✓ Vocabulario: {tok.tam_vocabulario()} tokens")
    tok.analizar_vocabulario(texto)
    tok.guardar(f"{dir_s}/tokenizador.json")

    ids = tok.codificar(texto)
    print(f"Tokens: {len(ids):,}  (compresión {len(texto)/len(ids):.2f}x)")

    cfg = config_fn(tok.tam_vocabulario())
    modelo = molineteai.GPT2Entrenable(cfg)
    n = modelo.num_parametros()
    print(f"\nModelo: vocab={cfg.tam_vocabulario}, embd={cfg.n_embd}, "
          f"capas={cfg.n_capas}, cabezas={cfg.n_cabezas}")
    print(f"Parámetros: {n:,} ({n/1e6:.2f}M) — Memoria: ~{n*4/1e6:.0f} MB")

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

    print(f"\n{'─'*70}\nGeneración de texto\n{'─'*70}")
    prompts = [
        ("To be, or not to be", 0.8),
        ("ROMEO.", 0.8),
        ("The king", 0.8),
        ("To be, or not to be", 0.3),
        ("To be, or not to be", 1.2),
    ]
    for prompt, temp in prompts:
        ids_p = tok.codificar(prompt)
        ids_g = modelo.generar(ids_p, 80, temp)
        print(f"\n── \"{prompt}\" (temp={temp}) ──")
        print(tok.decodificar(ids_g))

    print(f"\n✅ Archivos en {dir_s}/")
    print("  ├── tokenizador.json")
    print("  ├── checkpoint_best.bin")
    print("  └── registro_entrenamiento.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Opciones individuales de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def run_05_diminuto():
    titulo("05 — GPT-2 Diminuto en Shakespeare")
    print("~170K parámetros | Tiempo estimado: 3-8 minutos\n")
    import molineteai
    _entrenar_modelo(
        "diminuto", molineteai.Config.diminuta,
        vocab=512, max_chars=200_000,
        pasos=3000, lr=3e-3, paciencia=3000,
    )


def run_06_pequeno():
    titulo("06 — GPT-2 Pequeño en Shakespeare")
    print("~200K parámetros | Tiempo estimado: 10-20 minutos\n")
    if not _verificar_molineteai() or not verificar_corpus():
        return

    import molineteai

    marca = int(time.time())
    dir_s = f"data/shakespeare_pequeno_{marca}"
    os.makedirs(dir_s, exist_ok=True)
    print(f"Directorio de salida: {dir_s}/\n")

    with open("shakespeare.txt", encoding="utf-8") as f:
        texto = f.read(500_000)
    print(f"Corpus: {len(texto):,} bytes")

    barra_progreso("Tokenizador (vocab=1024)", segundos=0.5)
    tok = molineteai.TokenizadorBPE(1024)
    tok.entrenar(texto, 1024)
    print(f"✓ {tok.tam_vocabulario()} tokens")
    tok.analizar_vocabulario(texto)
    tok.guardar(f"{dir_s}/tokenizador.json")

    config = molineteai.Config(
        tam_vocabulario=tok.tam_vocabulario(),
        n_embd=128, n_capas=3, n_cabezas=1,
        tam_bloque=128, tasa_dropout=0.1,
    )
    modelo = molineteai.GPT2Entrenable(config)
    print(f"Parámetros: {modelo.num_parametros():,}")

    modelo.entrenar(
        tok, texto, pasos=100_000, tasa_aprendizaje=2e-3,
        long_secuencia=128, dir_salida=dir_s, paciencia=5000,
        fraccion_calentamiento=0.1, norma_recorte=1.0,
        fraccion_validacion=0.1, decaimiento_peso=0.01,
    )

    print(f"\n{'─'*70}\nGeneración\n{'─'*70}")
    for prompt, temp in [("To be, or not to be", 0.8), ("ROMEO.", 0.8), ("O Romeo", 0.8)]:
        ids_g = modelo.generar(tok.codificar(prompt), 80, temp)
        print(f"\n── \"{prompt}\" (t={temp}) ──\n{tok.decodificar(ids_g)}")
    print(f"\n✅ Archivos en {dir_s}/")


def run_07_mediano():
    titulo("07 — GPT-2 Mediano en Shakespeare")
    print("~4M parámetros | Tiempo estimado: 1-3 horas\n")
    print("💡 Tip: ejecuta en segundo plano con:  nohup python molineteai.py > log.txt &\n")
    import molineteai
    _entrenar_modelo(
        "mediano", molineteai.Config.mediana,
        vocab=1536, max_chars=None,
        pasos=100_000, lr=3e-4, paciencia=5000,
    )


def run_08_gpt2():
    titulo("08 — GPT-2 Small Completo en Shakespeare")
    print("~163M parámetros | Tiempo estimado: 4-12 horas\n")
    print("⚠️  Requiere mucha RAM. Considera usar nohup para ejecución en segundo plano.\n")
    confirmar = pedir_input("¿Continuar? (s/n): ", "n")
    if confirmar.lower() != "s":
        print("Cancelado.")
        return
    import molineteai
    _entrenar_modelo(
        "gpt2", molineteai.Config.gpt2_small,
        vocab=20534, max_chars=None,
        pasos=200_000, lr=1e-4, paciencia=10000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entrenador con presets (opción 09)
# ─────────────────────────────────────────────────────────────────────────────

#  (embd, capas, cab, ctx, vocab, lr, pasos, paciencia, descripción)
PRESETS_INFO = {
    "pocket-bard":   (256, 6, 8, 448,  8192, 3e-4, 50000, 5000, "~9M params — mejor config desde cero"),
    "cyclops":       ( 64, 2, 1,  64,  8192, 1e-3, 30000, 5000, "Una cabeza de alta resolución"),
    "spider":        ( 64, 2, 8,  64,  8192, 1e-3, 30000, 5000, "Ocho cabezas de baja resolución"),
    "wide":          (256, 4, 4, 1024, 8192, 3e-4, 30000, 5000, "Ancho y superficial"),
    "narrow":        (128, 6, 1, 1024, 8192, 3e-4, 30000, 5000, "Estrecho y profundo"),
    "short-context": (256, 4, 4,  128, 8192, 3e-4, 30000, 5000, "Contexto corto (128 tokens)"),
    "long-context":  (256, 4, 4, 1024, 8192, 3e-4, 30000, 5000, "Contexto largo (1024 tokens)"),
    "tinystories":   (256, 6, 8,  448, 8192, 1e-3, 50000, 2000, "Pre-entrenamiento TinyStories"),
}


def run_entrenar_presets():
    titulo("09 — Entrenador con Presets")

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
    print()

    custom_pasos = pedir_input(f"Pasos (Enter para usar {pasos}): ")
    if custom_pasos:
        try: pasos = int(custom_pasos)
        except ValueError: pass

    custom_lr = pedir_input(f"Learning rate (Enter para usar {lr}): ")
    if custom_lr:
        try: lr = float(custom_lr)
        except ValueError: pass

    datos = pedir_input("Archivo de datos (Enter para shakespeare.txt): ", "shakespeare.txt")

    if not _verificar_molineteai():
        return
    if not os.path.exists(datos):
        print(f"\n⚠️  '{datos}' no encontrado.")
        return

    import molineteai

    if embd % cab != 0:
        print(f"⚠️  n_embd ({embd}) no es divisible por n_cabezas ({cab})")
        return

    marca = int(time.time())
    prefijo = preset.replace("-", "_")
    dir_s = f"data/{prefijo}_{marca}"
    os.makedirs(dir_s, exist_ok=True)
    print(f"\nDirectorio de salida: {dir_s}/\n")

    with open(datos, encoding="utf-8") as f:
        texto = f.read()
    print(f"Corpus: {len(texto)/1e6:.2f} MB")

    barra_progreso(f"Tokenizador (vocab={vocab})", segundos=0.5)
    tok = molineteai.TokenizadorBPE(vocab)
    tok.entrenar(texto, vocab)
    print(f"✓ {tok.tam_vocabulario()} tokens")
    tok.guardar(f"{dir_s}/tokenizador.json")

    config = molineteai.Config(
        tam_vocabulario=tok.tam_vocabulario(),
        n_embd=embd, n_capas=capas, n_cabezas=cab,
        tam_bloque=ctx, tasa_dropout=0.1,
    )
    modelo = molineteai.GPT2Entrenable(config)
    n = modelo.num_parametros()
    print(f"Parámetros: {n:,} ({n/1e6:.2f}M)")

    modelo.entrenar(
        tok, texto, pasos=pasos, tasa_aprendizaje=lr,
        long_secuencia=ctx, dir_salida=dir_s, paciencia=pac,
        fraccion_calentamiento=0.1, norma_recorte=1.0,
        fraccion_validacion=0.1, decaimiento_peso=0.01,
    )

    print(f"\n{'─'*70}\nGeneración\n{'─'*70}")
    for prompt, temp in [("To be, or not to be", 0.8), ("ROMEO.", 0.9)]:
        ids_g = modelo.generar(tok.codificar(prompt), 60, temp)
        print(f"\n── \"{prompt}\" (t={temp}) ──\n{tok.decodificar(ids_g)[:150]}")

    print(f"\n✅ Archivos en {dir_s}/")
