"""
modulos/chat.py
━━━━━━━━━━━━━━━
Ejemplo 10 — Chat con Modelo Entrenado.

Carga un checkpoint .bin generado por Rust y permite chatear
con el modelo en tiempo real. Todo el procesamiento de tokens
y la generación ocurren en Rust (molineteai).

Soporta comandos: salir, temp X, max X.

Exporta:
    run_chat()
"""

import os
import threading
import time

from modulos.ui import titulo, pedir_input, barra_progreso, imprimir_lento


def _verificar_molineteai() -> bool:
    try:
        import molineteai  # noqa: F401
        return True
    except ImportError:
        print("\n⚠️  El módulo 'molineteai' no está instalado.")
        print("   Compílalo con:  maturin develop --release\n")
        return False


def run_chat():
    titulo("10 — Chat con Modelo Entrenado")
    if not _verificar_molineteai():
        return

    import molineteai

    # ── Buscar checkpoints disponibles ──────────────────────────────────────
    checkpoints = []
    if os.path.exists("data"):
        for root, _, files in os.walk("data"):
            for f in files:
                if f.endswith(".bin"):
                    checkpoints.append(os.path.join(root, f))
    checkpoints.sort()

    if checkpoints:
        print("Checkpoints encontrados:\n")
        for i, ck in enumerate(checkpoints):
            print(f"  [{i+1}] {ck}")
        print()
        sel = pedir_input(f"Selecciona un checkpoint (1-{len(checkpoints)}, o escribe la ruta): ")
        try:
            idx = int(sel) - 1
            ruta_ck = checkpoints[idx] if 0 <= idx < len(checkpoints) else sel
        except ValueError:
            ruta_ck = sel
    else:
        ruta_ck = pedir_input("Ruta del checkpoint (.bin): ")

    if not ruta_ck or not os.path.exists(ruta_ck):
        print(f"\n⚠️  Checkpoint no encontrado: '{ruta_ck}'")
        print("Primero entrena un modelo con las opciones 5-9.")
        return

    print(f"\nCargando '{ruta_ck}'...")

    # ── Animación de carga ───────────────────────────────────────────────────
    cargado = threading.Event()

    def animar_carga():
        longitud = 40
        i = 0
        while not cargado.is_set():
            llenos = i % (longitud + 1)
            barra = "█" * llenos + "░" * (longitud - llenos)
            print(f"\rCargando modelo: [{barra}] ", end="", flush=True)
            i += 1
            time.sleep(0.05)
        barra_llena = "█" * longitud
        print(f"\rCargando modelo: [{barra_llena}] ✓", flush=True)

    hilo = threading.Thread(target=animar_carga, daemon=True)
    hilo.start()

    # ── Cargar checkpoint completo (modelo + tokenizador) desde Rust ─────────
    try:
        modelo, tok = molineteai.GPT2Entrenable.cargar_checkpoint(ruta_ck)
    except Exception as e:
        cargado.set()
        hilo.join()
        print(f"\n⚠️  Error al cargar: {e}")
        return

    cargado.set()
    hilo.join()

    # ── Cargar tokenizador si no venía en el checkpoint ──────────────────────
    if tok is None:
        dir_ck = os.path.dirname(ruta_ck)
        ruta_tok = os.path.join(dir_ck, "tokenizador.json")
        if os.path.exists(ruta_tok):
            tok = molineteai.TokenizadorBPE.cargar(ruta_tok)
            print(f"Tokenizador cargado desde: {ruta_tok}")
        else:
            ruta_tok = pedir_input("Ruta del tokenizador .json: ")
            if not os.path.exists(ruta_tok):
                print("⚠️  Tokenizador no encontrado.")
                return
            tok = molineteai.TokenizadorBPE.cargar(ruta_tok)

    print(f"Modelo: {repr(modelo)}")
    print(f"Tokenizador: {repr(tok)}")

    # ── Parámetros de generación ─────────────────────────────────────────────
    temp_str = pedir_input("\nTemperatura de generación (Enter para 0.8): ", "0.8")
    try:
        temperatura = float(temp_str)
    except ValueError:
        temperatura = 0.8

    max_tok_str = pedir_input("Máximo de tokens a generar (Enter para 100): ", "100")
    try:
        max_tok = int(max_tok_str)
    except ValueError:
        max_tok = 100

    print()
    imprimir_lento(
        "¡Modelo listo! Escribe tu prompt. Comandos: 'salir', 'temp X', 'max X'\n",
        ms_por_letra=12,
    )

    # ── Bucle de chat ─────────────────────────────────────────────────────────
    while True:
        try:
            entrada = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not entrada:
            continue

        if entrada.lower() in ("salir", "exit", "quit"):
            imprimir_lento("¡Hasta pronto!", ms_por_letra=30)
            break

        if entrada.lower().startswith("temp "):
            try:
                temperatura = float(entrada.split()[1])
                print(f"  → Temperatura cambiada a {temperatura}")
            except (IndexError, ValueError):
                print("  → Uso: temp 0.8")
            continue

        if entrada.lower().startswith("max "):
            try:
                max_tok = int(entrada.split()[1])
                print(f"  → Máximo de tokens cambiado a {max_tok}")
            except (IndexError, ValueError):
                print("  → Uso: max 100")
            continue

        # Codificar prompt, generar y decodificar — todo en Rust
        ids_prompt = tok.codificar(entrada)
        ids_gen    = modelo.generar(ids_prompt, max_tok, temperatura)
        texto_gen  = tok.decodificar(ids_gen)

        print("Molinete: ", end="", flush=True)
        imprimir_lento(texto_gen, ms_por_letra=30)
        print()
