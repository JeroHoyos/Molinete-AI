"""
modules/comparar.py
━━━━━━━━━━━━━━━━━━━
Ejemplo 12 — Comparador de modelos.

Carga hasta 4 paneles a la vez y genera con el mismo prompt en todos,
para ver lado a lado cómo escribe cada tamaño de modelo. El mismo
modelo puede repetirse en varios paneles (se carga una sola vez y se
comparte), cada panel con su propia temperatura.

Protocolo de eventos (__MOL__):
    comp_checkpoints  → lista de modelos disponibles
    comp_loading      → empieza la carga del slot N
    comp_loaded       → slot N cargado (o fallido); incluye temp inicial
    comp_ready        → todos listos, se acepta el primer prompt
    comp_prompt       → prompt recibido, empieza la ronda
    comp_gen          → el slot N está generando
    comp_result       → resultado del slot N (texto + tokens + ids + temp + segs)
    comp_round_done   → ronda completada

Comandos del bucle:
    temp X      → temperatura de todos los paneles
    temp N X    → temperatura solo del panel N (1-based)
    max X       → máximo de tokens (global)
    salir       → terminar

Exporta:
    run_comparar(), _sesion_comparar(), parsear_indices()
"""

import os
import time

from modules.ui import pedir_input, verificar_molineteai, emit
from modules.chat import _buscar_modelos

MAX_MODELOS     = 4
TEMP_DEFECTO    = 0.8
MAX_TOK_DEFECTO = 80


def parsear_indices(sel: str, n_modelos: int) -> list[int]:
    """Convierte "1,3,3" en índices 0-based; admite repetidos (hasta MAX_MODELOS)."""
    indices = []
    for parte in sel.split(","):
        parte = parte.strip()
        if parte.isdigit():
            i = int(parte) - 1
            if 0 <= i < n_modelos:
                indices.append(i)
    return indices[:MAX_MODELOS]


def _sesion_comparar(modelos: list[dict], indices: list[int]) -> None:
    """Carga los paneles indicados y entra en el bucle de rondas de generación."""
    import molineteai

    # Un modelo repetido se carga una sola vez y se comparte entre paneles
    cache = {}
    cargados = []
    for slot, i in enumerate(indices):
        m = modelos[i]
        emit("comp_loading", slot=slot, display=m["display"], nombre=m["nombre"])
        print(f"Cargando [{slot + 1}/{len(indices)}] {m['nombre']}...")
        try:
            if i in cache:
                modelo, tok = cache[i]
            else:
                ruta = m["ck_mejor"] or m["ck_ultimo"]
                modelo, tok = molineteai.GPT2Entrenable.cargar_checkpoint(ruta)
                if tok is None:
                    ruta_tok = os.path.join(os.path.dirname(ruta), "tokenizador.json")
                    tok = molineteai.TokenizadorBPE.cargar(ruta_tok)
                cache[i] = (modelo, tok)
            cargados.append({"slot": slot, "modelo": modelo, "tok": tok, "temp": TEMP_DEFECTO})
            emit("comp_loaded", slot=slot, ok=True,
                 display=m["display"], nombre=m["nombre"],
                 pasos=m.get("pasos"), mejor_val=m.get("mejor_val"),
                 mejor_perp=m.get("mejor_perp"), vocab=tok.tam_vocabulario(),
                 temp=TEMP_DEFECTO)
        except Exception as e:
            print(f"⚠️  Error al cargar '{m['nombre']}': {e}")
            emit("comp_loaded", slot=slot, ok=False,
                 display=m["display"], nombre=m["nombre"], error=str(e))

    if not cargados:
        emit("chat_info", text="Ningún modelo se pudo cargar.")
        return

    max_tok = MAX_TOK_DEFECTO
    emit("comp_ready", n=len(cargados), temperatura=TEMP_DEFECTO, max_tok=max_tok)

    # ── Bucle de comparación ─────────────────────────────────────────────────
    while True:
        try:
            entrada = input("").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not entrada:
            continue

        if entrada.lower() in ("salir", "exit", "quit"):
            emit("chat_info", text="Comparación finalizada.")
            break

        partes = entrada.split()

        if partes[0].lower() == "temp":
            # "temp 0.9" → todos los paneles; "temp 2 0.9" → solo el panel 2
            try:
                if len(partes) == 2:
                    t = float(partes[1])
                    for c in cargados:
                        c["temp"] = t
                    emit("chat_info", text=f"Temperatura de todos los paneles → {t}")
                elif len(partes) == 3:
                    slot = int(partes[1]) - 1
                    t = float(partes[2])
                    for c in cargados:
                        if c["slot"] == slot:
                            c["temp"] = t
                            break
                else:
                    raise ValueError
            except ValueError:
                emit("chat_info", text="Uso: temp 0.8  |  temp 2 0.8")
            continue

        if partes[0].lower() == "max":
            try:
                max_tok = int(partes[1])
                emit("chat_info", text=f"Máximo de tokens → {max_tok}")
            except (IndexError, ValueError):
                emit("chat_info", text="Uso: max 80")
            continue

        emit("comp_prompt", text=entrada)

        for c in cargados:
            slot = c["slot"]
            emit("comp_gen", slot=slot)
            tok, modelo = c["tok"], c["modelo"]
            ids_prompt = tok.codificar(entrada)
            t0 = time.perf_counter()
            ids_gen = modelo.generar(ids_prompt, max_tok, c["temp"])
            segs = round(time.perf_counter() - t0, 1)

            if len(ids_gen) >= len(ids_prompt) and ids_gen[:len(ids_prompt)] == ids_prompt:
                ids_cont = ids_gen[len(ids_prompt):]
                emit("comp_result", slot=slot,
                     prompt=tok.decodificar(ids_prompt),
                     text=tok.decodificar(ids_cont),
                     tokens=[tok.decodificar([i]) for i in ids_cont],
                     tokens_prompt=[tok.decodificar([i]) for i in ids_prompt],
                     ids=list(ids_cont),
                     ids_prompt=list(ids_prompt),
                     temp=c["temp"], segs=segs)
            else:
                emit("comp_result", slot=slot,
                     text=tok.decodificar(ids_gen),
                     tokens=[tok.decodificar([i]) for i in ids_gen],
                     ids=list(ids_gen),
                     temp=c["temp"], segs=segs)

        emit("comp_round_done")
        print()


def run_comparar():
    if not verificar_molineteai():
        return

    modelos = _buscar_modelos()
    emit("comp_checkpoints", modelos=[
        {k: v for k, v in m.items() if k != "carpeta"}
        for m in modelos
    ])

    if not modelos:
        print("\nNo se encontraron modelos entrenados en data/.")
        return

    # El frontend envía los índices elegidos, ej. "1,3,3"
    sel = pedir_input("")
    indices = parsear_indices(sel, len(modelos))

    if not indices:
        emit("chat_info", text="Selección vacía: no hay modelos que comparar.")
        return

    _sesion_comparar(modelos, indices)
