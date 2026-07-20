"""
modules/comparar.py
━━━━━━━━━━━━━━━━━━━
Ejemplo 12 — Comparador de modelos.

Carga hasta 4 checkpoints a la vez y genera con el mismo prompt en
todos, para ver lado a lado cómo escribe cada tamaño de modelo.

Protocolo de eventos (__MOL__):
    comp_checkpoints  → lista de modelos disponibles
    comp_loading      → empieza la carga del slot N
    comp_loaded       → slot N cargado (o fallido)
    comp_ready        → todos listos, se acepta el primer prompt
    comp_prompt       → prompt recibido, empieza la ronda
    comp_gen          → el slot N está generando
    comp_result       → resultado del slot N (texto + tokens + ids)
    comp_round_done   → ronda completada

Exporta:
    run_comparar()
"""

import os

from modules.ui import pedir_input, verificar_molineteai, emit
from modules.chat import _buscar_modelos

MAX_MODELOS = 4


def run_comparar():
    if not verificar_molineteai():
        return

    import molineteai

    modelos = _buscar_modelos()
    emit("comp_checkpoints", modelos=[
        {k: v for k, v in m.items() if k != "carpeta"}
        for m in modelos
    ])

    if not modelos:
        print("\nNo se encontraron modelos entrenados en data/.")
        return

    # El frontend envía los índices elegidos, ej. "1,3,4"
    sel = pedir_input("")
    indices = []
    for parte in sel.split(","):
        parte = parte.strip()
        if parte.isdigit():
            i = int(parte) - 1
            if 0 <= i < len(modelos) and i not in indices:
                indices.append(i)
    indices = indices[:MAX_MODELOS]

    if not indices:
        emit("chat_info", text="Selección vacía: no hay modelos que comparar.")
        return

    # ── Cargar los checkpoints seleccionados ─────────────────────────────────
    cargados = []
    for slot, i in enumerate(indices):
        m = modelos[i]
        ruta = m["ck_mejor"] or m["ck_ultimo"]
        emit("comp_loading", slot=slot, display=m["display"], nombre=m["nombre"])
        print(f"Cargando [{slot + 1}/{len(indices)}] {m['nombre']}...")
        try:
            modelo, tok = molineteai.GPT2Entrenable.cargar_checkpoint(ruta)
            if tok is None:
                ruta_tok = os.path.join(os.path.dirname(ruta), "tokenizador.json")
                tok = molineteai.TokenizadorBPE.cargar(ruta_tok)
            cargados.append({"slot": slot, "modelo": modelo, "tok": tok})
            emit("comp_loaded", slot=slot, ok=True,
                 display=m["display"], nombre=m["nombre"],
                 pasos=m.get("pasos"), mejor_val=m.get("mejor_val"),
                 mejor_perp=m.get("mejor_perp"), vocab=tok.tam_vocabulario())
        except Exception as e:
            print(f"⚠️  Error al cargar '{ruta}': {e}")
            emit("comp_loaded", slot=slot, ok=False,
                 display=m["display"], nombre=m["nombre"], error=str(e))

    if not cargados:
        emit("chat_info", text="Ningún modelo se pudo cargar.")
        return

    temperatura = 0.8
    max_tok = 80

    emit("comp_ready", n=len(cargados), temperatura=temperatura, max_tok=max_tok)

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

        if entrada.lower().startswith("temp "):
            try:
                temperatura = float(entrada.split()[1])
                emit("chat_info", text=f"Temperatura → {temperatura}")
            except (IndexError, ValueError):
                emit("chat_info", text="Uso: temp 0.8")
            continue

        if entrada.lower().startswith("max "):
            try:
                max_tok = int(entrada.split()[1])
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
            ids_gen = modelo.generar(ids_prompt, max_tok, temperatura)

            if len(ids_gen) >= len(ids_prompt) and ids_gen[:len(ids_prompt)] == ids_prompt:
                ids_cont = ids_gen[len(ids_prompt):]
                emit("comp_result", slot=slot,
                     prompt=tok.decodificar(ids_prompt),
                     text=tok.decodificar(ids_cont),
                     tokens=[tok.decodificar([i]) for i in ids_cont],
                     tokens_prompt=[tok.decodificar([i]) for i in ids_prompt],
                     ids=list(ids_cont),
                     ids_prompt=list(ids_prompt))
            else:
                emit("comp_result", slot=slot,
                     text=tok.decodificar(ids_gen),
                     tokens=[tok.decodificar([i]) for i in ids_gen],
                     ids=list(ids_gen))

        emit("comp_round_done")
        print()
