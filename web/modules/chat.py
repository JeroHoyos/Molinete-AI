"""
modules/chat.py
━━━━━━━━━━━━━━━
Ejemplo 10 — Chat con Modelo Entrenado.

Carga un checkpoint .bin generado por Rust y permite chatear
con el modelo en tiempo real. Todo el procesamiento de tokens
y la generación ocurren en Rust (molineteai).

Soporta comandos: salir, temp X, max X.

Exporta:
    run_chat()
"""

import csv
import math
import os
import shutil
import threading
import time

from modules.ui import pedir_input, verificar_molineteai, emit

# Mapeo de prefijos de carpeta a nombres de display (por tamaño en parámetros)
_NOMBRES_DISPLAY = {
    "diminuto":    "GPT-2 50K",
    "pequeno":     "GPT-2 200K",
    "mediano":     "GPT-2 4M",
    "gpt2_small":  "GPT-2 163M",
    "pocket_bard": "Pocket Bard",
    "cyclops":     "Cyclops",
    "spider":      "Spider",
    "wide":        "Wide",
    "narrow":      "Narrow",
    "short_context": "Short Context",
    "long_context":  "Long Context",
}


def _prefijo_carpeta(nombre: str) -> str:
    """Extrae el prefijo antes del timestamp final (ej. 'diminuto_1234' → 'diminuto')."""
    partes = nombre.rsplit("_", 1)
    if len(partes) == 2 and partes[1].isdigit():
        return partes[0]
    return nombre


_MESES = ("ene", "feb", "mar", "abr", "may", "jun",
          "jul", "ago", "sep", "oct", "nov", "dic")


def _fecha_carpeta(nombre: str) -> str | None:
    """Fecha legible a partir del timestamp del nombre ('diminuto_1751234567')."""
    partes = nombre.rsplit("_", 1)
    if len(partes) != 2 or not partes[1].isdigit():
        return None
    try:
        t = time.localtime(int(partes[1]))
    except (ValueError, OverflowError, OSError):
        return None
    return f"{t.tm_mday} {_MESES[t.tm_mon - 1]} · {t.tm_hour:02d}:{t.tm_min:02d}"


def _nombre_display(nombre: str) -> str:
    prefix = _prefijo_carpeta(nombre)
    return _NOMBRES_DISPLAY.get(prefix, prefix.replace("_", " ").title())


def _leer_stats(carpeta: str) -> dict:
    """Lee registro_entrenamiento.csv y devuelve estadísticas resumidas."""
    stats = {"pasos": 0, "mejor_val": None, "mejor_perp": None, "pasos_totales": None}
    csv_path = os.path.join(carpeta, "registro_entrenamiento.csv")
    if not os.path.exists(csv_path):
        return stats
    try:
        with open(csv_path, encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        if not filas:
            return stats
        stats["pasos"] = int(filas[-1]["paso"])
        vals = []
        for r in filas:
            v = r.get("perdida_validacion", "").strip()
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        if vals:
            mejor = min(vals)
            stats["mejor_val"] = round(mejor, 4)
            stats["mejor_perp"] = round(math.exp(mejor), 1)
    except Exception:
        pass
    return stats


def _buscar_modelos() -> list[dict]:
    """Escanea data/ y devuelve lista de modelos disponibles con metadatos."""
    modelos = []
    if not os.path.exists("data"):
        return modelos
    for nombre in sorted(os.listdir("data")):
        carpeta = os.path.join("data", nombre)
        if not os.path.isdir(carpeta):
            continue
        ck_mejor  = os.path.join(carpeta, "punto_control_mejor.bin")
        ck_ultimo = os.path.join(carpeta, "punto_control_ultimo.bin")
        # Compatibilidad con nombre anterior
        ck_final  = os.path.join(carpeta, "punto_control_final.bin")

        tiene_mejor  = os.path.exists(ck_mejor)
        tiene_ultimo = os.path.exists(ck_ultimo) or os.path.exists(ck_final)

        if not tiene_mejor and not tiene_ultimo:
            continue

        stats = _leer_stats(carpeta)
        modelos.append({
            "idx":          len(modelos) + 1,
            "nombre":       nombre,
            "display":      _nombre_display(nombre),
            "fecha":        _fecha_carpeta(nombre),
            "carpeta":      carpeta,
            "ck_mejor":     ck_mejor  if tiene_mejor  else None,
            "ck_ultimo":    ck_ultimo if os.path.exists(ck_ultimo) else (ck_final if os.path.exists(ck_final) else None),
            "tiene_mejor":  tiene_mejor,
            "tiene_ultimo": tiene_ultimo,
            **stats,
        })
    return modelos


def _emitir_checkpoints(modelos):
    emit("chat_checkpoints", modelos=[
        {k: v for k, v in m.items() if k != "carpeta"}  # no exponer rutas internas
        for m in modelos
    ])


def run_chat():
    if not verificar_molineteai():
        return

    import molineteai

    # ── Buscar modelos disponibles ───────────────────────────────────────────
    modelos = _buscar_modelos()
    _emitir_checkpoints(modelos)

    ruta_ck = None
    m_sel = None

    if modelos:
        # El frontend muestra las tarjetas con los modelos vía chat_checkpoints.
        # Bloqueamos stdin esperando la selección; "borrar N" elimina la
        # carpeta del modelo N y reemite la lista actualizada.
        while True:
            sel = pedir_input("").strip()

            if sel.lower().startswith("borrar "):
                arg = sel.split(None, 1)[1].strip()
                if arg.isdigit() and 1 <= int(arg) <= len(modelos):
                    m = modelos[int(arg) - 1]
                    try:
                        shutil.rmtree(m["carpeta"])
                        print(f"Borrado: {m['carpeta']}/")
                        emit("chat_info", text=f"Modelo '{m['nombre']}' borrado.")
                    except Exception as e:
                        emit("chat_info", text=f"No se pudo borrar '{m['nombre']}': {e}")
                    modelos = _buscar_modelos()
                    _emitir_checkpoints(modelos)
                    if not modelos:
                        print("No quedan modelos entrenados.")
                        return
                continue

            try:
                idx = int(sel) - 1
                if 0 <= idx < len(modelos):
                    m_sel = modelos[idx]
                    # Preferir mejor checkpoint si existe
                    ruta_ck = m_sel["ck_mejor"] or m_sel["ck_ultimo"]
            except ValueError:
                ruta_ck = sel
            break
    else:
        print("\nNo se encontraron modelos entrenados en data/.")
        print("Primero entrena un modelo con las opciones 5–9.\n")
        ruta_ck = pedir_input("O escribe la ruta de un .bin manualmente: ").strip()

    if not ruta_ck or not os.path.exists(ruta_ck):
        print(f"\n⚠️  Checkpoint no encontrado: '{ruta_ck}'")
        emit("chat_info", text=f"Checkpoint no encontrado: '{ruta_ck}'")
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

    try:
        modelo, tok = molineteai.GPT2Entrenable.cargar_checkpoint(ruta_ck)
    except Exception as e:
        cargado.set()
        hilo.join()
        print(f"\n⚠️  Error al cargar: {e}")
        emit("chat_info", text=f"Error al cargar el checkpoint: {e}")
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

    temperatura = 0.8
    max_tok = 100

    m_sel = m_sel or {}
    emit("chat_ready",
         checkpoint=ruta_ck,
         temperatura=temperatura,
         max_tok=max_tok,
         display=m_sel.get("display"),
         nombre=m_sel.get("nombre"),
         pasos=m_sel.get("pasos"),
         mejor_val=m_sel.get("mejor_val"),
         mejor_perp=m_sel.get("mejor_perp"),
         vocab=tok.tam_vocabulario())

    # ── Bucle de chat ─────────────────────────────────────────────────────────
    while True:
        try:
            entrada = input("").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not entrada:
            continue

        if entrada.lower() in ("salir", "exit", "quit"):
            emit("chat_info", text="Chat finalizado.")
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
                emit("chat_info", text="Uso: max 100")
            continue

        ids_prompt = tok.codificar(entrada)
        tokens_prompt = [tok.decodificar([i]) for i in ids_prompt]
        emit("chat_user", text=entrada, tokens=tokens_prompt, ids=list(ids_prompt))

        ids_gen = modelo.generar(ids_prompt, max_tok, temperatura)

        # Separar el prompt de la continuación generada para que el
        # frontend pueda distinguirlos visualmente en la burbuja.
        if len(ids_gen) >= len(ids_prompt) and ids_gen[:len(ids_prompt)] == ids_prompt:
            ids_cont = ids_gen[len(ids_prompt):]
            emit("chat_model",
                 text=tok.decodificar(ids_cont),
                 prompt=tok.decodificar(ids_prompt),
                 tokens=[tok.decodificar([i]) for i in ids_cont],
                 tokens_prompt=tokens_prompt,
                 ids=list(ids_cont),
                 ids_prompt=list(ids_prompt))
        else:
            emit("chat_model",
                 text=tok.decodificar(ids_gen),
                 tokens=[tok.decodificar([i]) for i in ids_gen],
                 ids=list(ids_gen))
        print()
