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

import csv
import math
import os
import threading
import time

from modulos.ui import titulo, pedir_input, barra_progreso, emit

# Mapeo de prefijos de carpeta a nombres de display
_NOMBRES_DISPLAY = {
    "diminuto":    "GPT-2 Diminuto",
    "pequeno":     "GPT-2 Pequeño",
    "mediano":     "GPT-2 Mediano",
    "gpt2_small":  "GPT-2 Small",
    "pocket_bard": "Pocket Bard",
    "cyclops":     "Cyclops",
    "spider":      "Spider",
    "wide":        "Wide",
    "narrow":      "Narrow",
    "short_context": "Short Context",
    "long_context":  "Long Context",
}


def _verificar_molineteai() -> bool:
    try:
        import molineteai  # noqa: F401
        return True
    except ImportError:
        print("\n⚠️  El módulo 'molineteai' no está instalado.")
        print("   Compílalo con:  maturin develop --release\n")
        return False


def _prefijo_carpeta(nombre: str) -> str:
    """Extrae el prefijo antes del timestamp final (ej. 'diminuto_1234' → 'diminuto')."""
    partes = nombre.rsplit("_", 1)
    if len(partes) == 2 and partes[1].isdigit():
        return partes[0]
    return nombre


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
            "carpeta":      carpeta,
            "ck_mejor":     ck_mejor  if tiene_mejor  else None,
            "ck_ultimo":    ck_ultimo if os.path.exists(ck_ultimo) else (ck_final if os.path.exists(ck_final) else None),
            "tiene_mejor":  tiene_mejor,
            "tiene_ultimo": tiene_ultimo,
            **stats,
        })
    return modelos


def run_chat():
    titulo("10 — Chat con Modelo Entrenado")
    if not _verificar_molineteai():
        return

    import molineteai

    # ── Buscar modelos disponibles ───────────────────────────────────────────
    modelos = _buscar_modelos()

    emit("chat_checkpoints", modelos=[
        {k: v for k, v in m.items() if k != "carpeta"}  # no exponer rutas internas
        for m in modelos
    ])

    ruta_ck = None

    if modelos:
        print("\nModelos disponibles:\n")
        for m in modelos:
            perp = f"perp={m['mejor_perp']}" if m['mejor_perp'] else ""
            tags = []
            if m["tiene_mejor"]:  tags.append("mejor")
            if m["tiene_ultimo"]: tags.append("último")
            print(f"  [{m['idx']}] {m['display']} — {m['pasos']:,} pasos {perp} [{', '.join(tags)}]")
        print()
        sel = pedir_input(f"Selecciona un modelo (1-{len(modelos)}, o escribe la ruta .bin): ")

        try:
            idx = int(sel) - 1
            if 0 <= idx < len(modelos):
                m = modelos[idx]
                # Preferir mejor checkpoint si existe
                ruta_ck = m["ck_mejor"] or m["ck_ultimo"]
        except ValueError:
            ruta_ck = sel.strip()
    else:
        print("\nNo se encontraron modelos entrenados en data/.")
        print("Primero entrena un modelo con las opciones 5–9.\n")
        ruta_ck = pedir_input("O escribe la ruta de un .bin manualmente: ").strip()

    if not ruta_ck or not os.path.exists(ruta_ck):
        print(f"\n⚠️  Checkpoint no encontrado: '{ruta_ck}'")
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

    emit("chat_ready",
         checkpoint=ruta_ck,
         temperatura=temperatura,
         max_tok=max_tok,
         model_repr=repr(modelo))

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

        emit("chat_user", text=entrada)

        ids_prompt = tok.codificar(entrada)
        ids_gen    = modelo.generar(ids_prompt, max_tok, temperatura)
        texto_gen  = tok.decodificar(ids_gen)

        emit("chat_model", text=texto_gen)
        print()
