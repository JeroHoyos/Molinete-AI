"""
modulos/ui.py
━━━━━━━━━━━━━
Helpers de consola para Molinete AI.
Optimizados para mostrarse en la interfaz web (sin limpiar pantalla,
sin animaciones lentas que bloqueen la salida).

Exporta:
    SEPARADOR       — Separador de línea
    limpiar_pantalla()
    mostrar_arte()
    imprimir_lento(texto, ms_por_letra)
    barra_progreso(mensaje, segundos)
    pedir_input(prompt, default)
    titulo(texto)
    seccion(texto)
    ok(texto)
    info(texto)
    advertencia(texto)
    tabla(filas, cabeceras)
"""

import sys
import time
import json

SEPARADOR = "─" * 68


def emit(type_: str, **kwargs):
    """
    Emite un evento estructurado al frontend web.
    Las líneas __MOL__{json} son interceptadas por el frontend;
    el resto del stdout va al terminal como siempre.
    """
    data = {"type": type_, **kwargs}
    print(f'__MOL__{json.dumps(data, ensure_ascii=False)}', flush=True)

CABECERA = """\
  ╔══════════════════════════════════════╗
  ║  🌬  MOLINETE AI · GPT-2 en Rust    ║
  ╚══════════════════════════════════════╝"""


def limpiar_pantalla():
    """No-op en el contexto web (el terminal no se puede limpiar)."""
    pass


def mostrar_arte():
    """Muestra una cabecera compacta en vez del arte ASCII completo."""
    print(CABECERA)
    print()


def imprimir_lento(texto: str, ms_por_letra: float = 0):
    """En la web imprime inmediatamente (sin delay)."""
    print(texto)


def barra_progreso(mensaje: str, segundos: float = 0):
    """
    Muestra una barra de progreso animada durante `segundos`.
    En la web se omite para no bloquear la salida real.
    """
    print(f"  {mensaje}…")


def pedir_input(prompt: str, default: str = "") -> str:
    """Lee una entrada del usuario con soporte para valor por defecto."""
    try:
        valor = input(prompt).strip()
        return valor if valor else default
    except (KeyboardInterrupt, EOFError):
        print()
        return default


def titulo(texto: str):
    """Imprime un título de sección principal."""
    print()
    print(f"  ┌{'─' * (len(texto) + 4)}┐")
    print(f"  │  {texto}  │")
    print(f"  └{'─' * (len(texto) + 4)}┘")
    print()


def seccion(texto: str):
    """Imprime un subtítulo de sección."""
    print()
    print(f"  ▶  {texto}")
    print(f"  {'─' * (len(texto) + 5)}")


def ok(texto: str):
    """Imprime un mensaje de éxito."""
    print(f"  ✓  {texto}")


def info(texto: str):
    """Imprime un mensaje informativo."""
    print(f"  ·  {texto}")


def advertencia(texto: str):
    """Imprime una advertencia."""
    print(f"  ⚠  {texto}")


def tabla(filas: list, cabeceras: list = None):
    """
    Imprime una tabla de texto alineada.

    filas      — lista de listas/tuplas con los valores de cada fila
    cabeceras  — lista de strings para la cabecera (opcional)
    """
    todas = ([cabeceras] if cabeceras else []) + list(filas)
    if not todas:
        return

    anchos = [max(len(str(fila[i])) for fila in todas) for i in range(len(todas[0]))]
    sep = "  +" + "+".join("-" * (a + 2) for a in anchos) + "+"

    def fila_str(f):
        return "  |" + "|".join(f" {str(v):<{anchos[i]}} " for i, v in enumerate(f)) + "|"

    print(sep)
    if cabeceras:
        print(fila_str(cabeceras))
        print(sep)
        for f in filas:
            print(fila_str(f))
    else:
        for f in filas:
            print(fila_str(f))
    print(sep)
