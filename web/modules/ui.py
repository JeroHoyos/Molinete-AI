"""
modules/ui.py
━━━━━━━━━━━━━
Helpers de consola para Molinete AI.
Optimizados para mostrarse en la interfaz web (sin limpiar pantalla,
sin animaciones lentas que bloqueen la salida).

Exporta:
    SEPARADOR                — Separador de línea
    emit(type_, **kwargs)    — Evento estructurado __MOL__ para el frontend
    verificar_molineteai()
    imprimir_lento(texto, ms_por_letra)
    barra_progreso(mensaje, segundos)
    pedir_input(prompt, default)
    titulo(texto)
"""

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


def verificar_molineteai() -> bool:
    """
    Comprueba que el módulo Rust (molineteai) está compilado e instalado.
    Si falta, imprime instrucciones y devuelve False.
    """
    try:
        import molineteai  # noqa: F401
        return True
    except ImportError:
        print("\n⚠️  El módulo 'molineteai' no está instalado.")
        print("   Compílalo con:  maturin develop --release\n")
        return False


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
