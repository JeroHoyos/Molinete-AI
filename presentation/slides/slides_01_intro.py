import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_slides import Slide
from manim_code_blocks import *
import numpy as np
import random
import math
import os
from colores import *
from snippets import RUST_SNIPPETS
from objetos import *


class SlidesIntro:
    def slide_que_es_transformer(self):

        titulo, linea = self.crear_titulo("¿Qué es un Transformer?", palabra_clave="Transformer", color_clave=NARANJA_TERRACOTA)
        adornos = self._crear_adornos_esquinas(escala=0.8)

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)

        frase_1 = Text(
            "Un Transformer es una máquina que predice.",
            font=FUENTE, font_size=26, color=TINTA_NEGRA,
            t2c={"predice": NARANJA_TERRACOTA}
        ).move_to(UP * 2.5)

        frase_2 = Text(
            "Dada una secuencia de palabras...",
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
        ).next_to(frase_1, DOWN, buff=0.35)

        frase_3 = Text(
            "...adivina cuál viene a continuación.",
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"cuál viene a continuación": NARANJA_TERRACOTA}
        ).next_to(frase_2, DOWN, buff=0.25)

        self.play(FadeIn(frase_1, shift=UP * 0.2))
        self.play(FadeIn(frase_2, shift=UP * 0.15))
        self.play(FadeIn(frase_3, shift=UP * 0.15))

        texto_escrito = ["Hoy", "no", "puedo", "quedar,", "estoy", "muy"]
        tokens_burbuja = VGroup(*[
            Text(w, font=FUENTE, font_size=22, color=TINTA_NEGRA)
            for w in texto_escrito
        ]).arrange(RIGHT, buff=0.18)

        cursor = Rectangle(
            width=0.04, height=0.28,
            fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_width=0
        ).next_to(tokens_burbuja, RIGHT, buff=0.06)

        contenido_burbuja = VGroup(tokens_burbuja, cursor)

        burbuja = RoundedRectangle(
            corner_radius=0.18, width=contenido_burbuja.width + 0.7, height=0.75,
            fill_color=FONDO_CAJA,
            fill_opacity=1,
            stroke_color=TIERRA_MANCHEGA, stroke_width=1.5
        )
        contenido_burbuja.move_to(burbuja)
        grupo_burbuja = VGroup(burbuja, contenido_burbuja)

        label_movil = Text(
            "Tu móvil:", font=FUENTE, font_size=18, color=TINTA_NEGRA, slant=ITALIC
        )
        bloque_burbuja = VGroup(label_movil, grupo_burbuja)\
            .arrange(RIGHT, buff=0.35).next_to(frase_3, DOWN, buff=0.6)

        datos_sug = [
            ("cansado", NARANJA_TERRACOTA, NARANJA_TERRACOTA, "92%"),
            ("ocupado",  ACERO,            METAL_CLARO,       "6%"),
            ("bien",     ACERO,            METAL_CLARO,       "2%"),
        ]

        cajas_sug = VGroup()
        for palabra, color_txt, color_borde, pct in datos_sug:
            rect = RoundedRectangle(
                corner_radius=0.1, width=2.0, height=0.65,
                fill_color=FONDO_CAJA if color_txt == NARANJA_TERRACOTA else PAPEL_CREMA,
                fill_opacity=1,
                stroke_color=color_borde, stroke_width=1.5
            )
            txt = Text(
                palabra, font=FUENTE, font_size=20, color=color_txt,
                weight=BOLD if color_txt == NARANJA_TERRACOTA else NORMAL
            ).move_to(rect).shift(DOWN * 0.07)
            pct_txt = Text(
                pct, font=FUENTE, font_size=14, color=color_borde
            ).next_to(txt, UP, buff=0.05)
            cajas_sug.add(VGroup(rect, txt, pct_txt))

        cajas_sug.arrange(RIGHT, buff=0.3).next_to(bloque_burbuja, DOWN, buff=0.35)

        caja_todo = SurroundingRectangle(
            VGroup(bloque_burbuja, cajas_sug),
            color=NARANJA_TERRACOTA, corner_radius=0.15,
            buff=0.3, stroke_width=1.5,
            fill_color=MARRON_OSCURO, fill_opacity=0.05
        )

        label_eleccion = Text(
            "↑ mayor probabilidad", font=FUENTE, font_size=15,
            color=NARANJA_TERRACOTA
        ).next_to(cajas_sug[0], UP, buff=0.12)

        self.play(FadeIn(caja_todo))
        self.play(
            FadeIn(label_movil), FadeIn(burbuja),
            LaggedStart(*[FadeIn(t, shift=UP*0.1) for t in tokens_burbuja], lag_ratio=0.1),
            run_time=1.2
        )
        for _ in range(2):
            self.play(FadeOut(cursor, run_time=0.3))
            self.play(FadeIn(cursor, run_time=0.3))

        self.play(
            LaggedStart(*[FadeIn(c, shift=UP*0.15) for c in cajas_sug], lag_ratio=0.2),
            run_time=0.9
        )
        self.play(FadeIn(label_eleccion, shift=DOWN*0.1))

        self._siguiente()

        self.play(
            FadeOut(frase_1), FadeOut(frase_2), FadeOut(frase_3),
            FadeOut(caja_todo), FadeOut(label_movil),
            FadeOut(burbuja), FadeOut(tokens_burbuja), FadeOut(cursor),
            FadeOut(cajas_sug), FadeOut(label_eleccion),
            run_time=0.8
        )

        caja_maq = RoundedRectangle(
            corner_radius=0.2, width=2.5, height=1.5,
            fill_color=MARRON_OSCURO, fill_opacity=0.9,
            stroke_color=NARANJA_TERRACOTA, stroke_width=3
        )
        txt_maq = Text(
            "Modelo IA", font=FUENTE, font_size=24, color=FONDO_CAJA, weight=BOLD
        ).move_to(caja_maq)
        maquina = VGroup(caja_maq, txt_maq).to_edge(LEFT, buff=1.0).shift(DOWN * 0.5)

        chat_x_right = RIGHT * 5.2
        chat_x_left  = RIGHT * 1.5
        self.play(FadeIn(maquina, shift=RIGHT * 0.5))

        msg1_usuario = self._crear_burbuja_transformer("¿Quién es?", es_usuario=True)\
            .move_to(chat_x_right + UP * 0.8).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg1_usuario, shift=UP * 0.2))
        self._animar_pensamiento_transformer("Soy yo", "El cartero", "98.5%", "1.5%", caja_maq)
        msg2_ia = self._crear_burbuja_transformer("Soy yo", es_usuario=False)\
            .next_to(msg1_usuario, DOWN, buff=0.3).align_to(chat_x_left, LEFT)
        self.play(FadeIn(msg2_ia, shift=UP * 0.2))

        msg3_usuario = self._crear_burbuja_transformer("¿Qué vienes a buscar?", es_usuario=True)\
            .next_to(msg2_ia, DOWN, buff=0.3).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg3_usuario, shift=UP * 0.2))
        self._animar_pensamiento_transformer("A ti", "Nada", "99.9%", "0.05%", caja_maq)
        msg_final_ia = self._crear_burbuja_transformer("A ti", es_usuario=False)\
            .next_to(msg3_usuario, DOWN, buff=0.3).align_to(chat_x_left, LEFT)
        self.play(FadeIn(msg_final_ia, shift=UP * 0.2))

        shift_up = UP * (msg1_usuario.get_center()[1] - msg3_usuario.get_center()[1])
        self.play(
            FadeOut(msg1_usuario, shift=shift_up),
            FadeOut(msg2_ia,      shift=shift_up),
            msg3_usuario.animate.shift(shift_up),
            msg_final_ia.animate.shift(shift_up),
            run_time=1.2
        )

        msg5_usuario = self._crear_burbuja_transformer("Ya es tarde", es_usuario=True)\
            .next_to(msg_final_ia, DOWN, buff=0.3).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg5_usuario, shift=UP * 0.2))
        self._animar_pensamiento_transformer("¿Por qué?", "Vete", "95.0%", "5.0%", caja_maq)
        msg6_ia = self._crear_burbuja_transformer("¿Por qué?", es_usuario=False)\
            .next_to(msg5_usuario, DOWN, buff=0.3).align_to(chat_x_left, LEFT)
        self.play(FadeIn(msg6_ia, shift=UP * 0.2))

        shift_up_2 = UP * (msg3_usuario.get_center()[1] - msg5_usuario.get_center()[1])
        self.play(
            FadeOut(msg3_usuario,  shift=shift_up_2),
            FadeOut(msg_final_ia,  shift=shift_up_2),
            msg5_usuario.animate.shift(shift_up_2),
            msg6_ia.animate.shift(shift_up_2),
            run_time=1.2
        )

        texto_despecho = "Porque ahora soy yo la que\nquiere estar sin ti"
        msg7_usuario = self._crear_burbuja_transformer(texto_despecho, es_usuario=True)\
            .next_to(msg6_ia, DOWN, buff=0.3).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg7_usuario, shift=UP * 0.2))
        self._siguiente()

        chat_visible_final = VGroup(msg5_usuario, msg6_ia, msg7_usuario)
        self.play(
            FadeOut(chat_visible_final),
            FadeOut(maquina),
            run_time=1
        )

        self.limpiar_pantalla()

    def slide_roadmap(self):
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.7, buff=0.7)
        adornos[0][-1].add_updater(lambda m, dt: m.rotate(-dt * 0.5))

        titulo, linea = self.crear_titulo(
            "Hoja de Ruta",
            palabra_clave="Ruta",
            color_clave=NARANJA_TERRACOTA,
        )
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)

        pasos = [
            ("Tensores",     NARANJA_TERRACOTA, crear_rueda_carreta),
            ("Arquitectura", MARRON_OSCURO,     crear_molino),
            ("Entrenamiento", NARANJA_TERRACOTA, crear_pergamino),
            ("Demo",         MARRON_OSCURO,     crear_herradura),
        ]

        nodos   = VGroup()
        textos  = VGroup()
        iconos  = VGroup()
        flechas = VGroup()

        for i, (etiqueta, color, fn_icono) in enumerate(pasos):
            anillo = Circle(
                radius=0.6,
                stroke_color=color, stroke_width=5,
                fill_color=PAPEL_CREMA, fill_opacity=0.95,
            )
            numero = Text(str(i + 1), font=FUENTE, font_size=40,
                          color=color, weight=BOLD)
            numero.move_to(anillo.get_center())
            nodos.add(VGroup(anillo, numero))

            iconos.add(fn_icono().scale(0.60))
            textos.add(Text(etiqueta, font=FUENTE, font_size=22,
                            color=MARRON_OSCURO, weight=BOLD))

        nodos.arrange(RIGHT, buff=1.6).move_to(DOWN * 0.5)

        for i in range(len(pasos)):
            iconos[i].next_to(nodos[i], UP,   buff=0.4)
            textos[i].next_to(nodos[i], DOWN, buff=0.3)

        for i in range(len(pasos) - 1):
            flechas.add(Arrow(
                nodos[i].get_right()  + RIGHT * 0.05,
                nodos[i+1].get_left() + LEFT  * 0.05,
                buff=0.0, color=MARRON_OSCURO,
                stroke_width=4, max_tip_length_to_length_ratio=0.25,
            ))

        for i in range(len(pasos)):
            self.play(
                DrawBorderThenFill(nodos[i][0]),
                Write(nodos[i][1]),
                run_time=0.5,
            )
            self.play(
                FadeIn(iconos[i], scale=0.6),
                Write(textos[i]),
                run_time=0.6,
            )
            if i < len(pasos) - 1:
                self.play(Create(flechas[i]), run_time=0.5)

        self._siguiente()
        self.limpiar_pantalla()

    def slide_por_que_rust(self):
        titulo, linea = self.crear_titulo(
            "¿Por qué Rust?", palabra_clave="Rust", color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        COLOR_SI   = "#4CAF50"
        COLOR_NO   = ROJO_CONTRA
        COLOR_RUST = NARANJA_TERRACOTA

        ANCHO_COL_CRITERIO = 3.8
        ANCHO_COL_LANG     = 2.5
        ALTO_FILA          = 0.72
        RADIO              = 0.12

        CRITERIOS = [
            "Velocidad de cómputo",
            "Seguridad de memoria",
            "Hilos reales (sin GIL)",
            "Fácil de aprender",
            "Diferenciador en AI",
        ]

        DATOS = [
            (False, True,  True ),
            (False, False, True ),
            (False, True,  True ),
            (True,  False, False),
            (False, False, True ),
        ]

        LANGS = ["Python", "C++", "Rust"]

        def _celda_criterio(texto):
            bg = RoundedRectangle(
                corner_radius=RADIO,
                width=ANCHO_COL_CRITERIO, height=ALTO_FILA,
                fill_color=FONDO_CAJA, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=1.5,
            )
            lbl = Text(texto, font=FUENTE, font_size=20, color=TINTA_NEGRA)
            lbl.move_to(bg)
            return VGroup(bg, lbl)

        def _celda_valor(valor, es_rust=False):
            if valor is True:
                simbolo, color_bg, color_txt = "✓", COLOR_SI,  BLANCO
            elif valor is False:
                simbolo, color_bg, color_txt = "✗", COLOR_NO,  BLANCO
            else:
                simbolo, color_bg, color_txt = "±", PAPEL_TAN, TINTA_NEGRA

            bg = RoundedRectangle(
                corner_radius=RADIO,
                width=ANCHO_COL_LANG, height=ALTO_FILA,
                fill_color=color_bg, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=1.5,
            )
            lbl = Text(simbolo, font=FUENTE, font_size=26, color=color_txt, weight=BOLD)
            lbl.move_to(bg)
            return VGroup(bg, lbl)

        def _cabecera(texto, es_rust=False):
            color_bg  = COLOR_RUST if es_rust else MARRON_OSCURO
            color_txt = BLANCO
            bg = RoundedRectangle(
                corner_radius=RADIO,
                width=ANCHO_COL_LANG if texto != "" else ANCHO_COL_CRITERIO,
                height=ALTO_FILA * 0.9,
                fill_color=color_bg, fill_opacity=1,
                stroke_color=color_bg, stroke_width=0,
            )
            lbl = Text(texto, font=FUENTE, font_size=22, color=color_txt, weight=BOLD)
            lbl.move_to(bg)
            return VGroup(bg, lbl)

        cab_criterio = _cabecera("")
        cab_python   = _cabecera("Python")
        cab_cpp      = _cabecera("C++")
        cab_rust     = _cabecera("Rust", es_rust=True)

        fila_cab = VGroup(cab_criterio, cab_python, cab_cpp, cab_rust)
        fila_cab.arrange(RIGHT, buff=0.1)

        filas_datos = []
        celdas_rust_si = []

        for i, (criterio, (py, cpp, rs)) in enumerate(zip(CRITERIOS, DATOS)):
            c_crit = _celda_criterio(criterio)
            c_py   = _celda_valor(py)
            c_cpp  = _celda_valor(cpp)
            c_rs   = _celda_valor(rs, es_rust=True)

            if rs is True:
                celdas_rust_si.append(c_rs)

            fila = VGroup(c_crit, c_py, c_cpp, c_rs)
            fila.arrange(RIGHT, buff=0.12)
            filas_datos.append(fila)

        tabla_completa = VGroup(fila_cab, *filas_datos)
        tabla_completa.arrange(DOWN, buff=0.10)
        tabla_completa.next_to(linea, DOWN, buff=0.35)
        tabla_completa.to_edge(LEFT, buff=0.7)

        logo_py  = ImageMobject(os.path.join("assets", "logo_python.png")).set_height(1.1)
        logo_cpp = ImageMobject(os.path.join("assets", "logo_cpp.png")).set_height(1.1)
        logo_rs  = ImageMobject(os.path.join("assets", "logo_rust.png")).set_height(1.3)

        logos = Group(logo_py, logo_cpp, logo_rs).arrange(DOWN, buff=0.55)
        logos.next_to(tabla_completa, RIGHT, buff=0.6)
        logos.set_y(tabla_completa.get_center()[1])

        self.play(
            FadeIn(fila_cab, shift=DOWN * 0.15),
            FadeIn(logo_py,  shift=LEFT * 0.2),
            FadeIn(logo_cpp, shift=LEFT * 0.2),
            FadeIn(logo_rs,  shift=LEFT * 0.2),
            run_time=0.9,
        )

        for fila in filas_datos:
            self.play(FadeIn(fila, shift=RIGHT * 0.2), run_time=0.55)

        veredicto = Text(
            "Velocidad de C++  ·  Sin GIL: hilos reales",
            font=FUENTE, font_size=19, color=COLOR_RUST, weight=BOLD,
        )
        caja_veredicto = SurroundingRectangle(
            veredicto,
            color=COLOR_RUST, fill_color=FONDO_CAJA, fill_opacity=0.97,
            corner_radius=0.15, buff=0.22, stroke_width=2.5,
        )
        grupo_veredicto = VGroup(caja_veredicto, veredicto)
        grupo_veredicto.next_to(tabla_completa, DOWN, buff=0.35)

        self.play(FadeIn(grupo_veredicto, shift=UP * 0.2), run_time=0.7)
        self._siguiente()

        # ── ACTO 2: Python como interfaz, Rust como motor ─────────────────────
        self.play(
            FadeOut(tabla_completa),
            FadeOut(Group(logo_py, logo_cpp, logo_rs)),
            FadeOut(grupo_veredicto),
            run_time=0.6,
        )

        ANCHO_PANEL = 7.0
        ALTO_PANEL  = 1.6

        # ── Panel Python ──────────────────────────────────────────────────────
        py_bg = RoundedRectangle(
            corner_radius=0.18, width=ANCHO_PANEL, height=ALTO_PANEL,
            fill_color=FONDO_CAJA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=2,
        ).set_x(0).next_to(linea, DOWN, buff=0.7)

        py_logo = ImageMobject(os.path.join("assets", "logo_python.png")).set_height(0.9)
        py_lbl  = Text("Interfaz", font=FUENTE, font_size=28,
                        color=MARRON_OSCURO, weight=BOLD)
        Group(py_logo, py_lbl).arrange(RIGHT, buff=0.35).move_to(py_bg)

        # ── Conector ──────────────────────────────────────────────────────────
        flecha = Arrow(
            py_bg.get_bottom(), py_bg.get_bottom() + DOWN * 0.7,
            color=NARANJA_TERRACOTA, stroke_width=4,
            max_tip_length_to_length_ratio=0.45, buff=0,
        ).set_x(0)

        # ── Panel Rust ────────────────────────────────────────────────────────
        rs_bg = RoundedRectangle(
            corner_radius=0.18, width=ANCHO_PANEL, height=ALTO_PANEL,
            fill_color=FONDO_CAJA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=2,
        ).next_to(flecha.get_tip(), DOWN, buff=0.15).set_x(0)

        rs_logo = ImageMobject(os.path.join("assets", "logo_rust.png")).set_height(1.0)
        rs_lbl  = Text("Motor", font=FUENTE, font_size=28,
                        color=MARRON_OSCURO, weight=BOLD)
        Group(rs_logo, rs_lbl).arrange(RIGHT, buff=0.35).move_to(rs_bg)

        # ── Animación ─────────────────────────────────────────────────────────
        self.play(FadeIn(py_bg, shift=DOWN * 0.15), FadeIn(py_logo, shift=DOWN * 0.15),
                  FadeIn(py_lbl, shift=DOWN * 0.15), run_time=0.6)
        self.play(GrowArrow(flecha), run_time=0.45)
        self.play(FadeIn(rs_bg, shift=UP * 0.15), FadeIn(rs_logo, shift=UP * 0.15),
                  FadeIn(rs_lbl, shift=UP * 0.15), run_time=0.6)

        self._siguiente()
        self.limpiar_pantalla()

    def slide_molinete_ai(self):
        titulo, linea = self.crear_titulo(
            "¿Por qué Molinete?",
            palabra_clave="Molinete",
            color_clave=NARANJA_TERRACOTA,
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        try:
            imagen = ImageMobject(os.path.join("assets", "quijote_vs_molinos.png"))
        except Exception:
            imagen = Rectangle(
                width=6.8, height=5.8,
                fill_color=ARENA_MANCHEGA, fill_opacity=0.3,
                stroke_color=NARANJA_TERRACOTA, stroke_width=1.5,
            )
        imagen.set_height(5.8).move_to(RIGHT * 4.2 + DOWN * 0.3)
        marco = SurroundingRectangle(
            imagen, buff=0.06,
            color=NARANJA_TERRACOTA, stroke_width=2.0, corner_radius=0.05,
        )

        num_1 = Text("1.", font=FUENTE, font_size=42,
                     color=NARANJA_TERRACOTA, weight=BOLD)
        tit_1 = Text("El Quijote", font=FUENTE, font_size=30,
                     color=MARRON_OSCURO, weight=BOLD)
        enc_1 = VGroup(num_1, tit_1).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        ico_1 = crear_escudo_y_lanza().scale(0.35)
        txt_1 = Text(
            "Quijote arremetía contra molinos creyéndolos gigantes.",
            font=FUENTE, font_size=19, color=TINTA_NEGRA,
        )
        bloque_1 = VGroup(enc_1, ico_1, txt_1).arrange(DOWN, buff=0.18, aligned_edge=LEFT)

        num_2 = Text("2.", font=FUENTE, font_size=42,
                     color=HIERRO, weight=BOLD)
        tit_2 = Text("Dark Souls", font=FUENTE, font_size=30,
                     color=HIERRO, weight=BOLD)
        enc_2 = VGroup(num_2, tit_2).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        ico_2 = crear_yelmo_mambrino().scale(0.5)
        txt_2 = Text(
            "El boss de Las Catacumbas,\nel más fácil del juego.",
            font=FUENTE, font_size=19, color=TINTA_NEGRA, line_spacing=1.4,
        )
        bloque_2 = VGroup(enc_2, ico_2, txt_2).arrange(DOWN, buff=0.18, aligned_edge=LEFT)

        sep = Line(ORIGIN, RIGHT * 4.8,
                   stroke_color=NARANJA_TERRACOTA, stroke_width=1.0, stroke_opacity=0.45)
        conclusion = Text(
            "Por eso: Molinete.",
            font=FUENTE, font_size=26,
            color=NARANJA_TERRACOTA, weight=BOLD,
        )

        col_izq = VGroup(bloque_1, bloque_2, sep, conclusion).arrange(
            DOWN, buff=0.35, aligned_edge=LEFT,
        )
        col_izq.move_to(LEFT * 2.2 + DOWN * 0.3)

        self.play(
            AnimationGroup(
                FadeIn(imagen,   shift=LEFT  * 0.2, run_time=0.9),
                FadeIn(marco,                       run_time=0.9),
                FadeIn(bloque_1, shift=RIGHT * 0.2, run_time=0.8),
                FadeIn(bloque_2, shift=RIGHT * 0.2, run_time=0.8),
                Create(sep,                         run_time=0.5),
                Write(conclusion,                   run_time=0.7),
                lag_ratio=0.0,
            )
        )
        self._siguiente()
        self.limpiar_pantalla()
