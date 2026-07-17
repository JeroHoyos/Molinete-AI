import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideMolineteAi:
    def slide_molinete_ai(self):
        titulo, linea = self.crear_titulo(
            "¿Por qué Molinete?",
            palabra_clave="Molinete",
            color_clave=NARANJA_TERRACOTA,
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.6)
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)

        enc_1 = Text("El Quijote", font=FUENTE, font_size=28,
                     color=MARRON_OSCURO, weight=BOLD)
        imagen_quijote = ImageMobject(
            os.path.join("assets", "quijote_vs_molinos.png")
        ).set_height(4.25)
        marco_1 = SurroundingRectangle(
            imagen_quijote, buff=0.06,
            color=NARANJA_TERRACOTA, stroke_width=2.0, corner_radius=0.05,
        )
        grupo_img_1 = Group(imagen_quijote, marco_1)

        enc_2 = Text("Dark Souls", font=FUENTE, font_size=28,
                     color=HIERRO, weight=BOLD)
        imagen_souls = ImageMobject(
            os.path.join("assets", "molinete_dark_souls.png")
        ).set_height(4.25)
        marco_2 = SurroundingRectangle(
            imagen_souls, buff=0.06,
            color=HIERRO, stroke_width=2.0, corner_radius=0.05,
        )
        grupo_img_2 = Group(imagen_souls, marco_2)

        # Reparto horizontal equilibrado: mismo aire a los lados y en medio
        hueco = (13.0 - grupo_img_1.width - grupo_img_2.width) / 3
        Group(grupo_img_1, grupo_img_2).arrange(RIGHT, buff=hueco).set_x(0)

        # Centrado vertical de las imágenes en la zona libre bajo el título
        y_top    = linea.get_bottom()[1] - 0.15
        y_bot    = -3.9
        y_centro = (y_top + y_bot) / 2
        grupo_img_1.set_y(y_centro)
        grupo_img_2.set_y(y_centro)

        enc_1.next_to(grupo_img_1, UP, buff=0.28)
        enc_2.next_to(grupo_img_2, UP, buff=0.28)
        enc_2.set_y(enc_1.get_y())

        # Compensar el alto del encabezado para que el conjunto quede centrado
        ajuste = (enc_1.height + 0.28) / 2
        for m in (grupo_img_1, enc_1, grupo_img_2, enc_2):
            m.shift(DOWN * ajuste)

        self.play(
            FadeIn(enc_1, shift=RIGHT * 0.2),
            FadeIn(imagen_quijote, shift=UP * 0.15), FadeIn(marco_1),
            run_time=0.9,
        )
        self._siguiente()

        self.play(
            FadeIn(enc_2, shift=LEFT * 0.2),
            FadeIn(imagen_souls, shift=UP * 0.15), FadeIn(marco_2),
            run_time=0.9,
        )
        self._siguiente()
        self.limpiar_pantalla()
