import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


class SlideEntrenamiento:
    def slide_entrenamiento(self):

        titulo, linea = self.crear_titulo(
            "Entrenamiento: Ajustando Perillas",
            palabra_clave="Perillas",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        def crear_caja(texto, ancho=2.5, bg_color=FONDO_CAJA, txt_color=TINTA_NEGRA,
                       opacidad=1.0, borde_color=MARRON_OSCURO, borde_grosor=1, peso=NORMAL):
            caja = RoundedRectangle(
                corner_radius=0.1, width=ancho, height=0.6,
                fill_color=bg_color, fill_opacity=opacidad,
                stroke_color=borde_color, stroke_width=borde_grosor
            )
            txt = Text(texto, font=FUENTE, font_size=16, color=txt_color, weight=peso)
            return VGroup(caja, txt)

        def crear_perilla(angulo):
            base = Circle(radius=0.25, fill_color=FONDO_CAJA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2)
            indicador = Line(base.get_center(), base.get_center() + UP*0.25, color=NARANJA_TERRACOTA, stroke_width=4)
            indicador.rotate(angulo, about_point=base.get_center())
            return VGroup(base, indicador)

        estado_ui = crear_caja("Iniciando Motor de Entrenamiento...", ancho=7.5).to_edge(UP, buff=1.2)
        self.play(FadeIn(estado_ui, shift=DOWN))

        EJE_Y = DOWN * 0.2

        self.play(
            estado_ui[1].animate.set_text("→ Forward Pass"),
            estado_ui[0].animate.set_stroke(NARANJA_TERRACOTA, width=2)
        )

        lbl_in = Text("Contexto", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        tokens_in = VGroup(*[crear_caja(word, ancho=1.1) for word in ["En un", "lugar", "de la"]]).arrange(DOWN, buff=0.1)
        grupo_entrada = VGroup(lbl_in, tokens_in).arrange(DOWN, buff=0.3).move_to(LEFT * 4.5 + EJE_Y)

        modelo_bg = RoundedRectangle(corner_radius=0.2, width=3.5, height=2.6, fill_color=PAPEL_CREMA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2)
        lbl_modelo = Text("Transformer\n(Red Neuronal)", font=FUENTE, font_size=18, color=TINTA_NEGRA, weight=BOLD).next_to(modelo_bg.get_top(), DOWN, buff=0.2)

        grupo_perillas = VGroup(
            crear_perilla(PI/4), crear_perilla(-PI/3), crear_perilla(PI)
        ).arrange(RIGHT, buff=0.4).move_to(modelo_bg.get_center() + DOWN*0.1)

        lbl_pesos = Text("Parámetros (Pesos)", font=FUENTE, font_size=14, color=TINTA_NEGRA).next_to(grupo_perillas, DOWN, buff=0.25)
        grupo_modelo = VGroup(modelo_bg, lbl_modelo, grupo_perillas, lbl_pesos).move_to(EJE_Y)

        lbl_out = Text("Predicción", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        prob_incorrecta = crear_caja("playa? (85%)", bg_color=PAPEL_TAN, opacidad=0.8, borde_color=NARANJA_TERRACOTA, borde_grosor=2)
        prob_correcta = crear_caja("Mancha (2%)", bg_color=CAJA_INFERIOR, txt_color=MARRON_OSCURO, opacidad=0.5, borde_color=CAJA_INFERIOR)
        grupo_probs = VGroup(prob_incorrecta, prob_correcta).arrange(DOWN, buff=0.1)
        grupo_salida = VGroup(lbl_out, grupo_probs).arrange(DOWN, buff=0.3).move_to(RIGHT * 4.5 + EJE_Y)

        flujo_in = DashedLine(grupo_entrada.get_right(), modelo_bg.get_left(), color=MARRON_OSCURO, stroke_width=3)
        tubo_out = Line(modelo_bg.get_right(), grupo_salida.get_left(), color=MARRON_OSCURO, stroke_width=4)

        self.play(FadeIn(grupo_entrada, shift=RIGHT))
        self.play(Create(flujo_in))
        self.play(FadeIn(grupo_modelo, scale=0.9))
        self.play(GrowFromCenter(tubo_out))
        self.play(FadeIn(grupo_salida, shift=LEFT))
        self._siguiente()

        self.play(
            estado_ui[1].animate.set_text("→ Loss"),
            estado_ui[0].animate.set_stroke(NARANJA_TERRACOTA, width=2)
        )

        lbl_target = Text("Target Real:", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        caja_target = crear_caja("Mancha (100%)")
        grupo_target = VGroup(lbl_target, caja_target).arrange(DOWN, buff=0.1).next_to(grupo_salida, DOWN, buff=0.4)

        nodo_loss = MathTex(r"\mathcal{L}", font_size=40, color=NARANJA_TERRACOTA)
        medidor_bg = RoundedRectangle(width=1.5, height=0.2, corner_radius=0.1, stroke_color=MARRON_OSCURO, fill_color=FONDO_CAJA, fill_opacity=1)
        medidor_fill = RoundedRectangle(width=1.3, height=0.2, corner_radius=0.1, stroke_width=0, fill_color=NARANJA_TERRACOTA, fill_opacity=1).align_to(medidor_bg, LEFT)

        caja_error = VGroup(
            Text("Error:", font=FUENTE, font_size=14, color=TINTA_NEGRA),
            VGroup(medidor_bg, medidor_fill),
            Text("¡Alta!", font=FUENTE, font_size=14, color=NARANJA_TERRACOTA, weight=BOLD)
        ).arrange(DOWN, buff=0.1)

        panel_loss = VGroup(nodo_loss, caja_error).arrange(RIGHT, buff=0.4).move_to(DOWN * 2.7)

        self.play(FadeIn(grupo_target, shift=UP))

        self.play(
            Indicate(prob_incorrecta, color=NARANJA_TERRACOTA, scale_factor=1.1),
            Indicate(caja_target, color=NARANJA_TERRACOTA, scale_factor=1.1)
        )

        self.play(FadeIn(panel_loss, shift=UP))
        self.play(Flash(medidor_fill, color=NARANJA_TERRACOTA, line_length=0.2))
        self._siguiente()

        self.play(
            estado_ui[1].animate.set_text("→ Backprop"),
            estado_ui[0].animate.set_stroke(NARANJA_TERRACOTA, width=3)
        )

        tubo_back = Arrow(panel_loss.get_top(), modelo_bg.get_bottom(), color=NARANJA_TERRACOTA, stroke_width=4, buff=0.1)
        lbl_grad = MathTex(r"\nabla W", font_size=24, color=NARANJA_TERRACOTA).next_to(tubo_back, RIGHT, buff=0.1)

        self.play(Create(tubo_back), Write(lbl_grad))

        self.play(
            Rotate(grupo_perillas[0][1], angle=-PI/2, about_point=grupo_perillas[0][0].get_center()),
            Rotate(grupo_perillas[1][1], angle=PI/1.5, about_point=grupo_perillas[1][0].get_center()),
            Rotate(grupo_perillas[2][1], angle=-PI/4, about_point=grupo_perillas[2][0].get_center()),
            modelo_bg.animate.set_stroke(NARANJA_TERRACOTA, width=3),
            run_time=2,
            rate_func=there_and_back_with_pause
        )

        lbl_pesos_nuevos = Text("Parámetros (Ajustados)", font=FUENTE, font_size=14, color=MARRON_OSCURO, weight=BOLD).move_to(lbl_pesos)

        self.play(
            Transform(lbl_pesos, lbl_pesos_nuevos),
            modelo_bg.animate.set_stroke(MARRON_OSCURO, width=2),
            FadeOut(tubo_back, lbl_grad)
        )
        self._siguiente()

        self.play(
            estado_ui[1].animate.set_text("FASE 4: Nuevo Forward Pass (Éxito)"),
            estado_ui[0].animate.set_stroke(MARRON_OSCURO, width=2)
        )

        flujo_exito = DashedLine(grupo_entrada.get_right(), modelo_bg.get_left(), color=NARANJA_TERRACOTA, stroke_width=4)
        tubo_exito = Line(modelo_bg.get_right(), grupo_salida.get_left(), color=NARANJA_TERRACOTA, stroke_width=5)

        self.play(
            Indicate(modelo_bg, color=PAPEL_TAN),
            Transform(flujo_in, flujo_exito),
            Transform(tubo_out, tubo_exito),
            run_time=1.5
        )

        prob_correcta_nueva = crear_caja("Mancha (98%)", bg_color=PAPEL_TAN, borde_color=MARRON_OSCURO, borde_grosor=2, peso=BOLD).move_to(prob_incorrecta)
        prob_incorrecta_nueva = crear_caja("playa? (1%)", bg_color=CAJA_INFERIOR, txt_color=MARRON_OSCURO, opacidad=0.5, borde_color=CAJA_INFERIOR).move_to(prob_correcta)

        self.play(
            FadeOut(panel_loss, grupo_target),
            Transform(prob_incorrecta, prob_correcta_nueva),
            Transform(prob_correcta, prob_incorrecta_nueva)
        )

        self.play(Wiggle(prob_incorrecta, scale_value=1.05))
        self._siguiente()

        self.limpiar_pantalla()


