import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideEntrenamiento:
    def slide_entrenamiento(self):

        titulo, linea = self.crear_titulo(
            "Entrenamiento",
            palabra_clave="Entrenamiento",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

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

        EJE_Y = UP * 0.35

        lbl_in = Text("Contexto", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        tokens_in = VGroup(*[crear_caja(word, ancho=1.1) for word in ["En un", "lugar", "de la"]]).arrange(DOWN, buff=0.1)
        grupo_entrada = VGroup(lbl_in, tokens_in).arrange(DOWN, buff=0.3).move_to(LEFT * 4.5 + EJE_Y)

        modelo_bg = RoundedRectangle(corner_radius=0.2, width=3.5, height=2.6, fill_color=PAPEL_CREMA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2)
        lbl_modelo = Text("El modelo", font=FUENTE, font_size=18, color=TINTA_NEGRA, weight=BOLD).next_to(modelo_bg.get_top(), DOWN, buff=0.2)

        _ang_perillas = [PI/4, -PI/3, PI, -PI/4, PI/2, -PI/1.5]
        grupo_perillas = VGroup(*[crear_perilla(a) for a in _ang_perillas])\
            .arrange_in_grid(rows=2, cols=3, buff=0.32)\
            .scale(1.08).move_to(modelo_bg.get_center() + DOWN*0.05)

        lbl_pesos = Text("Perillas", font=FUENTE, font_size=14, color=TINTA_NEGRA).next_to(grupo_perillas, DOWN, buff=0.25)
        grupo_modelo = VGroup(modelo_bg, lbl_modelo, grupo_perillas, lbl_pesos).move_to(EJE_Y)

        lbl_out = Text("Predicción", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        prob_incorrecta = crear_caja("playa", bg_color=PAPEL_TAN, opacidad=0.8, borde_color=NARANJA_TERRACOTA, borde_grosor=2)
        prob_correcta = crear_caja("Mancha", bg_color=CAJA_INFERIOR, txt_color=MARRON_OSCURO, opacidad=0.5, borde_color=CAJA_INFERIOR)
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

        lbl_target = Text("La palabra correcta era:", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        caja_target = crear_caja("Mancha", bg_color=VERDE_OLIVA, txt_color=BLANCO, borde_color=VERDE_OLIVA)
        grupo_target = VGroup(lbl_target, caja_target).arrange(DOWN, buff=0.1).next_to(grupo_salida, DOWN, buff=0.2)

        self.play(FadeIn(grupo_target, shift=UP))
        self.play(
            Indicate(prob_incorrecta, color=NARANJA_TERRACOTA, scale_factor=1.1),
            Indicate(caja_target, color=VERDE_OLIVA, scale_factor=1.1)
        )

        # ── Medidor circular: seguridad en la palabra correcta ──────────
        GC = np.array([0.0, -2.0, 0.0])
        R = 0.72

        def _col(v):
            return interpolate_color(ManimColor(NARANJA_TERRACOTA), ManimColor(VERDE_OLIVA), v)

        seguridad = ValueTracker(0.0)

        aro_bg = Circle(radius=R, stroke_color=MARRON_OSCURO, stroke_opacity=0.15,
                        stroke_width=15, fill_opacity=0).move_to(GC)

        aro_fg = always_redraw(lambda: Arc(
            radius=R, start_angle=PI / 2, angle=-TAU * max(seguridad.get_value(), 0.0001),
            arc_center=GC, stroke_color=_col(seguridad.get_value()), stroke_width=15))

        num = always_redraw(lambda: Text(
            f"{int(round(seguridad.get_value() * 100))}%", font=FUENTE, font_size=34,
            color=_col(seguridad.get_value()), weight=BOLD).move_to(GC))

        lbl_acierto = Text("Qué tanto le atina a la palabra correcta", font=FUENTE, font_size=18,
                           color=MARRON_OSCURO, weight=BOLD).next_to(aro_bg, DOWN, buff=0.2)

        self.play(Create(aro_bg), FadeIn(lbl_acierto, shift=UP * 0.1))
        self.add(aro_fg, num)
        self.play(seguridad.animate.set_value(0.02), run_time=0.6)

        _giros = [-PI/2, PI/1.5, -PI/4, PI/3, -PI/2.5, PI/2]
        self.play(
            *[Rotate(grupo_perillas[i][1], angle=_giros[i],
                     about_point=grupo_perillas[i][0].get_center()) for i in range(6)],
            modelo_bg.animate.set_stroke(NARANJA_TERRACOTA, width=3),
            run_time=2,
            rate_func=there_and_back_with_pause
        )

        lbl_pesos_nuevos = Text("Perillas ajustadas", font=FUENTE, font_size=14, color=MARRON_OSCURO, weight=BOLD).move_to(lbl_pesos)

        self.play(
            Transform(lbl_pesos, lbl_pesos_nuevos),
            modelo_bg.animate.set_stroke(MARRON_OSCURO, width=2),
        )

        flujo_exito = DashedLine(grupo_entrada.get_right(), modelo_bg.get_left(), color=VERDE_OLIVA, stroke_width=4)
        tubo_exito = Line(modelo_bg.get_right(), grupo_salida.get_left(), color=VERDE_OLIVA, stroke_width=5)

        self.play(
            Indicate(modelo_bg, color=PAPEL_TAN),
            Transform(flujo_in, flujo_exito),
            Transform(tubo_out, tubo_exito),
            run_time=1.5
        )

        prob_correcta_nueva = crear_caja("Mancha", bg_color=VERDE_OLIVA, txt_color=BLANCO, borde_color=VERDE_OLIVA, borde_grosor=2, peso=BOLD).move_to(prob_incorrecta)
        prob_incorrecta_nueva = crear_caja("playa", bg_color=CAJA_INFERIOR, txt_color=MARRON_OSCURO, opacidad=0.5, borde_color=CAJA_INFERIOR).move_to(prob_correcta)

        # ahora acierta: el anillo se llena y el porcentaje sube
        self.play(
            Transform(prob_incorrecta, prob_correcta_nueva),
            Transform(prob_correcta, prob_incorrecta_nueva),
            seguridad.animate.set_value(0.98),
            aro_bg.animate.set_stroke(VERDE_OLIVA, opacity=0.22),
            run_time=1.5,
        )

        self.play(
            Wiggle(prob_incorrecta, scale_value=1.05),
            Flash(aro_bg, color=VERDE_OLIVA, line_length=0.3),
        )
        self._siguiente()

        self.limpiar_pantalla()


