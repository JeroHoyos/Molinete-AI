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


class SlidesEntrenamiento:
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


    def slide_descenso_gradiente(self):

        titulo, linea = self.crear_titulo(
            "Descenso del Gradiente: Bajando la Colina",
            palabra_clave="Gradiente:",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

        def func_costo(x):
            return np.sin(x) + 0.3 * (x ** 2) + 2

        def gradiente_costo(x):
            return np.cos(x) + 0.6 * x

        ejes = Axes(
            x_range=[-3, 5, 1],
            y_range=[0, 10, 2],
            x_length=6.5,
            y_length=4.5,
            axis_config={"color": MARRON_OSCURO, "include_numbers": False}
        ).shift(LEFT * 1.8 + DOWN * 0.3)

        lbl_x = ejes.get_x_axis_label(
            Text("Parámetro  w", font=FUENTE, font_size=15, color=MARRON_OSCURO),
            edge=RIGHT, direction=DOWN
        )
        lbl_y = ejes.get_y_axis_label(
            Text("Loss  L", font=FUENTE, font_size=15, color=MARRON_OSCURO),
            edge=UP, direction=UP
        )

        curva = ejes.plot(func_costo, color=NARANJA_TERRACOTA, stroke_width=4)
        area  = ejes.get_area(curva, opacity=0.08, color=NARANJA_TERRACOTA)

        x_minimo = -0.89
        p_minimo = ejes.c2p(x_minimo, func_costo(x_minimo))

        def crear_stickman():
            cabeza  = Circle(radius=0.15, color=MARRON_OSCURO,
                            fill_color=PAPEL_CREMA, fill_opacity=1)
            cuerpo  = Line(cabeza.get_bottom(),
                        cabeza.get_bottom() + DOWN * 0.4,
                        color=MARRON_OSCURO, stroke_width=3)
            brazos  = VGroup(
                Line(cuerpo.get_center() + UP * 0.1,
                    cuerpo.get_center() + UP * 0.3 + LEFT * 0.2,
                    color=MARRON_OSCURO, stroke_width=3),
                Line(cuerpo.get_center() + UP * 0.1,
                    cuerpo.get_center() + UP * 0.3 + RIGHT * 0.2,
                    color=MARRON_OSCURO, stroke_width=3)
            )
            piernas = VGroup(
                Line(cuerpo.get_bottom(),
                    cuerpo.get_bottom() + DOWN * 0.3 + LEFT * 0.2,
                    color=MARRON_OSCURO, stroke_width=3),
                Line(cuerpo.get_bottom(),
                    cuerpo.get_bottom() + DOWN * 0.3 + RIGHT * 0.2,
                    color=MARRON_OSCURO, stroke_width=3)
            )
            mastil  = Line(brazos[0].get_end(),
                        brazos[0].get_end() + UP * 0.5,
                        color=TINTA_NEGRA, stroke_width=2)
            bandera = Polygon(
                mastil.get_end(),
                mastil.get_end() + DOWN * 0.15 + LEFT * 0.3,
                mastil.get_end() + DOWN * 0.3,
                fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_width=0
            )
            return VGroup(cabeza, cuerpo, brazos, piernas, mastil, bandera)

        stickman = crear_stickman().scale(0.6).next_to(p_minimo, UP, buff=0)
        lbl_meta = Text("mínimo", font=FUENTE, font_size=13,
                        color=NARANJA_TERRACOTA, weight=BOLD).next_to(stickman, UP, buff=0.08)

        lr = 0.8
        eq_update = MathTex(
            r"w \leftarrow w - \underbrace{\eta}_{\text{lr}} \cdot \underbrace{\nabla L}_{\text{pendiente}}",
            font_size=30, color=MARRON_OSCURO
        )
        eq_frame = SurroundingRectangle(eq_update, color=MARRON_OSCURO,
                                        buff=0.25, corner_radius=0.1)
        eq_grupo = VGroup(eq_frame, eq_update)

        lbl_lr   = Text(f"η = {lr}  (learning rate)", font=FUENTE,
                        font_size=16, color=TINTA_NEGRA)
        lbl_grad = Text("∇L = pendiente en w", font=FUENTE,
                        font_size=16, color=NARANJA_TERRACOTA)

        txt_pos      = Text("w = 4.00", font="Monospace", font_size=16, color=MARRON_OSCURO)
        txt_grad_val = Text("∇L = —",   font="Monospace", font_size=16, color=NARANJA_TERRACOTA)
        txt_loss     = Text("L = —",    font="Monospace", font_size=16, color=ROJO_TOMATE)

        panel = VGroup(eq_grupo, lbl_lr, lbl_grad,
                    txt_pos, txt_grad_val, txt_loss
                    ).arrange(DOWN, buff=0.32, aligned_edge=LEFT
                    ).next_to(ejes, RIGHT, buff=0.55).align_to(ejes, UP).shift(DOWN * 0.3)

        self.play(Create(ejes), Write(lbl_x), Write(lbl_y), run_time=0.9)
        self.play(Create(curva), FadeIn(area), run_time=1.0)
        self.play(FadeIn(stickman, shift=UP * 0.3), Write(lbl_meta), run_time=0.6)
        self.play(FadeIn(panel, shift=LEFT * 0.2), run_time=0.5)
        self._siguiente()

        x_val = 4.0
        punto = Dot(ejes.c2p(x_val, func_costo(x_val)),
                    color=TINTA_NEGRA, radius=0.13)
        self.play(FadeIn(punto, scale=0.5))

        rastros = VGroup()

        for i in range(7):
            grad_actual = gradiente_costo(x_val)
            x_next      = x_val - lr * grad_actual
            x_next      = max(-2.8, min(4.8, x_next))

            p_actual = ejes.c2p(x_val,  func_costo(x_val))
            p_next   = ejes.c2p(x_next, func_costo(x_next))

            nuevo_pos  = Text(f"w = {x_val:.2f}",
                            font="Monospace", font_size=16, color=MARRON_OSCURO
                            ).move_to(txt_pos, aligned_edge=LEFT)
            nuevo_grad = Text(f"∇L = {grad_actual:+.2f}",
                            font="Monospace", font_size=16,
                            color=VERDE_OLIVA if abs(grad_actual) < 0.5 else NARANJA_TERRACOTA
                            ).move_to(txt_grad_val, aligned_edge=LEFT)
            nuevo_loss = Text(f"L = {func_costo(x_val):.2f}",
                            font="Monospace", font_size=16, color=ROJO_TOMATE
                            ).move_to(txt_loss, aligned_edge=LEFT)

            self.play(
                Transform(txt_pos,      nuevo_pos),
                Transform(txt_grad_val, nuevo_grad),
                Transform(txt_loss,     nuevo_loss),
                run_time=0.35
            )

            dx      = 0.55
            t_start = ejes.c2p(x_val - dx, func_costo(x_val) - grad_actual * dx)
            t_end   = ejes.c2p(x_val + dx, func_costo(x_val) + grad_actual * dx)
            tangente = Line(t_start, t_end, color=AZUL_NOCHE,
                            stroke_width=2.5).set_opacity(0.55)
            self.play(Create(tangente), run_time=0.25)

            angulo_salto = -TAU / 6 if grad_actual > 0 else TAU / 6
            flecha_salto = CurvedArrow(p_actual, p_next, angle=angulo_salto,
                                    color=NARANJA_TERRACOTA, stroke_width=2.8)
            self.play(Create(flecha_salto), run_time=0.35)

            rastro = Dot(p_actual, color=MARRON_OSCURO, radius=0.055).set_opacity(0.35)
            rastros.add(rastro)
            self.add(rastro)

            self.play(
                punto.animate.move_to(p_next),
                FadeOut(tangente),
                FadeOut(flecha_salto),
                run_time=0.5
            )
            x_val = x_next

        self.play(
            Wiggle(stickman, scale_value=1.18, rotation_angle=0.04 * TAU),
            Flash(punto, color=NARANJA_TERRACOTA, line_length=0.45,
                flash_radius=0.32, num_lines=12),
            run_time=1.0
        )
        self._siguiente()

        adornos[1].clear_updaters()
        self.play(
            FadeOut(ejes), FadeOut(lbl_x), FadeOut(lbl_y),
            FadeOut(curva), FadeOut(area),
            FadeOut(stickman), FadeOut(lbl_meta),
            FadeOut(punto), FadeOut(rastros),
            FadeOut(panel), FadeOut(adornos)
        )
        self.limpiar_pantalla()


    def slide_backpropagation(self):
        titulo, linea = self.crear_titulo(
            "Backpropagation: Regla de la Cadena",
            palabra_clave="Regla de la Cadena",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        pregunta = Text(
            "¿Qué parámetro tuvo la culpa?",
            font=FUENTE, font_size=24, weight=BOLD,
            color=TINTA_NEGRA, line_spacing=1.4
        ).move_to(UP * 1.8)
        self.play(Write(pregunta))
        self._siguiente()

        cadena_palabras = ["L", "GELU", "suma", "mul", "w₀"]
        cadena_nodos = VGroup()
        for i, p in enumerate(cadena_palabras):
            rect = RoundedRectangle(corner_radius=0.1, width=1.4, height=0.6,
                                    fill_color=PAPEL_CREMA, fill_opacity=0.9,
                                    stroke_color=MARRON_OSCURO, stroke_width=2)
            lbl = Text(p, font=FUENTE, font_size=20,
                       color=NARANJA_TERRACOTA if i == 0 else TINTA_NEGRA, weight=BOLD)
            lbl.move_to(rect)
            cadena_nodos.add(VGroup(rect, lbl))

        cadena_nodos.arrange(RIGHT, buff=0.55).next_to(pregunta, DOWN, buff=0.55)

        flechas_cadena = VGroup(*[
            Arrow(cadena_nodos[i].get_right(), cadena_nodos[i + 1].get_left(),
                  buff=0.08, color=MARRON_OSCURO,
                  max_tip_length_to_length_ratio=0.3, stroke_width=2.5)
            for i in range(len(cadena_nodos) - 1)
        ])

        lbl_cadena = Text("¿Cuánto contribuyó cada uno?",
                          font=FUENTE, font_size=19, color=MARRON_OSCURO
                          ).next_to(cadena_nodos, DOWN, buff=0.35)

        animaciones_lagged = []
        for i in range(len(cadena_nodos)):
            anims = [FadeIn(cadena_nodos[i], shift=RIGHT * 0.15)]
            if i < len(flechas_cadena):
                anims.append(Create(flechas_cadena[i]))
            animaciones_lagged.append(AnimationGroup(*anims))

        if animaciones_lagged:
            self.play(LaggedStart(*animaciones_lagged, lag_ratio=0.2))

        self.play(Write(lbl_cadena))

        self.play(FadeOut(pregunta), FadeOut(cadena_nodos),
                  FadeOut(flechas_cadena), FadeOut(lbl_cadena))

        lbl_herramienta = Text("Regla de la Cadena",
                               font=FUENTE, font_size=26, weight=BOLD,
                               color=TINTA_NEGRA).move_to(UP * 2.6)
        self.play(Write(lbl_herramienta))

        analogia_lbl = Text(
            "Multiplica las pendientes del camino:",
            font=FUENTE, font_size=20, color=MARRON_OSCURO, line_spacing=1.4
        ).next_to(lbl_herramienta, DOWN, buff=0.4)
        self.play(FadeIn(analogia_lbl, shift=UP * 0.2))
        self._siguiente()

        eq_cadena = MathTex(
            r"\frac{\partial L}{\partial w} = "
            r"\frac{\partial L}{\partial \hat{y}} \cdot "
            r"\frac{\partial \hat{y}}{\partial \text{sum}} \cdot "
            r"\frac{\partial \text{sum}}{\partial \text{mul}} \cdot "
            r"\frac{\partial \text{mul}}{\partial w}",
            font_size=34, color=TINTA_NEGRA
        ).next_to(analogia_lbl, DOWN, buff=0.5)

        self.play(Write(eq_cadena), run_time=2.0)
        self._siguiente()

        caja_eq = SurroundingRectangle(eq_cadena, color=NARANJA_TERRACOTA,
                                       buff=0.2, corner_radius=0.1, stroke_width=2.5)
        nota_local = Text(
            "Gradientes locales · de derecha a izquierda",
            font=FUENTE, font_size=19, color=NARANJA_TERRACOTA, line_spacing=1.3
        ).next_to(eq_cadena, DOWN, buff=0.4)
        self.play(Create(caja_eq))
        self.play(Write(nota_local))

        self.play(FadeOut(lbl_herramienta), FadeOut(analogia_lbl),
                  FadeOut(eq_cadena), FadeOut(caja_eq), FadeOut(nota_local))

        lbl_grafo = Text("forward → backward",
                         font=FUENTE, font_size=24, weight=BOLD,
                         color=TINTA_NEGRA).move_to(UP * 2.7)
        self.play(Write(lbl_grafo))

        EJE_Y = UP * 1.0

        def crear_nodo_op(texto, pos, es_texto=False):
            circ = Circle(radius=0.48, fill_color=FONDO_CAJA, fill_opacity=1,
                          stroke_color=MARRON_OSCURO, stroke_width=3)
            circ.move_to(pos)
            if es_texto:
                etq = Text(texto, font=FUENTE, font_size=15,
                           color=MARRON_OSCURO, weight=BOLD).move_to(circ)
            else:
                etq = MathTex(texto, font_size=38,
                              color=MARRON_OSCURO).move_to(circ)
            return VGroup(circ, etq), circ

        grp_mul, nd_mul = crear_nodo_op(r"\times", LEFT * 3.5 + EJE_Y)
        grp_sum, nd_sum = crear_nodo_op(r"+", LEFT * 0.0 + EJE_Y)
        grp_gelu, nd_gelu = crear_nodo_op("GELU", RIGHT * 3.5 + EJE_Y, es_texto=True)

        tx0 = MathTex("x_0", font_size=30, color=TINTA_NEGRA
                      ).move_to(nd_mul.get_left() + LEFT * 1.4 + UP * 0.55)
        tw0 = MathTex("w_0", font_size=30, color=AZUL_NOCHE
                      ).move_to(nd_mul.get_left() + LEFT * 1.4 + DOWN * 0.55)
        tb = MathTex("b", font_size=30, color=AZUL_NOCHE
                     ).move_to(nd_sum.get_bottom() + DOWN * 1.1)
        ty = MathTex(r"\hat{y}", font_size=34, color=NARANJA_TERRACOTA
                     ).move_to(nd_gelu.get_right() + RIGHT * 1.4)

        def arista(a, b, color=MARRON_OSCURO, sw=2):
            return Line(a, b, color=color, stroke_width=sw, z_index=-1).add_tip(
                tip_length=0.18, tip_width=0.18)

        f_x0 = arista(tx0.get_right(), nd_mul.get_left() + UP * 0.2)
        f_w0 = arista(tw0.get_right(), nd_mul.get_left() + DOWN * 0.2, color=AZUL_NOCHE)
        f_b = arista(tb.get_top(), nd_sum.get_bottom(), color=AZUL_NOCHE)
        f_ms = arista(nd_mul.get_right(), nd_sum.get_left())
        f_sg = arista(nd_sum.get_right(), nd_gelu.get_left())
        f_out = arista(nd_gelu.get_right(), ty.get_left(), color=NARANJA_TERRACOTA)

        lbl_ms = Text("mul", font=FUENTE, font_size=14,
                      color=MARRON_OSCURO).next_to(f_ms, UP, buff=0.08)
        lbl_sg = Text("sum", font=FUENTE, font_size=14,
                      color=MARRON_OSCURO).next_to(f_sg, UP, buff=0.08)

        grafo_fwd = VGroup(grp_mul, grp_sum, grp_gelu,
                           tx0, tw0, tb, ty,
                           f_x0, f_w0, f_b, f_ms, f_sg, f_out,
                           lbl_ms, lbl_sg)

        lbl_fwd = Text("① Forward pass", font=FUENTE, font_size=18,
                       weight=BOLD, color=VERDE_OLIVA).next_to(lbl_grafo, DOWN, buff=0.15)
        self.play(Write(lbl_fwd))
        self.play(LaggedStart(
            AnimationGroup(FadeIn(tx0), FadeIn(tw0)),
            AnimationGroup(Create(f_x0), Create(f_w0)),
            DrawBorderThenFill(grp_mul),
            AnimationGroup(Create(f_ms), FadeIn(lbl_ms), FadeIn(tb), Create(f_b)),
            DrawBorderThenFill(grp_sum),
            AnimationGroup(Create(f_sg), FadeIn(lbl_sg)),
            DrawBorderThenFill(grp_gelu),
            AnimationGroup(Create(f_out), Write(ty)),
            lag_ratio=0.25, run_time=2.5
        ))
        self._siguiente()

        lbl_bwd = Text("② Backward pass  (gradientes de derecha a izquierda)",
                       font=FUENTE, font_size=18, weight=BOLD,
                       color=NARANJA_TERRACOTA).move_to(lbl_fwd)
        self.play(ReplacementTransform(lbl_fwd, lbl_bwd))

        rutas_back = [
            (nd_gelu.get_left(), nd_sum.get_right(), r"\partial\hat{y}/\partial\text{sum}"),
            (nd_sum.get_left(), nd_mul.get_right(), r"\partial\text{sum}/\partial\text{mul}"),
            (nd_mul.get_left() + DOWN * 0.2, tw0.get_right(), r"\partial\text{mul}/\partial w_0"),
            (nd_sum.get_bottom(), tb.get_top(), r"\partial\text{sum}/\partial b"),
        ]

        for p_start, p_end, grad_tex in rutas_back:
            flash_line = Line(p_start, p_end, color=NARANJA_TERRACOTA, stroke_width=5)
            grad_lbl = MathTex(grad_tex, font_size=20, color=NARANJA_TERRACOTA
                               ).move_to(flash_line.get_center() + UP * 0.35)
            self.play(ShowPassingFlash(flash_line, time_width=0.55), run_time=0.7)
            self.play(FadeIn(grad_lbl, shift=UP * 0.1), run_time=0.35)
            self.play(FadeOut(grad_lbl), run_time=0.25)

        caja_w0 = SurroundingRectangle(tw0, color=NARANJA_TERRACOTA,
                                       buff=0.1, corner_radius=0.08, stroke_width=2.5)
        caja_b = SurroundingRectangle(tb, color=NARANJA_TERRACOTA,
                                      buff=0.1, corner_radius=0.08, stroke_width=2.5)
        lbl_upd_w = MathTex(r"-\eta\,\Delta w_0", font_size=20,
                            color=NARANJA_TERRACOTA).next_to(caja_w0, LEFT, buff=0.15)
        lbl_upd_b = MathTex(r"-\eta\,\Delta b", font_size=20,
                            color=NARANJA_TERRACOTA).next_to(caja_b, LEFT, buff=0.15)

        self.play(Create(caja_w0), Create(caja_b))
        self.play(FadeIn(lbl_upd_w, shift=RIGHT * 0.15),
                  FadeIn(lbl_upd_b, shift=RIGHT * 0.15))
        self.play(
            Indicate(lbl_upd_w, color=ORO_VIEJO, scale_factor=1.2),
            Indicate(lbl_upd_b, color=ORO_VIEJO, scale_factor=1.2),
        )

        self.play(FadeOut(grafo_fwd), FadeOut(lbl_bwd),
                  FadeOut(caja_w0), FadeOut(caja_b),
                  FadeOut(lbl_upd_w), FadeOut(lbl_upd_b), FadeOut(lbl_grafo))

        lbl_red = Text("En la red completa: millones de parámetros, mismo principio",
                       font=FUENTE, font_size=22, weight=BOLD,
                       color=TINTA_NEGRA).move_to(UP * 2.7)
        self.play(Write(lbl_red))

        capas_config = [3, 5, 4, 2]
        nodos_red = VGroup()
        for i, n in enumerate(capas_config):
            capa = VGroup(*[
                Circle(radius=0.22, fill_color=FONDO_CAJA, fill_opacity=1,
                       stroke_color=MARRON_OSCURO, stroke_width=2.5)
                for _ in range(n)
            ]).arrange(DOWN, buff=0.45)
            capa.move_to(RIGHT * (i * 2.6 - 3.9) + DOWN * 0.35)
            nodos_red.add(capa)

        conexiones_fwd_red = VGroup()
        conexiones_back_por_capa = []
        for i in range(len(capas_config) - 1):
            grupo_back = VGroup()
            for n1 in nodos_red[i]:
                for n2 in nodos_red[i + 1]:
                    ln_fwd = Line(n1.get_right(), n2.get_left(),
                                  stroke_width=1.5, color=MARRON_OSCURO,
                                  z_index=-1).set_opacity(0.25)
                    conexiones_fwd_red.add(ln_fwd)
                    ln_back = Line(n2.get_left(), n1.get_right(),
                                   stroke_width=4, color=NARANJA_TERRACOTA)
                    grupo_back.add(ln_back)
            conexiones_back_por_capa.append(grupo_back)

        nombres_capas = ["Entrada", "Capa 1", "Capa 2", "Salida"]
        lbls_capas = VGroup(*[
            Text(nombres_capas[i], font=FUENTE, font_size=14, color=MARRON_OSCURO
                 ).next_to(nodos_red[i], DOWN, buff=0.3)
            for i in range(len(capas_config))
        ])

        self.play(LaggedStart(
            *[GrowFromCenter(n) for capa in nodos_red for n in capa],
            lag_ratio=0.06
        ), run_time=1.2)
        self.play(Create(conexiones_fwd_red), FadeIn(lbls_capas), run_time=1.0)

        txt_loss_red = MathTex(r"L", font_size=30,
                               color=ROJO_TOMATE).next_to(nodos_red[-1], RIGHT, buff=0.5)
        self.play(Write(txt_loss_red),
                  Indicate(nodos_red[-1], color=ROJO_TOMATE, scale_factor=1.12))
        self._siguiente()

        lbl_bwd_red = Text("Gradientes fluyendo hacia atrás →",
                           font=FUENTE, font_size=19, weight=BOLD,
                           color=NARANJA_TERRACOTA).next_to(lbl_red, DOWN, buff=0.18)
        self.play(Write(lbl_bwd_red))

        for i in reversed(range(len(capas_config) - 1)):
            destellos = [ShowPassingFlash(l.copy(), time_width=0.45)
                         for l in conexiones_back_por_capa[i]]
            self.play(
                AnimationGroup(*destellos),
                Indicate(nodos_red[i], color=NARANJA_TERRACOTA, scale_factor=1.1),
                run_time=1.0
            )

        conclusion = Text(
            "124M params · un solo paso",
            font=FUENTE, font_size=20, weight=BOLD,
            color=NARANJA_TERRACOTA, line_spacing=1.3
        ).next_to(nodos_red, DOWN, buff=0.55).set_x(0)
        self.play(Write(conclusion))
        self.play(Indicate(conclusion, color=ORO_VIEJO, scale_factor=1.05))
        self._siguiente()

        self.limpiar_pantalla()


    def slide_adam(self):

        titulo_intro, linea_intro = self.crear_titulo(
            "Optimizador Adam",
            palabra_clave="Adam",
            color_clave=NARANJA_TERRACOTA
        )
        self.play(Write(titulo_intro), GrowFromCenter(linea_intro))

        formula_adam = MathTex(
            r"w_{t + 1} \leftarrow w_{t} - ",
            r"\frac{\eta}{\sqrt{v_{t}} + \epsilon} ",
            r"\cdot m_{t}",
            color=TINTA_NEGRA, font_size=60
        ).shift(UP * 2)

        self.play(Write(formula_adam))
        self._siguiente()

        ejes_m = Axes(x_range=[0, 4], y_range=[0, 2], x_length=4, y_length=2).shift(LEFT*3.2 + DOWN*1.5)

        func_m = lambda x: 0.5 * np.cos(3*x) - 0.2 * x + 1.5
        terreno_m = ejes_m.plot(func_m, color=MARRON_OSCURO)

        lbl_m = Text("m: Momentum", font_size=24, color=NARANJA_TERRACOTA).next_to(ejes_m, UP)

        bola_gd = Dot(color=MARRON_OSCURO).move_to(ejes_m.c2p(0.5, func_m(0.5)))
        bola_adam = Dot(color=NARANJA_TERRACOTA).move_to(ejes_m.c2p(0.5, func_m(0.5)))

        self.play(
            formula_adam[1].animate.set_opacity(0.3),
            formula_adam[2].animate.set_color(NARANJA_TERRACOTA).scale(1.2),
            Create(ejes_m), Create(terreno_m), Write(lbl_m),
            FadeIn(bola_gd), FadeIn(bola_adam)
        )

        tracker_gd = ValueTracker(0.5)
        tracker_adam = ValueTracker(0.5)

        bola_gd.add_updater(lambda d: d.move_to(ejes_m.c2p(tracker_gd.get_value(), func_m(tracker_gd.get_value()))))
        bola_adam.add_updater(lambda d: d.move_to(ejes_m.c2p(tracker_adam.get_value(), func_m(tracker_adam.get_value()))))

        self.play(
            tracker_gd.animate.set_value(1.2),
            tracker_adam.animate.set_value(3.2),
            run_time=2.5, rate_func=smooth
        )

        bola_gd.clear_updaters()
        bola_adam.clear_updaters()
        self._siguiente()

        ejes_v = Axes(x_range=[0, 4], y_range=[0, 4], x_length=4, y_length=2).shift(RIGHT*3.2 + DOWN*1.5)
        func_v = lambda x: 0.8 * (x - 2.5)**2 + 0.5
        terreno_v = ejes_v.plot(func_v, color=MARRON_OSCURO)

        lbl_v = Text("v: Varianza", font_size=24, color=MARRON_OSCURO).next_to(ejes_v, UP)

        self.play(
            formula_adam[2].animate.set_color(TINTA_NEGRA).set_opacity(0.3).scale(1/1.2),
            formula_adam[1].animate.set_opacity(1).set_color(NARANJA_TERRACOTA).scale(1.2),
            Create(ejes_v), Create(terreno_v), Write(lbl_v)
        )

        x_inicio = 0.8
        punto_base = Dot(ejes_v.c2p(x_inicio, func_v(x_inicio)), color=TINTA_NEGRA)
        self.play(FadeIn(punto_base))

        x_sobreimpulso = 3.8
        flecha_mala = Arrow(
            start=ejes_v.c2p(x_inicio, func_v(x_inicio)),
            end=ejes_v.c2p(x_sobreimpulso, func_v(x_sobreimpulso)),
            color=MARRON_OSCURO, buff=0.1
        )
        lbl_mala = Text("Gradiente puro: paso gigante", font_size=16, color=MARRON_OSCURO).next_to(flecha_mala, UP)

        self.play(GrowArrow(flecha_mala), Write(lbl_mala))
        self.wait(0.5)

        x_ideal = 2.0
        flecha_buena = Arrow(
            start=ejes_v.c2p(x_inicio, func_v(x_inicio)),
            end=ejes_v.c2p(x_ideal, func_v(x_ideal)),
            color=NARANJA_TERRACOTA, buff=0.1
        )
        lbl_buena = Text("Adam: paso adaptativo", font_size=16, color=NARANJA_TERRACOTA).next_to(flecha_buena, DOWN)

        self.play(
            ReplacementTransform(flecha_mala, flecha_buena),
            ReplacementTransform(lbl_mala, lbl_buena)
        )
        self.wait(0.5)

        tracker_v = ValueTracker(x_inicio)
        punto_v = Dot(color=NARANJA_TERRACOTA)
        punto_v.add_updater(lambda d: d.move_to(ejes_v.c2p(tracker_v.get_value(), func_v(tracker_v.get_value()))))

        self.add(punto_v)
        self.remove(punto_base)

        self.play(
            tracker_v.animate.set_value(x_ideal),
            run_time=2,
            rate_func=smooth
        )
        punto_v.clear_updaters()

        self.play(FadeOut(VGroup(
            ejes_m, terreno_m, lbl_m, bola_gd, bola_adam,
            ejes_v, terreno_v, lbl_v, flecha_buena, lbl_buena, punto_v,
            formula_adam, titulo_intro, linea_intro
        )))

        titulo, linea = self.crear_titulo("ADAM vs GD: El Descenso Final", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        def func_costo(x):
            return np.cos(3*x)*0.5 + 0.2 * (x**2) + 2

        ejes_gd = Axes(x_range=[-3, 5, 1], y_range=[0, 8, 2], x_length=5.5, y_length=4,
                       axis_config={"color": MARRON_OSCURO}).shift(LEFT * 3.4 + DOWN * 0.5)
        ejes_adam = Axes(x_range=[-3, 5, 1], y_range=[0, 8, 2], x_length=5.5, y_length=4,
                         axis_config={"color": NARANJA_TERRACOTA}).shift(RIGHT * 3.4 + DOWN * 0.5)

        lbl_titulo_gd = Text("Gradient Descent Normal", font_size=24, color=MARRON_OSCURO).next_to(ejes_gd, UP)
        lbl_titulo_adam = Text("Optimizador ADAM", font_size=24, color=NARANJA_TERRACOTA).next_to(ejes_adam, UP)

        self.play(
            Create(VGroup(ejes_gd, ejes_gd.plot(func_costo, color=MARRON_OSCURO))),
            Create(VGroup(ejes_adam, ejes_adam.plot(func_costo, color=NARANJA_TERRACOTA))),
            Write(lbl_titulo_gd),
            Write(lbl_titulo_adam)
        )

        pasos_gd, pasos_adam = [4.0], [4.0]
        x_g, x_a = 4.0, 4.0
        m, v = 0, 0

        for t in range(1, 60):
            g_a = -np.sin(3*x_a)*1.5 + 0.4*x_a
            m = 0.9 * m + 0.1 * g_a
            v = 0.999 * v + 0.001 * (g_a**2)
            m_hat = m / (1 - 0.9**t)
            v_hat = v / (1 - 0.999**t)

            if abs(x_a) < 0.05 and t > 25:
                x_a = 0.0
            else:
                x_a -= 0.35 * m_hat / (np.sqrt(v_hat) + 1e-8)

            pasos_adam.append(x_a)
            g_g = -np.sin(3*x_g)*1.5 + 0.4*x_g
            x_g -= 0.12 * g_g
            pasos_gd.append(x_g)

        trail_gd_points = [ejes_gd.c2p(x, func_costo(x)) for x in pasos_gd]
        trail_adam_points = [ejes_adam.c2p(x, func_costo(x)) for x in pasos_adam]

        linea_rastro_gd = VMobject(color=MARRON_OSCURO).set_points_smoothly(trail_gd_points)
        linea_rastro_adam = VMobject(color=NARANJA_TERRACOTA).set_points_smoothly(trail_adam_points)

        punto_gd = Dot(trail_gd_points[0], color=MARRON_OSCURO)
        punto_adam = Dot(trail_adam_points[0], color=NARANJA_TERRACOTA)

        self.play(FadeIn(punto_gd), FadeIn(punto_adam))

        self.play(
            MoveAlongPath(punto_gd, linea_rastro_gd),
            Create(linea_rastro_gd),
            MoveAlongPath(punto_adam, linea_rastro_adam),
            Create(linea_rastro_adam),
            run_time=4, rate_func=smooth
        )

        lbl_resultado_gd = Text("Mínimo Local", font_size=20, color=MARRON_OSCURO).next_to(punto_gd, DOWN)
        lbl_resultado_adam = Text("Mínimo Global", font_size=20, color=NARANJA_TERRACOTA).next_to(punto_adam, DOWN)

        self.play(
            Write(lbl_resultado_gd),
            Write(lbl_resultado_adam),
            Flash(punto_adam, color=NARANJA_TERRACOTA)
        )

        self._siguiente()

        panel_textos = VGroup(
            Text("¿Por qué ganó ADAM?", font_size=26, color=TINTA_NEGRA, weight=BOLD),
            Text("• El Momentum saltó los baches locales.", font_size=20, color=MARRON_OSCURO),
            Text("• La Varianza controló los pasos bruscos.", font_size=20, color=MARRON_OSCURO)
        ).arrange(DOWN, aligned_edge=LEFT)

        fondo_panel = Rectangle(
            width=panel_textos.width + 1, height=panel_textos.height + 0.7,
            color=PAPEL_CREMA, fill_color=FONDO_CAJA, fill_opacity=1
        )
        panel_final = VGroup(fondo_panel, panel_textos).to_edge(DOWN, buff=0.2)

        self.play(FadeIn(panel_final, shift=UP))
        self._siguiente()
        self.limpiar_pantalla()


    def slide_dropout(self):

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Dropout: El Arte de Olvidar",
            palabra_clave="Dropout",
            color_clave=NARANJA_TERRACOTA
        )

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))


        tamaños_capas = [4, 5, 5, 5, 4]
        colores_capas = [BLUE_D, NARANJA_TERRACOTA, NARANJA_TERRACOTA, NARANJA_TERRACOTA, GREEN_D]
        nombres_capas = ["Input", "Dense 1", "Dense 2", "Dense 3", "Output"]

        nodos = VGroup()
        etiquetas = VGroup()

        for size, color, nombre in zip(tamaños_capas, colores_capas, nombres_capas):
            capa = VGroup(*[Dot(radius=0.15, color=color) for _ in range(size)]).arrange(DOWN, buff=0.4)
            nodos.add(capa)
            etiqueta = Text(nombre, font=FUENTE, font_size=16, color=MARRON_OSCURO, weight=BOLD)
            etiquetas.add(etiqueta)

        nodos.arrange(RIGHT, buff=1.8).shift(DOWN * 0.2)

        for i, etiqueta in enumerate(etiquetas):
            etiqueta.next_to(nodos[i], UP, buff=0.3)

        conexiones = VGroup()
        for i in range(len(tamaños_capas) - 1):
            capa_act = nodos[i]
            capa_sig = nodos[i+1]
            grupo_conexiones = VGroup()
            for n1 in capa_act:
                for n2 in capa_sig:
                    grupo_conexiones.add(Line(n1.get_center(), n2.get_center(),
                                             stroke_width=1.5, color=MARRON_OSCURO, stroke_opacity=0.3))
            conexiones.add(grupo_conexiones)

        red_grupo = VGroup(conexiones, nodos, etiquetas)
        self.play(FadeIn(red_grupo))


        txt_problema = Text(
            "Durante el entrenamiento, no todas las neuronas aprenden por igual...",
            font=FUENTE, font_size=20, color=TINTA_NEGRA
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(txt_problema, shift=UP))
        self._siguiente()
        self.play(FadeOut(txt_problema))


        dominantes = {1: [0, 2], 2: [1, 3], 3: [0, 2]}
        ociosas    = {1: [1, 3, 4], 2: [0, 2, 4], 3: [1, 3, 4]}

        anims_dom = []
        for capa_idx, idxs in dominantes.items():
            for idx in idxs:
                anims_dom.append(
                    nodos[capa_idx][idx].animate.set_color(ORO_VIEJO).set_opacity(1.0)
                )
        self.play(*anims_dom, run_time=1.0)

        txt_dominantes = Text(
            "Unas pocas neuronas acaparan la mayor parte de la representación.",
            font=FUENTE, font_size=20, color=TINTA_NEGRA, weight=BOLD
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(txt_dominantes, shift=UP))
        self._siguiente()
        self.play(FadeOut(txt_dominantes))


        anims_oci = []
        for capa_idx, idxs in ociosas.items():
            for idx in idxs:
                anims_oci.append(
                    nodos[capa_idx][idx].animate.set_color(ACERO).set_opacity(0.25)
                )
        self.play(*anims_oci, run_time=1.0)

        txt_ociosas = Text(
            "Mientras tanto, otras apenas participan y quedan relegadas.",
            font=FUENTE, font_size=20, color=TINTA_NEGRA
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(txt_ociosas, shift=UP))
        self._siguiente()
        self.play(FadeOut(txt_ociosas))


        txt_consecuencia = Text(
            "El modelo deja de aprender patrones… y comienza a memorizar.",
            font=FUENTE, font_size=20, color=TINTA_NEGRA, weight=BOLD
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(txt_consecuencia, shift=UP))
        self._siguiente()
        self.play(FadeOut(txt_consecuencia))


        anims_reset = []
        for capa_idx, color in enumerate(colores_capas):
            for nodo in nodos[capa_idx]:
                anims_reset.append(nodo.animate.set_color(color).set_opacity(1.0))
        for grupo in conexiones:
            for linea in grupo:
                anims_reset.append(linea.animate.set_stroke(opacity=0.3))
        self.play(*anims_reset, run_time=0.8)


        txt_solucion = Text(
            "Dropout: forzar a la red a no depender de nadie en particular.",
            font=FUENTE, font_size=20, color=TINTA_NEGRA, weight=BOLD
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(txt_solucion, shift=UP))
        self._siguiente()
        self.play(FadeOut(txt_solucion))


        def aplicar_dropout_capa(indice_capa, indices_apagar, texto_explicativo):
            animaciones = []
            capa = nodos[indice_capa]

            for idx in indices_apagar:
                animaciones.append(capa[idx].animate.set_color(ACERO).set_opacity(0.2))

                if indice_capa > 0:
                    for n1_idx in range(tamaños_capas[indice_capa - 1]):
                        line_idx = n1_idx * tamaños_capas[indice_capa] + idx
                        animaciones.append(conexiones[indice_capa - 1][line_idx].animate.set_stroke(opacity=0.02))

                if indice_capa < len(tamaños_capas) - 1:
                    for n2_idx in range(tamaños_capas[indice_capa + 1]):
                        line_idx = idx * tamaños_capas[indice_capa + 1] + n2_idx
                        animaciones.append(conexiones[indice_capa][line_idx].animate.set_stroke(opacity=0.02))

            texto = Text(texto_explicativo, font=FUENTE, font_size=20, color=TINTA_NEGRA).to_edge(DOWN, buff=0.5)
            self.play(*animaciones, FadeIn(texto, shift=UP), run_time=1.5)
            self.play(FadeOut(texto))

        aplicar_dropout_capa(1, [1, 4],
            "Aplicar dropout en Dense 1"
        )

        aplicar_dropout_capa(2, [0, 2, 3],
            "Aplicar dropout en Dense 2"
        )

        aplicar_dropout_capa(3, [1, 4],
            "Aplicar dropout en Dense 3"
        )

        texto_metafora = Text(
            "\"No levantes un reino sobre un solo guerrero;\n"
            "haz que cada espada sepa luchar por sí misma.\"",
            font=FUENTE, font_size=24, color=TINTA_NEGRA, slant=ITALIC
        ).to_edge(DOWN, buff=0.5)

        self.play(Write(texto_metafora))
        self._siguiente()
        self.limpiar_pantalla()


    def slide_training_metrics(self):

        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))

        titulo, linea = self.crear_titulo(
            "Métricas de Entrenamiento: Loss y Perplejidad",
            palabra_clave="Métricas",
            color_clave=NARANJA_TERRACOTA
        )
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        lbl_a1 = Text("Loss = sorpresa del modelo ante la palabra correcta",
                      font=FUENTE, font_size=25, weight=BOLD, color=TINTA_NEGRA
                      ).next_to(linea, DOWN, buff=0.38)
        self.play(FadeIn(lbl_a1, shift=DOWN * 0.1))

        ctx_bg  = RoundedRectangle(corner_radius=0.14, width=7.8, height=0.62,
                                    fill_color=CAJA_INFERIOR, fill_opacity=0.65,
                                    stroke_color=MARRON_OSCURO, stroke_width=1.8)
        ctx_txt = Text('"...Fortuna es una mujer  ___"', font=FUENTE, font_size=20,
                       color=TINTA_NEGRA, t2c={"___": NARANJA_TERRACOTA}).move_to(ctx_bg)
        ctx_grp = VGroup(ctx_bg, ctx_txt).next_to(lbl_a1, DOWN, buff=0.32)
        self.play(FadeIn(ctx_grp))

        palabras_pred = ["borracha", "bella", "rica", "alta"]
        probs_mal     = [0.06, 0.29, 0.22, 0.43]
        probs_bien    = [0.74, 0.12, 0.09, 0.05]
        BAR_W, BAR_MAX_H = 0.52, 1.55
        COL_SEP = BAR_W + 0.32

        def _panel_barras(probs, color_winner, titulo_str, loss_str, loss_color):
            rects, pcts, wrds = [], [], []
            for i, (p, pal) in enumerate(zip(probs, palabras_pred)):
                h = max(0.06, p * BAR_MAX_H)
                rect = Rectangle(width=BAR_W, height=h,
                                 fill_color=color_winner if i == 0 else PAPEL_TAN,
                                 fill_opacity=0.88, stroke_width=1.1,
                                 stroke_color=MARRON_OSCURO)
                rect.move_to(np.array([i * COL_SEP, h / 2, 0]))
                pct = Text(f"{int(p*100)}%", font="Monospace", font_size=13,
                           color=TINTA_NEGRA).move_to(np.array([i * COL_SEP, h + 0.19, 0]))
                wrd = Text(pal, font=FUENTE, font_size=13,
                           color=TINTA_NEGRA).move_to(np.array([i * COL_SEP, -0.25, 0]))
                rects.append(rect); pcts.append(pct); wrds.append(wrd)

            barras = VGroup(*[VGroup(r, p, w) for r, p, w in zip(rects, pcts, wrds)])
            barras.center()
            tit      = Text(titulo_str, font=FUENTE, font_size=17, weight=BOLD,
                            color=color_winner).next_to(barras, UP, buff=0.35)
            loss_lbl = Text(loss_str, font="Monospace", font_size=16, weight=BOLD,
                            color=loss_color).next_to(barras, DOWN, buff=0.28)
            return VGroup(tit, barras, loss_lbl)

        panel_mal  = _panel_barras(probs_mal,  ROJO_TOMATE, "Sin entrenar", "Loss ≈ 8.1", ROJO_TOMATE)
        panel_bien = _panel_barras(probs_bien, VERDE_OLIVA, "Entrenado",   "Loss ≈ 2.5", VERDE_OLIVA)

        comparacion = VGroup(panel_mal, panel_bien).arrange(RIGHT, buff=1.6)
        comparacion.next_to(ctx_grp, DOWN, buff=0.28).set_x(0)

        sep = DashedLine(
            comparacion.get_top() + UP * 0.1,
            comparacion.get_bottom() + DOWN * 0.1,
            color=MARRON_OSCURO, stroke_width=1.2, dash_length=0.10,
        ).set_x(comparacion.get_center()[0])

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.12) for b in panel_mal[1]], lag_ratio=0.1),
            FadeIn(panel_mal[0]), run_time=0.9,
        )
        self.play(Write(panel_mal[2]))
        self.play(Create(sep))
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.12) for b in panel_bien[1]], lag_ratio=0.1),
            FadeIn(panel_bien[0]), run_time=0.9,
        )
        self.play(Write(panel_bien[2]))

        formula = MathTex(r"L = -\log P(\text{borracha}\mid\text{contexto})",
                          font_size=22, color=MARRON_OSCURO)
        formula.next_to(comparacion, DOWN, buff=0.25)
        self.play(Write(formula))

        self._siguiente()
        self.play(FadeOut(lbl_a1, ctx_grp, panel_mal, panel_bien, sep, formula))


        lbl_a2 = Text("Perplejidad = ¿cuántas opciones baraja el modelo?",
                      font=FUENTE, font_size=24, weight=BOLD, color=TINTA_NEGRA
                      ).next_to(linea, DOWN, buff=0.38)
        self.play(Write(lbl_a2))

        eq_ppl = MathTex(r"PPL = e^{\,L}", font_size=46, color=NARANJA_TERRACOTA)
        eq_ppl.next_to(lbl_a2, DOWN, buff=0.38)
        self.play(Write(eq_ppl))

        def _tarjeta_ppl(ppl_label, descripcion, n_boxes, color):
            tit = Text(f"PPL {ppl_label}", font="Monospace", font_size=24,
                       weight=BOLD, color=color)
            n_show = min(n_boxes, 7)
            cajas_viz = VGroup(*[
                RoundedRectangle(corner_radius=0.05, width=0.40, height=0.32,
                                 fill_color=color if i == 0 else PAPEL_TAN,
                                 fill_opacity=0.72 if i == 0 else 0.38,
                                 stroke_color=MARRON_OSCURO, stroke_width=1.0)
                for i in range(n_show)
            ]).arrange(RIGHT, buff=0.07)
            if n_boxes > n_show:
                puntos_lbl = Text("···", font="Monospace", font_size=16,
                                  color=MARRON_OSCURO).next_to(cajas_viz, RIGHT, buff=0.1)
                cajas_viz = VGroup(cajas_viz, puntos_lbl)
            desc = Text(descripcion, font=FUENTE, font_size=15,
                        color=MARRON_OSCURO, line_spacing=1.2)
            contenido = VGroup(tit, cajas_viz, desc).arrange(DOWN, buff=0.22)
            fondo = SurroundingRectangle(contenido, color=color,
                                         fill_color=FONDO_CAJA, fill_opacity=1,
                                         buff=0.30, corner_radius=0.16, stroke_width=2.2)
            return VGroup(fondo, contenido)

        card_alto = _tarjeta_ppl("3 360", "Elegir entre\n3360 opciones.", 3360, ROJO_TOMATE)
        card_bajo = _tarjeta_ppl("17", "Elegir entre\n17 opciones.", 17, VERDE_OLIVA)

        tarjetas = VGroup(card_alto, card_bajo).arrange(RIGHT, buff=1.1)
        tarjetas.next_to(eq_ppl, DOWN, buff=0.38).set_x(0)
        if tarjetas.width > 12.8:
            tarjetas.scale(12.8 / tarjetas.width)

        self.play(FadeIn(card_alto, scale=0.92, shift=RIGHT * 0.15), run_time=0.8)
        self.play(FadeIn(card_bajo, scale=0.92, shift=LEFT * 0.15), run_time=0.8)

        flecha_t = Arrow(card_alto.get_right(), card_bajo.get_left(),
                         color=VERDE_OLIVA, stroke_width=3,
                         max_tip_length_to_length_ratio=0.2)
        lbl_flt  = Text("entrenar", font=FUENTE, font_size=14, color=VERDE_OLIVA,
                        slant=ITALIC).next_to(flecha_t, UP, buff=0.1)
        self.play(GrowArrow(flecha_t), FadeIn(lbl_flt, shift=DOWN * 0.1))

        self._siguiente()
        self.play(FadeOut(lbl_a2, eq_ppl, card_alto, card_bajo, flecha_t, lbl_flt))


        lbl_a3 = Text("A medida que el Loss cae, el texto mejora",
                      font=FUENTE, font_size=25, weight=BOLD, color=TINTA_NEGRA
                      ).next_to(linea, DOWN, buff=0.38)
        self.play(Write(lbl_a3))

        def curva_loss(t):
            return 2.85 + 5.27 * np.exp(-t / 4200)

        ax = Axes(
            x_range=[0, 16000, 4000],
            y_range=[0, 9, 3],
            x_length=6.2,
            y_length=3.6,
            axis_config={"include_tip": False, "color": MARRON_OSCURO},
        ).shift(LEFT * 2.0 + DOWN * 0.5)

        x_lbl = Text("Pasos", font=FUENTE, font_size=13,
                     color=MARRON_OSCURO).next_to(ax, DOWN, buff=0.15)
        y_lbl = Text("Loss", font=FUENTE, font_size=13,
                     color=MARRON_OSCURO).next_to(ax, LEFT, buff=0.12)

        curva = ax.plot(curva_loss, x_range=[0, 16000],
                        color=NARANJA_TERRACOTA, stroke_width=4)

        self.play(Create(ax), Write(x_lbl), Write(y_lbl))
        self.play(Create(curva), run_time=2.0)

        checkpoints = [
            (0,     8.12, "3 360", ROJO_TOMATE,
             '"q7e llam8n p0r a#í\nF0rtun4 es una muj8r..."'),
            (4000,  4.45, "85",    PAPEL_TAN,
             '"Esta que llaman por\nahí Fortuna es una mujer..."'),
            (16000, 2.85, "17",    VERDE_OLIVA,
             '"Esta que llaman por ahí\nFortuna es una mujer borracha."'),
        ]

        dot_actual = None
        burbuja_actual = None

        for paso, loss_v, ppl_v, color, texto in checkpoints:
            pos_pt = ax.c2p(paso, curva_loss(paso))
            nuevo_dot = Dot(pos_pt, radius=0.12, color=color,
                            fill_opacity=1, stroke_color=BLANCO, stroke_width=1.5)

            rect_b = RoundedRectangle(corner_radius=0.16, width=4.8, height=1.62,
                                      fill_color=PAPEL_CREMA, fill_opacity=1,
                                      stroke_color=color, stroke_width=2.5)
            txt_b  = Text(texto, font=FUENTE, font_size=17, color=TINTA_NEGRA,
                          line_spacing=1.2)
            txt_b.scale_to_fit_width(rect_b.width - 0.55).move_to(rect_b)
            info_b = Text(f"Paso {paso:,}   Loss {loss_v}   PPL {ppl_v}",
                          font="Monospace", font_size=13, color=color
                          ).next_to(rect_b, DOWN, buff=0.1, aligned_edge=RIGHT)
            nueva_burbuja = VGroup(rect_b, txt_b, info_b).move_to(RIGHT * 3.2 + DOWN * 0.35)

            if dot_actual is None:
                self.play(FadeIn(nuevo_dot, scale=0.5))
                self.play(FadeIn(nueva_burbuja, shift=LEFT * 0.2))
            else:
                self.play(
                    ReplacementTransform(dot_actual, nuevo_dot),
                    FadeTransform(burbuja_actual, nueva_burbuja),
                    run_time=1.1,
                )
            dot_actual = nuevo_dot
            burbuja_actual = nueva_burbuja
            self._siguiente()

        adornos[1].clear_updaters()
        self.play(FadeOut(lbl_a3, ax, x_lbl, y_lbl, curva, dot_actual, burbuja_actual))
        self.limpiar_pantalla()

    def slide_temperature(self):

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "La Temperatura: Controlando la Locura",
            palabra_clave="Temperatura",
            color_clave=NARANJA_TERRACOTA
        )

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        label_eq = Text("Softmax con Temperatura (T)", font=FUENTE, font_size=18, color=MARRON_OSCURO)

        eq_temp = MathTex(
            r"P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}",
            color=TINTA_NEGRA
        ).scale(1.3)

        eq_temp.set_color_by_tex("T", NARANJA_TERRACOTA)

        math_group = VGroup(label_eq, eq_temp).arrange(DOWN, buff=0.3).move_to(ORIGIN).shift(UP * 0.5)

        self.play(FadeIn(label_eq, shift=DOWN*0.2), Write(eq_temp), run_time=1.5)
        self.play(eq_temp.animate.scale(1.1).set_glow(0.3), rate_func=there_and_back, run_time=1)
        self._siguiente()

        explicacion = Tex(
            r"$T \to 0$: Determinista, conservador (Como Sancho)\\$T > 1$: Aleatorio, creativo, ``alucinaciones'' (Como el Quijote)",
            font_size=28, color=MARRON_OSCURO, tex_environment="flushleft"
        ).next_to(math_group, DOWN, buff=0.8)

        self.play(FadeIn(explicacion, shift=UP))

        self.play(FadeOut(math_group, explicacion))

        rect_prompt = RoundedRectangle(corner_radius=0.15, height=1.2, width=8)
        rect_prompt.set_fill(color=MARRON_OSCURO, opacity=0.1).set_stroke(color=MARRON_OSCURO, width=1.5)

        user_label = Text("Usuario", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(rect_prompt, UP, aligned_edge=LEFT).shift(DOWN*0.1 + RIGHT*0.2)

        texto_prompt = Text(
            "Prompt: \"En un lugar de la Mancha...\"",
            font=FUENTE, font_size=22, color=TINTA_NEGRA, weight=BOLD
        ).move_to(rect_prompt)

        grupo_prompt = VGroup(rect_prompt, user_label, texto_prompt).to_edge(UP, buff=1.5)

        self.play(FadeIn(grupo_prompt, shift=DOWN))

        def crear_respuesta(temp_val, intento, texto, color_perfil, titulo_perfil):
            sombra = RoundedRectangle(corner_radius=0.15, height=2.2, width=8)
            sombra.set_fill(MARRON_OSCURO, opacity=0.1).set_stroke(width=0)
            sombra.shift(RIGHT * 0.08 + DOWN * 0.08)

            rect = RoundedRectangle(corner_radius=0.15, height=2.2, width=8)
            rect.set_fill(color=PAPEL_CREMA, opacity=1).set_stroke(color=MARRON_OSCURO, width=1.5)

            icon = Circle(radius=0.25, color=color_perfil, fill_opacity=1)
            label_icon = Text(titulo_perfil[5], font=FUENTE, font_size=20, color=BLANCO, weight=BOLD).move_to(icon)
            user_icon = VGroup(icon, label_icon).next_to(rect, LEFT, buff=0.3).shift(UP * 0.5)

            username = Text(f"{titulo_perfil} (T={temp_val})", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(rect, UP, aligned_edge=LEFT).shift(UP*0.1)

            contenido = Paragraph(
                texto, font=FUENTE, font_size=22, color=TINTA_NEGRA,
                line_spacing=1.3, alignment="left"
            ).scale_to_fit_width(rect.width - 0.8).move_to(rect)

            bubble_group = VGroup(sombra, rect, user_icon, username, contenido)

            info = Text(f"Generación - Intento #{intento}",
                        font="Monospace", font_size=16, color=color_perfil).next_to(rect, DOWN, buff=0.15, aligned_edge=RIGHT)

            return VGroup(bubble_group, info).next_to(grupo_prompt, DOWN, buff=1.0)

        r_sancho_1 = crear_respuesta("0.1", 1, "\"...de cuyo nombre no quiero acordarme.\"", MARRON_OSCURO, "Sancho")
        r_sancho_2 = crear_respuesta("0.1", 2, "\"...de cuyo nombre no quiero acordarme.\"", MARRON_OSCURO, "Sancho")

        r_quijote_1 = crear_respuesta("1.5", 1, "\"...donde los dragones mecánicos beben aceite de oliva.\"", NARANJA_TERRACOTA, "El Quijote")
        r_quijote_2 = crear_respuesta("1.5", 2, "\"...los molinos me hablan en código binario al amanecer.\"", NARANJA_TERRACOTA, "El Quijote")

        actual = r_sancho_1
        self.play(FadeIn(actual, shift=UP))
        self._siguiente()

        self.play(FadeTransform(actual, r_sancho_2), run_time=1)
        actual = r_sancho_2
        self._siguiente()

        self.play(FadeTransform(actual, r_quijote_1), run_time=1.5)
        actual = r_quijote_1
        self._siguiente()

        self.play(FadeTransform(actual, r_quijote_2), run_time=1)
        self._siguiente()

        self.limpiar_pantalla()

