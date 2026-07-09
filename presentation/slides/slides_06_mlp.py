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


class SlidesMLP:
    def slide_layer_normalization(self):


        def crear_cajita(texto, bg_color, borde_color=MARRON_OSCURO, w=2.6, h=0.7, tam_fuente=20):
            caja = RoundedRectangle(corner_radius=0.1, width=w, height=h,
                                    fill_color=bg_color, fill_opacity=1,
                                    stroke_color=borde_color, stroke_width=2)
            lbl = Text(texto, font_size=tam_fuente, color=TINTA_NEGRA).move_to(caja.get_center())
            return VGroup(caja, lbl)

        def crear_vector_visual(numeros, bg_color, borde_color=MARRON_OSCURO):
            return VGroup(*[
                crear_cajita(num, bg_color, borde_color, w=1.6, h=0.7, tam_fuente=18)
                for num in numeros
            ]).arrange(DOWN, buff=0.1)


        titulo_p1 = Text("Layer ", font_size=42, weight=BOLD, color=TINTA_NEGRA)
        titulo_p2 = Text("Normalization", font_size=42, weight=BOLD, color=NARANJA_TERRACOTA)
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 4, RIGHT * 4, color=MARRON_OSCURO).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)

        adornos = self._crear_adornos_esquinas()
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo_completo, linea, adornos=adornos, fondo=llanuras_fondo)


        lbl_antes   = Text("Sin norma",     font=FUENTE, font_size=20,
                            color=NARANJA_TERRACOTA, weight=BOLD)
        lbl_despues = Text("Con LayerNorm", font=FUENTE, font_size=20,
                            color=MARRON_OSCURO, weight=BOLD)

        pares = [
            ("8459.1",  "1.34"),
            ("-7302.4", "-1.15"),
            ("0.00001", "0.00"),
            ("5120.9",  "0.89"),
            ("-9999.9", "-1.52"),
        ]
        inestables = [p[0] for p in pares]
        estables   = [p[1] for p in pares]

        vec_inestable = crear_vector_visual(
            inestables, bg_color=SALMON_CLARO, borde_color=NARANJA_TERRACOTA
        )
        vec_estable = crear_vector_visual(
            estables, bg_color=CREMA_CALIDA, borde_color=MARRON_OSCURO
        )


        lbl_antes.next_to(vec_inestable, UP, buff=0.3)
        lbl_despues.next_to(vec_estable, UP, buff=0.3)

        bloque_izq = VGroup(lbl_antes,   vec_inestable)
        bloque_der = VGroup(lbl_despues, vec_estable)

        VGroup(bloque_izq, bloque_der).arrange(RIGHT, buff=1.8).move_to(ORIGIN)

        stats_antes = Text(
            "μ ≈ -1344   σ ≈ 7210",
            font=FUENTE, font_size=17, color=NARANJA_TERRACOTA
        ).next_to(vec_inestable, DOWN, buff=0.3)

        stats_despues = Text(
            "μ = 0   σ = 1",
            font=FUENTE, font_size=17, color=MARRON_OSCURO, weight=BOLD
        ).next_to(vec_estable, DOWN, buff=0.3)


        self.play(
            FadeIn(lbl_antes, shift=UP * 0.1),
            LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in vec_inestable],
                        lag_ratio=0.1),
            run_time=1.0
        )
        self.play(
            Indicate(vec_inestable, color=NARANJA_TERRACOTA, scale_factor=1.05),
            FadeIn(stats_antes, shift=UP * 0.1)
        )
        self.play(
            LaggedStart(*[
                ReplacementTransform(vec_inestable[i].copy(), vec_estable[i])
                for i in range(len(pares))
            ], lag_ratio=0.15),
            FadeIn(lbl_despues, shift=UP * 0.1),
            run_time=1.4
        )
        self.play(FadeIn(stats_despues, shift=UP * 0.1))
        self._siguiente()

        self.play(
            FadeOut(bloque_izq), FadeOut(bloque_der),
            FadeOut(stats_antes), FadeOut(stats_despues),
            run_time=0.7
        )


        formula = MathTex(
            r"\text{output} = \frac{\text{input} - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta",
            substrings_to_isolate=[r"\epsilon", r"\times \gamma + \beta"],
            color=TINTA_NEGRA
        ).scale(1.2).move_to(DOWN * 0.5)

        lbl_formula = Text("Por cada token:", font_size=20, weight=BOLD,
                            color=MARRON_OSCURO).next_to(formula, UP, buff=0.5)

        self.play(Write(lbl_formula), FadeIn(formula))

        parte_eps = formula.get_part_by_tex(r"\epsilon")
        caja_eps  = SurroundingRectangle(parte_eps, color=NARANJA_TERRACOTA, buff=0.05)
        nota_eps  = Text("ε previene división por cero", font_size=18,
                          color=NARANJA_TERRACOTA).next_to(caja_eps, DOWN, buff=0.5)

        self.play(Create(caja_eps), FadeIn(nota_eps, shift=UP))
        self.play(FadeOut(caja_eps), FadeOut(nota_eps))

        parte_params = formula.get_part_by_tex(r"\times \gamma + \beta")
        caja_params  = SurroundingRectangle(parte_params, color=MARRON_OSCURO, buff=0.1)
        nota_params  = Text("γ, β: parámetros aprendibles", font_size=18,
                             color=MARRON_OSCURO).next_to(caja_params, DOWN, buff=0.5)

        self.play(Create(caja_params), FadeIn(nota_params, shift=UP))
        self._siguiente()

        self.play(
            *[FadeOut(m) for m in [lbl_formula, formula, caja_params, nota_params]]
        )

        self.limpiar_pantalla()


    def slide_arquitectura_neurona(self):

        titulo, linea = self.crear_titulo("La Capa MLP: Una Función Matemática", palabra_clave="Función", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        eq_top = MathTex(r"f : \mathbb{R}^{768} \rightarrow \mathbb{R}^{768}", color=TINTA_NEGRA, font_size=42).next_to(linea, DOWN, buff=0.3)
        self.play(Write(eq_top))

        r = 0.15
        c_linea = ACERO

        capa_in = VGroup(*[Circle(radius=r, fill_color=PAPEL_TAN, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(5)]).arrange(DOWN, buff=0.2)

        hid_1 = VGroup(*[Circle(radius=r, fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)
        hid_2 = VGroup(*[Circle(radius=r, fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)
        hid_3 = VGroup(*[Circle(radius=r, fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)

        capas_profundas = VGroup(hid_1, hid_2, hid_3).arrange(RIGHT, buff=0.8)
        capa_out = VGroup(*[Circle(radius=r, fill_color=MARRON_OSCURO, fill_opacity=1, stroke_color=TINTA_NEGRA, stroke_width=2) for _ in range(5)]).arrange(DOWN, buff=0.2)

        red = VGroup(capa_in, capas_profundas, capa_out).arrange(RIGHT, buff=1.8).move_to(DOWN * 0.5)

        conexiones_in_h1 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in capa_in for n2 in hid_1])
        conexiones_h1_h2 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in hid_1 for n2 in hid_2])
        conexiones_h2_h3 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in hid_2 for n2 in hid_3])
        conexiones_h3_out = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in hid_3 for n2 in capa_out])

        brace_in = Brace(capa_in, direction=LEFT, color=MARRON_OSCURO)
        lbl_in = MathTex(r"x \in \mathbb{R}^{768}", color=TINTA_NEGRA, font_size=28).next_to(brace_in, LEFT)

        brace_hid = Brace(capas_profundas, direction=UP, color=NARANJA_TERRACOTA)
        lbl_hid = MathTex(r"3072", color=NARANJA_TERRACOTA, font_size=32).next_to(brace_hid, UP)

        brace_out = Brace(capa_out, direction=RIGHT, color=MARRON_OSCURO)
        lbl_out = MathTex(r"f(x) \in \mathbb{R}^{768}", color=TINTA_NEGRA, font_size=28).next_to(brace_out, RIGHT)

        flecha_1 = MathTex(r"\xrightarrow{\quad \phi_1 \quad}", color=NARANJA_TERRACOTA).next_to(capas_profundas[0], DOWN, buff=0.5)
        flecha_2 = MathTex(r"\xrightarrow{\quad \phi_2 \quad}", color=NARANJA_TERRACOTA).next_to(capas_profundas[1], DOWN, buff=0.5)
        flecha_3 = MathTex(r"\xrightarrow{\quad \phi_3 \quad}", color=NARANJA_TERRACOTA).next_to(capas_profundas[2], DOWN, buff=0.5)
        eq_composicion = MathTex(r"f(x) = \phi_3(\phi_2(\phi_1(x)))", color=TINTA_NEGRA, font_size=32).next_to(VGroup(flecha_1, flecha_3), DOWN, buff=0.3)

        self.play(FadeIn(capa_in), GrowFromCenter(brace_in), Write(lbl_in))
        self._siguiente()

        self.play(LaggedStartMap(Create, conexiones_in_h1, lag_ratio=0.01), run_time=0.8)
        self.play(FadeIn(hid_1, shift=LEFT*0.2), Write(flecha_1))

        self.play(LaggedStartMap(Create, conexiones_h1_h2, lag_ratio=0.01), run_time=0.6)
        self.play(FadeIn(hid_2, shift=LEFT*0.2), Write(flecha_2))

        self.play(LaggedStartMap(Create, conexiones_h2_h3, lag_ratio=0.01), run_time=0.6)
        self.play(FadeIn(hid_3, shift=LEFT*0.2), Write(flecha_3))

        self.play(GrowFromCenter(brace_hid), Write(lbl_hid), Write(eq_composicion))
        self._siguiente()

        self.play(LaggedStartMap(Create, conexiones_h3_out, lag_ratio=0.01), run_time=0.8)
        self.play(FadeIn(capa_out, shift=LEFT*0.2), GrowFromCenter(brace_out), Write(lbl_out))
        self._siguiente()

        self.limpiar_pantalla()


    def slide_zoom_neurona(self):

        titulo, linea = self.crear_titulo("La Capa MLP: Dentro de una Neurona", palabra_clave="Neurona", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        r_mini = 0.08
        columna_in = VGroup(*[Circle(radius=r_mini, fill_color=PAPEL_TAN, fill_opacity=1, stroke_color=MARRON_OSCURO) for _ in range(3)]).arrange(DOWN, buff=0.1)
        columna_hid = VGroup(*[Circle(radius=r_mini, fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_color=MARRON_OSCURO) for _ in range(5)]).arrange(DOWN, buff=0.1)

        mini_red = VGroup(columna_in, columna_hid).arrange(RIGHT, buff=0.6)

        mini_red.to_edge(LEFT, buff=0.8)

        conexiones_mini = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=0.5, color=ACERO) for n1 in columna_in for n2 in columna_hid])

        self.play(FadeIn(mini_red), FadeIn(conexiones_mini))

        neurona_objetivo = columna_hid[2]
        resaltador = Circle(radius=r_mini * 2, color=MARRON_OSCURO, stroke_width=3).move_to(neurona_objetivo)
        self.play(Create(resaltador))
        self._siguiente()

        posicion_neurona = RIGHT * 2.2 + UP * 0.4
        neurona_gigante = Circle(radius=1.6, fill_color=NARANJA_TERRACOTA, fill_opacity=0.1, stroke_color=NARANJA_TERRACOTA, stroke_width=4).move_to(posicion_neurona)

        sumatoria = MathTex(r"\Sigma", color=TINTA_NEGRA, font_size=55).move_to(neurona_gigante).shift(LEFT * 0.6)
        separador = Line(neurona_gigante.get_top() + DOWN*0.1, neurona_gigante.get_bottom() + UP*0.1, color=NARANJA_TERRACOTA)

        ejes_act = Axes(x_range=[-2, 2], y_range=[-0.5, 1.5], x_length=1.0, y_length=1.0, axis_config={"color": MARRON_OSCURO, "include_ticks": False}).move_to(neurona_gigante).shift(RIGHT * 0.6)
        curva_act = ejes_act.plot(lambda x: x if x > 0 else 0, color=MARRON_OSCURO)
        grupo_activacion = VGroup(ejes_act, curva_act)

        lineas_zoom = VGroup(
            DashedLine(resaltador.get_top(), neurona_gigante.get_top(), color=ACERO),
            DashedLine(resaltador.get_bottom(), neurona_gigante.get_bottom(), color=ACERO)
        )

        self.play(Create(lineas_zoom), FadeIn(neurona_gigante))
        self.play(Write(sumatoria), Create(separador), FadeIn(grupo_activacion))

        self.play(FadeOut(lineas_zoom))

        entradas_text = VGroup(
            MathTex(r"x_1", color=TINTA_NEGRA),
            MathTex(r"x_2", color=TINTA_NEGRA),
            MathTex(r"\vdots", color=TINTA_NEGRA),
            MathTex(r"x_n", color=TINTA_NEGRA)
        ).arrange(DOWN, buff=0.4)

        entradas_text.move_to(LEFT * 2.0 + UP * 0.4)

        flechas_in = VGroup(*[Arrow(e.get_right(), neurona_gigante.get_left(), buff=0.15, color=MARRON_OSCURO, max_tip_length_to_length_ratio=0.08) for e in entradas_text if e.tex_string != r"\vdots"])

        pesos_text = VGroup(
            MathTex(r"w_1", color=NARANJA_TERRACOTA).move_to(flechas_in[0].get_center() + UP * 0.35).scale(0.8),
            MathTex(r"w_2", color=NARANJA_TERRACOTA).move_to(flechas_in[1].get_center() + DOWN * 0.2).scale(0.8),
            MathTex(r"w_n", color=NARANJA_TERRACOTA).move_to(flechas_in[2].get_center() + DOWN * 0.35).scale(0.8)
        )

        bias_text = MathTex(r"b", color=MARRON_OSCURO).next_to(neurona_gigante, UP, buff=0.4)
        flecha_bias = Arrow(bias_text.get_bottom(), neurona_gigante.get_top(), buff=0.1, color=MARRON_OSCURO)

        flecha_out = Arrow(neurona_gigante.get_right(), neurona_gigante.get_right() + RIGHT * 1.2, color=MARRON_OSCURO)
        salida_text = MathTex(r"\phi(x)", color=TINTA_NEGRA).next_to(flecha_out, RIGHT)

        self.play(LaggedStart(
            *[AnimationGroup(FadeIn(e, shift=RIGHT), GrowArrow(f)) for e, f in zip([entradas_text[0], entradas_text[1], entradas_text[3]], flechas_in)],
            FadeIn(entradas_text[2]),
            lag_ratio=0.2
        ))
        self.play(FadeIn(pesos_text, shift=UP))
        self.play(FadeIn(bias_text, shift=DOWN), GrowArrow(flecha_bias))
        self.play(GrowArrow(flecha_out), Write(salida_text))
        self._siguiente()

        eq_final = MathTex(r"\phi(x)", r"=", r"\sigma", r"(", r"\sum x_i \cdot w_i", r"+ b", r")", color=TINTA_NEGRA, font_size=46).move_to(DOWN * 1.8)

        eq_final[2].set_color(MARRON_OSCURO)
        eq_final[4].set_color(NARANJA_TERRACOTA)
        eq_final[5].set_color(MARRON_OSCURO)

        nota_w = Text("Suma ponderada", font=FUENTE, font_size=16, color=NARANJA_TERRACOTA).next_to(eq_final[4], DOWN, buff=0.6)
        flecha_w = Arrow(nota_w.get_top(), eq_final[4].get_bottom(), buff=0.1, color=NARANJA_TERRACOTA)

        nota_act = Text("Activación no lineal", font=FUENTE, font_size=16, color=MARRON_OSCURO).next_to(eq_final[2], DOWN, buff=0.6).shift(LEFT * 0.5)
        flecha_act = Arrow(nota_act.get_top(), eq_final[2].get_bottom(), buff=0.1, color=MARRON_OSCURO)

        self.play(Write(eq_final))
        self.play(FadeIn(nota_w), GrowArrow(flecha_w))
        self.play(FadeIn(nota_act), GrowArrow(flecha_act))
        self._siguiente()

        self.limpiar_pantalla()


    def slide_activacion(self):

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Función de Activación: GELU",
            palabra_clave="GELU",
            color_clave=NARANJA_TERRACOTA
        )

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        formula_principal = MathTex(
            r"\text{GELU}(x) = x \cdot \Phi(x)",
            color=TINTA_NEGRA, font_size=42
        )
        formula_aprox = MathTex(
            r"\approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3)\right)\right)",
            color=MARRON_OSCURO, font_size=26
        )

        grupo_formulas = VGroup(formula_principal, formula_aprox).arrange(DOWN, buff=0.4)

        texto_contexto = Text(
            "Apagado de las neuronas",
            font=FUENTE, font_size=20, color=NARANJA_TERRACOTA, line_spacing=1.2
        )

        grupo_izq = VGroup(grupo_formulas, texto_contexto).arrange(DOWN, buff=0.8)
        grupo_izq.to_edge(LEFT, buff=1.0).shift(UP * 1.0)

        self.play(Write(formula_principal))
        self.play(FadeIn(formula_aprox, shift=UP))
        self.play(FadeIn(texto_contexto, shift=RIGHT))

        ejes = Axes(
            x_range=[-3, 3, 1], y_range=[-1, 3, 1],
            x_length=6, y_length=4.5,
            axis_config={"color": MARRON_OSCURO, "include_numbers": True, "font_size": 16}
        ).to_edge(RIGHT, buff=0.8).shift(DOWN * 0.5)

        curva_relu = ejes.plot(lambda x: np.maximum(0, x), color=BEIGE_MEDIO, stroke_width=4)
        lbl_relu = Text("ReLU", font=FUENTE, font_size=26, color=BEIGE_MEDIO).next_to(ejes.c2p(2, 2), UL, buff=0.2)

        curva_gelu = ejes.plot(
            lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
            color=NARANJA_TERRACOTA, stroke_width=5
        )
        lbl_gelu = Text("GELU", font=FUENTE, font_size=26, color=NARANJA_TERRACOTA).next_to(ejes.c2p(2.5, 2.5), DR, buff=0.1)

        punto_minimo = ejes.c2p(-0.75, -0.17)
        lbl_suavizado = Text("Apagado suave", font=FUENTE, font_size=16, color=MARRON_QUIJOTE).next_to(punto_minimo, DOWN, buff=0.5).shift(RIGHT * 1)
        flecha_suav = Arrow(lbl_suavizado.get_left(), punto_minimo, buff=0.1, color=MARRON_QUIJOTE, tip_length=0.15)

        self.play(Create(ejes), Write(lbl_relu))
        self.play(Create(curva_relu))
        self._siguiente()

        nodo = Circle(radius=0.7, fill_color=PAPEL_CREMA, fill_opacity=1, stroke_color=BEIGE_MEDIO, stroke_width=4)
        lbl_nodo = Text("ReLU", font=FUENTE, font_size=20, color=BEIGE_MEDIO).move_to(nodo)

        flecha_in = Arrow(ORIGIN, RIGHT * 1.5, buff=0.1, color=MARRON_OSCURO)
        val_in = MathTex("x = -1", font_size=28, color=TINTA_NEGRA).next_to(flecha_in, UP, buff=0.1)
        grupo_in = VGroup(val_in, flecha_in)

        flecha_out = Arrow(ORIGIN, RIGHT * 1.5, buff=0.1, color=BEIGE_MEDIO)
        val_out = MathTex("0", font_size=32, color=BEIGE_MEDIO).next_to(flecha_out, UP, buff=0.1)
        grupo_out = VGroup(val_out, flecha_out)

        diagrama_neurona = VGroup(grupo_in, nodo, grupo_out).arrange(RIGHT, buff=0.1)
        diagrama_neurona.to_corner(DL, buff=1.0).shift(UP * 0.5)

        lbl_nodo.move_to(nodo)

        cruz_muerte = Cross(val_out, stroke_color=RED, stroke_width=5, scale_factor=0.6)

        self.play(FadeIn(nodo), Write(lbl_nodo))
        self.play(GrowArrow(flecha_in), FadeIn(val_in, shift=RIGHT))

        val_in_anim = val_in.copy()
        self.play(val_in_anim.animate.move_to(nodo).scale(0.5).set_opacity(0), run_time=0.8)

        self.play(GrowArrow(flecha_out), FadeIn(val_out, shift=RIGHT))
        self.play(Create(cruz_muerte))
        self._siguiente()

        lbl_nodo_gelu = Text("GELU", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA).move_to(nodo)

        self.play(
            ReplacementTransform(curva_relu, curva_gelu),
            ReplacementTransform(lbl_relu, lbl_gelu),
            nodo.animate.set_stroke(color=NARANJA_TERRACOTA),
            ReplacementTransform(lbl_nodo, lbl_nodo_gelu),
            flecha_out.animate.set_color(NARANJA_TERRACOTA),
            FadeOut(cruz_muerte)
        )

        val_out_gelu = MathTex("-0.15", font_size=32, color=NARANJA_TERRACOTA).next_to(flecha_out, UP, buff=0.1)
        chispa = Star(n=5, outer_radius=0.25, inner_radius=0.12, color=MARRON_QUIJOTE, fill_opacity=1).next_to(val_out_gelu, RIGHT, buff=0.2)

        val_in_anim_2 = val_in.copy()
        self.play(val_in_anim_2.animate.move_to(nodo).scale(0.5).set_opacity(0), run_time=0.8)

        self.play(
            ReplacementTransform(val_out, val_out_gelu),
            Create(chispa)
        )

        self.play(FadeIn(lbl_suavizado, shift=UP), GrowArrow(flecha_suav))
        self._siguiente()

        self.limpiar_pantalla()


    def slide_capa_transformer(self):

        escala = 0.65

        def crear_nodo(texto, ancho=2.5 * escala, alto=0.8 * escala, resaltado=False):
            bg = NARANJA_TERRACOTA if resaltado else PAPEL_CREMA
            borde = MARRON_OSCURO
            txt_color = BLANCO if resaltado else TINTA_NEGRA
            caja = RoundedRectangle(
                corner_radius=0.15 * escala, width=ancho, height=alto,
                fill_color=bg, fill_opacity=1, stroke_color=borde, stroke_width=2.5 * escala
            )
            txt = Text(texto, font=FUENTE, font_size=20 * escala, color=txt_color)
            return VGroup(caja, txt)

        titulo, linea = self.crear_titulo(
            "Arquitectura: Transformer Layer",
            palabra_clave="Transformer",
            color_clave=NARANJA_TERRACOTA,
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(
            titulo, linea,
            fondo=VGroup(llanuras_fondo),
            adornos=adornos,
        )


        X_RES  = -3.0
        X_BLOQ =  2.2
        G = 2.0 * escala


        Y_INPUT  =  2.8
        Y_BIF1   =  1.8
        Y_ADD1   =  0.9
        Y_LN1    =  1.8
        Y_ATTN   =  0.9
        Y_BIF2   = -0.1
        Y_ADD2   = -1.0
        Y_LN2    = -0.1
        Y_MLP    = -1.0
        Y_OUTPUT = -2.2


        nodo_input  = crear_nodo("Input") .move_to([X_RES, Y_INPUT,  0])
        nodo_output = crear_nodo("Output").move_to([X_RES, Y_OUTPUT, 0])

        add_1 = VGroup(
            Circle(radius=0.28*escala, fill_color=PAPEL_CREMA, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=2*escala),
            Text("+", font_size=24*escala, color=TINTA_NEGRA, weight=BOLD)
        ).move_to([X_RES, Y_ADD1, 0])

        add_2 = VGroup(
            Circle(radius=0.28*escala, fill_color=PAPEL_CREMA, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=2*escala),
            Text("+", font_size=24*escala, color=TINTA_NEGRA, weight=BOLD)
        ).move_to([X_RES, Y_ADD2, 0])

        nodo_ln1  = crear_nodo("Layer Norm")                  .move_to([X_BLOQ, Y_LN1,  0])
        nodo_attn = crear_nodo("Self-Attention", resaltado=True).move_to([X_BLOQ, Y_ATTN, 0])
        nodo_ln2  = crear_nodo("Layer Norm")                  .move_to([X_BLOQ, Y_LN2,  0])
        nodo_mlp  = crear_nodo("MLP", resaltado=True)         .move_to([X_BLOQ, Y_MLP,  0])


        seg_input_bif1 = Line(
            nodo_input.get_bottom(), [X_RES, Y_BIF1, 0],
            stroke_color=MARRON_OSCURO, stroke_width=G
        )

        f_bif1_add1 = Arrow(
            [X_RES, Y_BIF1, 0], add_1.get_top(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )

        seg_add1_bif2 = Line(
            add_1.get_bottom(), [X_RES, Y_BIF2, 0],
            stroke_color=MARRON_OSCURO, stroke_width=G
        )
        f_bif2_add2 = Arrow(
            [X_RES, Y_BIF2, 0], add_2.get_top(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )

        f_add2_out = Arrow(
            add_2.get_bottom(), nodo_output.get_top(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )


        f_bif1_ln1 = Arrow(
            [X_RES, Y_BIF1, 0], nodo_ln1.get_left(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )

        f_ln1_attn = Arrow(
            nodo_ln1.get_bottom(), nodo_attn.get_top(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )

        f_attn_add1 = Arrow(
            nodo_attn.get_left(), add_1.get_right(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )


        f_bif2_ln2 = Arrow(
            [X_RES, Y_BIF2, 0], nodo_ln2.get_left(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )
        f_ln2_mlp = Arrow(
            nodo_ln2.get_bottom(), nodo_mlp.get_top(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )
        f_mlp_add2 = Arrow(
            nodo_mlp.get_left(), add_2.get_right(), buff=0,
            color=MARRON_OSCURO, stroke_width=G, max_tip_length_to_length_ratio=0.15
        )


        diagrama = VGroup(
            nodo_input, nodo_output,
            add_1, add_2,
            nodo_ln1, nodo_attn, nodo_ln2, nodo_mlp,
            seg_input_bif1, f_bif1_add1,
            seg_add1_bif2, f_bif2_add2,
            f_add2_out,
            f_bif1_ln1, f_ln1_attn, f_attn_add1,
            f_bif2_ln2, f_ln2_mlp, f_mlp_add2,
        )
        diagrama.move_to(ORIGIN)

        self.play(FadeIn(diagrama))


        caja1 = SurroundingRectangle(
            VGroup(seg_input_bif1, f_bif1_add1, add_1,
                seg_add1_bif2, f_bif2_add2, add_2, f_add2_out),
            color=NARANJA_TERRACOTA, buff=0.2, stroke_width=3
        )
        self.play(Create(caja1))
        self.play(FadeOut(caja1))


        caja2 = VGroup(
            SurroundingRectangle(nodo_ln1, color=MARRON_OSCURO, buff=0.1, stroke_width=3),
            SurroundingRectangle(nodo_ln2, color=MARRON_OSCURO, buff=0.1, stroke_width=3),
        )
        self.play(Create(caja2))
        self.play(FadeOut(caja2))


        caja3 = VGroup(
            SurroundingRectangle(nodo_attn, color=NARANJA_TERRACOTA, buff=0.15, stroke_width=3),
            SurroundingRectangle(nodo_mlp,  color=NARANJA_TERRACOTA, buff=0.15, stroke_width=3),
        )
        self.play(Create(caja3))
        self.play(FadeOut(caja3))

        self._siguiente()
        self.limpiar_pantalla()


    def slide_residual(self):

        escala = 0.85

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Conexiones Residuales: El Atajo de Sancho",
            palabra_clave="Residuales",
            color_clave=NARANJA_TERRACOTA
        )

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        pos_x = LEFT * 4
        pos_f = LEFT * 0.5
        pos_add = RIGHT * 2.5
        pos_y = RIGHT * 4.5

        label_x = MathTex("x", font_size=48 * escala, color=TINTA_NEGRA).move_to(pos_x)

        nodo_f = RoundedRectangle(corner_radius=0.2, width=1.5 * escala, height=2.2 * escala, fill_color=MARRON_QUIJOTE, fill_opacity=1, stroke_color=BLANCO, stroke_width=3)
        label_f = MathTex("f", font_size=48 * escala, color=BLANCO).move_to(nodo_f)
        grupo_f = VGroup(nodo_f, label_f).move_to(pos_f)

        nodo_add = Circle(radius=0.4 * escala, fill_color=MARRON_QUIJOTE, fill_opacity=1, stroke_color=BLANCO, stroke_width=3)
        label_add = MathTex("+", font_size=40 * escala, color=BLANCO).move_to(nodo_add)
        grupo_add = VGroup(nodo_add, label_add).move_to(pos_add)

        label_y = MathTex("y", font_size=48 * escala, color=TINTA_NEGRA).move_to(pos_y)

        eq_final = MathTex("y = x + f(x)", font_size=42 * escala, color=TINTA_NEGRA).next_to(grupo_f, DOWN, buff=1.2)

        arrow_x_f = Arrow(label_x.get_right(), grupo_f.get_left(), buff=0.2, color=MARRON_OSCURO, stroke_width=3)
        arrow_f_add = Arrow(grupo_f.get_right(), grupo_add.get_left(), buff=0.2, color=MARRON_OSCURO, stroke_width=3)
        arrow_add_y = Arrow(grupo_add.get_right(), label_y.get_left(), buff=0.2, color=MARRON_OSCURO, stroke_width=3)

        punto_inicio_skip = pos_x + RIGHT * 0.8
        p1 = punto_inicio_skip
        p2 = p1 + UP * 2.2
        p3 = np.array([grupo_add.get_center()[0], p2[1], 0])
        p4 = grupo_add.get_top() + UP * 0.1

        line_up = Line(p1, p2, color=MARRON_OSCURO, stroke_width=3)
        line_across = Line(p2, p3, color=MARRON_OSCURO, stroke_width=3)
        arrow_down = Arrow(p3, p4, buff=0, color=MARRON_OSCURO, stroke_width=3)
        skip_connection = VGroup(line_up, line_across, arrow_down)

        self.play(FadeIn(label_x, shift=RIGHT))
        self.play(GrowArrow(arrow_x_f), FadeIn(grupo_f, shift=RIGHT))
        self.play(GrowArrow(arrow_f_add), FadeIn(grupo_add, scale=0.5))
        self.play(Create(skip_connection))
        self.play(GrowArrow(arrow_add_y), FadeIn(label_y, shift=RIGHT))
        self.play(Write(eq_final))
        self._siguiente()

        pos_texto = DOWN * 3

        txt_desc_1 = Text("La señal viaja...", font=FUENTE, font_size=24 * escala, color=TINTA_NEGRA).move_to(pos_texto)
        txt_desc_2 = Text("f(x) transforma... la señal se desvanece", font=FUENTE, font_size=24 * escala, color=MARRON_QUIJOTE).move_to(pos_texto)
        txt_desc_3 = Text("Sancho lleva la copia por el atajo", font=FUENTE, font_size=24 * escala, color=NARANJA_TERRACOTA).move_to(pos_texto)
        txt_desc_4 = Text("Realidad + visión se suman en +", font=FUENTE, font_size=24 * escala, color=MARRON_OSCURO).move_to(pos_texto)

        def crear_imagen_pixelada(resolucion="alta"):
            cuadros = []
            filas, cols = (6, 6) if resolucion == "alta" else (3, 3)
            lado = 0.15 if resolucion == "alta" else 0.3
            colores = [NARANJA_TERRACOTA, MARRON_OSCURO, MARRON_QUIJOTE, OCRE_CERVANTINO]

            for i in range(filas):
                for j in range(cols):
                    color = colores[(i*j) % len(colores)]
                    cuadro = Square(side_length=lado, fill_color=color, fill_opacity=1, stroke_width=0.5, stroke_color=BLANCO)
                    cuadros.append(cuadro)

            img = VGroup(*cuadros).arrange_in_grid(rows=filas, cols=cols, buff=0)
            if resolucion == "baja":
                img.set_opacity(0.6)
            return img

        img_alta_x = crear_imagen_pixelada("alta").move_to(pos_x).shift(UP*1.2)
        img_baja_f = crear_imagen_pixelada("baja").move_to(pos_f).shift(UP*1.2)
        img_alta_copia = crear_imagen_pixelada("alta").move_to(pos_x).shift(UP*1.2)
        img_final_y = crear_imagen_pixelada("alta").move_to(pos_y).shift(UP*1.2)

        txt_input = Text("Imagen Real", font=FUENTE, font_size=18, color=MARRON_OSCURO).next_to(img_alta_x, UP)
        self.play(FadeIn(img_alta_x, shift=DOWN), FadeIn(txt_input), FadeIn(txt_desc_1))
        self._siguiente()

        self.play(ReplacementTransform(txt_desc_1, txt_desc_2))
        txt_f = Text("Visión Alterada f(x)", font=FUENTE, font_size=18, color=MARRON_QUIJOTE).next_to(img_baja_f, UP)
        self.play(
            ReplacementTransform(img_alta_x.copy(), img_baja_f),
            FadeIn(txt_f)
        )
        self._siguiente()

        self.play(ReplacementTransform(txt_desc_2, txt_desc_3))
        txt_skip = Text("La Copia de Sancho", font=FUENTE, font_size=18, color=NARANJA_TERRACOTA).next_to(line_across, UP)
        self.play(FadeIn(txt_skip))
        self.play(
            img_alta_copia.animate.move_to(p2).shift(UP*0.5),
            run_time=0.8
        )
        self.play(
            img_alta_copia.animate.move_to(p3).shift(UP*0.5),
            run_time=1.5
        )
        self.play(
            img_alta_copia.animate.move_to(pos_add).shift(UP*1.2),
            run_time=0.8
        )
        self._siguiente()

        self.play(ReplacementTransform(txt_desc_3, txt_desc_4))
        txt_output = Text("Realidad Recuperada", font=FUENTE, font_size=18, color=MARRON_OSCURO).next_to(img_final_y, UP)

        self.play(
            FadeOut(img_baja_f, shift=RIGHT),
            img_alta_copia.animate.move_to(pos_y).shift(UP*1.2),
            FadeIn(txt_output)
        )

        caja_eq = SurroundingRectangle(eq_final, color=NARANJA_TERRACOTA, buff=0.2, stroke_width=2)
        self.play(Create(caja_eq))
        self._siguiente()

        self.limpiar_pantalla()

