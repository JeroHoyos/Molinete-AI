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


class SlidesTensores:
    def slide_que_es_un_tensor(self):

        titulo, linea = self.crear_titulo(
            "¿Qué es un Tensor?",
            palabra_clave="Tensor?",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.15))

        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)


        lbl_0d = Text("Escalar", font=FUENTE, font_size=22,
                    color=MARRON_OSCURO, weight=BOLD)
        lbl_1d = Text("Vector",  font=FUENTE, font_size=22,
                    color=MARRON_OSCURO, weight=BOLD)
        lbl_2d = Text("Matriz",  font=FUENTE, font_size=22,
                    color=MARRON_OSCURO, weight=BOLD)
        lbl_3d = Text("Tensor",  font=FUENTE, font_size=22,
                    color=MARRON_OSCURO, weight=BOLD)


        escalar = self.crear_bloque("7", ancho=1.0, alto=1.0)

        vector = VGroup(*[
            self.crear_bloque(v, ancho=0.9, alto=0.55)
            for v in ["1", "5", "9", "2"]
        ]).arrange(DOWN, buff=0.08)

        valores_mat = ["3","1","4", "2","5","9", "2","6","5", "3","5","8"]
        filas_2d = VGroup(*[
            VGroup(*[
                self.crear_bloque(valores_mat[r*3 + c], ancho=0.7, alto=0.55)
                for c in range(3)
            ]).arrange(RIGHT, buff=0.06)
            for r in range(4)
        ]).arrange(DOWN, buff=0.06)

        def hacer_capa(valores, color_fondo, opacidad):
            capa = VGroup(*[
                VGroup(*[
                    self.crear_bloque(
                        valores[r*3 + c],
                        ancho=0.58, alto=0.48,
                        color_fondo=color_fondo
                    )
                    for c in range(3)
                ]).arrange(RIGHT, buff=0.05)
                for r in range(3)
            ]).arrange(DOWN, buff=0.05)
            capa.set_opacity(opacidad)
            return capa

        capa_back = hacer_capa(["1","2","3","4","5","6","7","8","9"], PAPEL_CREMA,       0.30)
        capa_mid  = hacer_capa(["9","8","7","6","5","4","3","2","1"], PAPEL_TAN,         0.60)
        capa_top  = hacer_capa(["2","4","6","8","0","2","4","6","8"], NARANJA_TERRACOTA, 1.00)
        capa_mid.shift(RIGHT * 0.18 + UP * 0.18)
        capa_top.shift(RIGHT * 0.36 + UP * 0.36)
        tensor_3d = VGroup(capa_back, capa_mid, capa_top)


        forma_0d = Text("forma: []",      font=FUENTE, font_size=17, color=PAPEL_TAN)
        forma_1d = Text("forma: [4]",     font=FUENTE, font_size=17, color=PAPEL_TAN)
        forma_2d = Text("forma: [4, 3]",  font=FUENTE, font_size=17, color=PAPEL_TAN)
        forma_3d = Text("forma: [3,3,3]", font=FUENTE, font_size=17, color=PAPEL_TAN)


        contenidos = VGroup(escalar, vector, filas_2d, tensor_3d)
        contenidos.arrange(RIGHT, buff=1.2)


        ZONA_TOP = linea.get_bottom()[1] - 0.5
        ZONA_BOT = -2.8
        ZONA_MID_Y = (ZONA_TOP + ZONA_BOT) / 2


        for mob in contenidos:
            mob.set_y(ZONA_MID_Y)


        for lbl, mob, forma in zip(
            [lbl_0d, lbl_1d, lbl_2d, lbl_3d],
            contenidos,
            [forma_0d, forma_1d, forma_2d, forma_3d],
        ):
            lbl.next_to(mob, UP, buff=0.3).set_x(mob.get_x())
            forma.next_to(mob, DOWN, buff=0.25).set_x(mob.get_x())


        lbl_top_y = max(lbl.get_top()[1] for lbl in [lbl_0d, lbl_1d, lbl_2d, lbl_3d])
        for lbl in [lbl_0d, lbl_1d, lbl_2d, lbl_3d]:
            lbl.set_y(lbl_top_y - lbl.height / 2)


        separadores = VGroup()
        pares = [(escalar, vector), (vector, filas_2d), (filas_2d, tensor_3d)]
        for izq, der in pares:
            x_sep = (izq.get_right()[0] + der.get_left()[0]) / 2
            sep = DashedLine(
                UP * 2.8, DOWN * 2.8,
                color=MARRON_OSCURO, stroke_width=1.0,
                dash_length=0.12, dashed_ratio=0.4,
            ).set_x(x_sep)
            separadores.add(sep)


        nota_ram = Text(
            "En RAM: arreglo plano 1D — la forma y los saltos son solo metadatos",
            font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD
        ).to_edge(DOWN, buff=0.35)
        caja_ram = SurroundingRectangle(
            nota_ram, color=NARANJA_TERRACOTA,
            fill_color=FONDO_CAJA, fill_opacity=0.95,
            corner_radius=0.12, buff=0.15, stroke_width=2,
        )


        self.play(
            LaggedStart(
                Write(lbl_0d), Write(lbl_1d), Write(lbl_2d), Write(lbl_3d),
                lag_ratio=0.15
            ),
            Create(separadores),
            run_time=1.0,
        )

        self.play(
            GrowFromCenter(escalar),
            LaggedStart(*[FadeIn(b, shift=UP*0.15) for b in vector],   lag_ratio=0.08),
            LaggedStart(*[FadeIn(f, shift=UP*0.1)  for f in filas_2d], lag_ratio=0.08),
            AnimationGroup(
                FadeIn(capa_back, shift=UP*0.1),
                FadeIn(capa_mid,  shift=UP*0.1),
                FadeIn(capa_top,  shift=UP*0.1),
                lag_ratio=0.15,
            ),
            run_time=1.4,
        )

        self.play(
            LaggedStart(
                FadeIn(forma_0d, shift=UP*0.1),
                FadeIn(forma_1d, shift=UP*0.1),
                FadeIn(forma_2d, shift=UP*0.1),
                FadeIn(forma_3d, shift=UP*0.1),
                lag_ratio=0.12,
            ),
            run_time=0.8,
        )

        self.play(FadeIn(caja_ram), Write(nota_ram), run_time=0.9)
        self.play(Indicate(nota_ram, color=ORO_VIEJO, scale_factor=1.04))

        self._siguiente()

        adornos[1].clear_updaters()
        self.limpiar_pantalla()


    def slide_softmax(self):


        titulo, linea = self.crear_titulo(
            "Softmax", palabra_clave="Probabilidades",
            color_clave=NARANJA_TERRACOTA
        )
        adornos = self._crear_adornos_esquinas()
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)


        ANC = 1.15

        def bloque(v, fondo=FONDO_CAJA, texto=MARRON_OSCURO):
            return self.crear_bloque(v, ancho=ANC, color_fondo=fondo, color_texto=texto)

        def columna(etiqueta, valores, color_etiq=MARRON_OSCURO, **kw_bloque):
            lbl = Text(etiqueta, font=FUENTE, font_size=17,
                       color=color_etiq, weight=BOLD)
            bloques = VGroup(*[bloque(v, **kw_bloque) for v in valores])\
                .arrange(DOWN, buff=0.1)
            return VGroup(lbl, bloques).arrange(DOWN, buff=0.18)

        def conector(operacion, es_tex=False):
            flecha = Arrow(LEFT, RIGHT, color=MARRON_OSCURO,
                           stroke_width=2, max_tip_length_to_length_ratio=0.25).scale(0.5)
            op = (MathTex(operacion, font_size=20, color=NARANJA_TERRACOTA)
                  if es_tex
                  else Text(operacion, font=FUENTE, font_size=16,
                             color=NARANJA_TERRACOTA, weight=BOLD))
            op.next_to(flecha, UP, buff=0.08)
            return VGroup(flecha, op)


        formula_grande = MathTex(
            r"\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}",
            color=TINTA_NEGRA, font_size=64
        ).move_to(ORIGIN)

        self.play(Write(formula_grande), run_time=1.2)
        self._siguiente()
        self.play(FadeOut(formula_grande), run_time=0.7)


        c1 = columna("Logits",  ["2.0", "1.0", "0.1"])
        k1 = conector(r"\exp(x)", es_tex=True)
        c2 = columna("exp(x)", ["7.39", "2.72", "1.10"],
                     fondo=CREMA_CALIDA)
        k2 = conector(r"\div\,\Sigma", es_tex=True)
        c3 = columna("Prob",   ["66%", "24%", "10%"],
                     color_etiq=NARANJA_TERRACOTA,
                     fondo=NARANJA_TERRACOTA, texto=PAPEL_CREMA)

        flujo1 = VGroup(c1, k1, c2, k2, c3)\
            .arrange(RIGHT, buff=0.55).move_to(DOWN * 0.3)

        self.play(FadeIn(c1, shift=UP * 0.2))
        self.play(
            Write(k1),
            ReplacementTransform(c1[1].copy(), c2[1]),
            FadeIn(c2[0])
        )
        self.play(
            Write(k2),
            ReplacementTransform(c2[1].copy(), c3[1]),
            FadeIn(c3[0])
        )
        self._siguiente()

        self.play(FadeOut(flujo1), run_time=0.7)


        subtit_2 = Text("Problema: overflow", font=FUENTE, font_size=26,
                         color=NARANJA_TERRACOTA).next_to(linea, DOWN, buff=0.4)
        self.play(FadeIn(subtit_2, shift=UP * 0.15))

        c_e1 = columna("Logits", ["800.0", "2.0", "-1.0"])
        c_e1[1][0].set_fill(color=ROJO_TOMATE)
        c_e1[1][0][0].set_color(PAPEL_CREMA)

        k_e1 = conector(r"\exp(x)", es_tex=True)

        c_e2 = columna("exp(x)", ["inf", "7.39", "0.37"],
                        fondo=CREMA_CALIDA)
        c_e2[1][0].set_fill(color=ROJO_TOMATE)
        c_e2[1][0][0].set_color(PAPEL_CREMA)

        flujo_err = VGroup(c_e1, k_e1, c_e2)\
            .arrange(RIGHT, buff=0.7).next_to(subtit_2, DOWN, buff=0.5)

        self.play(FadeIn(c_e1, shift=UP * 0.2))
        self.play(Flash(c_e1[1][0], color=NARANJA_TERRACOTA, line_length=0.18))
        self.play(Write(k_e1))
        self.play(ReplacementTransform(c_e1[1].copy(), c_e2[1]), FadeIn(c_e2[0]))
        self.play(Wiggle(c_e2[1][0], scale_value=1.15))

        self._siguiente()
        self.play(FadeOut(flujo_err), FadeOut(subtit_2), run_time=0.7)


        subtit_3 = Text("Fix: restar el máximo", font=FUENTE, font_size=26,
                         color=NARANJA_TERRACOTA).next_to(linea, DOWN, buff=0.4)
        self.play(FadeIn(subtit_3, shift=UP * 0.15))

        c_f1 = columna("Logits",     ["800.0", "2.0", "-1.0"])
        k_f1 = conector("− max(x)")
        c_f2 = columna("Shifted",    ["0.0", "-798.0", "-801.0"],
                        color_etiq=NARANJA_TERRACOTA,
                        fondo=CREMA_CALIDA)
        k_f2 = conector(r"\exp(x)", es_tex=True)
        c_f3 = columna("exp seguro", ["1.0", "≈ 0", "≈ 0"],
                        fondo=VERDE_OLIVA, texto=PAPEL_CREMA)
        k_f3 = conector(r"\div\,\Sigma", es_tex=True)
        c_f4 = columna("Prob",       ["100%", "0%", "0%"],
                        color_etiq=NARANJA_TERRACOTA,
                        fondo=NARANJA_TERRACOTA, texto=PAPEL_CREMA)

        flujo_fix = VGroup(c_f1, k_f1, c_f2, k_f2, c_f3, k_f3, c_f4)\
            .arrange(RIGHT, buff=0.3).next_to(subtit_3, DOWN, buff=0.5)

        self.play(FadeIn(c_f1))
        self.play(Write(k_f1),
                  ReplacementTransform(c_f1[1].copy(), c_f2[1]), FadeIn(c_f2[0]))
        self.play(Write(k_f2),
                  ReplacementTransform(c_f2[1].copy(), c_f3[1]), FadeIn(c_f3[0]))
        self.play(Write(k_f3),
                  ReplacementTransform(c_f3[1].copy(), c_f4[1]), FadeIn(c_f4[0]))

        self.play(Flash(c_f2[1][0], color=NARANJA_TERRACOTA, line_length=0.18))

        self._siguiente()
        self.limpiar_pantalla()

    def slide_strides(self):

        titulo, linea = self.crear_titulo(
            "Strides: Saltando en Memoria 1D",
            palabra_clave="Strides:",
            color_clave=NARANJA_TERRACOTA
        )

        camino_mancha = FunctionGraph(lambda x: 0.5 * math.sin(x) - 0.5, color=MARRON_OSCURO).set_opacity(0.15)
        camino_punteado = DashedVMobject(camino_mancha, num_dashes=45, dashed_ratio=0.5)

        lanza_fondo = Line(LEFT * 7 + DOWN * 2, RIGHT * 7 + UP * 2, color=NARANJA_TERRACOTA, stroke_width=2).set_opacity(0.15)

        decoracion_quijote = VGroup(camino_punteado, lanza_fondo).set_z_index(-2)

        adornos = self._crear_adornos_esquinas()
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.15))

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=VGroup(llanuras_fondo, decoracion_quijote))

        arr_1d = VGroup(*[self.crear_bloque(str(i)) for i in range(6)])
        arr_1d.arrange(RIGHT, buff=0.1).shift(UP * 1.5)

        lbl_1d = Text("RAM — 1D", font=FUENTE, font_size=20, color=MARRON_OSCURO, weight=BOLD).next_to(arr_1d, UP, buff=0.3)

        self.play(FadeIn(arr_1d, shift=UP*0.2), FadeIn(lbl_1d, shift=UP*0.2))
        self._siguiente()

        fila1 = VGroup(*[arr_1d[i].copy() for i in range(3)]).arrange(RIGHT, buff=0.1)
        fila2 = VGroup(*[arr_1d[i].copy() for i in range(3, 6)]).arrange(RIGHT, buff=0.1)

        mat_shape = VGroup(fila1, fila2).arrange(DOWN, buff=0.1).shift(DOWN * 0.2)
        lbl_2d = Text("Shape lógica (2×3)", font=FUENTE, font_size=20, color=MARRON_OSCURO, weight=BOLD).next_to(mat_shape, DOWN, buff=0.3)

        self.play(
            TransformFromCopy(VGroup(*arr_1d[0:3]), fila1),
            TransformFromCopy(VGroup(*arr_1d[3:6]), fila2),
            FadeIn(lbl_2d, shift=DOWN*0.2),
            run_time=1.5
        )

        self.play(
            arr_1d[0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            arr_1d[3][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            mat_shape[0][0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            mat_shape[1][0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
        )

        arco_1d = CurvedArrow(arr_1d[0].get_top(), arr_1d[3].get_top(), angle=-PI/2, color=NARANJA_TERRACOTA)
        txt_stride_1d = Text("stride = 3", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA, weight=BOLD).next_to(arco_1d, UP, buff=0.1)

        arco_2d = CurvedArrow(mat_shape[0][0].get_left(), mat_shape[1][0].get_left(), angle=PI/2, color=NARANJA_TERRACOTA).shift(LEFT*0.1)
        txt_stride_2d = Text("+1 Fila", font=FUENTE, font_size=16, color=NARANJA_TERRACOTA).next_to(arco_2d, LEFT, buff=0.1)

        self.play(
            Create(arco_1d), Write(txt_stride_1d),
            Create(arco_2d), Write(txt_stride_2d)
        )

        self._siguiente()

        adornos[1].clear_updaters()
        self.limpiar_pantalla()


    def _acto_operaciones_tensoriales(self, linea: Mobject) -> None:

        def _vec_col(valores, color_fill):
            celdas = VGroup()
            for v in valores:
                rect = RoundedRectangle(corner_radius=0.08, width=0.70, height=0.54,
                                        fill_color=color_fill, fill_opacity=0.82,
                                        stroke_color=MARRON_OSCURO, stroke_width=1.5)
                txt = Text(str(v), font="Monospace", font_size=17,
                           color=TINTA_NEGRA).move_to(rect)
                celdas.add(VGroup(rect, txt))
            return celdas.arrange(DOWN, buff=0.07)

        def _signo(s):
            return Text(s, font=FUENTE, font_size=32, weight=BOLD, color=MARRON_OSCURO)

        def _card(titulo_str, expr):
            lbl = Text(titulo_str, font=FUENTE, font_size=20,
                       weight=BOLD, color=NARANJA_TERRACOTA)
            fondo = RoundedRectangle(
                corner_radius=0.18,
                width=max(expr.width + 0.7, lbl.width + 0.5),
                height=expr.height + lbl.height + 0.7,
                fill_color=FONDO_CAJA, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=1.8,
            )
            contenido = VGroup(lbl, expr).arrange(DOWN, buff=0.22)
            contenido.move_to(fondo)
            return VGroup(fondo, contenido)


        v1a  = _vec_col([1, 2, 3], PAPEL_TAN)
        v1b  = _vec_col([4, 5, 6], PAPEL_TAN)
        v1c  = _vec_col([5, 7, 9], NARANJA_TERRACOTA)
        ex1  = VGroup(v1a, _signo("+"), v1b, _signo("="), v1c
                      ).arrange(RIGHT, buff=0.20)
        card1 = _card("Suma Elementwise", ex1)


        rect_esc = RoundedRectangle(corner_radius=0.08, width=0.70, height=0.54,
                                    fill_color=VERDE_OLIVA, fill_opacity=0.80,
                                    stroke_color=MARRON_OSCURO, stroke_width=1.5)
        txt_esc  = Text("10", font="Monospace", font_size=17,
                        color=PAPEL_CREMA).move_to(rect_esc)
        escalar2 = VGroup(rect_esc, txt_esc)

        v2a = _vec_col([1, 2, 3], PAPEL_TAN)
        v2c = _vec_col([11, 12, 13], NARANJA_TERRACOTA)
        ex2 = VGroup(v2a, _signo("+"), escalar2, _signo("="), v2c
                     ).arrange(RIGHT, buff=0.18)
        card2 = _card("Broadcasting", ex2)


        rect_k = RoundedRectangle(corner_radius=0.08, width=0.70, height=0.54,
                                   fill_color=AZUL_NOCHE, fill_opacity=0.75,
                                   stroke_color=MARRON_OSCURO, stroke_width=1.5)
        txt_k  = Text("2", font="Monospace", font_size=17,
                      color=PAPEL_CREMA).move_to(rect_k)
        escalar3 = VGroup(rect_k, txt_k)

        v3a = _vec_col([3, 1, 5], PAPEL_TAN)
        v3c = _vec_col([6, 2, 10], NARANJA_TERRACOTA)
        ex3 = VGroup(escalar3, _signo("×"), v3a, _signo("="), v3c
                     ).arrange(RIGHT, buff=0.20)
        card3 = _card("Multiplicación Escalar", ex3)


        cards = VGroup(card1, card2, card3).arrange(RIGHT, buff=0.55)
        cards.set_x(0)
        if cards.width > 13.2:
            cards.scale(13.2 / cards.width)


        zona_top = linea.get_bottom()[1] - 0.3
        zona_bot = -3.6
        cards.set_y((zona_top + zona_bot) / 2)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.92, shift=UP * 0.15) for c in cards],
                        lag_ratio=0.35),
            run_time=1.2,
        )

        self._siguiente()
        self.play(FadeOut(cards))

    def slide_matmul(self):


        llanuras_fondo = crear_llanuras_manchegas()

        camino_mancha    = FunctionGraph(lambda x: 0.5 * math.sin(x) - 0.5,
                                        color=MARRON_OSCURO).set_opacity(0.15)
        camino_punteado  = DashedVMobject(camino_mancha, num_dashes=45, dashed_ratio=0.5)
        lanza_fondo      = Line(LEFT * 7 + DOWN * 2, RIGHT * 7 + UP * 2,
                                color=NARANJA_TERRACOTA, stroke_width=2).set_opacity(0.15)
        decoracion_fondo = VGroup(camino_punteado, lanza_fondo).set_z_index(-2)

        adornos = self._crear_adornos_esquinas()
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.15))


        titulo, linea = self.crear_titulo(
            "Operaciones Tensoriales",
            palabra_clave="Tensoriales",
            color_clave=NARANJA_TERRACOTA,
        )
        self._animar_entrada_slide(
            titulo, linea,
            adornos=adornos,
            fondo=VGroup(llanuras_fondo, decoracion_fondo),
        )


        self._acto_operaciones_tensoriales(linea)


        nuevo_titulo = Text(
            "MatMul: El Corazón del Transformer",
            font=FUENTE, font_size=35, color=TINTA_NEGRA,
            t2c={"MatMul": NARANJA_TERRACOTA},
        ).to_edge(UP)
        nueva_linea = Underline(nuevo_titulo, color=NARANJA_TERRACOTA, stroke_width=4)
        self.play(
            ReplacementTransform(titulo, nuevo_titulo),
            ReplacementTransform(linea, nueva_linea),
            run_time=0.7,
        )
        linea = nueva_linea


        A_rows, A_cols, B_cols = 2, 4, 3
        k = A_cols

        A = [
            [1, 2, 3, 0],
            [4, 1, 2, 3],
        ]
        B = [
            [2, 1, 0],
            [0, 2, 1],
            [1, 0, 3],
            [1, 1, 1],
        ]
        C_vals = [
            [sum(A[i][kk] * B[kk][j] for kk in range(k)) for j in range(B_cols)]
            for i in range(A_rows)
        ]

        val_A = [str(A[i][j]) for i in range(A_rows) for j in range(A_cols)]
        val_B = [str(B[i][j]) for i in range(k) for j in range(B_cols)]
        val_C_vacias = [""] * (A_rows * B_cols)

        mat_A     = self.crear_matriz_bloques(A_rows, A_cols, valores=val_A)
        signo_por = Text("×", font=FUENTE, font_size=40, color=TINTA_NEGRA)
        mat_B     = self.crear_matriz_bloques(k, B_cols, valores=val_B)
        signo_ig  = Text("=", font=FUENTE, font_size=40, color=TINTA_NEGRA)
        mat_C     = self.crear_matriz_bloques(A_rows, B_cols, valores=val_C_vacias)

        grupo_matmul = VGroup(mat_A, signo_por, mat_B, signo_ig, mat_C).arrange(RIGHT, buff=0.4)
        grupo_matmul.center().shift(DOWN * 0.1)

        lbl_A = Text(f"A  ({A_rows}×{A_cols})", font=FUENTE, font_size=18, color=MARRON_OSCURO).next_to(mat_A, UP, buff=0.15)
        lbl_B = Text(f"B  ({k}×{B_cols})",      font=FUENTE, font_size=18, color=MARRON_OSCURO).next_to(mat_B, UP, buff=0.15)
        lbl_C = Text(f"C  ({A_rows}×{B_cols})", font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD).next_to(mat_C, UP, buff=0.15)

        self.play(FadeIn(grupo_matmul, shift=UP * 0.2), FadeIn(VGroup(lbl_A, lbl_B, lbl_C)))

        calculo_label = Text("", font=FUENTE, font_size=26, color=MARRON_OSCURO).next_to(grupo_matmul, DOWN, buff=0.6)
        self.add(calculo_label)


        txt_resultados = VGroup()

        for i in range(A_rows):
            for j in range(B_cols):
                fila_i = mat_A[i]
                col_j  = VGroup(*[mat_B[kk][j] for kk in range(k)])
                celda_C = mat_C[i][j]

                anim_highlight = [
                    *[b[0].animate.set_fill(PAPEL_TAN, opacity=0.85) for b in fila_i],
                    *[b[0].animate.set_fill(PAPEL_TAN, opacity=0.85) for b in col_j],
                ]

                terminos  = " + ".join(f"({A[i][kk]}×{B[kk][j]})" for kk in range(k))
                resultado = C_vals[i][j]
                nuevo_label = Text(
                    f"{terminos} = {resultado}",
                    font=FUENTE, font_size=20, color=MARRON_OSCURO,
                    t2c={str(resultado): NARANJA_TERRACOTA},
                ).next_to(grupo_matmul, DOWN, buff=0.6)

                self.play(*anim_highlight, ReplacementTransform(calculo_label, nuevo_label), run_time=0.5)

                bloque_res = self.crear_bloque(
                    str(resultado),
                    color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=0.8
                ).move_to(nuevo_label.get_center())

                self.play(ReplacementTransform(nuevo_label, bloque_res), run_time=0.35)
                self.play(
                    bloque_res.animate.move_to(celda_C.get_center()),
                    celda_C[0].animate.set_fill(NARANJA_TERRACOTA, opacity=0.75),
                    run_time=0.45,
                )
                self.remove(bloque_res)

                txt_res = Text(str(resultado), font=FUENTE, font_size=20, color=PAPEL_CREMA).move_to(celda_C)
                self.add(txt_res)
                txt_resultados.add(txt_res)

                calculo_label = Text("", font=FUENTE, font_size=26, color=MARRON_OSCURO).next_to(grupo_matmul, DOWN, buff=0.6)
                self.add(calculo_label)

                anim_reset = [
                    *[b[0].animate.set_fill(FONDO_CAJA, opacity=1) for b in fila_i],
                    *[b[0].animate.set_fill(FONDO_CAJA, opacity=1) for b in col_j],
                ]
                self.play(*anim_reset, run_time=0.25)
        self._siguiente()
        self.remove(calculo_label)


        self.play(
            FadeOut(grupo_matmul),
            FadeOut(txt_resultados),
            FadeOut(lbl_A), FadeOut(lbl_B), FadeOut(lbl_C),
        )

        pregunta = Text(
            "¿Por qué optimizar la multiplicación de matrices?",
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
        ).next_to(linea, DOWN, buff=0.35)
        self.play(Write(pregunta))


        radio = 1.4
        datos_pastel = [
            (0.833, NARANJA_TERRACOTA, "MatMul\n83.3 %"),
            (0.064, VERDE_OLIVA,       "Elementwise\n6.4 %"),
            (0.058, OCRE_CERVANTINO,   "Norm.\n5.8 %"),
            (0.045, BEIGE_MEDIO,       "Comm.\n4.5 %"),
        ]

        sectores       = VGroup()
        labels_grafica = VGroup()
        lineas_grafica = VGroup()
        angulo_actual  = PI / 2

        for i, (porc, color, nombre) in enumerate(datos_pastel):
            angulo_sector = porc * TAU
            sector = Sector(
                radius=radio, angle=-angulo_sector, start_angle=angulo_actual,
                color=color, fill_opacity=0.92, stroke_color=BLANCO, stroke_width=1.5,
            )
            sectores.add(sector)

            angulo_medio = angulo_actual - angulo_sector / 2
            direccion    = np.array([np.cos(angulo_medio), np.sin(angulo_medio), 0])

            if i == 0:

                fs        = 20
                fw        = BOLD
                lbl_color = BLANCO
                lbl = Text(nombre, font=FUENTE, font_size=fs, color=lbl_color,
                        weight=fw, line_spacing=0.85)
                lbl.move_to(direccion * (radio * 0.55))
                labels_grafica.add(lbl)
                lineas_grafica.add(VMobject())
            else:
                fs        = 14
                fw        = NORMAL
                lbl_color = color
                lbl = Text(nombre, font=FUENTE, font_size=fs, color=lbl_color,
                        weight=fw, line_spacing=0.85)


                punto_borde    = direccion * radio
                punto_exterior = direccion * (radio + 0.55)
                linea_ext      = Line(punto_borde, punto_exterior, color=color, stroke_width=1.5)


                if direccion[0] >= 0.2:
                    dir_texto = RIGHT
                elif direccion[0] <= -0.2:
                    dir_texto = LEFT
                elif direccion[1] > 0:
                    dir_texto = UP
                else:
                    dir_texto = DOWN
                lbl.next_to(punto_exterior, dir_texto, buff=0.2)

                lineas_grafica.add(linea_ext)
                labels_grafica.add(lbl)

            angulo_actual -= angulo_sector

        borde_grafica    = Circle(radius=radio, color=MARRON_OSCURO, stroke_width=2)
        grafica_completa = VGroup(sectores, borde_grafica, lineas_grafica, labels_grafica)


        razones_data = [
            (NARANJA_TERRACOTA,
            "83.3 % de los FLOPs",
            "Atención, proyecciones Q/K/V y el bloque MLP\nson todas multiplicaciones de matrices."),
            (VERDE_OLIVA,
            "El cuello de botella real",
            "Cada token generado dispara decenas de MatMuls.\nOptimizarlas acelera toda la inferencia."),
            (ORO_VIEJO,
            "Técnicas aplicables",
            "SIMD, cache blocking, Rayon y batching se\ncombinen para multiplicar la ganancia total."),
        ]

        razones_grupo = VGroup()
        for color, titulo_r, cuerpo_r in razones_data:
            icono  = Square(side_length=0.2, color=color, fill_opacity=1, stroke_width=0)
            tit    = Text(titulo_r, font=FUENTE, font_size=19, color=color, weight=BOLD)
            cue    = Text(cuerpo_r, font=FUENTE, font_size=14, color=TINTA_NEGRA, line_spacing=0.9)
            textos = VGroup(tit, cue).arrange(DOWN, buff=0.05, aligned_edge=LEFT)
            fila   = VGroup(icono, textos).arrange(RIGHT, buff=0.25, aligned_edge=UP)
            razones_grupo.add(fila)
        razones_grupo.arrange(DOWN, buff=0.45, aligned_edge=LEFT)


        y_linea  = linea.get_bottom()[1]
        y_fondo  = -3.8
        y_centro = (y_linea + y_fondo) / 2

        contenido_principal = VGroup(grafica_completa, razones_grupo)
        contenido_principal.arrange(RIGHT, buff=0.9)
        contenido_principal.move_to(np.array([0, y_centro, 0]))


        margen_bajo_pregunta = pregunta.get_bottom()[1] - 0.3
        if contenido_principal.get_top()[1] > margen_bajo_pregunta:
            contenido_principal.next_to(pregunta, DOWN, buff=0.4)
            contenido_principal.set_x(0)

        self.play(
            LaggedStart(*[GrowFromCenter(s) for s in sectores], lag_ratio=0.15),
            run_time=1.2,
        )
        self.play(Create(borde_grafica), run_time=0.4)
        self.play(FadeIn(labels_grafica[0]))

        animaciones_etiquetas = [
            AnimationGroup(Create(lineas_grafica[j]), FadeIn(labels_grafica[j]))
            for j in range(1, len(datos_pastel))
        ]
        self.play(LaggedStart(*animaciones_etiquetas, lag_ratio=0.3), run_time=1.2)

        for fila in razones_grupo:
            self.play(
                GrowFromCenter(fila[0]),
                FadeIn(fila[1], shift=RIGHT * 0.2),
                run_time=0.7,
            )
        self._siguiente()


        adornos[1].clear_updaters()
        self.limpiar_pantalla()


    def slide_simd(self):
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "SIMD: Una Lanza, Cuatro Gigantes",
            palabra_clave="SIMD",
            color_clave=NARANJA_TERRACOTA
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        CELDA_W  = 1.10
        CELDA_H  = 0.80
        FONT_VAL = 21
        FONT_LBL = 16

        COLOR_A   = MARRON_OSCURO
        COLOR_B   = PAPEL_TAN
        COLOR_RES = MARRON_QUIJOTE
        COLOR_OP  = NARANJA_TERRACOTA

        def hacer_fila(vals, bg_color):
            celdas = VGroup(*[
                RoundedRectangle(
                    corner_radius=0.08,
                    width=CELDA_W, height=CELDA_H,
                    fill_color=bg_color, fill_opacity=0.20,
                    stroke_color=bg_color, stroke_width=2.5,
                )
                for _ in range(4)
            ]).arrange(RIGHT, buff=0.14)
            etqs = VGroup(*[
                Text(str(v), font=FUENTE, font_size=FONT_VAL, color=TINTA_NEGRA).move_to(celdas[i])
                for i, v in enumerate(vals)
            ])
            return celdas, etqs

        vals_a = ["a₃", "a₂", "a₁", "a₀"]
        vals_b = ["b₃", "b₂", "b₁", "b₀"]
        vals_r = ["a₃+b₃", "a₂+b₂", "a₁+b₁", "a₀+b₀"]

        CENTRO_Y = DOWN * 0.75

        celdas_a, etqs_a = hacer_fila(vals_a, COLOR_A)
        celdas_b, etqs_b = hacer_fila(vals_b, COLOR_B)
        celdas_r, etqs_r = hacer_fila(vals_r, COLOR_RES)

        VGroup(celdas_a, etqs_a).move_to(CENTRO_Y + UP * 1.60)
        VGroup(celdas_b, etqs_b).move_to(CENTRO_Y + UP * 0.38)
        VGroup(celdas_r, etqs_r).move_to(CENTRO_Y + DOWN * 1.05)


        def reg_label(texto, color, ref):
            return Text(texto, font=FUENTE, font_size=FONT_LBL, color=color, weight=BOLD) \
                       .next_to(ref, RIGHT, buff=0.28)

        lbl_s0 = reg_label("$s0", COLOR_A,   celdas_a)
        lbl_s1 = reg_label("$s1", COLOR_B,   celdas_b)
        lbl_s2 = reg_label("$s2", COLOR_RES, celdas_r)


        plus_sign = Text("+", font=FUENTE, font_size=40, color=MARRON_OSCURO, weight=BOLD) \
                        .next_to(celdas_b, LEFT, buff=0.30)


        sep_line = Line(
            celdas_r.get_left()  + LEFT  * 0.12 + UP * (CELDA_H / 2 + 0.18),
            celdas_r.get_right() + RIGHT * 0.05 + UP * (CELDA_H / 2 + 0.18),
            stroke_color=MARRON_OSCURO, stroke_width=2.5
        )


        instr = Text("padd8  $s2, $s0, $s1",
                     font="Courier New", font_size=24, color=NARANJA_TERRACOTA) \
                    .next_to(linea, DOWN, buff=0.32)


        self.play(Write(instr), run_time=0.7)
        self.play(
            LaggedStart(
                FadeIn(VGroup(celdas_a, lbl_s0),            shift=DOWN * 0.15),
                FadeIn(VGroup(plus_sign, celdas_b, lbl_s1), shift=DOWN * 0.15),
                FadeIn(VGroup(sep_line, celdas_r, lbl_s2),  shift=DOWN * 0.15),
                lag_ratio=0.35
            ),
            run_time=1.0
        )
        self.play(
            FadeIn(etqs_a, lag_ratio=0.07),
            FadeIn(etqs_b, lag_ratio=0.07),
            run_time=0.7
        )


        modo_lbl = Text("Modo Escalar — 1 suma por ciclo",
                        font=FUENTE, font_size=19, color=TINTA_NEGRA) \
                       .to_edge(DOWN, buff=0.55)
        self.play(FadeIn(modo_lbl, shift=UP * 0.15))

        ciclo_lbl = Text("ciclo 1", font=FUENTE, font_size=15,
                         color=COLOR_OP, weight=BOLD).next_to(instr, RIGHT, buff=0.35)
        self.play(FadeIn(ciclo_lbl))

        for idx in range(4):
            col = 3 - idx
            ciclo_num = idx + 1

            self.play(
                celdas_a[col].animate.set_fill(COLOR_OP, opacity=0.50).set_stroke(COLOR_OP, width=3.5),
                celdas_b[col].animate.set_fill(COLOR_OP, opacity=0.50).set_stroke(COLOR_OP, width=3.5),
                run_time=0.30
            )
            self.play(
                Write(etqs_r[col]),
                celdas_r[col].animate.set_fill(COLOR_RES, opacity=0.50).set_stroke(COLOR_RES, width=3.0),
                run_time=0.38
            )

            anims = [
                celdas_a[col].animate.set_fill(COLOR_A, opacity=0.20).set_stroke(COLOR_A, width=2.5),
                celdas_b[col].animate.set_fill(COLOR_B, opacity=0.20).set_stroke(COLOR_B, width=2.5),
            ]
            if idx < 3:
                nuevo_ciclo = Text(f"ciclo {ciclo_num + 1}", font=FUENTE, font_size=15,
                                   color=COLOR_OP, weight=BOLD).next_to(instr, RIGHT, buff=0.35)
                anims.append(Transform(ciclo_lbl, nuevo_ciclo))
            self.play(*anims, run_time=0.24)

        resumen_escalar = Text("4 ciclos → 4 sumas",
                               font=FUENTE, font_size=20, color=ROJO_CONTRA, weight=BOLD) \
                             .to_edge(DOWN, buff=0.55)
        self.play(FadeOut(modo_lbl), FadeIn(resumen_escalar, shift=UP * 0.12))
        self.wait(1.5)


        self._siguiente()


        self.play(
            FadeOut(ciclo_lbl),
            FadeOut(resumen_escalar),
            *[celdas_r[i].animate.set_fill(COLOR_RES, opacity=0.10).set_stroke(COLOR_RES, width=2.5)
              for i in range(4)],
            *[etqs_r[i].animate.set_opacity(0) for i in range(4)],
            run_time=0.45
        )


        modo_simd_lbl = Text("Una instrucción vectorial — 4 sumas en paralelo",
                             font=FUENTE, font_size=19, color=NARANJA_TERRACOTA, weight=BOLD) \
                            .to_edge(DOWN, buff=0.55)
        self.play(FadeIn(modo_simd_lbl, shift=UP * 0.15))


        marco_vec = SurroundingRectangle(
            VGroup(celdas_a, celdas_b),
            color=NARANJA_TERRACOTA, stroke_width=4, buff=0.16, corner_radius=0.14
        )

        lbl_vec = Text("Registro vectorial ×4", font=FUENTE, font_size=15,
                       color=NARANJA_TERRACOTA, weight=BOLD) \
                      .next_to(marco_vec, UP, buff=0.20)

        self.play(Create(marco_vec), FadeIn(lbl_vec, shift=DOWN * 0.12), run_time=0.55)

        self.play(
            *[celdas_a[i].animate.set_fill(NARANJA_TERRACOTA, opacity=0.45).set_stroke(NARANJA_TERRACOTA, width=3.5)
              for i in range(4)],
            *[celdas_b[i].animate.set_fill(NARANJA_TERRACOTA, opacity=0.45).set_stroke(NARANJA_TERRACOTA, width=3.5)
              for i in range(4)],
            run_time=0.35
        )


        for i in range(4):
            etqs_r[i].set_opacity(1)

        self.play(
            *[FadeIn(etqs_r[i]) for i in range(4)],
            *[celdas_r[i].animate.set_fill(COLOR_RES, opacity=0.60).set_stroke(COLOR_RES, width=3.5)
              for i in range(4)],
            Flash(celdas_r.get_center(), color=NARANJA_TERRACOTA, line_length=0.55, num_lines=14),
            run_time=0.60
        )

        resumen_simd = Text("1 ciclo → 4 sumas   (×4 más rápido)",
                            font=FUENTE, font_size=20, color=OCRE_CERVANTINO, weight=BOLD) \
                           .to_edge(DOWN, buff=0.55)
        self.play(FadeOut(modo_simd_lbl), FadeIn(resumen_simd, shift=UP * 0.12))
        self.wait(1.5)


        self._siguiente()
        self.limpiar_pantalla()


    def slide_cache_blocking(self):
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Cache Blocking: El Arte de Encajar",
            palabra_clave="Cache",
            color_clave=NARANJA_TERRACOTA
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        ram_box = RoundedRectangle(corner_radius=0.15, height=4.0, width=2.6,
                                   stroke_color=MARRON_OSCURO, stroke_width=3,
                                   fill_color=PAPEL_CREMA, fill_opacity=0.85)
        ram_box.move_to(LEFT * 3.2 + DOWN * 0.5)
        label_ram = Text("RAM", font=FUENTE, font_size=24, color=TINTA_NEGRA,
                         weight=BOLD).move_to(ram_box.get_top() + DOWN * 0.35)
        sub_ram = Text("Memoria principal", font=FUENTE, font_size=13,
                       color=MARRON_OSCURO).next_to(label_ram, DOWN, buff=0.04)


        matriz_ram = VGroup(*[
            Square(side_length=0.33,
                   stroke_color=MARRON_OSCURO, stroke_opacity=0.55,
                   fill_color=BEIGE_MEDIO, fill_opacity=0.65)
            for _ in range(36)
        ]).arrange_in_grid(6, 6, buff=0.03).next_to(sub_ram, DOWN, buff=0.14)


        cpu_box = RoundedRectangle(corner_radius=0.18, height=4.0, width=3.6,
                                   stroke_color=MARRON_OSCURO, stroke_width=3,
                                   fill_color=PAPEL_CREMA, fill_opacity=0.25)
        cpu_box.move_to(RIGHT * 2.6 + DOWN * 0.5)
        label_cpu = Text("CPU", font=FUENTE, font_size=24, color=TINTA_NEGRA,
                         weight=BOLD).move_to(cpu_box.get_top() + DOWN * 0.35)

        cache_box = RoundedRectangle(corner_radius=0.12, height=1.6, width=1.6,
                                     stroke_color=NARANJA_TERRACOTA, stroke_width=3,
                                     fill_color=NARANJA_TERRACOTA, fill_opacity=0.18)
        cache_box.next_to(label_cpu, DOWN, buff=0.40)
        label_cache = Text("Caché L1", font=FUENTE, font_size=14,
                           color=TINTA_NEGRA, weight=BOLD).move_to(cache_box.get_center())

        flecha_bus = Arrow(
            ram_box.get_right(), cpu_box.get_left(),
            color=MARRON_OSCURO, stroke_width=3, buff=0.10
        )
        label_bus = Text("Bus de datos", font=FUENTE, font_size=13,
                         color=MARRON_OSCURO).next_to(flecha_bus, UP, buff=0.08)

        self.play(DrawBorderThenFill(ram_box), Write(label_ram), FadeIn(sub_ram))
        self.play(Create(matriz_ram, lag_ratio=0.04), run_time=1.1)
        self.play(DrawBorderThenFill(cpu_box), Write(label_cpu))
        self.play(GrowArrow(flecha_bus), FadeIn(label_bus))
        self.play(DrawBorderThenFill(cache_box), Write(label_cache))


        texto_prob = Text(
            "MatMul necesita datos que no caben completos en caché",
            font=FUENTE, font_size=19, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=0.22)
        self.play(Write(texto_prob))


        col0 = VGroup(*[matriz_ram[i * 6 + 0] for i in range(6)])
        col1 = VGroup(*[matriz_ram[i * 6 + 1] for i in range(6)])
        col0_copy = col0.copy()
        col1_copy = col1.copy()

        self.play(
            col0_copy.animate.set_fill(MARRON_QUIJOTE, opacity=0.90),
            col1_copy.animate.set_fill(NARANJA_TERRACOTA, opacity=0.70),
        )


        grupo_cols = VGroup(col0_copy, col1_copy)
        self.play(
            grupo_cols.animate.move_to(cache_box.get_center()).scale(0.70),
            run_time=1.1
        )


        texto_lento = Text(
            "RAM: ~100 ns de latencia  ·  caché L1: ~1 ns",
            font=FUENTE, font_size=17, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.22)
        self.play(FadeTransform(texto_prob, texto_lento))

        texto_error = Text(
            "No caben → hay que releer desde RAM en cada bloque",
            font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(texto_lento, DOWN, buff=0.18)
        self.play(FadeIn(texto_error, shift=UP * 0.1))

        self.play(
            Wiggle(cache_box, scale_value=1.12, rotation_angle=0.04),
            Flash(cache_box, color=NARANJA_TERRACOTA, line_length=0.38, num_lines=10)
        )
        self.play(FadeOut(grupo_cols))
        self.wait(1.5)


        self._siguiente()


        texto_sol = Text(
            "Solución: llevar bloques 2×2 de a poco a la caché",
            font=FUENTE, font_size=19, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=0.22)
        self.play(
            FadeOut(texto_error),
            FadeTransform(texto_lento, texto_sol)
        )


        bloques_2x2 = [
            [r * 6 + c, r * 6 + c + 1, (r+1) * 6 + c, (r+1) * 6 + c + 1]
            for c in range(0, 6, 2)
            for r in range(0, 6, 2)
        ]
        colores_ciclo = [NARANJA_TERRACOTA, MARRON_QUIJOTE, MARRON_OSCURO]

        for k, indices in enumerate(bloques_2x2):
            color = colores_ciclo[k % len(colores_ciclo)]
            bloque_orig = VGroup(*[matriz_ram[i] for i in indices])
            bloque_copia = bloque_orig.copy()


            self.play(
                bloque_copia.animate.set_fill(color, opacity=0.88),
                run_time=0.22
            )

            self.play(
                bloque_copia.animate.move_to(cache_box.get_center()).scale(0.80),
                run_time=0.45
            )

            self.play(
                Flash(cache_box, color=color, line_length=0.22, num_lines=7),
                run_time=0.30
            )

            self.play(FadeOut(bloque_copia, shift=RIGHT * 0.4), run_time=0.28)

        resumen = Text(
            "Bloque a bloque: la caché nunca se desborda",
            font=FUENTE, font_size=19, color=TINTA_NEGRA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.25)
        self.play(FadeOut(texto_sol), FadeIn(resumen, shift=UP * 0.12))
        self.play(Indicate(cache_box, color=NARANJA_TERRACOTA, scale_factor=1.12))
        self.wait(1.5)


        self._siguiente()
        self.limpiar_pantalla()


    def slide_parallel(self):

        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Paralelización: La Fuerza de los Escuderos",
            palabra_clave="Paralelización",
            color_clave=NARANJA_TERRACOTA
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(
            titulo, linea,
            fondo=VGroup(llanuras_fondo),
            adornos=adornos
        )


        texto_problema = Text(
            "Un solo núcleo trabaja · los demás están ociosos",
            font=FUENTE, font_size=20, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=0.22)
        self.play(Write(texto_problema))


        matriz = VGroup(*[
            Square(side_length=0.54,
                   stroke_color=MARRON_OSCURO, stroke_width=1.5,
                   fill_color=PAPEL_CREMA, fill_opacity=0.65)
            for _ in range(36)
        ]).arrange_in_grid(6, 6, buff=0.04).move_to(DOWN * 0.55 + RIGHT * 1.6)

        self.play(Create(matriz, lag_ratio=0.03), run_time=0.9)


        ociosos = VGroup(*[
            Circle(
                radius=0.26,
                fill_color=BEIGE_MEDIO, fill_opacity=0.50,
                stroke_color=MARRON_OSCURO, stroke_width=1.8
            )
            for _ in range(3)
        ]).arrange(DOWN, buff=0.65).move_to(LEFT * 4.8 + DOWN * 0.3)


        nucleo = Circle(
            radius=0.26,
            fill_color=MARRON_OSCURO, fill_opacity=0.95,
            stroke_color=NARANJA_TERRACOTA, stroke_width=2.5
        ).move_to(LEFT * 4.8 + UP * 2.2)

        self.play(
            FadeIn(nucleo),
            LaggedStart(*[FadeIn(o, scale=0.7) for o in ociosos], lag_ratio=0.2),
            run_time=0.8
        )


        filas = [VGroup(*[matriz[r * 6 + c] for c in range(6)]) for r in range(6)]

        for fila in filas:
            self.play(
                nucleo.animate.next_to(fila[0], LEFT, buff=0.18),
                run_time=0.28
            )
            self.play(
                fila.animate.set_fill(MARRON_OSCURO, opacity=0.68),
                run_time=0.40
            )

        lbl_lento = Text(
            "Núcleos sin trabajo · tiempo perdido",
            font=FUENTE, font_size=17, color=ROJO_CONTRA, weight=BOLD
        ).next_to(matriz, DOWN, buff=0.24)
        self.play(FadeIn(lbl_lento, shift=UP * 0.12))
        self.play(Wiggle(ociosos, scale_value=1.10, rotation_angle=0.03))

        self._siguiente()


        self.play(
            FadeOut(texto_problema), FadeOut(lbl_lento),
            FadeOut(nucleo), FadeOut(ociosos),
            *[c.animate.set_fill(PAPEL_CREMA, opacity=0.65) for c in matriz],
            run_time=0.6
        )


        texto_solucion = Text(
            "Dividimos la matriz en franjas independientes",
            font=FUENTE, font_size=20, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.22)
        self.play(Write(texto_solucion))


        chunk1 = VGroup(*[matriz[i] for i in range(12)])
        chunk2 = VGroup(*[matriz[i] for i in range(12, 24)])
        chunk3 = VGroup(*[matriz[i] for i in range(24, 36)])
        chunks = [chunk1, chunk2, chunk3]
        colores_chunks  = [MARRON_OSCURO, NARANJA_TERRACOTA, MARRON_QUIJOTE]
        strokes_chunks  = [NARANJA_TERRACOTA, MARRON_OSCURO, NARANJA_TERRACOTA]


        self.play(
            chunk1.animate.shift(UP * 0.30),
            chunk3.animate.shift(DOWN * 0.30),
            run_time=0.75
        )


        texto_paralelo = Text(
            "Cada núcleo avanza en su franja · todos a la vez",
            font=FUENTE, font_size=20, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.22)
        self.play(FadeTransform(texto_solucion, texto_paralelo))


        bolitas_par = VGroup(*[
            Circle(
                radius=0.26,
                fill_color=colores_chunks[i], fill_opacity=0.95,
                stroke_color=strokes_chunks[i], stroke_width=2.5
            ).next_to(chunks[i][0], LEFT, buff=2.4)
            for i in range(3)
        ])

        self.play(
            LaggedStart(*[FadeIn(b, scale=0.6) for b in bolitas_par], lag_ratio=0.18),
            run_time=0.6
        )


        filas_por_chunk = [
            [VGroup(*[chunks[k][r * 6 + c] for c in range(6)]) for r in range(2)]
            for k in range(3)
        ]

        for paso in range(2):

            self.play(
                *[bolitas_par[k].animate.next_to(filas_por_chunk[k][paso][0], LEFT, buff=2.4)
                  for k in range(3)],
                run_time=0.40
            )

            self.play(
                *[filas_por_chunk[k][paso].animate.set_fill(colores_chunks[k], opacity=0.80)
                  for k in range(3)],
                run_time=0.50
            )

        self.play(
            *[Flash(chunks[k].get_center(), color=colores_chunks[k],
                    line_length=0.38, num_lines=8)
              for k in range(3)],
            run_time=0.70
        )


        self.play(
            chunk1.animate.shift(DOWN * 0.30),
            chunk3.animate.shift(UP   * 0.30),
            FadeOut(texto_paralelo),
            run_time=0.6
        )

        caja_concl = RoundedRectangle(
            corner_radius=0.18, width=7.8, height=1.05,
            fill_color=PAPEL_CREMA, fill_opacity=0.92,
            stroke_color=NARANJA_TERRACOTA, stroke_width=3
        ).next_to(linea, DOWN, buff=0.26)
        concl = Text(
            "Tiempo ≈ T / núcleos  (si el trabajo se divide bien)",
            font=FUENTE, font_size=21, color=MARRON_OSCURO, weight=BOLD
        ).move_to(caja_concl)

        self.play(DrawBorderThenFill(caja_concl), Write(concl))
        self.play(
            Indicate(matriz, color=NARANJA_TERRACOTA, scale_factor=1.04),
            Flash(matriz.get_center(), color=NARANJA_TERRACOTA, line_length=0.55, num_lines=14)
        )

        self._siguiente()
        self.limpiar_pantalla()


    def slide_batched_matmul(self):
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Batched MatMul: Una Llamada Para Ejecutarlos A Todos",
            palabra_clave="Batched",
            color_clave=NARANJA_TERRACOTA
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(
            titulo, linea,
            fondo=llanuras_fondo,
            adornos=adornos
        )

        Y_BOLSAS = DOWN * 0.80
        Y_ANNOT  = DOWN * 2.80
        BUFF_SUB = 0.55

        colores_cab   = [MARRON_OSCURO, NARANJA_TERRACOTA, MARRON_QUIJOTE]
        offsets_x     = [-3.6, 0.0, 3.6]
        nombres_bolsa = ["Operación A", "Operación B", "Operación C"]
        COLOR_LETRA = WHITE

        unidades = VGroup()
        for i in range(3):
            rect = RoundedRectangle(
                corner_radius=0.22, width=2.45, height=1.55,
                fill_color=PAPEL_CREMA, fill_opacity=1.0,
                stroke_color=colores_cab[i], stroke_width=3.0
            )
            letra = Text(
                ["A", "B", "C"][i], font=FUENTE, font_size=28,
                color=COLOR_LETRA, weight=BOLD
            ).move_to(rect).set_z_index(5)
            unidad = VGroup(rect, letra)
            unidad.move_to(RIGHT * offsets_x[i] + Y_BOLSAS)
            unidad.set_z_index(2)
            unidades.add(unidad)

        labels_bolsa = VGroup(*[
            Text(nombres_bolsa[i], font=FUENTE, font_size=14,
                color=colores_cab[i], weight=BOLD)
            .next_to(unidades[i], UP, buff=0.15)
            .set_z_index(3)
            for i in range(3)
        ])

        sub1 = Text(
            "Tres operaciones independientes · múltiples llamadas al procesador",
            font=FUENTE, font_size=18, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=BUFF_SUB)
        self.play(Write(sub1))

        self.play(
            LaggedStart(*[
                AnimationGroup(FadeIn(unidades[i]), FadeIn(labels_bolsa[i]))
                for i in range(3)
            ], lag_ratio=0.28),
            run_time=1.0
        )

        annot_viajes = VGroup(*[
            Text("1 llamada", font=FUENTE, font_size=13,
                color=ROJO_CONTRA, weight=BOLD)
            .next_to(unidades[i], DOWN, buff=0.18)
            .set_z_index(3)
            for i in range(3)
        ])

        for i in range(3):
            self.play(
                unidades[i][0].animate.set_fill(colores_cab[i], opacity=1.0),
                FadeIn(annot_viajes[i], shift=UP * 0.08),
                run_time=0.40
            )

        lbl_costo = Text(
            "3 operaciones · 3 llamadas · acceso a memoria ineficiente",
            font=FUENTE, font_size=16, color=ROJO_CONTRA, weight=BOLD
        ).move_to(Y_ANNOT).set_z_index(3)
        self.play(FadeIn(lbl_costo, shift=UP * 0.10))
        self.play(Wiggle(unidades, scale_value=1.05, rotation_angle=0.02))

        self._siguiente()

        self.play(
            FadeOut(sub1), FadeOut(lbl_costo),
            FadeOut(annot_viajes), FadeOut(labels_bolsa),
            *[unidades[i][0].animate.set_fill(PAPEL_CREMA, opacity=1.0)
            for i in range(3)],
            run_time=0.50
        )

        sub2 = Text(
            "Un batch · una sola llamada · ejecución paralela",
            font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=BUFF_SUB)
        self.play(Write(sub2))

        contenedor = RoundedRectangle(
            corner_radius=0.20, width=8.5, height=1.55,
            fill_color=PAPEL_CREMA, fill_opacity=1.0,
            stroke_color=MARRON_OSCURO, stroke_width=3.6
        ).move_to(Y_BOLSAS).set_z_index(1)

        ancho = 8.5
        destinos_x = [-ancho / 3, 0.0, ancho / 3]

        self.play(FadeIn(contenedor), run_time=0.40)

        self.play(
            *[unidades[i].animate
                .move_to(contenedor.get_center() + RIGHT * destinos_x[i])
                .scale(0.68)
            for i in range(3)],
            run_time=0.95
        )

        div_xs = [-ancho / 6, ancho / 6]
        divisores = VGroup(*[
            DashedLine(
                contenedor.get_top() + RIGHT * dx,
                contenedor.get_bottom() + RIGHT * dx,
                color=MARRON_OSCURO, stroke_width=1.6, dash_length=0.11
            ).set_z_index(4)
            for dx in div_xs
        ])
        self.play(Create(divisores), run_time=0.45)

        annot_contiguo = Text(
            "A · B · C se ejecutan juntas en una sola operación",
            font=FUENTE, font_size=14, color=MARRON_OSCURO
        ).move_to(Y_ANNOT).set_z_index(3)
        self.play(FadeIn(annot_contiguo, shift=UP * 0.10))

        self.play(
            *[unidades[i][0].animate.set_fill(colores_cab[i], opacity=1.0)
            for i in range(3)],
            run_time=0.55
        )

        self.play(
            *[Flash(unidades[i].get_center(), color=colores_cab[i],
                    line_length=0.30, num_lines=8)
            for i in range(3)],
            run_time=0.60
        )

        self.play(Indicate(contenedor, color=NARANJA_TERRACOTA, scale_factor=1.03))

        msg_final = Text(
            "Multiplicaciones independientes agrupadas en una sola llamada eficiente",
            font=FUENTE, font_size=16, color=MARRON_OSCURO, weight=BOLD
        ).move_to(Y_ANNOT).set_z_index(3)

        self.play(
            FadeTransform(annot_contiguo, msg_final),
            run_time=0.55
        )

        self._siguiente()
        self.limpiar_pantalla()