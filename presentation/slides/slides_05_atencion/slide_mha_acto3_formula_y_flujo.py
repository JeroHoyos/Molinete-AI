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


class SlideMhaActo3FormulaYFlujo:
    def slide_mha_acto3_formula_y_flujo(self):


        titulo, linea = self.crear_titulo(
            "Atención: Formula y diagrama de flujo.",
            palabra_clave="Atención",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)


        formula = MathTex(
            r"\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            color=TINTA_NEGRA, font_size=62
        ).move_to(ORIGIN)
        self.play(FadeIn(formula, shift=UP, scale=0.9))
        self._siguiente()
        self.play(FadeOut(formula), run_time=0.8)
        self.play(FadeOut(adornos), run_time=0.5)


        SW       = 2.2
        TIP      = 0.22
        C_FLECHA = MARRON_OSCURO

        def arr(p1, p2):
            return Arrow(
                p1, p2, buff=0.07,
                color=C_FLECHA,
                stroke_width=SW,
                max_tip_length_to_length_ratio=TIP
            )

        def seg(p1, p2):
            return Line(p1, p2, stroke_color=C_FLECHA, stroke_width=SW)

        def L_up_right(x0, y0, x1, y1):
            return VGroup(seg([x0, y0, 0], [x0, y1, 0]),
                          arr([x0, y1, 0], [x1, y1, 0]))


        COLOR_W     = PAPEL_TAN
        COLOR_W_STR = MARRON_OSCURO
        W_W, W_H    = 1.15, 0.65

        def caja_W(letra):
            r = RoundedRectangle(
                corner_radius=0.16, width=W_W, height=W_H,
                fill_color=COLOR_W, fill_opacity=1,
                stroke_color=COLOR_W_STR, stroke_width=2.2
            )
            lbl = MathTex(rf"W_{{{letra}}}", font_size=20, color=MARRON_OSCURO)
            lbl.move_to(r)
            return VGroup(r, lbl)


        def mat_bloques(filas, cols, bw, bh, color_fondo=FONDO_CAJA):
            buff_b = 0.04
            mat = VGroup()
            for i in range(filas):
                fila = VGroup()
                for j in range(cols):
                    b = RoundedRectangle(
                        corner_radius=0.07, width=bw, height=bh,
                        fill_color=color_fondo, fill_opacity=1,
                        stroke_color=MARRON_OSCURO, stroke_width=1.4
                    )
                    fila.add(b)
                fila.arrange(RIGHT, buff=buff_b)
                mat.add(fila)
            mat.arrange(DOWN, buff=buff_b)
            return mat

        def etiq_math(texto, obj, dir_, buff=0.08, fs=12, col=TINTA_NEGRA):
            lbl = MathTex(texto, font_size=fs, color=col)
            lbl.next_to(obj, dir_, buff=buff)
            return lbl


        cx_X   = -6.2
        cx_W   = -4.7
        cx_QKV = -2.9
        cx_QKT =  0.2
        cx_AP  =  2.9
        cx_Y   =  5.8

        yq    =  1.5
        yk    =  0.0
        yv    = -1.5
        y_mid =  0.75
        y_Y   =  0.10


        X_mat = mat_bloques(8, 2, 0.24, 0.30, FONDO_CAJA).move_to([cx_X, 0, 0])
        lbl_X = MathTex("X", font_size=22, color=TINTA_NEGRA
                ).next_to(X_mat, DOWN, buff=0.08)

        xbif     = X_mat.get_right()[0] + 0.15
        dot_bif  = Dot(radius=0.08, color=MARRON_OSCURO).move_to([xbif, 0, 0])
        ln_x_dot = seg(X_mat.get_right(), dot_bif.get_center())


        Wq = caja_W("Q").move_to([cx_W, yq, 0])
        Wk = caja_W("K").move_to([cx_W, yk, 0])
        Wv = caja_W("V").move_to([cx_W, yv, 0])


        Q_mat = mat_bloques(4, 6, 0.22, 0.24, FONDO_CAJA  ).move_to([cx_QKV, yq, 0])
        K_mat = mat_bloques(4, 6, 0.22, 0.24, CREMA_CALIDA).move_to([cx_QKV, yk, 0])
        V_mat = mat_bloques(4, 6, 0.22, 0.24, SALMON_CLARO).move_to([cx_QKV, yv, 0])

        eQ = etiq_math(r"Q = XW_Q", Q_mat, DOWN, fs=11)
        eK = etiq_math(r"K = XW_K", K_mat, DOWN, fs=11)
        eV = etiq_math(r"V = XW_V", V_mat, DOWN, fs=11)


        QKT = mat_bloques(6, 6, 0.24, 0.24, FONDO_CAJA).move_to([cx_QKT, y_mid, 0])
        lbl_QKT = etiq_math(r"QK^T", QKT, UP, fs=13)


        AP_n = 6
        AP_celdas = VGroup()
        for i in range(AP_n):
            fila = VGroup()
            for j in range(AP_n):
                mask = j > i
                b = RoundedRectangle(
                    corner_radius=0.07, width=0.28, height=0.28,
                    fill_color=MARRON_OSCURO if mask else FONDO_CAJA,
                    fill_opacity=0.75 if mask else 1.0,
                    stroke_color=MARRON_OSCURO,
                    stroke_width=1.1 if mask else 1.5
                )
                fila.add(b)
            fila.arrange(RIGHT, buff=0.04)
            AP_celdas.add(fila)
        AP_celdas.arrange(DOWN, buff=0.04)
        AP_celdas.move_to([cx_AP, y_mid, 0])


        formula_esquina = MathTex(
            r"\mathrm{Softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)",
            color=TINTA_NEGRA, font_size=15
        ).next_to(AP_celdas, UP, buff=0.10).align_to(AP_celdas, RIGHT)


        Y_mat    = mat_bloques(4, 6, 0.22, 0.24, FONDO_CAJA).move_to([cx_Y, y_Y, 0])

        lbl_Y_eq = MathTex(
            r"\mathrm{Softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V",
            color=TINTA_NEGRA, font_size=13
        ).next_to(Y_mat, DOWN, buff=0.10)


        rt_q = L_up_right(xbif, 0, Wq.get_left()[0] - 0.06, yq)
        rt_k = arr(dot_bif.get_right(), Wk.get_left())
        rt_v = L_up_right(xbif, 0, Wv.get_left()[0] - 0.06, yv)


        a_Wq_Q = arr(Wq.get_right(), Q_mat.get_left())
        a_Wk_K = arr(Wk.get_right(), K_mat.get_left())
        a_Wv_V = arr(Wv.get_right(), V_mat.get_left())


        QK_group = VGroup(Q_mat, K_mat)
        brace_obj = Brace(
            QK_group, direction=RIGHT,
            color=MARRON_OSCURO,
            buff=0.10
        )

        brace_tip = brace_obj.get_tip()
        flecha_qkt = arr(brace_tip, QKT.get_left())

        brace_QK = VGroup(brace_obj, flecha_qkt)


        a_QKT_AP = arr(QKT.get_right(), AP_celdas.get_left())


        x_brace_APV  = AP_celdas.get_right()[0] + 0.22


        y_brace_top = AP_celdas.get_top()[1]
        y_brace_bot = yv

        brace_span = Line(
            [x_brace_APV, y_brace_top, 0],
            [x_brace_APV, y_brace_bot, 0]
        )
        brace_APV = Brace(brace_span, direction=RIGHT, color=MARRON_OSCURO, buff=0.08)


        tip_x        = brace_APV.get_tip()[0]
        tip_actual_y = brace_APV.get_tip()[1]


        flecha_Y  = arr([tip_x, y_Y, 0], Y_mat.get_left())


        bot_x     = brace_APV.get_bottom()[0]
        a_V_brace = Arrow(
            V_mat.get_right(), [bot_x, yv, 0],
            buff=0.07, color=C_FLECHA,
            stroke_width=SW,
            max_tip_length_to_length_ratio=0.04
        )


        self.play(
            FadeIn(X_mat), FadeIn(lbl_X),
            Create(ln_x_dot), FadeIn(dot_bif),
            run_time=0.8
        )
        self.play(
            LaggedStart(Create(rt_q), Create(rt_k), Create(rt_v), lag_ratio=0.15),
            LaggedStart(FadeIn(Wq), FadeIn(Wk), FadeIn(Wv), lag_ratio=0.15),
            run_time=1.0
        )


        self.play(
            LaggedStart(Create(a_Wq_Q), Create(a_Wk_K), Create(a_Wv_V), lag_ratio=0.12),
            LaggedStart(
                AnimationGroup(FadeIn(Q_mat), FadeIn(eQ)),
                AnimationGroup(FadeIn(K_mat), FadeIn(eK)),
                AnimationGroup(FadeIn(V_mat), FadeIn(eV)),
                lag_ratio=0.15
            ),
            run_time=1.3
        )
        self._siguiente()


        self.play(
            GrowFromCenter(brace_obj),
            run_time=0.7
        )
        self.play(
            Create(flecha_qkt),
            run_time=0.5
        )
        self.play(FadeIn(QKT), FadeIn(lbl_QKT), run_time=0.8)


        self.play(Create(a_QKT_AP), run_time=0.6)
        self.play(FadeIn(AP_celdas), run_time=0.9)
        self.play(FadeIn(formula_esquina, scale=0.9), run_time=0.5)
        self._siguiente()


        self.play(GrowFromCenter(brace_APV), run_time=0.7)
        self.play(
            Create(flecha_Y),
            Create(a_V_brace),
            run_time=0.6
        )
        self.play(
            FadeIn(Y_mat), FadeIn(lbl_Y_eq),
            run_time=0.8
        )
        self.play(
            Flash(Y_mat, color=NARANJA_TERRACOTA, line_length=0.22, num_lines=8),
            run_time=0.7
        )
        self._siguiente()

        self.limpiar_pantalla()

