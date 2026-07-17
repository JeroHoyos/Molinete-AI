import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideSimd:
    def slide_simd(self):
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "SIMD",
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

        def resumen_flecha(izq, der, color):
            flecha_res = Arrow(
                ORIGIN, RIGHT * 0.55, buff=0, color=color,
                stroke_width=3.5, max_tip_length_to_length_ratio=0.4,
            )
            return VGroup(
                Text(izq, font=FUENTE, font_size=20, color=color, weight=BOLD),
                flecha_res,
                Text(der, font=FUENTE, font_size=20, color=color, weight=BOLD),
            ).arrange(RIGHT, buff=0.22).to_edge(DOWN, buff=0.55)


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


        modo_lbl = Text("Modo escalar: 1 suma por ciclo",
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

        resumen_escalar = resumen_flecha("4 ciclos", "4 sumas", ROJO_CONTRA)
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


        modo_simd_lbl = Text("Una instrucción vectorial: 4 sumas en paralelo",
                             font=FUENTE, font_size=19, color=NARANJA_TERRACOTA, weight=BOLD) \
                            .to_edge(DOWN, buff=0.55)
        self.play(FadeIn(modo_simd_lbl, shift=UP * 0.15))


        marco_vec = SurroundingRectangle(
            VGroup(celdas_a, celdas_b),
            color=NARANJA_TERRACOTA, stroke_width=4, buff=0.16, corner_radius=0.14
        )

        lbl_vec = Text("Registro vectorial ×4", font=FUENTE, font_size=15,
                       color=NARANJA_TERRACOTA, weight=BOLD) \
                      .next_to(marco_vec, LEFT, buff=0.3)

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

        resumen_simd = resumen_flecha("1 ciclo", "4 sumas  (×4 más rápido)", OCRE_CERVANTINO)
        self.play(FadeOut(modo_simd_lbl), FadeIn(resumen_simd, shift=UP * 0.12))
        self.wait(1.5)


        self._siguiente()
        self.limpiar_pantalla()


