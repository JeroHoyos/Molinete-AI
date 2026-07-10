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


class SlideMatmul:
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


        ORIGEN_PT = np.array([-4.9, -1.35, 0])
        UNIDAD = 1.6
        M2 = np.array([[2, 1], [0, 1]])

        eje_x = Arrow(
            ORIGEN_PT + LEFT * 0.6, ORIGEN_PT + RIGHT * 5.4, buff=0,
            color=MARRON_OSCURO, stroke_width=2.5,
            max_tip_length_to_length_ratio=0.03,
        )
        eje_y = Arrow(
            ORIGEN_PT + DOWN * 0.55, ORIGEN_PT + UP * 2.15, buff=0,
            color=MARRON_OSCURO, stroke_width=2.5,
            max_tip_length_to_length_ratio=0.06,
        )

        cuadrado = Polygon(
            ORIGEN_PT,
            ORIGEN_PT + RIGHT * UNIDAD,
            ORIGEN_PT + RIGHT * UNIDAD + UP * UNIDAD,
            ORIGEN_PT + UP * UNIDAD,
            fill_color=OCRE_CERVANTINO, fill_opacity=0.35,
            stroke_color=MARRON_OSCURO, stroke_width=2,
        )
        vec_i = Arrow(
            ORIGEN_PT, ORIGEN_PT + RIGHT * UNIDAD, buff=0,
            color=NARANJA_TERRACOTA, stroke_width=5,
            max_tip_length_to_length_ratio=0.18,
        )
        vec_j = Arrow(
            ORIGEN_PT, ORIGEN_PT + UP * UNIDAD, buff=0,
            color=VERDE_OLIVA, stroke_width=5,
            max_tip_length_to_length_ratio=0.18,
        )

        def _celda_m(valor, color_txt):
            rect = RoundedRectangle(
                corner_radius=0.08, width=0.72, height=0.6,
                fill_color=FONDO_CAJA, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=1.5,
            )
            t = Text(str(valor), font=FUENTE, font_size=24,
                     color=color_txt, weight=BOLD).move_to(rect)
            return VGroup(rect, t)

        colores_col = [NARANJA_TERRACOTA, VERDE_OLIVA]
        celdas_M = VGroup(*[
            VGroup(*[_celda_m(M2[r][c], colores_col[c]) for c in range(2)])
            .arrange(RIGHT, buff=0.08)
            for r in range(2)
        ]).arrange(DOWN, buff=0.08)
        lbl_M = Text("M  =", font=FUENTE, font_size=28,
                     color=TINTA_NEGRA, weight=BOLD)
        chip_M = VGroup(lbl_M, celdas_M).arrange(RIGHT, buff=0.3)

        titular_lineal = Text(
            "M transforma el espacio",
            font=FUENTE, font_size=24, color=TINTA_NEGRA, weight=BOLD,
            t2c={"M": NARANJA_TERRACOTA},
        )
        col_derecha = VGroup(titular_lineal, chip_M).arrange(DOWN, buff=0.45)
        col_derecha.move_to([3.4, 0.9, 0])

        self.play(
            GrowArrow(eje_x), GrowArrow(eje_y),
            FadeIn(cuadrado), GrowArrow(vec_i), GrowArrow(vec_j),
            run_time=0.8,
        )
        self.play(
            Write(titular_lineal),
            FadeIn(chip_M, shift=DOWN * 0.2),
            run_time=0.9,
        )
        self._siguiente()

        fantasma = cuadrado.copy().set_fill(opacity=0)\
            .set_stroke(PAPEL_TAN, width=2, opacity=0.9)
        fantasma = DashedVMobject(fantasma, num_dashes=24)
        self.add(fantasma)

        self.play(
            ApplyMatrix(M2, VGroup(cuadrado, vec_i, vec_j), about_point=ORIGEN_PT),
            run_time=1.6,
        )

        punta_i = ORIGEN_PT + RIGHT * 2 * UNIDAD
        punta_j = ORIGEN_PT + (RIGHT + UP) * UNIDAD

        coord_i = Text("(2, 0)", font=FUENTE, font_size=19,
                       color=NARANJA_TERRACOTA, weight=BOLD)\
            .next_to(punta_i, DOWN, buff=0.15)
        coord_j = Text("(1, 1)", font=FUENTE, font_size=19,
                       color=VERDE_OLIVA, weight=BOLD)\
            .next_to(punta_j, UP, buff=0.15)

        col_M_1 = VGroup(celdas_M[0][0], celdas_M[1][0])
        col_M_2 = VGroup(celdas_M[0][1], celdas_M[1][1])
        enlace_i = CurvedArrow(
            col_M_1.get_bottom() + DOWN * 0.1,
            coord_i.get_right() + RIGHT * 0.12,
            angle=0.4, color=NARANJA_TERRACOTA, stroke_width=2.5,
        )
        enlace_j = CurvedArrow(
            col_M_2.get_top() + UP * 0.1,
            coord_j.get_right() + RIGHT * 0.12,
            angle=-0.4, color=VERDE_OLIVA, stroke_width=2.5,
        )

        self.play(
            FadeIn(coord_i, shift=UP * 0.1), FadeIn(coord_j, shift=DOWN * 0.1),
            Create(enlace_i), Create(enlace_j),
            run_time=0.9,
        )
        self._siguiente()

        self.play(
            FadeOut(VGroup(eje_x, eje_y, cuadrado, vec_i, vec_j, fantasma,
                           chip_M, titular_lineal, coord_i, coord_j,
                           enlace_i, enlace_j)),
            run_time=0.6,
        )


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


        radio = 1.7
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


        def _mini_pastel():
            return VGroup(
                Circle(radius=0.28, fill_color=BEIGE_MEDIO, fill_opacity=1,
                       stroke_color=MARRON_OSCURO, stroke_width=1.5),
                Sector(radius=0.28, start_angle=PI / 2, angle=-0.833 * TAU,
                       fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_width=0),
            )

        def _mini_embudo():
            return VGroup(
                Polygon(
                    [-0.3, 0.28, 0], [0.3, 0.28, 0], [0.07, -0.04, 0], [-0.07, -0.04, 0],
                    fill_color=VERDE_OLIVA, fill_opacity=1,
                    stroke_color=MARRON_OSCURO, stroke_width=1.5,
                ),
                Rectangle(
                    width=0.14, height=0.26,
                    fill_color=VERDE_OLIVA, fill_opacity=1,
                    stroke_color=MARRON_OSCURO, stroke_width=1.5,
                ).move_to([0, -0.18, 0]),
            )

        def _mini_chevrons():
            return VGroup(*[
                VMobject(stroke_color=ORO_VIEJO, stroke_width=5.5)
                .set_points_as_corners([[dx, 0.2, 0], [dx + 0.2, 0, 0], [dx, -0.2, 0]])
                for dx in (0.0, 0.24, 0.48)
            ])

        razones_grupo = VGroup(*[
            VGroup(icono, Text(texto_r, font=FUENTE, font_size=21,
                               color=TINTA_NEGRA, weight=BOLD))
            .arrange(RIGHT, buff=0.35)
            for icono, texto_r in [
                (_mini_pastel(),   "83 % de los FLOPs"),
                (_mini_embudo(),   "El cuello de botella"),
                (_mini_chevrons(), "SIMD, cache y hilos"),
            ]
        ]).arrange(DOWN, buff=0.55, aligned_edge=LEFT)


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

        ex1.remove(v1c)
        ex2.remove(v2c)
        ex3.remove(v3c)

        self.play(
            LaggedStart(*[FadeIn(c, scale=0.92, shift=UP * 0.15) for c in cards],
                        lag_ratio=0.35),
            run_time=1.2,
        )

        operaciones = [
            (v1c, lambda i: VGroup(v1a[i], v1b[i])),
            (v2c, lambda i: VGroup(v2a[i], escalar2)),
            (v3c, lambda i: VGroup(escalar3, v3a[i])),
        ]
        for resultado, fuente in operaciones:
            self.play(
                LaggedStart(*[
                    TransformFromCopy(fuente(i), resultado[i])
                    for i in range(3)
                ], lag_ratio=0.25),
                run_time=1.0,
            )

        self._siguiente()
        self.play(FadeOut(cards), FadeOut(v1c), FadeOut(v2c), FadeOut(v3c))

