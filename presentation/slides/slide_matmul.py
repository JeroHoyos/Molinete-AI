import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import math
import os
from colores import *
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
            "Multiplicación de Matrices",
            font=FUENTE, font_size=35, color=TINTA_NEGRA,
            t2c={"Matrices": NARANJA_TERRACOTA},
        ).to_edge(UP)
        nueva_linea = Underline(nuevo_titulo, color=NARANJA_TERRACOTA, stroke_width=4)
        self.play(
            ReplacementTransform(titulo, nuevo_titulo),
            ReplacementTransform(linea, nueva_linea),
            run_time=0.7,
        )
        linea = nueva_linea


        self._acto_matmul()

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


    def _acto_matmul(self) -> None:
        # ── Multiplicación de matrices, con números ────────────────────
        A = [
            [1, 2, 3, 0],
            [0, 1, 2, 3],
            [2, 0, 1, 1],
        ]
        B = [
            [2, 1, 0],
            [0, 2, 1],
            [1, 0, 3],
            [1, 1, 1],
        ]
        C_vals = [
            [sum(A[i][kk] * B[kk][j] for kk in range(4)) for j in range(3)]
            for i in range(3)
        ]
        val_A = [str(v) for fila in A for v in fila]
        val_B = [str(v) for fila in B for v in fila]

        mat_A     = self.crear_matriz_bloques(3, 4, valores=val_A, ancho=0.85, alto=0.85)
        signo_por = Text("×", font=FUENTE, font_size=46, color=TINTA_NEGRA)
        mat_B     = self.crear_matriz_bloques(4, 3, valores=val_B, ancho=0.85, alto=0.85)
        signo_ig  = Text("=", font=FUENTE, font_size=46, color=TINTA_NEGRA)
        mat_C     = self.crear_matriz_bloques(3, 3, ancho=0.85, alto=0.85)
        # los Text("") vacíos de las celdas anclan el bounding box al origen
        # y desbaratan el arrange y el marco final: fuera
        for fila in mat_C:
            for b in fila:
                b.remove(b[1])

        for fila in mat_A:
            for b in fila:
                b[0].set_fill(NARANJA_CLARO, 0.55)
        for fila in mat_B:
            for b in fila:
                b[0].set_fill(MENTA_PALIDA, 0.55)

        grupo_matmul = VGroup(mat_A, signo_por, mat_B, signo_ig, mat_C)
        grupo_matmul.arrange(RIGHT, buff=0.5).center().shift(UP * 0.15)

        self.play(FadeIn(grupo_matmul, shift=UP * 0.2))

        def col_de(j):
            return VGroup(*[mat_B[kk][j] for kk in range(4)])

        fila_beam = SurroundingRectangle(
            mat_A[0], color=NARANJA_TERRACOTA,
            corner_radius=0.1, buff=0.06, stroke_width=4,
        )
        col_beam = SurroundingRectangle(
            col_de(0), color=VERDE_OLIVA,
            corner_radius=0.1, buff=0.06, stroke_width=4,
        )

        self.play(Create(fila_beam), Create(col_beam), run_time=0.6)

        txt_resultados = VGroup()

        def txt_de(i, j):
            t = Text(str(C_vals[i][j]), font=FUENTE, font_size=24,
                     color=PAPEL_CREMA, weight=BOLD).move_to(mat_C[i][j])
            txt_resultados.add(t)
            return t

        # primera celda: fila y columna se encienden y la celda nace
        celda00 = mat_C[0][0]
        self.play(
            Indicate(fila_beam, color=NARANJA_TERRACOTA, scale_factor=1.04),
            Indicate(col_beam, color=VERDE_OLIVA, scale_factor=1.04),
            celda00[0].animate.set_fill(NARANJA_TERRACOTA, 0.8),
            FadeIn(txt_de(0, 0), scale=0.4),
            Flash(celda00, color=NARANJA_TERRACOTA, line_length=0.22, num_lines=10),
            run_time=0.7,
        )

        # el barrido: la columna recorre B y C se llena a su paso
        self.play(
            col_beam.animate.move_to(col_de(2)),
            LaggedStart(*[
                AnimationGroup(
                    mat_C[0][j][0].animate.set_fill(NARANJA_TERRACOTA, 0.8),
                    FadeIn(txt_de(0, j), scale=0.4),
                ) for j in (1, 2)
            ], lag_ratio=0.45),
            run_time=0.8,
        )
        for i in (1, 2):
            self.play(
                fila_beam.animate.move_to(mat_A[i]),
                col_beam.animate.move_to(col_de(0)),
                run_time=0.35,
            )
            self.play(
                col_beam.animate.move_to(col_de(2)),
                LaggedStart(*[
                    AnimationGroup(
                        mat_C[i][j][0].animate.set_fill(NARANJA_TERRACOTA, 0.8),
                        FadeIn(txt_de(i, j), scale=0.4),
                    ) for j in range(3)
                ], lag_ratio=0.35),
                run_time=0.9,
            )

        marco_final = SurroundingRectangle(
            mat_C, color=NARANJA_TERRACOTA,
            corner_radius=0.12, buff=0.12, stroke_width=4,
        )
        self.play(
            FadeOut(fila_beam), FadeOut(col_beam),
            Create(marco_final),
            run_time=0.8,
        )
        self._siguiente()

        self.play(FadeOut(grupo_matmul),
                  FadeOut(txt_resultados), FadeOut(marco_final))


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
                    TransformFromCopy(fuente(i), resultado[i], path_arc=-0.6)
                    for i in range(3)
                ], lag_ratio=0.25),
                run_time=1.0,
            )

        self._siguiente()
        self.play(FadeOut(cards), FadeOut(v1c), FadeOut(v2c), FadeOut(v3c))

