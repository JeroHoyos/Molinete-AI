import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideParallel:
    def slide_parallel(self):

        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Paralelización",
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


        ociosos = Group(*[
            ImageMobject(os.path.join("assets", "sancho_rust.png")).set_height(0.75)
            for _ in range(2)
        ]).arrange(DOWN, buff=0.6).move_to(LEFT * 4.8 + DOWN * 0.4)
        for sancho in ociosos:
            sancho.set_opacity(0.45)

        zzz = Text("z z z", font=FUENTE, font_size=15,
                   color=ACERO, slant=ITALIC).next_to(ociosos, RIGHT, buff=0.2)

        nucleo = ImageMobject(
            os.path.join("assets", "quijote_rust.png")
        ).set_height(0.85).move_to(LEFT * 4.8 + UP * 2.2)

        t_lbl = Text("t = 1", font=FUENTE, font_size=18,
                     color=MARRON_OSCURO, weight=BOLD).next_to(matriz, UP, buff=0.2)

        self.play(
            FadeIn(nucleo),
            LaggedStart(*[FadeIn(o, scale=0.7) for o in ociosos], lag_ratio=0.2),
            FadeIn(zzz),
            FadeIn(t_lbl),
            run_time=0.8
        )


        filas = [VGroup(*[matriz[r * 6 + c] for c in range(6)]) for r in range(6)]

        for idx, fila in enumerate(filas):
            self.play(
                nucleo.animate.next_to(fila[0], LEFT, buff=0.25),
                run_time=0.28
            )
            anims_fila = [fila.animate.set_fill(MARRON_OSCURO, opacity=0.68)]
            if idx > 0:
                nuevo_t = Text(f"t = {idx + 1}", font=FUENTE, font_size=18,
                               color=MARRON_OSCURO, weight=BOLD).move_to(t_lbl)
                anims_fila.append(ReplacementTransform(t_lbl, nuevo_t))
                t_lbl = nuevo_t
            self.play(*anims_fila, run_time=0.40)

        lbl_lento = Text(
            "6t con un solo núcleo",
            font=FUENTE, font_size=17, color=ROJO_CONTRA, weight=BOLD
        ).next_to(matriz, DOWN, buff=0.24)
        self.play(FadeIn(lbl_lento, shift=UP * 0.12))
        self.play(Wiggle(ociosos, scale_value=1.10, rotation_angle=0.03))

        self._siguiente()


        self.play(
            FadeOut(texto_problema), FadeOut(lbl_lento),
            FadeOut(nucleo), FadeOut(ociosos), FadeOut(zzz), FadeOut(t_lbl),
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


        rutas_escuderos = ["quijote_rust.png", "sancho_rust.png", "sancho_rust.png"]
        escuderos = Group(*[
            ImageMobject(os.path.join("assets", ruta)).set_height(0.78)
            .next_to(chunks[i][0], LEFT, buff=0.4)
            for i, ruta in enumerate(rutas_escuderos)
        ])

        t_par = Text("t = 1", font=FUENTE, font_size=18,
                     color=NARANJA_TERRACOTA, weight=BOLD).next_to(chunks[0], UP, buff=0.2)

        self.play(
            LaggedStart(*[FadeIn(e, scale=0.6) for e in escuderos], lag_ratio=0.18),
            run_time=0.6
        )


        filas_por_chunk = [
            [VGroup(*[chunks[k][r * 6 + c] for c in range(6)]) for r in range(2)]
            for k in range(3)
        ]

        for paso in range(2):

            self.play(
                *[escuderos[k].animate.next_to(filas_por_chunk[k][paso][0], LEFT, buff=0.4)
                  for k in range(3)],
                run_time=0.40
            )

            anims_paso = [
                *[filas_por_chunk[k][paso].animate.set_fill(colores_chunks[k], opacity=0.80)
                  for k in range(3)],
            ]
            if paso == 0:
                anims_paso.append(FadeIn(t_par))
            else:
                nuevo_t_par = Text(f"t = {paso + 1}", font=FUENTE, font_size=18,
                                   color=NARANJA_TERRACOTA, weight=BOLD).move_to(t_par)
                anims_paso.append(ReplacementTransform(t_par, nuevo_t_par))
                t_par = nuevo_t_par
            self.play(*anims_paso, run_time=0.50)

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
            FadeOut(escuderos), FadeOut(t_par),
            run_time=0.6
        )

        caja_concl = RoundedRectangle(
            corner_radius=0.18, width=7.8, height=1.05,
            fill_color=PAPEL_CREMA, fill_opacity=0.92,
            stroke_color=NARANJA_TERRACOTA, stroke_width=3
        ).next_to(linea, DOWN, buff=0.26)
        concl = Text(
            "Tiempo = T / núcleos  (de 6t a 2t)",
            font=FUENTE, font_size=21, color=MARRON_OSCURO, weight=BOLD
        ).move_to(caja_concl)

        self.play(DrawBorderThenFill(caja_concl), Write(concl))
        self.play(
            Indicate(matriz, color=NARANJA_TERRACOTA, scale_factor=1.04),
            Flash(matriz.get_center(), color=NARANJA_TERRACOTA, line_length=0.55, num_lines=14)
        )

        self._siguiente()
        self.limpiar_pantalla()


