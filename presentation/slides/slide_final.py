import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideFinal:
    def slide_final(self):

        def estrella_deco(pos, outer_r=0.22, inner_r=0.10):
            return Star(n=5, outer_radius=outer_r, inner_radius=inner_r,
                        color=ORO_VIEJO, fill_opacity=1, stroke_width=0).move_to(pos)

        def construir_marco_y_estrellas():
            ext = RoundedRectangle(
                corner_radius=0.35, width=13.2, height=7.2,
                stroke_color=NARANJA_TERRACOTA, stroke_width=5,
                fill_color=PAPEL_CREMA, fill_opacity=0.08
            ).move_to(ORIGIN)
            int_ = RoundedRectangle(
                corner_radius=0.22, width=12.6, height=6.6,
                stroke_color=MARRON_OSCURO, stroke_width=2, fill_opacity=0
            ).move_to(ORIGIN)

            estrellas = VGroup(
                estrella_deco(ext.get_corner(UL) + RIGHT*0.35 + DOWN*0.35),
                estrella_deco(ext.get_corner(UR) + LEFT*0.35  + DOWN*0.35),
                estrella_deco(ext.get_corner(DL) + RIGHT*0.35 + UP*0.35),
                estrella_deco(ext.get_corner(DR) + LEFT*0.35  + UP*0.35),
            )
            return ext, int_, estrellas

        def construir_textos():
            gracias = Text("¡Muchas Gracias!", font=FUENTE, font_size=66,
                           weight=BOLD, color=NARANJA_TERRACOTA).move_to(UP * 2.4)
            linea = Line(LEFT*4.5, RIGHT*4.5, color=NARANJA_TERRACOTA, stroke_width=3).next_to(gracias, DOWN, buff=0.18)
            sub = Text("Por tu atención y participación",
                       font=FUENTE, font_size=22, color=MARRON_OSCURO).next_to(linea, DOWN, buff=0.2)

            estrellas_tit = VGroup(*[
                estrella_deco(gracias.get_center() + RIGHT*(i-3)*1.1 + UP*0.55, 0.14, 0.06)
                for i in range(7)
            ])
            return gracias, linea, sub, estrellas_tit

        def construir_molino():
            base = Polygon([-0.85, -1.5, 0], [0.85, -1.5, 0], [0.52, 1.0, 0], [-0.52, 1.0, 0],
                           color=LADRILLO, fill_opacity=1, stroke_color=MADERA_OSCURA, stroke_width=2)
            puerta = RoundedRectangle(corner_radius=0.18, width=0.55, height=0.75,
                                      color=MADERA_OSCURA, fill_color=MADERA_CLARA, fill_opacity=1, stroke_width=2).move_to(base.get_bottom() + UP*0.38)
            ventana = Circle(radius=0.16, color=MADERA_OSCURA, fill_color=AZUL_NOCHE,
                             fill_opacity=0.8, stroke_width=2).move_to(base.get_center() + UP*0.32)
            techo = Polygon([-0.6, 1.0, 0], [0, 1.85, 0], [0.6, 1.0, 0],
                            color=TEJA, fill_opacity=1, stroke_color=MADERA_OSCURA, stroke_width=2)
            cuerpo = VGroup(base, puerta, ventana, techo)

            def crear_aspa():
                palo = Line(ORIGIN, UP*2.5, color=MADERA_OSCURA, stroke_width=5)
                vela = Polygon([0.08, 0.35, 0], [0.75, 0.35, 0], [0.75, 2.25, 0], [0.08, 2.25, 0],
                               color=MADERA_CLARA, fill_color=PERGAMINO, fill_opacity=0.92, stroke_width=1.5, stroke_color=MADERA_OSCURA)
                lineas = VGroup(*[Line([0.08, y, 0], [0.75, y, 0], color=MADERA_OSCURA, stroke_width=1.5) for y in np.linspace(0.55, 2.05, 5)])
                return VGroup(palo, vela, lineas)

            aspas = VGroup(*[crear_aspa().rotate(i * 90 * DEGREES, about_point=ORIGIN) for i in range(4)])
            centro_aspas_pos = techo.get_bottom() + UP * 0.32
            aspas.move_to(centro_aspas_pos)
            eje = Dot(centro_aspas_pos, color=HIERRO, radius=0.14)

            molino = VGroup(cuerpo, aspas, eje).scale(0.7).to_edge(LEFT, buff=0.8).shift(DOWN * 1.8)
            return cuerpo, aspas, eje, aspas.get_center()

        def construir_qr():
            qr_real = ImageMobject(os.path.join("assets", "qr_github_molineteai.png")).scale(0.85)
            fondo_qr = RoundedRectangle(corner_radius=0.25, width=qr_real.width + 0.5, height=qr_real.height + 0.5,
                                        color=NARANJA_TERRACOTA, stroke_width=4, fill_color=PAPEL_CREMA, fill_opacity=1)
            estr_l = estrella_deco(fondo_qr.get_corner(UL) + RIGHT*0.28 + DOWN*0.28)
            estr_r = estrella_deco(fondo_qr.get_corner(UR) + LEFT*0.28  + DOWN*0.28)

            grupo_qr = Group(fondo_qr, qr_real, estr_l, estr_r).to_edge(RIGHT, buff=1.0).shift(DOWN * 1.2)

            lbl_qr = Text("Repositorio del proyecto", font=FUENTE, font_size=17, weight=BOLD, color=MARRON_OSCURO).next_to(grupo_qr, UP, buff=0.28)
            url_lbl = Text("github.com/molineteai", font=FUENTE, font_size=15, color=NARANJA_TERRACOTA).next_to(grupo_qr, DOWN, buff=0.18)

            return fondo_qr, qr_real, estr_l, estr_r, lbl_qr, url_lbl

        def construir_creditos():
            return VGroup(
                Text("Proyecto:", font=FUENTE, font_size=19, color=MARRON_OSCURO),
                Text("Molinete AI", font=FUENTE, font_size=24, weight=BOLD, color=NARANJA_TERRACOTA),
                Text("Implementación de GPT-2 en Rust", font=FUENTE, font_size=17, color=TINTA_NEGRA),
            ).arrange(DOWN, buff=0.2).move_to(UP * 0.2)

        marco_ext, marco_int, estrellas_esq = construir_marco_y_estrellas()
        gracias, linea_deco, sub, estrellas_tit = construir_textos()
        cuerpo_molino, aspas, eje, centro_giro_aspas = construir_molino()
        fondo_qr, qr_img, estr_qr_l, estr_qr_r, lbl_qr, url_lbl = construir_qr()
        creditos = construir_creditos()

        quijote = crear_rust_quijote().scale(0.85).next_to(cuerpo_molino, RIGHT, buff=0.6, aligned_edge=DOWN)
        sancho = crear_rust_sancho().scale(0.85).next_to(quijote, RIGHT, buff=0.4, aligned_edge=DOWN)

        self.add(crear_llanuras_manchegas())

        self.play(Create(marco_ext), Create(marco_int), run_time=1.0)
        self.play(LaggedStart(*[GrowFromCenter(e) for e in estrellas_esq], lag_ratio=0.2), run_time=0.8)

        self.play(Write(gracias), run_time=1.0)
        self.play(Create(linea_deco), FadeIn(sub, shift=UP*0.2))
        self.play(LaggedStart(*[GrowFromCenter(s) for s in estrellas_tit], lag_ratio=0.1), run_time=0.9)

        self.play(LaggedStart(*[FadeIn(c, shift=UP*0.15) for c in creditos], lag_ratio=0.25), run_time=1.0)

        self.play(
            FadeIn(cuerpo_molino, shift=DOWN*0.5),
            GrowFromCenter(aspas),
            FadeIn(eje),
            FadeIn(quijote, shift=DOWN*0.2),
            FadeIn(sancho, shift=DOWN*0.2),
            run_time=1.2
        )

        self.play(
            DrawBorderThenFill(fondo_qr), FadeIn(qr_img),
            GrowFromCenter(estr_qr_l), GrowFromCenter(estr_qr_r),
            Write(lbl_qr), FadeIn(url_lbl, shift=UP*0.15),
            run_time=1.4
        )

        self.play(Rotate(aspas, angle=2*PI*4, about_point=centro_giro_aspas, run_time=10, rate_func=linear))
