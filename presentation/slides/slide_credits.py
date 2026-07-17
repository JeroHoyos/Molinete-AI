import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideCredits:
    def slide_credits(self):

        llanuras = crear_llanuras_manchegas()

        titulo_creditos, linea_creditos = self.crear_titulo(
            "Esta presentación se basa en:",
            palabra_clave="basa en:",
            color_clave=MARRON_OSCURO
        )

        escudo = crear_escudo_y_lanza().scale(0.8).to_corner(UL).shift(UP * 0.5 + LEFT * 0.3)

        molino = crear_molino().to_corner(DL).shift(UP * 0.5 + RIGHT * 0.5)
        libros = crear_pila_libros().to_corner(DR).shift(UP * 0.5 + LEFT * 0.5)
        tintero = crear_tintero_y_pluma().next_to(libros, LEFT, buff=0.8)

        estrellas = VGroup(
            crear_estrella().move_to(UP * 1.8 + LEFT * 6.0),
            crear_estrella().move_to(DOWN * 1.3 + LEFT * 5.6),
            crear_estrella().move_to(UP * 1.0 + RIGHT * 6.0)
        )

        imagen_articulo = ImageMobject(os.path.join("assets", "creditos_guia_original.png"))
        imagen_articulo.scale_to_fit_width(8.0)

        alto_header = 0.55
        ancho_caja  = imagen_articulo.width + 0.24
        alto_caja   = imagen_articulo.height + alto_header + 0.24

        sombra_ventana = RoundedRectangle(
            corner_radius=0.12, width=ancho_caja, height=alto_caja,
            fill_color=NEGRO_SUAVE, fill_opacity=0.25, stroke_width=0
        ).shift(DOWN * 0.13 + RIGHT * 0.13)

        ventana_bg = RoundedRectangle(
            corner_radius=0.12, width=ancho_caja, height=alto_caja,
            fill_color=FONDO_CAJA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=3,
        )

        ventana_header = Rectangle(
            width=ancho_caja, height=alto_header,
            fill_color=MARRON_OSCURO, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=3,
        ).align_to(ventana_bg, UP)

        botones_ventana = VGroup(
            Circle(radius=0.07, fill_color=ROJO_MAC,     fill_opacity=1, stroke_width=0),
            Circle(radius=0.07, fill_color=AMARILLO_MAC, fill_opacity=1, stroke_width=0),
            Circle(radius=0.07, fill_color=VERDE_MAC,    fill_opacity=1, stroke_width=0),
        ).arrange(RIGHT, buff=0.16).move_to(ventana_header.get_left() + RIGHT * 0.55)

        barra_url = RoundedRectangle(
            corner_radius=0.12, width=2.4, height=0.32,
            fill_color=PAPEL_CREMA, fill_opacity=0.18, stroke_width=0
        ).move_to(ventana_header)
        texto_url = Text("tag1.com", font="Monospace", font_size=15,
                         color=PAPEL_CREMA).move_to(barra_url)

        imagen_articulo.move_to(ventana_bg.get_center() + DOWN * (alto_header / 2))
        ventana = Group(sombra_ventana, ventana_bg, ventana_header,
                        botones_ventana, barra_url, texto_url, imagen_articulo)
        ventana.next_to(linea_creditos, DOWN, buff=0.4)

        credito_serie = Text(
            "Building an LLM From Scratch in Rust",
            font=FUENTE, font_size=26, weight=BOLD, color=NARANJA_TERRACOTA
        )
        credito_autor = Text(
            "Jeremy Andrews · Tag1 Consulting",
            font=FUENTE, font_size=22, color=MARRON_OSCURO
        )
        texto_repo = Text(
            "github.com/tag1consulting/feste",
            font="Monospace", font_size=16, color=TINTA_NEGRA
        )
        caja_repo = RoundedRectangle(
            corner_radius=0.1,
            width=texto_repo.width + 0.5, height=texto_repo.height + 0.3,
            fill_color=FONDO_CAJA, fill_opacity=0.95,
            stroke_color=MARRON_OSCURO, stroke_width=1.5
        ).move_to(texto_repo)
        credito_repo = VGroup(caja_repo, texto_repo)

        creditos_texto = VGroup(credito_serie, credito_autor, credito_repo)
        creditos_texto.arrange(DOWN, buff=0.18).next_to(ventana, DOWN, buff=0.35)

        self.play(FadeIn(llanuras, run_time=1.0))

        self._animar_entrada_slide(
            titulo_creditos, linea_creditos,
            adornos=VGroup(escudo, molino, libros, tintero, estrellas),
        )

        molino[-1].add_updater(lambda m, dt: m.rotate(-dt * 0.6))
        for estrella in estrellas:
            estrella.add_updater(lambda m, dt: m.rotate(dt * 0.5))

        self.play(
            FadeIn(sombra_ventana, shift=UP * 0.1),
            DrawBorderThenFill(ventana_bg),
            run_time=0.8
        )
        self.play(
            FadeIn(ventana_header, shift=DOWN * 0.2),
            FadeIn(botones_ventana, shift=RIGHT * 0.1),
            FadeIn(barra_url),
            Write(texto_url),
            run_time=0.6
        )
        self.play(FadeIn(imagen_articulo, scale=1.05), run_time=0.7)

        self.play(
            LaggedStart(
                FadeIn(credito_serie, shift=UP * 0.2),
                FadeIn(credito_autor, shift=UP * 0.2),
                FadeIn(credito_repo, shift=UP * 0.2),
                lag_ratio=0.3,
            ),
            run_time=1.2,
        )

        num_corazones = 18
        animaciones_corazones = []
        grupo_corazones = VGroup().set_z_index(-1)

        for _ in range(num_corazones):
            x_ini = np.random.uniform(-7.5, 7.5)
            y_ini = np.random.uniform(-5.0, -1.0)
            color_elegido = np.random.choice([NARANJA_TERRACOTA, MARRON_OSCURO, PAPEL_TAN, ROJO_TOMATE])
            escala_aleatoria = np.random.uniform(0.5, 1.3)

            corazon = self._crear_corazon(color_elegido, escala_aleatoria).move_to([x_ini, y_ini, 0])
            grupo_corazones.add(corazon)

            destino = corazon.get_center() + UP * np.random.uniform(3.5, 7.0) + RIGHT * np.random.uniform(-1.5, 1.5)

            animaciones_corazones.append(
                Succession(
                    FadeIn(corazon, scale=0.3, run_time=1),
                    corazon.animate(run_time=np.random.uniform(3.5, 6.0), rate_func=rate_functions.ease_in_out_sine)
                           .move_to(destino)
                           .set_opacity(0)
                           .rotate(np.random.uniform(-PI/5, PI/5))
                )
            )

        self.add(grupo_corazones)

        self.play(
            LaggedStart(*animaciones_corazones, lag_ratio=0.15),
            run_time=5.0
        )

        self._siguiente()

        molino[-1].clear_updaters()
        for estrella in estrellas:
            estrella.clear_updaters()

        elementos_en_pantalla = Group(*self.mobjects)
        self.play(FadeOut(elementos_en_pantalla, scale=0.9), run_time=1.5)

        self.limpiar_pantalla()
