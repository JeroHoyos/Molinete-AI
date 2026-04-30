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


class SlidesInicio:
    def slide_pronto_iniciamos(self):

        gato_caballero = ImageMobject(os.path.join("assets", "gato_armadura.png")).scale(0.5)
        texto_inicio = Text("Pronto iniciamos", font=FUENTE, font_size=50, weight=BOLD, color=MARRON_OSCURO)

        cat_and_text = Group(gato_caballero, texto_inicio).arrange(DOWN, buff=0.8)
        cat_and_text.move_to(ORIGIN)

        pantalla_completa = Group(cat_and_text)

        self.play(FadeIn(pantalla_completa, shift=UP))

        self._siguiente()
        self.limpiar_pantalla()

    def slide_introduction(self):

        llanuras_fondo = crear_llanuras_manchegas()
        amanecer = Rectangle(
            width=config.frame_width, height=config.frame_height,
            fill_color=NARANJA_TERRACOTA, fill_opacity=0.07, stroke_width=0
        )

        titulo, linea = self.crear_titulo(
            "Construyendo un Transformer con Rust y Python", palabra_clave="Rust y Python"
        )
        subtitulo = Text("Jerónimo Hoyos Botero", font=FUENTE, font_size=25,
                         color=MARRON_OSCURO)
        VGroup(titulo, linea, subtitulo).arrange(DOWN, buff=0.2).to_edge(UP, buff=0.4)
        linea_deco = Line(LEFT * 2.8, RIGHT * 2.8, stroke_color=MARRON_OSCURO,
                          stroke_width=0.9).next_to(subtitulo, DOWN, buff=0.18)

        armas_decor = (crear_escudo_y_lanza().scale(0.55)
                       .to_corner(UL, buff=0.2).shift(DOWN * 0.10))
        escritorio_decor = (crear_tintero_y_pluma().scale(0.65)
                            .next_to(titulo, RIGHT, buff=0.3).shift(DOWN * 0.05))
        sol_decor = crear_sol_cervantino().scale(0.6).to_corner(UR, buff=0.35)

        molino_grande = crear_molino().scale(1.0).to_corner(DR, buff=0.2)
        molino_pequeño = (crear_molino().scale(0.58)
                          .next_to(molino_grande, LEFT, buff=0.0)
                          .shift(DOWN * 0.1 + RIGHT * 0.3))
        molinos_paisaje = VGroup(molino_pequeño, molino_grande)
        molino_grande[-1].add_updater(lambda m, dt: m.rotate(-dt * 0.85))
        molino_pequeño[-1].add_updater(lambda m, dt: m.rotate(-dt * 0.50))

        libros_decor = crear_pila_libros().scale(0.8).to_corner(DL, buff=0.3).shift(UP * 0.6)

        estrellas = VGroup(
            crear_estrella().scale(0.9).move_to(armas_decor.get_top() + UP*0.15 + RIGHT*0.25),
            crear_estrella().scale(0.7).move_to(libros_decor.get_top() + UP*0.25 + RIGHT*0.15),
        )

        POS_QUIJOTE = DOWN * 2.8 + LEFT * 3.2
        POS_SANCHO  = DOWN * 2.9 + LEFT * 1.0
        quijote = crear_rust_quijote().scale(0.88).move_to(LEFT * 12 + DOWN * 2.8)
        sancho  = crear_rust_sancho().scale(0.88).move_to(LEFT * 12 + DOWN * 2.9)

        POS_BADGE = LEFT * 4.2 + UP * 0.9
        sombra_badge = RoundedRectangle(
            corner_radius=0.15, width=2.5, height=0.95,
            fill_color=NEGRO_SUAVE, fill_opacity=0.15, stroke_width=0
        ).move_to(POS_BADGE + DR * 0.07)
        cajas_badge = VGroup(*[
            RoundedRectangle(
                corner_radius=0.13, width=2.4, height=0.9,
                fill_color=FONDO_CAJA, fill_opacity=0.97,
                stroke_color=MARRON_OSCURO, stroke_width=1.5
            ).shift(UP * i * 0.03 + RIGHT * i * 0.03)
            for i in range(3)
        ]).move_to(POS_BADGE)
        etiqueta_molinete = Text(
            "Molinete AI", font=FUENTE, font_size=21, weight=BOLD, color=TINTA_NEGRA
        ).move_to(cajas_badge)

        flecha_molinete = Arrow(
            LEFT * 3.0 + UP * 0.9, LEFT * 1.6 + UP * 0.9,
            color=MARRON_OSCURO, stroke_width=5,
            max_tip_length_to_length_ratio=0.25,
        )

        POS_PROBS = LEFT * 4.2 + DOWN * 1.2

        self.play(FadeIn(llanuras_fondo), FadeIn(amanecer), run_time=1.1)
        self.play(
            Write(titulo), GrowFromCenter(linea),
            FadeIn(subtitulo, shift=DOWN * 0.1), Create(linea_deco),
            run_time=0.9,
        )
        self.play(
            DrawBorderThenFill(escritorio_decor, run_time=1.1),
            SpinInFromNothing(sol_decor),
            GrowFromCenter(molinos_paisaje, lag_ratio=0.2),
            FadeIn(libros_decor, shift=UP * 0.2),
            DrawBorderThenFill(armas_decor, run_time=1.0),
            Create(estrellas, lag_ratio=0.3),
            run_time=1.2,
        )

        self.play(
            FadeIn(sombra_badge),
            FadeIn(cajas_badge, shift=UP * 0.1),
            FadeIn(etiqueta_molinete),
            run_time=0.65,
        )

        self.add(quijote, sancho)
        self.play(
            GrowArrow(flecha_molinete),
            quijote.animate.move_to(POS_QUIJOTE),
            sancho.animate.move_to(POS_SANCHO),
            run_time=0.85, rate_func=rate_functions.ease_out_cubic,
        )

        for estrella in estrellas:
            estrella.add_updater(lambda m, dt: m.rotate(dt * 0.5))

        poema = [
            ["retorciendo", "el", "mostacho", "soldadesco,"],
            ["por", "ver", "que", "ya", "su", "bolsa", "le", "repica,"],
            ["a", "un", "corrillo", "llegó", "de", "gente", "rica"],
            ["y", "en", "el", "nombre", "de", "Dios", "pidió", "refresco."],
            ["Den", "voacedes,", "por", "Dios,", "a", "mi", "pobreza,"],
            ["les", "dice;", "donde", "no,", "por", "ocho", "santos"],
            ["que", "haré", "lo", "que", "hacer", "suelo", "sin", "tardanza."],
        ]

        INTERLINEA = 0.42
        FONT_POEMA = 17
        PAD_H      = 1.5
        PAD_V_TOP  = 0.55
        PAD_V_BOT  = 0.10

        ancho_max_verso = max(
            sum(
                Text(w, font=FUENTE, font_size=FONT_POEMA).width + 0.13
                for w in verso
            ) - 0.13
            for verso in poema
        )

        N_LINEAS   = len(poema)
        PERG_ALTO  = (PAD_V_TOP + N_LINEAS * INTERLINEA + PAD_V_BOT)
        PERG_ANCHO = max(ancho_max_verso + 2 * PAD_H, PERG_ALTO * 1.5)

        PERG_X     = -1.5 + PERG_ANCHO / 2
        PERG_Y     = 0.0
        PERG_CENTRO = np.array([PERG_X, PERG_Y, 0])

        pergamino = self._crear_pergamino_decorativo(ancho=PERG_ANCHO, alto=PERG_ALTO)
        pergamino.move_to(PERG_CENTRO)

        titulo_panel = Text(
            "Predicción de siguiente palabra",
            font=FUENTE, font_size=14, color=MARRON_OSCURO, slant=ITALIC
        ).move_to(pergamino[1].get_top() + DOWN * 0.22)

        flecha_molinete.put_start_and_end_on(
            LEFT * 3.0 + UP * 0.9,
            np.array([PERG_CENTRO[0] - PERG_ANCHO / 2, 0.9, 0]),
        )

        self.play(
            FadeIn(pergamino, run_time=0.55),
            FadeIn(titulo_panel, run_time=0.55),
        )

        TEXTO_X0 = PERG_CENTRO[0] - ancho_max_verso / 2
        TEXTO_Y0 = PERG_CENTRO[1] + PERG_ALTO / 2 - PAD_V_TOP

        curr_y     = TEXTO_Y0
        curr_probs = Mobject()

        for linea_texto in poema:
            curr_x = TEXTO_X0
            for word in linea_texto:
                new_probs = self._crear_panel_probs_rico(word, posicion=POS_PROBS)
                word_mob  = (Text(word, font=FUENTE, font_size=FONT_POEMA, color=TINTA_NEGRA)
                             .move_to([curr_x, curr_y, 0], aligned_edge=LEFT))

                if not curr_probs.submobjects:
                    self.play(FadeIn(new_probs),
                              FadeIn(word_mob, shift=LEFT * 0.05), run_time=0.27)
                else:
                    self.play(ReplacementTransform(curr_probs, new_probs),
                              FadeIn(word_mob, shift=LEFT * 0.05), run_time=0.27)

                curr_probs  = new_probs
                curr_x     += word_mob.width + 0.13
            curr_y -= INTERLINEA

        molino_grande[-1].clear_updaters()
        molino_pequeño[-1].clear_updaters()
        for estrella in estrellas:
            estrella.clear_updaters()

        self._siguiente()
        self.limpiar_pantalla()

    def slide_credits(self):

        llanuras = crear_llanuras_manchegas()

        titulo_creditos, linea_creditos = self.crear_titulo(
            "Esta presentación se basa en:",
            palabra_clave="basa en:",
            color_clave=MARRON_OSCURO
        )

        imagen_creditos = ImageMobject(os.path.join("assets", "creditos_guia_original.png"))
        imagen_creditos.scale(1.5).next_to(linea_creditos, DOWN, buff=0.8)

        escudo = crear_escudo_y_lanza().scale(0.8).to_corner(UL).shift(UP * 0.5 + LEFT * 0.3)

        molino = crear_molino().to_corner(DL).shift(UP * 0.5 + RIGHT * 0.5)
        libros = crear_pila_libros().to_corner(DR).shift(UP * 0.5 + LEFT * 0.5)
        tintero = crear_tintero_y_pluma().next_to(libros, LEFT, buff=0.8)

        estrellas = VGroup(
            crear_estrella().move_to(UP * 1.5 + LEFT * 4),
            crear_estrella().move_to(DOWN * 1.5 + LEFT * 5),
            crear_estrella().move_to(UP * 0.5 + RIGHT * 5)
        )

        self.play(FadeIn(llanuras, run_time=1.5))

        self._animar_entrada_slide(
            titulo_creditos, linea_creditos,
            fondo=Group(imagen_creditos, molino, escudo, libros, tintero, estrellas)
        )

        num_corazones = 20
        animaciones_corazones = []
        grupo_corazones = VGroup().set_z_index(-1)

        for _ in range(num_corazones):
            x_ini = np.random.uniform(-7.5, 7.5)
            y_ini = np.random.uniform(-5.0, -1.0)
            color_elegido = np.random.choice([NARANJA_TERRACOTA, MARRON_OSCURO, "#D4A373", "#E63946"])
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
            run_time=6.0
        )

        self._siguiente()

        elementos_en_pantalla = Group(*self.mobjects)
        self.play(FadeOut(elementos_en_pantalla, scale=0.9), run_time=1.5)

        self.limpiar_pantalla()
