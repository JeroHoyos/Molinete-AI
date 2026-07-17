import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideIntroduction:
    def slide_introduction(self):

        llanuras_fondo = crear_llanuras_manchegas()
        amanecer = Rectangle(
            width=config.frame_width, height=config.frame_height,
            fill_color=NARANJA_TERRACOTA, fill_opacity=0.07, stroke_width=0
        )

        titulo, linea = self.crear_titulo(
            "Construyendo un Transformer con Rust y Python",
            palabra_clave="Rust y Python", font_size=30,
        )
        subtitulo = Text("Jerónimo Hoyos Botero", font=FUENTE, font_size=23,
                         color=MARRON_OSCURO)
        VGroup(titulo, linea, subtitulo).arrange(DOWN, buff=0.22).to_edge(UP, buff=0.45)

        def _adorno_subtitulo(lado):
            raya = Line(ORIGIN, RIGHT * 0.85, stroke_color=MARRON_OSCURO,
                        stroke_width=1.1, stroke_opacity=0.75)
            diamante = (Square(side_length=0.09).rotate(PI / 4)
                        .set_fill(NARANJA_TERRACOTA, 1).set_stroke(width=0))
            piezas = [raya, diamante] if lado is LEFT else [diamante, raya]
            return VGroup(*piezas).arrange(RIGHT, buff=0.12).next_to(subtitulo, lado, buff=0.3)

        adornos_subtitulo = VGroup(_adorno_subtitulo(LEFT), _adorno_subtitulo(RIGHT))

        armas_decor = crear_escudo_y_lanza().scale(0.45).to_corner(UL, buff=0.15)
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
            crear_estrella().scale(0.9).move_to(armas_decor.get_bottom() + DOWN*0.35 + RIGHT*0.2),
            crear_estrella().scale(0.7).move_to(libros_decor.get_top() + UP*0.25 + RIGHT*0.15),
        )

        POS_QUIJOTE = DOWN * 2.85 + LEFT * 3.8
        POS_SANCHO  = DOWN * 2.95 + LEFT * 1.3
        quijote = crear_rust_quijote().scale(0.88).move_to(LEFT * 12 + DOWN * 2.85)
        sancho  = crear_rust_sancho().scale(0.88).move_to(LEFT * 12 + DOWN * 2.95)

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
            "Molinete AI", font=FUENTE, font_size=42, weight=BOLD, color=TINTA_NEGRA
        ).scale(0.5).move_to(cajas_badge)

        flecha_molinete = Arrow(
            LEFT * 3.0 + UP * 0.9, LEFT * 1.6 + UP * 0.9,
            color=MARRON_OSCURO, stroke_width=5,
            max_tip_length_to_length_ratio=0.25,
        )

        POS_PROBS = LEFT * 4.2 + DOWN * 1.2

        self.play(FadeIn(llanuras_fondo), FadeIn(amanecer), run_time=1.1)
        self.play(
            Write(titulo), GrowFromCenter(linea),
            FadeIn(subtitulo, shift=DOWN * 0.1),
            FadeIn(adornos_subtitulo, scale=0.8),
            run_time=0.9,
        )
        self.play(
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

        INTERLINEA = 0.40
        FONT_POEMA = 17
        PAD_H      = 1.1
        PAD_V_TOP  = 0.75
        PAD_V_BOT  = 0.45

        lineas_poema = [
            Text(" ".join(verso), font=FUENTE, font_size=FONT_POEMA * 2,
                 color=TINTA_NEGRA, disable_ligatures=True).scale(0.5)
            for verso in poema
        ]
        ancho_max_verso = max(l.width for l in lineas_poema)

        N_LINEAS   = len(poema)
        PERG_ALTO  = (PAD_V_TOP + (N_LINEAS - 1) * INTERLINEA + PAD_V_BOT)
        PERG_ANCHO = max(ancho_max_verso + 2 * PAD_H, PERG_ALTO * 1.5)

        PERG_X     = -1.5 + PERG_ANCHO / 2
        PERG_Y     = -0.05
        PERG_CENTRO = np.array([PERG_X, PERG_Y, 0])

        pergamino = self._crear_pergamino_decorativo(ancho=PERG_ANCHO, alto=PERG_ALTO)
        pergamino.move_to(PERG_CENTRO)

        titulo_panel = Text(
            "Predicción de siguiente palabra",
            font=FUENTE, font_size=32, color=MARRON_OSCURO, slant=ITALIC
        ).scale(0.5).move_to(pergamino[1].get_top() + DOWN * 0.24)

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

        for i, linea_poema in enumerate(lineas_poema):
            linea_poema.move_to([TEXTO_X0, TEXTO_Y0 - i * INTERLINEA, 0],
                                aligned_edge=LEFT)

        curr_probs = Mobject()

        for verso, linea_poema in zip(poema, lineas_poema):
            espacio_es_glifo = len(linea_poema.submobjects) == len(" ".join(verso))
            idx = 0
            for word in verso:
                word_mob  = linea_poema[idx: idx + len(word)]
                idx      += len(word) + (1 if espacio_es_glifo else 0)
                new_probs = self._crear_panel_probs_rico(word, posicion=POS_PROBS)

                if not curr_probs.submobjects:
                    self.play(FadeIn(new_probs),
                              FadeIn(word_mob, shift=LEFT * 0.05), run_time=0.27)
                else:
                    self.play(ReplacementTransform(curr_probs, new_probs),
                              FadeIn(word_mob, shift=LEFT * 0.05), run_time=0.27)

                curr_probs = new_probs

        molino_grande[-1].clear_updaters()
        molino_pequeño[-1].clear_updaters()
        for estrella in estrellas:
            estrella.clear_updaters()

        self._siguiente()
        self.limpiar_pantalla()

