import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideTemperature:
    def slide_temperature(self):

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "La Temperatura",
            palabra_clave="Temperatura",
            color_clave=NARANJA_TERRACOTA
        )

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        # ── Visualización: qué significa la temperatura ─────────────────
        subt = Text("La temperatura reparte la probabilidad entre las palabras",
                    font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(linea, DOWN, buff=0.4)

        palabras = ["acordarme", "recordar", "soñar", "volar", "reír"]
        logits = np.array([3.0, 1.6, 1.0, 0.4, 0.1])

        BW, GAP = 0.95, 0.6
        BASE_Y, MAXH = -1.7, 3.2
        n = len(palabras)
        step = BW + GAP
        x0 = -(n - 1) * step / 2 + 0.9   # corrido a la derecha por el termómetro

        def probs(T):
            z = logits / T
            z = z - z.max()
            e = np.exp(z)
            return e / e.sum()

        etiquetas = VGroup(*[
            Text(w, font=FUENTE, font_size=16, color=TINTA_NEGRA).move_to([x0 + i * step, BASE_Y - 0.32, 0])
            for i, w in enumerate(palabras)
        ])
        base_line = Line([x0 - BW, BASE_Y, 0], [x0 + (n - 1) * step + BW * 0.7, BASE_Y, 0],
                         color=MARRON_OSCURO, stroke_width=2)

        def make_bars(T):
            p = probs(T)
            g = VGroup()
            for i, pi in enumerate(p):
                h = max(pi * MAXH, 0.04)
                bar = Rectangle(width=BW, height=h, fill_color=NARANJA_TERRACOTA,
                                fill_opacity=0.9, stroke_color=MARRON_OSCURO, stroke_width=2)
                bar.move_to([x0 + i * step, BASE_Y + h / 2, 0])
                pct = Text(f"{pi*100:.0f}%", font=FUENTE, font_size=15,
                           color=MARRON_OSCURO).next_to(bar, UP, buff=0.08)
                g.add(VGroup(bar, pct))
            return g

        THX = x0 - BW - 1.1   # x del termómetro

        def termometro(T):
            frac = float(np.clip((T - 0.2) / 2.0, 0.05, 1.0))
            col = interpolate_color(ManimColor(AZUL_NOCHE), ManimColor(ROJO_CONTRA), frac)
            tubo = RoundedRectangle(corner_radius=0.18, width=0.36, height=2.8,
                                    stroke_color=MARRON_OSCURO, stroke_width=3,
                                    fill_color=PAPEL_CREMA, fill_opacity=1)
            tubo.move_to([THX, BASE_Y + 1.5, 0])
            bulbo = Circle(radius=0.33, stroke_color=MARRON_OSCURO, stroke_width=3,
                           fill_color=col, fill_opacity=1).move_to(tubo.get_bottom())
            fill_h = 0.25 + frac * 2.2
            fill = RoundedRectangle(corner_radius=0.1, width=0.20, height=fill_h,
                                    fill_color=col, fill_opacity=1, stroke_width=0)
            fill.move_to(bulbo.get_center() + UP * (fill_h / 2))
            return VGroup(tubo, fill, bulbo)

        def t_label(T):
            frac = float(np.clip((T - 0.2) / 2.0, 0.05, 1.0))
            col = interpolate_color(ManimColor(AZUL_NOCHE), ManimColor(ROJO_CONTRA), frac)
            return Text(f"T = {T}", font=FUENTE, font_size=26, color=col, weight=BOLD)\
                .move_to([THX, BASE_Y + 3.1, 0])

        estados = [0.4, 1.0, 2.0]

        T0 = estados[0]
        bars = make_bars(T0)
        thermo = termometro(T0)
        tlab = t_label(T0)

        self.play(FadeIn(subt, shift=DOWN * 0.2))
        self.play(
            Create(base_line),
            FadeIn(etiquetas, lag_ratio=0.1),
            FadeIn(thermo), FadeIn(tlab),
            *[GrowFromEdge(b[0], DOWN) for b in bars],
            *[FadeIn(b[1]) for b in bars],
        )
        self._siguiente()

        for T in estados[1:]:
            self.play(
                Transform(bars, make_bars(T)),
                Transform(thermo, termometro(T)),
                Transform(tlab, t_label(T)),
                run_time=1.3,
            )
            self._siguiente()

        self.play(FadeOut(VGroup(subt, base_line, etiquetas, bars, thermo, tlab)))

        rect_prompt = RoundedRectangle(corner_radius=0.15, height=1.2, width=8)
        rect_prompt.set_fill(color=MARRON_OSCURO, opacity=0.1).set_stroke(color=MARRON_OSCURO, width=1.5)

        user_label = Text("Usuario", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(rect_prompt, UP, aligned_edge=LEFT).shift(DOWN*0.1 + RIGHT*0.2)

        texto_prompt = Text(
            "Prompt: \"En un lugar de la Mancha...\"",
            font=FUENTE, font_size=22, color=TINTA_NEGRA, weight=BOLD
        ).move_to(rect_prompt)

        grupo_prompt = VGroup(rect_prompt, user_label, texto_prompt).to_edge(UP, buff=1.5)

        self.play(FadeIn(grupo_prompt, shift=DOWN))

        def crear_respuesta(temp_val, intento, texto, color_perfil, titulo_perfil, img_file):
            sombra = RoundedRectangle(corner_radius=0.15, height=2.2, width=7.2)
            sombra.set_fill(MARRON_OSCURO, opacity=0.1).set_stroke(width=0)
            sombra.shift(RIGHT * 0.08 + DOWN * 0.08)

            rect = RoundedRectangle(corner_radius=0.15, height=2.2, width=7.2)
            rect.set_fill(color=PAPEL_CREMA, opacity=1).set_stroke(color=color_perfil, width=1.8)

            username = Text(f"{titulo_perfil} (T={temp_val})", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(rect, UP, aligned_edge=LEFT).shift(UP*0.1)

            contenido = Paragraph(
                texto, font=FUENTE, font_size=22, color=TINTA_NEGRA,
                line_spacing=1.3, alignment="left"
            ).scale_to_fit_width(rect.width - 0.8).move_to(rect)

            info = Text(f"Generación - Intento #{intento}",
                        font="Monospace", font_size=16, color=color_perfil).next_to(rect, DOWN, buff=0.15, aligned_edge=RIGHT)

            parte_v = VGroup(sombra, rect, username, contenido, info)

            avatar = ImageMobject(os.path.join("assets", img_file)).set_height(1.5)
            avatar.next_to(rect, LEFT, buff=0.35)

            return Group(parte_v, avatar).next_to(grupo_prompt, DOWN, buff=1.0)

        r_sancho_1 = crear_respuesta("0.1", 1, "\"...de cuyo nombre no quiero acordarme.\"", MARRON_OSCURO, "Sancho", "sancho_rust.png")
        r_sancho_2 = crear_respuesta("0.1", 2, "\"...de cuyo nombre no quiero acordarme.\"", MARRON_OSCURO, "Sancho", "sancho_rust.png")

        r_quijote_1 = crear_respuesta("1.5", 1, "\"...donde los dragones mecánicos beben aceite de oliva.\"", NARANJA_TERRACOTA, "El Quijote", "quijote_rust.png")
        r_quijote_2 = crear_respuesta("1.5", 2, "\"...los molinos me hablan en código binario al amanecer.\"", NARANJA_TERRACOTA, "El Quijote", "quijote_rust.png")

        actual = r_sancho_1
        self.play(FadeIn(actual, shift=UP))
        self._siguiente()

        self.play(FadeOut(actual, shift=UP * 0.2), FadeIn(r_sancho_2, shift=UP * 0.2), run_time=1)
        actual = r_sancho_2
        self._siguiente()

        self.play(FadeOut(actual, shift=UP * 0.2), FadeIn(r_quijote_1, shift=UP * 0.2), run_time=1)
        actual = r_quijote_1
        self._siguiente()

        self.play(FadeOut(actual, shift=UP * 0.2), FadeIn(r_quijote_2, shift=UP * 0.2), run_time=1)
        self._siguiente()

        self.limpiar_pantalla()
