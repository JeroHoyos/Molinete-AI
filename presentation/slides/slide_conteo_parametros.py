import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import math
import os
from colores import *
from objetos import *


class SlideConteoParametros:
    def slide_conteo_parametros(self):

        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))

        titulo, linea = self.crear_titulo(
            "Conteo de Parámetros",
            palabra_clave="Parámetros",
            color_clave=NARANJA_TERRACOTA,
        )
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

        # ── Acto 1: las arquitecturas del taller y de donde salen ───────
        # sus parametros, una por una
        def _mini_matriz(filas, cols, lado, color):
            g = VGroup()
            paso = lado + 0.03
            for f in range(filas):
                for c in range(cols):
                    g.add(Square(side_length=lado, fill_color=color,
                                 fill_opacity=0.9, stroke_color=MARRON_OSCURO,
                                 stroke_width=0.8)
                          .move_to([c * paso, -f * paso, 0]))
            return g

        def _tarjeta(nombre, visual, formula_tex, color_borde):
            lbl = Text(nombre, font=FUENTE, font_size=13, weight=BOLD,
                       color=TINTA_NEGRA)
            frm = MathTex(formula_tex, font_size=28, color=MARRON_OSCURO)
            fila = VGroup(lbl, visual, frm).arrange(RIGHT, buff=0.22)
            bg = RoundedRectangle(
                corner_radius=0.12,
                width=max(fila.width + 0.4, 3.0),
                height=fila.height + 0.32,
                fill_color=FONDO_CAJA, fill_opacity=1,
                stroke_color=color_borde, stroke_width=2.2,
            )
            fila.move_to(bg)
            return VGroup(bg, fila)

        # embeddings: la tabla de V filas por d columnas
        caja_emb = _tarjeta(
            "Embeddings", _mini_matriz(3, 5, 0.12, CELESTE_PALIDO),
            r"V \cdot d", ACERO,
        ).move_to(LEFT * 4.0 + UP * 1.45)

        # atencion: las cuatro matrices d x d (Q, K, V y salida)
        att_visual = VGroup(*[
            Square(side_length=0.26, fill_color=SALMON_CLARO, fill_opacity=0.95,
                   stroke_color=MARRON_OSCURO, stroke_width=1.2)
            for _ in range(4)
        ]).arrange(RIGHT, buff=0.07)
        lbls_qkvo = VGroup(*[
            MathTex(s, font_size=16, color=MARRON_OSCURO).move_to(sq)
            for s, sq in zip(["Q", "K", "V", "O"], att_visual)
        ])
        caja_att = _tarjeta("Atención", VGroup(att_visual, lbls_qkvo),
                            r"4 \cdot d^{2}", LADRILLO_VIVO)

        # mlp: las dos matrices anchas (d x 4d y 4d x d)
        mlp_visual = VGroup(*[
            Rectangle(width=0.62, height=0.17, fill_color=MENTA_PALIDA,
                      fill_opacity=0.95, stroke_color=MARRON_OSCURO,
                      stroke_width=1.2)
            for _ in range(2)
        ]).arrange(DOWN, buff=0.07)
        caja_mlp = _tarjeta("MLP", mlp_visual, r"8 \cdot d^{2}", VERDE_OLIVA)

        pila_bloque = VGroup(caja_att, caja_mlp).arrange(DOWN, buff=0.16)
        marco_bloque = RoundedRectangle(
            corner_radius=0.14,
            width=pila_bloque.width + 0.5, height=pila_bloque.height + 0.44,
            fill_color=FONDO_CAJA, fill_opacity=0.7,
            stroke_color=NARANJA_TERRACOTA, stroke_width=2.4,
        ).move_to(LEFT * 4.0 + DOWN * 1.05)
        pila_bloque.move_to(marco_bloque)

        # el x L hecho pila: copias del marco asomando detras
        sombras_bloque = VGroup(*[
            marco_bloque.copy()
            .set_fill(FONDO_CAJA, opacity=0.55)
            .set_stroke(NARANJA_TERRACOTA, 1.8, opacity=0.3 + 0.2 * (1 - k))
            .shift((RIGHT * 0.15 + DOWN * 0.13) * (k + 1))
            for k in (1, 0)
        ])
        bloque_stack = VGroup(sombras_bloque, marco_bloque, pila_bloque)
        etiq_xL = MathTex(r"\times L", font_size=38, color=NARANJA_TERRACOTA)\
            .next_to(sombras_bloque, RIGHT, buff=0.18)
        flecha_stack = Arrow(caja_emb.get_bottom(), marco_bloque.get_top(),
                             buff=0.12, color=MARRON_OSCURO, stroke_width=3.2)

        formula = MathTex(
            "P", r"\approx", r"V \cdot d", "+", r"12 \cdot L \cdot d^{2}",
            font_size=46, color=TINTA_NEGRA,
        ).move_to(RIGHT * 2.4 + UP * 1.15)
        formula[2].set_color(ACERO)
        formula[4].set_color(NARANJA_TERRACOTA)

        self.play(FadeIn(caja_emb, shift=DOWN * 0.15), run_time=0.6)
        self.play(
            GrowArrow(flecha_stack),
            FadeIn(sombras_bloque, shift=(RIGHT + DOWN) * 0.08),
            DrawBorderThenFill(marco_bloque),
            FadeIn(pila_bloque),
            FadeIn(etiq_xL, scale=0.6),
            run_time=0.9,
        )
        self.play(Write(formula), run_time=1.0)
        self.play(
            Indicate(caja_emb, color=ACERO, scale_factor=1.05),
            Indicate(formula[2], color=ACERO, scale_factor=1.15),
            run_time=0.7,
        )
        self.play(
            Indicate(bloque_stack, color=NARANJA_TERRACOTA, scale_factor=1.03),
            Indicate(formula[4], color=NARANJA_TERRACOTA, scale_factor=1.15),
            run_time=0.7,
        )
        self._siguiente()

        datos_arq = [
            ("Diminuto",    64,  2,  r"\approx 50\,\mathrm{K}",  "≈ 50 K"),
            ("Pequeño",     128, 3,  r"\approx 200\,\mathrm{K}", "≈ 200 K"),
            ("Mediano",     256, 4,  r"\approx 4\,\mathrm{M}",   "≈ 4 M"),
            ("GPT-2 Small", 768, 12, r"\approx 163\,\mathrm{M}", "≈ 163 M"),
        ]

        chips_arq = VGroup()
        for nombre, _, _, _, _ in datos_arq:
            bg = RoundedRectangle(corner_radius=0.1, width=2.3, height=0.62,
                                  fill_color=FONDO_CAJA, fill_opacity=1,
                                  stroke_color=CAJA_INFERIOR, stroke_width=1.8)
            t = Text(nombre, font=FUENTE, font_size=15, weight=BOLD,
                     color=MARRON_OSCURO).move_to(bg.get_center() + UP * 0.12)
            chips_arq.add(VGroup(bg, t))
        chips_arq.arrange(RIGHT, buff=0.3).to_edge(DOWN, buff=0.45)

        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in chips_arq],
                              lag_ratio=0.1), run_time=0.8)

        config_txt, total_txt = None, None
        minis_tot = VGroup()
        for k, (nombre, d_v, L_v, tot_tex, tot_plano) in enumerate(datos_arq):
            chip = chips_arq[k]
            nuevo_cfg = MathTex(
                rf"d = {d_v}", r"\qquad", rf"L = {L_v}",
                font_size=38, color=TINTA_NEGRA,
            ).move_to(RIGHT * 2.4 + DOWN * 0.05)
            nuevo_cfg[2].set_color(NARANJA_TERRACOTA)
            nuevo_tot = MathTex(tot_tex, font_size=58, color=NARANJA_TERRACOTA)\
                .move_to(RIGHT * 2.4 + DOWN * 1.35)
            mini_tot = Text(tot_plano, font="Monospace", font_size=13, weight=BOLD,
                            color=NARANJA_TERRACOTA)\
                .move_to(chip[0].get_center() + DOWN * 0.14)
            minis_tot.add(mini_tot)

            anims_chip = [chip[0].animate.set_stroke(NARANJA_TERRACOTA, 2.6)]
            if k > 0:
                anims_chip.append(
                    chips_arq[k - 1][0].animate.set_stroke(CAJA_INFERIOR, 1.8))

            if config_txt is None:
                config_txt, total_txt = nuevo_cfg, nuevo_tot
                self.play(*anims_chip, Write(config_txt), run_time=0.6)
                self.play(Write(total_txt), run_time=0.5)
            else:
                self.play(*anims_chip, Transform(config_txt, nuevo_cfg),
                          run_time=0.6)
                self.play(Transform(total_txt, nuevo_tot), run_time=0.5)
            self.play(FadeIn(mini_tot, shift=UP * 0.08), run_time=0.3)
            self._siguiente()

        self.play(
            FadeOut(VGroup(caja_emb, flecha_stack, bloque_stack,
                           etiq_xL, formula, config_txt, total_txt,
                           chips_arq, minis_tot)),
            run_time=0.7,
        )

        # ── Ejes en escala logarítmica (y = log10 de parámetros) ────────
        ax = Axes(
            x_range=[0, 7, 1],
            y_range=[4, 13, 1],
            x_length=10.6,
            y_length=4.6,
            axis_config={"include_tip": False, "include_ticks": False,
                         "color": MARRON_OSCURO},
        ).move_to(DOWN * 0.55)

        grid = VGroup()
        ticks = VGroup()
        for yv in [4, 6, 8, 10, 12]:
            if yv > 4:
                grid.add(DashedLine(
                    ax.c2p(0, yv), ax.c2p(7, yv),
                    stroke_color=BEIGE_MEDIO, stroke_width=1.4,
                    dash_length=0.08, stroke_opacity=0.85,
                ))
            ticks.add(MathTex(r"10^{%d}" % yv, font_size=24, color=MARRON_OSCURO)
                      .next_to(ax.c2p(0, yv), LEFT, buff=0.18))

        self.play(Create(ax), run_time=0.8)
        self.play(FadeIn(grid),
                  LaggedStart(*[FadeIn(t) for t in ticks], lag_ratio=0.1),
                  run_time=0.8)

        # ── Datos: los cuatro tamaños del taller web + los gigantes ─────
        # (nombre, config, parámetros, etiqueta, color)
        niveles = [
            ("Diminuto",    "emb 64 · 2 capas",   5.0e4,  "50 K",  CREMA_CALIDA),
            ("Pequeño",     "emb 128 · 3 capas",  2.0e5,  "200 K", CAJA_INFERIOR),
            ("Mediano",     "emb 256 · 4 capas",  4.0e6,  "4 M",   PAPEL_TAN),
            ("GPT-2 Small", "emb 768 · 12 capas", 1.63e8, "163 M", NARANJA_TERRACOTA),
        ]
        gigantes = [
            ("GPT-2 XL", "OpenAI · 2019", 1.5e9,  "1 500 M",       MARRON_QUIJOTE),
            ("GPT-5.6",  "OpenAI · hoy",  2.0e12, "≈ 2 billones*", MARRON_OSCURO),
        ]

        ANCHO_BARRA = 1.0

        def crear_barra(idx, nombre, config_txt, valor, etiqueta, color):
            base = ax.c2p(idx, 4)
            tope = ax.c2p(idx, math.log10(valor))
            barra = RoundedRectangle(
                corner_radius=0.05, width=ANCHO_BARRA, height=tope[1] - base[1],
                fill_color=color, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=1.8,
            ).move_to(base, aligned_edge=DOWN)
            val = Text(etiqueta, font="Monospace", font_size=15, weight=BOLD,
                       color=MARRON_OSCURO).next_to(barra, UP, buff=0.09)
            nom = Text(nombre, font=FUENTE, font_size=16, weight=BOLD,
                       color=TINTA_NEGRA).next_to(barra, DOWN, buff=0.14)
            cfg = Text(config_txt, font=FUENTE, font_size=11,
                       color=ACERO).next_to(nom, DOWN, buff=0.06)
            if cfg.width > 1.35:
                cfg.scale(1.35 / cfg.width)
            return barra, val, VGroup(nom, cfg)

        def chip_mult(texto, barra_izq, barra_der):
            t = Text(texto, font="Monospace", font_size=14, weight=BOLD,
                     color=NARANJA_TERRACOTA)
            bg = RoundedRectangle(
                corner_radius=0.1, width=t.width + 0.26, height=t.height + 0.18,
                fill_color=FONDO_CAJA, fill_opacity=0.95,
                stroke_color=NARANJA_TERRACOTA, stroke_width=1.5,
            )
            t.move_to(bg)
            # en el hueco entre barras, justo sobre el tope de la barra baja,
            # para no pisar la etiqueta de valor de la barra alta
            x_medio = (barra_izq.get_top()[0] + barra_der.get_top()[0]) / 2
            y_chip = barra_izq.get_top()[1] + 0.35
            return VGroup(bg, t).move_to(np.array([x_medio, y_chip, 0]))

        # ── Acto 2: la escalera del taller (50K → 163M) ──────────────────
        barras_niveles = [crear_barra(i + 1, *datos) for i, datos in enumerate(niveles)]

        for barra, val, pie in barras_niveles:
            self.play(GrowFromEdge(barra, DOWN),
                      FadeIn(pie, shift=UP * 0.05), run_time=0.55)
            self.play(FadeIn(val, shift=UP * 0.1), run_time=0.3)

        chips = VGroup(
            chip_mult("×4",  barras_niveles[0][0], barras_niveles[1][0]),
            chip_mult("×20", barras_niveles[1][0], barras_niveles[2][0]),
            chip_mult("×40", barras_niveles[2][0], barras_niveles[3][0]),
        )
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in chips],
                              lag_ratio=0.25), run_time=1.0)
        self._siguiente()

        # ── Acto 3: el GPT-2 completo de OpenAI ──────────────────────────
        barra_xl, val_xl, pie_xl = crear_barra(5, *gigantes[0])
        chip_xl = chip_mult("×9", barras_niveles[3][0], barra_xl)
        self.play(GrowFromEdge(barra_xl, DOWN),
                  FadeIn(pie_xl, shift=UP * 0.05), run_time=0.7)
        self.play(FadeIn(val_xl, shift=UP * 0.1),
                  FadeIn(chip_xl, scale=0.6), run_time=0.5)
        self._siguiente()

        # ── Acto 4: GPT-5.6, otra escala ─────────────────────────────────
        barra_56, val_56, pie_56 = crear_barra(6, *gigantes[1])
        chip_56 = chip_mult("×1 300", barra_xl, barra_56)
        nota = Text(
            "* Cifra estimada: OpenAI no publica el número de parámetros",
            font=FUENTE, font_size=13, color=ACERO,
        ).to_edge(DOWN, buff=0.15)

        self.play(GrowFromEdge(barra_56, DOWN),
                  FadeIn(pie_56, shift=UP * 0.05), run_time=1.5)
        self.play(FadeIn(val_56, shift=UP * 0.1),
                  FadeIn(chip_56, scale=0.6),
                  FadeIn(nota), run_time=0.6)

        self.play(
            Indicate(barra_xl, color=NARANJA_TERRACOTA, scale_factor=1.06),
            Indicate(barra_56, color=NARANJA_TERRACOTA, scale_factor=1.03),
            run_time=1.0,
        )
        self._siguiente()

        adornos[1].clear_updaters()
        self.limpiar_pantalla()
