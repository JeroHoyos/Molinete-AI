import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *
from objetos import *


class SlideLayerNormalization:
    def slide_layer_normalization(self):


        def crear_cajita(texto, bg_color, borde_color=MARRON_OSCURO, w=2.6, h=0.7, tam_fuente=20):
            caja = RoundedRectangle(corner_radius=0.1, width=w, height=h,
                                    fill_color=bg_color, fill_opacity=1,
                                    stroke_color=borde_color, stroke_width=2)
            lbl = Text(texto, font_size=tam_fuente, color=TINTA_NEGRA).move_to(caja.get_center())
            return VGroup(caja, lbl)

        def crear_vector_visual(numeros, bg_color, borde_color=MARRON_OSCURO):
            return VGroup(*[
                crear_cajita(num, bg_color, borde_color, w=1.6, h=0.7, tam_fuente=18)
                for num in numeros
            ]).arrange(DOWN, buff=0.1)


        titulo, linea = self.crear_titulo(
            "Layer Normalization",
            palabra_clave="Normalization",
            color_clave=NARANJA_TERRACOTA,
        )
        adornos = self._crear_adornos_esquinas()
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)


        inestables = ["8459.1", "-7302.4", "0.00001", "5120.9", "-9999.9"]
        valores    = [float(v) for v in inestables]
        mu, sigma  = float(np.mean(valores)), float(np.std(valores))
        estables   = [f"{(v - mu) / sigma:.2f}" for v in valores]

        vec_inestable = crear_vector_visual(
            inestables, bg_color=SALMON_CLARO, borde_color=NARANJA_TERRACOTA
        ).move_to([-3.9, -0.55, 0])
        vec_estable = crear_vector_visual(
            estables, bg_color=CREMA_CALIDA, borde_color=MARRON_OSCURO
        ).move_to([3.9, -0.55, 0])

        lbl_antes = Text("Sin norma", font=FUENTE, font_size=20,
                         color=NARANJA_TERRACOTA, weight=BOLD
                         ).next_to(vec_inestable, UP, buff=0.3)
        lbl_despues = Text("Con LayerNorm", font=FUENTE, font_size=20,
                           color=MARRON_OSCURO, weight=BOLD
                           ).next_to(vec_estable, UP, buff=0.3)

        stats_antes = Text(
            f"μ ≈ {mu:.0f}   σ ≈ {sigma:.0f}",
            font=FUENTE, font_size=17, color=NARANJA_TERRACOTA
        ).next_to(vec_inestable, DOWN, buff=0.3)
        stats_despues = Text(
            "μ = 0   σ = 1",
            font=FUENTE, font_size=17, color=MARRON_OSCURO, weight=BOLD
        ).next_to(vec_estable, DOWN, buff=0.3)

        # ── Paso 1: el vector problemático ─────────────────────────────
        self.play(
            FadeIn(lbl_antes, shift=UP * 0.1),
            LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in vec_inestable],
                        lag_ratio=0.1),
            run_time=1.0
        )
        self.play(
            Indicate(vec_inestable, color=NARANJA_TERRACOTA, scale_factor=1.04),
            FadeIn(stats_antes, shift=UP * 0.1)
        )
        self._siguiente()

        # ── Paso 2: la máquina LayerNorm en el centro ──────────────────
        maquina_caja = RoundedRectangle(
            corner_radius=0.16, width=2.5, height=1.4,
            fill_color=FONDO_CAJA, fill_opacity=1,
            stroke_color=NARANJA_TERRACOTA, stroke_width=2.5,
        )
        maquina_lbl = Text("LayerNorm", font=FUENTE, font_size=20,
                           color=NARANJA_TERRACOTA, weight=BOLD)
        maquina_formula = MathTex(r"\frac{x - \mu}{\sigma}", font_size=30,
                                  color=MARRON_OSCURO)
        contenido_maquina = VGroup(maquina_lbl, maquina_formula).arrange(DOWN, buff=0.15)
        maquina = VGroup(maquina_caja, contenido_maquina.move_to(maquina_caja))
        maquina.move_to([0, -0.55, 0])

        flecha_in = Arrow(vec_inestable.get_right(), maquina.get_left(),
                          color=NARANJA_TERRACOTA, stroke_width=4, buff=0.2)
        flecha_out = Arrow(maquina.get_right(), vec_estable.get_left(),
                           color=NARANJA_TERRACOTA, stroke_width=4, buff=0.2)

        self.play(FadeIn(maquina, scale=0.92), GrowArrow(flecha_in), run_time=0.6)
        self.play(GrowArrow(flecha_out), FadeIn(lbl_despues, shift=UP * 0.1),
                  run_time=0.5)

        # ── Paso 3: los valores pasan de uno en uno por la máquina ─────
        for i in range(len(valores)):
            copia   = vec_inestable[i].copy()
            destino = vec_estable[i]
            destino.generate_target()
            destino.scale(0.45).move_to(maquina.get_center()).fade(1)

            if i < 2:
                # los dos primeros, con calma: entra, se transforma, sale
                self.play(
                    copia.animate.scale(0.5).move_to(maquina.get_center()),
                    run_time=0.35,
                )
                self.play(
                    FadeOut(copia, scale=0.3),
                    Indicate(maquina_caja, color=NARANJA_TERRACOTA, scale_factor=1.05),
                    MoveToTarget(destino),
                    run_time=0.45,
                )
            else:
                # el resto, de corrido
                self.play(
                    copia.animate.scale(0.5).move_to(maquina.get_center()).fade(1),
                    MoveToTarget(destino),
                    run_time=0.4,
                )
                self.remove(copia)

        # ── Paso 4: el resultado, domesticado ──────────────────────────
        self.play(FadeIn(stats_despues, shift=UP * 0.1))
        self._siguiente()

        self.play(
            FadeOut(vec_inestable), FadeOut(vec_estable),
            FadeOut(lbl_antes), FadeOut(lbl_despues),
            FadeOut(maquina), FadeOut(flecha_in), FadeOut(flecha_out),
            FadeOut(stats_antes), FadeOut(stats_despues),
            run_time=0.7
        )

        # ── Por qué importa: la precisión numérica tiene límites ───────
        def gauss(mu_g, sigma_g):
            return lambda x: np.exp(-0.5 * ((x - mu_g) / sigma_g) ** 2)

        def crear_panel(titulo_str, color_titulo, x_centro):
            ejes_p = Axes(
                x_range=[-6, 6, 2], y_range=[0, 1.25, 0.5],
                x_length=6.2, y_length=3.4,
                axis_config={"color": MARRON_OSCURO, "include_numbers": False,
                             "stroke_width": 2},
                tips=False,
            ).move_to([x_centro, -0.45, 0])
            tit_p = Text(titulo_str, font=FUENTE, font_size=20, weight=BOLD,
                         color=color_titulo).next_to(ejes_p, UP, buff=0.3)
            return ejes_p, tit_p

        ejes_sin, tit_sin = crear_panel("Sin LayerNorm", ROJO_TOMATE, -3.45)
        ejes_con, tit_con = crear_panel("Con LayerNorm", VERDE_OLIVA, 3.45)

        curva_sin = ejes_sin.plot(gauss(0.0, 0.9), color=NARANJA_TERRACOTA, stroke_width=4)
        curva_con = ejes_con.plot(gauss(0.0, 0.9), color=VERDE_OLIVA, stroke_width=4)

        badge_bg = RoundedRectangle(
            corner_radius=0.12, width=2.0, height=0.55,
            fill_color=MARRON_OSCURO, fill_opacity=0.9, stroke_width=0,
        ).move_to([0, -3.25, 0])
        badge_txt = Text("capa 1 / 4", font=FUENTE, font_size=18,
                         color=PAPEL_CREMA, weight=BOLD).move_to(badge_bg)

        self.play(
            Create(ejes_sin), Create(ejes_con),
            FadeIn(tit_sin), FadeIn(tit_con),
            FadeIn(badge_bg), FadeIn(badge_txt),
            run_time=1.0,
        )
        self.play(Create(curva_sin), Create(curva_con), run_time=0.7)

        # capa tras capa: sin norma la distribución deriva y se ensancha,
        # con norma siempre vuelve a media 0 y desviación 1
        capas_sin  = [(0.0, 0.9), (0.7, 1.7), (1.5, 2.9), (2.6, 4.4)]
        sigmas_con = [0.9, 1.06, 0.94, 1.0]
        for kcapa in range(1, 4):
            mu_s, sg_s = capas_sin[kcapa]
            nueva_sin = ejes_sin.plot(gauss(mu_s, sg_s),
                                      color=NARANJA_TERRACOTA, stroke_width=4)
            nueva_con = ejes_con.plot(gauss(0.0, sigmas_con[kcapa]),
                                      color=VERDE_OLIVA, stroke_width=4)
            nuevo_badge = Text(f"capa {kcapa + 1} / 4", font=FUENTE, font_size=18,
                               color=PAPEL_CREMA, weight=BOLD).move_to(badge_bg)
            self.play(
                Transform(curva_sin, nueva_sin),
                Transform(curva_con, nueva_con),
                ReplacementTransform(badge_txt, nuevo_badge),
                run_time=0.75,
            )
            badge_txt = nuevo_badge

        nota_nan = Text("las colas se salen: inf / NaN", font=FUENTE, font_size=16,
                        color=ROJO_TOMATE, weight=BOLD).next_to(ejes_sin, DOWN, buff=0.18)
        nota_ok = Text("siempre μ = 0, σ = 1", font=FUENTE, font_size=16,
                       color=VERDE_OLIVA, weight=BOLD).next_to(ejes_con, DOWN, buff=0.18)
        self.play(
            Indicate(curva_sin, color=ROJO_TOMATE, scale_factor=1.03),
            FadeIn(nota_nan, shift=UP * 0.1),
            FadeIn(nota_ok, shift=UP * 0.1),
        )
        self._siguiente()

        self.play(*[FadeOut(m) for m in [
            ejes_sin, ejes_con, tit_sin, tit_con,
            curva_sin, curva_con, badge_bg, badge_txt, nota_nan, nota_ok,
        ]], run_time=0.7)


        formula = MathTex(
            r"\text{output} = \frac{\text{input} - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta",
            substrings_to_isolate=[r"\mu", r"\sigma^2", r"\epsilon", r"\times \gamma + \beta"],
            color=TINTA_NEGRA
        ).scale(1.2).move_to(DOWN * 0.5)

        self.play(FadeIn(formula))

        parte_mu    = formula.get_part_by_tex(r"\mu")
        parte_sigma = formula.get_part_by_tex(r"\sigma^2")
        caja_mu     = SurroundingRectangle(parte_mu, color=VERDE_OLIVA, buff=0.05)
        caja_sigma  = SurroundingRectangle(parte_sigma, color=VERDE_OLIVA, buff=0.05)
        nota_stats  = Text("μ y σ se calculan sobre cada vector", font_size=18,
                           color=VERDE_OLIVA).next_to(formula, DOWN, buff=0.5)

        self.play(Create(caja_mu), Create(caja_sigma), FadeIn(nota_stats, shift=UP))
        self.play(FadeOut(caja_mu), FadeOut(caja_sigma), FadeOut(nota_stats))

        parte_eps = formula.get_part_by_tex(r"\epsilon")
        caja_eps  = SurroundingRectangle(parte_eps, color=NARANJA_TERRACOTA, buff=0.05)
        nota_eps  = Text("ε previene división por cero", font_size=18,
                          color=NARANJA_TERRACOTA).next_to(caja_eps, DOWN, buff=0.5)

        self.play(Create(caja_eps), FadeIn(nota_eps, shift=UP))
        self.play(FadeOut(caja_eps), FadeOut(nota_eps))

        parte_params = formula.get_part_by_tex(r"\times \gamma + \beta")
        caja_params  = SurroundingRectangle(parte_params, color=MARRON_OSCURO, buff=0.1)
        nota_params  = Text("γ, β: parámetros aprendibles", font_size=18,
                             color=MARRON_OSCURO).next_to(caja_params, DOWN, buff=0.5)

        self.play(Create(caja_params), FadeIn(nota_params, shift=UP))
        self._siguiente()

        self.play(
            *[FadeOut(m) for m in [formula, caja_params, nota_params]]
        )

        self.limpiar_pantalla()


