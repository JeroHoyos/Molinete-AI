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


class SlidesFinal:
    def slide_rust_python_bridge(self):
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Rust × Python: el puente PyO3",
            palabra_clave="PyO3",
            color_clave=NARANJA_TERRACOTA,
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(
            titulo, linea,
            fondo=VGroup(llanuras_fondo),
            adornos=adornos
        )

        # ── LAYOUT FIJO ───────────────────────────────────────────────────────────
        BUFF_SUB   = 0.42
        Y_EDITORES = DOWN * 0.55    # ← antes 1.60  (centrado visual)
        Y_TERMINAL = DOWN * 1.90    # ← antes 2.90

        # ── ACTO 1: dependencia en Cargo.toml ─────────────────────────────────────
        sub1 = Text(
            "Declaramos PyO3 como dependencia en Cargo",
            font=FUENTE, font_size=18, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=BUFF_SUB)
        self.play(Write(sub1))

        cargo_src = (
            '[lib]\ncrate-type = ["cdylib", "rlib"]\n\n'
            '[dependencies]\npyo3 = { version = "0.22",\n'
            '  features = ["extension-module"] }'
        )
        editor_cargo = self._hacer_editor(cargo_src, "toml", "Cargo.toml")
        editor_cargo.move_to(Y_EDITORES)

        self.play(FadeIn(editor_cargo, shift=DOWN * 0.18), run_time=0.75)

        # Resaltar la línea de pyo3 — es la clave
        borde_pyo3 = SurroundingRectangle(
            editor_cargo, color=NARANJA_TERRACOTA,
            buff=0.12, corner_radius=0.10, stroke_width=2.5
        )
        annot_cargo = Text(
            "una línea · PyO3 disponible",
            font=FUENTE, font_size=14, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(editor_cargo, UP, buff=0.38)   # ← UP y más separación

        self.play(Create(borde_pyo3), FadeIn(annot_cargo, shift=UP * 0.08))

        # ── PAUSA 1 ───────────────────────────────────────────────────────────────
        self._siguiente()

        self.play(
            FadeOut(sub1), FadeOut(borde_pyo3), FadeOut(annot_cargo),
            run_time=0.40
        )

        # ── ACTO 2: bindings Rust — editor cargo se mueve a la izquierda ──────────
        sub2 = Text(
            "Marcamos structs y métodos con macros · Rust los expone a Python",
            font=FUENTE, font_size=18, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=BUFF_SUB)
        self.play(Write(sub2))

        rust_src = (
            "use pyo3::prelude::*;\n\n"
            "#[pyclass]\n"
            "pub struct TokenizadorBPE { ... }\n\n"
            "#[pymethods]\n"
            "impl TokenizadorBPE {\n"
            "    #[new]\n"
            "    pub fn new(tam_vocab: usize) -> Self { ... }\n"
            "    pub fn codificar(&self, txt: &str) -> Vec<usize> { ... }\n"
            "    pub fn decodificar(&self, ids: Vec<usize>) -> String { ... }\n"
            "}"
        )
        editor_rust = self._hacer_editor(rust_src, "rust", "python_bindings.rs", escala=0.58)

        # Los dos editores se colocan lado a lado centrados
        fila = VGroup(editor_cargo, editor_rust).arrange(RIGHT, buff=0.55)
        if fila.width > 13.0:
            fila.scale_to_fit_width(13.0)
        fila.move_to(Y_EDITORES)

        self.play(
            editor_cargo.animate.move_to(fila[0].get_center()),
            FadeIn(editor_rust, shift=LEFT * 0.18),
            run_time=0.85
        )

        # Flechas de macro → Python encima del editor rust
        macro_labels = VGroup(
            Text("#[pyclass]",   font="Monospace", font_size=13, color=NARANJA_TERRACOTA, weight=BOLD),
            Text("#[pymethods]", font="Monospace", font_size=13, color=NARANJA_TERRACOTA, weight=BOLD),
        ).arrange(RIGHT, buff=0.55).next_to(editor_rust, UP, buff=0.38)   # ← más buff

        self.play(
            LaggedStart(*[FadeIn(m, shift=DOWN * 0.10) for m in macro_labels], lag_ratio=0.3),
            run_time=0.65
        )
        self.play(Indicate(editor_rust, color=NARANJA_TERRACOTA, scale_factor=1.02))

        # ── PAUSA 2 ───────────────────────────────────────────────────────────────
        self._siguiente()

        self.play(
            FadeOut(sub2), FadeOut(macro_labels),
            run_time=0.40
        )

        # ── ACTO 3: maturin compile → .so listo para importar ────────────────────
        sub3 = Text(
            "Un solo comando compila y enlaza · Python ya puede importarlo",
            font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=BUFF_SUB)
        self.play(Write(sub3))

        # Terminal centrada bajo los editores
        term_w, term_h = 6.2, 1.05
        term_bg = RoundedRectangle(
            corner_radius=0.12, width=term_w, height=term_h,
            fill_color=NEGRO_SUAVE, fill_opacity=1,
            stroke_color=HIERRO, stroke_width=2.2,
        )
        term_header = Rectangle(
            width=term_w, height=0.32,
            fill_color=HIERRO, fill_opacity=1, stroke_width=0,
        ).align_to(term_bg, UP)
        term_dots = VGroup(
            Circle(radius=0.055, fill_color=ROJO_MAC,     fill_opacity=1, stroke_width=0),
            Circle(radius=0.055, fill_color=AMARILLO_MAC, fill_opacity=1, stroke_width=0),
            Circle(radius=0.055, fill_color=VERDE_MAC,    fill_opacity=1, stroke_width=0),
        ).arrange(RIGHT, buff=0.12).move_to(term_header.get_left() + RIGHT * 0.38)
        term_title = Text(
            "terminal", font="Monospace", font_size=12, color=ACERO
        ).move_to(term_header)

        prompt  = Text("$ ", font="Monospace", font_size=18, color=VERDE_MAC)
        cmd_txt = Text(
            "maturin develop --release",
            font="Monospace", font_size=18, color=BLANCO
        )
        cmd_txt.set_opacity(0)
        linea_cmd = VGroup(prompt, cmd_txt).arrange(RIGHT, buff=0.0)
        linea_cmd.move_to(term_bg.get_center() + DOWN * 0.05)

        terminal = VGroup(term_bg, term_header, term_dots, term_title, linea_cmd)
        terminal.next_to(fila, DOWN, buff=0.40).set_x(0)

        # Ajuste si se sale de pantalla
        if terminal.get_bottom()[1] < -3.40:
            shift_up = abs(terminal.get_bottom()[1] + 3.40)
            fila.shift(UP * shift_up * 0.5)
            terminal.shift(UP * shift_up * 0.5)

        self.play(
            FadeIn(VGroup(term_bg, term_header, term_dots, term_title, prompt),
                shift=UP * 0.15),
            run_time=0.60
        )
        self.play(
            AddTextLetterByLetter(cmd_txt, time_per_char=0.05),
            run_time=1.10
        )

        # Flash en ambos editores — el puente está completo
        self.play(
            Flash(editor_cargo.get_center(), color=ORO_VIEJO,
                line_length=0.35, num_lines=10),
            Flash(editor_rust.get_center(),  color=ORO_VIEJO,
                line_length=0.35, num_lines=10),
            run_time=0.75
        )

        self._siguiente()
        self.limpiar_pantalla()


    def slide_model_in_action(self):

        titulo, linea = self.crear_titulo(
            "Molinete AI: Demostración en Vivo",
            palabra_clave="Vivo",
            color_clave=NARANJA_TERRACOTA,
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        WIN_W, WIN_H = 12.0, 5.8
        HDR_H, SB_W  = 0.52, 3.1
        SB_H = WIN_H - HDR_H

        app_win = RoundedRectangle(
            corner_radius=0.22, width=WIN_W, height=WIN_H,
            fill_color=FONDO_CAJA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=2)
        app_hdr = Rectangle(
            width=WIN_W, height=HDR_H,
            fill_color=MARRON_OSCURO, fill_opacity=1, stroke_width=0,
        ).align_to(app_win, UP)

        dot_r = Circle(radius=0.08, fill_color=ROJO_MAC,     fill_opacity=1, stroke_width=0)
        dot_a = Circle(radius=0.08, fill_color=AMARILLO_MAC, fill_opacity=1, stroke_width=0)
        dot_v = Circle(radius=0.08, fill_color=VERDE_MAC,    fill_opacity=1, stroke_width=0)
        mac_dots = VGroup(dot_r, dot_a, dot_v).arrange(RIGHT, buff=0.13)
        mac_dots.next_to(app_hdr.get_left(), RIGHT, buff=0.28)
        hdr_lbl = Text("Molinete Chat", font_size=14, color=FONDO_CAJA, weight=BOLD).move_to(app_hdr)

        sb_bg = Rectangle(
            width=SB_W, height=SB_H,
            fill_color=CAJA_INFERIOR, fill_opacity=1, stroke_width=0)
        sb_bg.move_to(app_win.get_corner(UL) + RIGHT*(SB_W/2) + DOWN*(HDR_H + SB_H/2))

        sb_divider = Line(sb_bg.get_corner(UR), sb_bg.get_corner(DR),
                          color=PAPEL_TAN, stroke_width=1.5)

        sb_title = Text("Modelos disponibles", font=FUENTE, font_size=13,
                        color=MARRON_OSCURO, weight=BOLD)
        sb_title.next_to(sb_bg.get_top(), DOWN, buff=0.28)

        MODELOS = [
            ("ChatGPT",     "#10A37F", False),
            ("Claude",      "#CC785C", False),
            ("Gemini",      "#4285F4", False),
            ("Molinete AI", NARANJA_TERRACOTA, True),
        ]
        model_cards = VGroup()
        for name, col, active in MODELOS:
            card_bg = RoundedRectangle(
                corner_radius=0.12, width=SB_W - 0.28, height=0.56,
                fill_color=NARANJA_TERRACOTA if active else FONDO_CAJA,
                fill_opacity=0.18 if active else 0.55,
                stroke_color=col,
                stroke_width=2.5 if active else 0.8)
            dot_m = Circle(radius=0.09, fill_color=col, fill_opacity=1, stroke_width=0)
            lbl_m = Text(name, font=FUENTE, font_size=14,
                         color=MARRON_OSCURO if active else TINTA_NEGRA,
                         weight=BOLD if active else "NORMAL")
            VGroup(dot_m, lbl_m).arrange(RIGHT, buff=0.15).move_to(card_bg)
            model_cards.add(VGroup(card_bg, dot_m, lbl_m))

        model_cards.arrange(DOWN, buff=0.13)
        model_cards.next_to(sb_title, DOWN, buff=0.2)
        model_cards.align_to(sb_bg, LEFT).shift(RIGHT*0.15)

        chat_sub_sep = Line(
            sb_bg.get_corner(UR),
            app_win.get_corner(UR) + DOWN*HDR_H,
            color=PAPEL_TAN, stroke_width=1.5)

        chat_lbl = Text("Chat con Molinete AI", font=FUENTE, font_size=15,
                        color=PAPEL_TAN, weight=BOLD)
        chat_lbl.next_to(chat_sub_sep, DOWN, buff=0.22)
        chat_lbl.set_x((sb_bg.get_right()[0] + app_win.get_right()[0]) / 2)

        INPUT_W = WIN_W - SB_W - 1.0
        input_rect = RoundedRectangle(
            corner_radius=0.18, width=INPUT_W, height=0.54,
            fill_color=BLANCO, fill_opacity=0.92,
            stroke_color=PAPEL_TAN, stroke_width=1.5)
        input_rect.move_to(app_win.get_corner(DR) + LEFT*(INPUT_W/2 + 0.55) + UP*0.46)
        hint_txt = Text("Escribe un mensaje…", font=FUENTE, font_size=12, color=PAPEL_TAN)
        hint_txt.move_to(input_rect).shift(LEFT*1.5)
        send_circle = Circle(radius=0.20, fill_color=NARANJA_TERRACOTA,
                             fill_opacity=1, stroke_width=0)
        send_circle.next_to(input_rect, RIGHT, buff=0.1)
        send_arrow = Text("→", font_size=14, color=BLANCO).move_to(send_circle)

        app_ui = VGroup(
            app_win, app_hdr, mac_dots, hdr_lbl,
            sb_bg, sb_divider, sb_title, model_cards,
            chat_sub_sep, chat_lbl,
            input_rect, hint_txt, send_circle, send_arrow,
        )
        app_ui.next_to(linea, DOWN, buff=0.28)

        self.play(FadeIn(VGroup(app_win, app_hdr, mac_dots, hdr_lbl), shift=UP*0.3), run_time=0.8)
        self.play(FadeIn(VGroup(sb_bg, sb_divider, sb_title)), run_time=0.4)
        for card in model_cards:
            self.play(FadeIn(card, shift=RIGHT*0.1), run_time=0.2)
        self._siguiente()

        self.play(Indicate(model_cards[-1][0], scale_factor=1.09, color=NARANJA_TERRACOTA), run_time=0.7)
        self.play(
            FadeIn(VGroup(chat_sub_sep, chat_lbl), shift=DOWN*0.1),
            FadeIn(VGroup(input_rect, hint_txt, send_circle, send_arrow), shift=UP*0.1),
            run_time=0.6,
        )
        self._siguiente()

        sb_right_x  = sb_bg.get_right()[0]
        win_right_x = app_win.get_right()[0]
        sep_y       = chat_sub_sep.get_start()[1]

        burbuja_u = self._crear_burbuja_transformer(
            "En un lugar de la Mancha,", es_usuario=True)
        burbuja_ia = self._crear_burbuja_transformer(
            "de cuyo nombre no quiero\nacordarme, no ha mucho...", es_usuario=False)
        burbuja_u.scale(0.80)
        burbuja_ia.scale(0.80)

        burbuja_u.move_to([
            win_right_x - burbuja_u.width/2 - 0.30,
            sep_y - burbuja_u.height/2 - 0.55,
            0,
        ])
        burbuja_ia.move_to([
            sb_right_x + burbuja_ia.width/2 + 0.30,
            burbuja_u.get_bottom()[1] - burbuja_ia.height/2 - 0.35,
            0,
        ])

        self.play(FadeIn(burbuja_u, shift=UP*0.15), run_time=0.5)
        self.wait(0.4)
        self.play(FadeIn(burbuja_ia, shift=UP*0.15), run_time=0.5)
        self.wait(0.8)
        self._siguiente()

        transicion_lbl = Text("Cambiando al Demo", font_size=24,
                              color=NARANJA_TERRACOTA, weight=BOLD)
        caja_transicion = SurroundingRectangle(
            transicion_lbl, color=MARRON_OSCURO, fill_color=FONDO_CAJA,
            fill_opacity=1, buff=0.4)
        grupo_transicion = VGroup(caja_transicion, transicion_lbl).move_to(app_win.get_center())

        fade_group = VGroup(
            sb_bg, sb_divider, sb_title, model_cards,
            chat_sub_sep, chat_lbl,
            input_rect, hint_txt, send_circle, send_arrow,
            burbuja_u, burbuja_ia,
        )
        self.play(fade_group.animate.set_opacity(0.1), FadeIn(grupo_transicion, scale=0.8))
        self._siguiente()
        self.limpiar_pantalla()


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
