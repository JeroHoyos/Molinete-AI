import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


class SlideRustPythonBridge:
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


        BUFF_SUB   = 0.42
        Y_EDITORES = DOWN * 0.55
        Y_TERMINAL = DOWN * 1.90


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


        borde_pyo3 = SurroundingRectangle(
            editor_cargo, color=NARANJA_TERRACOTA,
            buff=0.12, corner_radius=0.10, stroke_width=2.5
        )
        annot_cargo = Text(
            "una línea · PyO3 disponible",
            font=FUENTE, font_size=14, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(editor_cargo, UP, buff=0.38)

        self.play(Create(borde_pyo3), FadeIn(annot_cargo, shift=UP * 0.08))


        self._siguiente()

        self.play(
            FadeOut(sub1), FadeOut(borde_pyo3), FadeOut(annot_cargo),
            run_time=0.40
        )


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


        fila = VGroup(editor_cargo, editor_rust).arrange(RIGHT, buff=0.55)
        if fila.width > 13.0:
            fila.scale_to_fit_width(13.0)
        fila.move_to(Y_EDITORES)

        self.play(
            editor_cargo.animate.move_to(fila[0].get_center()),
            FadeIn(editor_rust, shift=LEFT * 0.18),
            run_time=0.85
        )


        macro_labels = VGroup(
            Text("#[pyclass]",   font="Monospace", font_size=13, color=NARANJA_TERRACOTA, weight=BOLD),
            Text("#[pymethods]", font="Monospace", font_size=13, color=NARANJA_TERRACOTA, weight=BOLD),
        ).arrange(RIGHT, buff=0.55).next_to(editor_rust, UP, buff=0.38)

        self.play(
            LaggedStart(*[FadeIn(m, shift=DOWN * 0.10) for m in macro_labels], lag_ratio=0.3),
            run_time=0.65
        )
        self.play(Indicate(editor_rust, color=NARANJA_TERRACOTA, scale_factor=1.02))


        self._siguiente()

        self.play(
            FadeOut(sub2), FadeOut(macro_labels),
            run_time=0.40
        )


        sub3 = Text(
            "Un solo comando compila y enlaza · Python ya puede importarlo",
            font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=BUFF_SUB)
        self.play(Write(sub3))


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


        self.play(
            Flash(editor_cargo.get_center(), color=ORO_VIEJO,
                line_length=0.35, num_lines=10),
            Flash(editor_rust.get_center(),  color=ORO_VIEJO,
                line_length=0.35, num_lines=10),
            run_time=0.75
        )

        self._siguiente()
        self.limpiar_pantalla()


