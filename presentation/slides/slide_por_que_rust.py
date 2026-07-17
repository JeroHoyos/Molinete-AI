import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlidePorQueRust:
    def slide_por_que_rust(self):
        titulo, linea = self.crear_titulo(
            "¿Por qué Rust?", palabra_clave="Rust", color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        COLOR_SI   = "#4CAF50"
        COLOR_NO   = ROJO_CONTRA
        COLOR_RUST = NARANJA_TERRACOTA

        ANCHO_COL_CRITERIO = 3.8
        ANCHO_COL_LANG     = 2.5
        ALTO_FILA          = 0.72
        RADIO              = 0.12

        CRITERIOS = [
            "Velocidad de cómputo",
            "Seguridad de memoria",
            "Hilos reales (sin GIL)",
            "Fácil de aprender",
            "Diferenciador en AI",
        ]

        DATOS = [
            (False, True,  True ),
            (False, False, True ),
            (False, True,  True ),
            (True,  False, False),
            (False, False, True ),
        ]

        def _celda_criterio(texto):
            bg = RoundedRectangle(
                corner_radius=RADIO,
                width=ANCHO_COL_CRITERIO, height=ALTO_FILA,
                fill_color=FONDO_CAJA, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=1.5,
            )
            lbl = Text(texto, font=FUENTE, font_size=20, color=TINTA_NEGRA)
            lbl.move_to(bg)
            return VGroup(bg, lbl)

        def _marca_si(color=BLANCO):
            trazo = VMobject(stroke_color=color, stroke_width=5)
            trazo.set_points_as_corners([
                [-0.17, 0.02, 0], [-0.05, -0.13, 0], [0.18, 0.15, 0],
            ])
            return trazo

        def _marca_no(color=BLANCO):
            return VGroup(
                Line(UL * 0.13, DR * 0.13, stroke_color=color, stroke_width=5),
                Line(UR * 0.13, DL * 0.13, stroke_color=color, stroke_width=5),
            )

        def _celda_valor(valor, es_rust=False):
            if valor is True:
                marca, color_bg = _marca_si(), COLOR_SI
            elif valor is False:
                marca, color_bg = _marca_no(), COLOR_NO
            else:
                marca = Text("±", font=FUENTE, font_size=26, color=TINTA_NEGRA, weight=BOLD)
                color_bg = PAPEL_TAN

            bg = RoundedRectangle(
                corner_radius=RADIO,
                width=ANCHO_COL_LANG, height=ALTO_FILA,
                fill_color=color_bg, fill_opacity=1,
                stroke_color=MARRON_OSCURO, stroke_width=1.5,
            )
            marca.move_to(bg)
            return VGroup(bg, marca)

        def _cabecera(texto, es_rust=False):
            color_bg  = COLOR_RUST if es_rust else MARRON_OSCURO
            color_txt = BLANCO
            bg = RoundedRectangle(
                corner_radius=RADIO,
                width=ANCHO_COL_LANG if texto != "" else ANCHO_COL_CRITERIO,
                height=ALTO_FILA * 0.9,
                fill_color=color_bg, fill_opacity=1,
                stroke_color=color_bg, stroke_width=0,
            )
            lbl = Text(texto, font=FUENTE, font_size=22, color=color_txt, weight=BOLD)
            lbl.move_to(bg)
            return VGroup(bg, lbl)

        cab_criterio = _cabecera("")
        cab_python   = _cabecera("Python")
        cab_cpp      = _cabecera("C++")
        cab_rust     = _cabecera("Rust", es_rust=True)

        fila_cab = VGroup(cab_criterio, cab_python, cab_cpp, cab_rust)
        fila_cab.arrange(RIGHT, buff=0.1)

        filas_datos = []

        for i, (criterio, (py, cpp, rs)) in enumerate(zip(CRITERIOS, DATOS)):
            c_crit = _celda_criterio(criterio)
            c_py   = _celda_valor(py)
            c_cpp  = _celda_valor(cpp)
            c_rs   = _celda_valor(rs, es_rust=True)

            fila = VGroup(c_crit, c_py, c_cpp, c_rs)
            fila.arrange(RIGHT, buff=0.12)
            filas_datos.append(fila)

        tabla_completa = VGroup(fila_cab, *filas_datos)
        tabla_completa.arrange(DOWN, buff=0.10)
        tabla_completa.next_to(linea, DOWN, buff=0.35)
        tabla_completa.to_edge(LEFT, buff=0.7)

        logo_py  = ImageMobject(os.path.join("assets", "logo_python.png")).set_height(1.1)
        logo_cpp = ImageMobject(os.path.join("assets", "logo_cpp.png")).set_height(1.1)
        logo_rs  = ImageMobject(os.path.join("assets", "logo_rust.png")).set_height(1.3)

        logos = Group(logo_py, logo_cpp, logo_rs).arrange(DOWN, buff=0.55)
        logos.next_to(tabla_completa, RIGHT, buff=0.6)
        logos.set_y(tabla_completa.get_center()[1])

        self.play(
            FadeIn(fila_cab, shift=DOWN * 0.15),
            FadeIn(logo_py,  shift=LEFT * 0.2),
            FadeIn(logo_cpp, shift=LEFT * 0.2),
            FadeIn(logo_rs,  shift=LEFT * 0.2),
            run_time=0.9,
        )

        for fila in filas_datos:
            self.play(FadeIn(fila, shift=RIGHT * 0.2), run_time=0.55)

        columna_rust = VGroup(cab_rust, *[fila[3] for fila in filas_datos])
        marco_rust = SurroundingRectangle(
            columna_rust, color=NARANJA_TERRACOTA,
            corner_radius=0.15, buff=0.09, stroke_width=3.5,
        )
        self.play(Create(marco_rust), run_time=0.7)
        self._siguiente()


        self.play(
            FadeOut(tabla_completa),
            FadeOut(marco_rust),
            FadeOut(Group(logo_py, logo_cpp, logo_rs)),
            run_time=0.6,
        )

        ANCHO_PANEL = 7.0
        ALTO_PANEL  = 1.6


        py_bg = RoundedRectangle(
            corner_radius=0.18, width=ANCHO_PANEL, height=ALTO_PANEL,
            fill_color=FONDO_CAJA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=2,
        ).set_x(0).next_to(linea, DOWN, buff=0.7)

        adornos = self._crear_adornos_esquinas(escala=0.6)

        py_logo = ImageMobject(os.path.join("assets", "logo_python.png")).set_height(0.9)
        py_lbl = VGroup(
            Text("Interfaz", font=FUENTE, font_size=28,
                 color=MARRON_OSCURO, weight=BOLD),
            Text("web, scripts y experimentos", font=FUENTE, font_size=17,
                 color=TINTA_NEGRA),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        Group(py_logo, py_lbl).arrange(RIGHT, buff=0.4).move_to(py_bg)


        flecha = Arrow(
            py_bg.get_bottom(), py_bg.get_bottom() + DOWN * 0.7,
            color=NARANJA_TERRACOTA, stroke_width=4,
            max_tip_length_to_length_ratio=0.45, buff=0,
        ).set_x(0)

        chip_bg = RoundedRectangle(
            corner_radius=0.1, width=1.15, height=0.5,
            fill_color=FONDO_CAJA, fill_opacity=1,
            stroke_color=NARANJA_TERRACOTA, stroke_width=2,
        )
        chip_txt = Text("PyO3", font=FUENTE, font_size=18,
                        color=MARRON_OSCURO, weight=BOLD).move_to(chip_bg)
        chip_pyo3 = VGroup(chip_bg, chip_txt).next_to(flecha, RIGHT, buff=0.25)


        rs_bg = RoundedRectangle(
            corner_radius=0.18, width=ANCHO_PANEL, height=ALTO_PANEL,
            fill_color=FONDO_CAJA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=2,
        ).next_to(flecha.get_tip(), DOWN, buff=0.15).set_x(0)

        rs_logo = ImageMobject(os.path.join("assets", "logo_rust.png")).set_height(1.0)
        rs_lbl = VGroup(
            Text("Motor", font=FUENTE, font_size=28,
                 color=MARRON_OSCURO, weight=BOLD),
            Text("tensores, atención y entrenamiento", font=FUENTE, font_size=17,
                 color=TINTA_NEGRA),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        Group(rs_logo, rs_lbl).arrange(RIGHT, buff=0.4).move_to(rs_bg)


        self.play(
            FadeIn(py_bg, shift=DOWN * 0.15), FadeIn(py_logo, shift=DOWN * 0.15),
            FadeIn(py_lbl, shift=DOWN * 0.15),
            LaggedStart(*[FadeIn(a, scale=0.5) for a in adornos], lag_ratio=0.15),
            run_time=0.8,
        )
        self.play(GrowArrow(flecha), FadeIn(chip_pyo3, shift=LEFT * 0.15), run_time=0.45)
        self.play(FadeIn(rs_bg, shift=UP * 0.15), FadeIn(rs_logo, shift=UP * 0.15),
                  FadeIn(rs_lbl, shift=UP * 0.15), run_time=0.6)

        self._siguiente()
        self.limpiar_pantalla()

