from manim import *
from manim_code_blocks import *
import numpy as np
import os
from colores import *

def crear_llanuras_manchegas():
    colina_fondo = Ellipse(width=18, height=6, fill_color=TIERRA_MANCHEGA, fill_opacity=0.06, stroke_width=0)
    colina_fondo.move_to(DOWN * 3.5 + LEFT * 3)

    colina_media = Ellipse(width=16, height=4, fill_color=ARENA_MANCHEGA, fill_opacity=0.08, stroke_width=0)
    colina_media.move_to(DOWN * 3.8 + RIGHT * 3)

    colina_frente = Ellipse(width=20, height=3.5, fill_color=BARRO_MANCHEGO, fill_opacity=0.05, stroke_width=0)
    colina_frente.move_to(DOWN * 4)

    llanuras = VGroup(colina_fondo, colina_media, colina_frente).set_z_index(-10)
    return llanuras

def crear_molino():
    base_molino = Polygon(
        [-0.45, -0.6, 0], [0.45, -0.6, 0], [0.3, 0.4, 0], [-0.3, 0.4, 0],
        fill_color=LADRILLO, fill_opacity=1, stroke_color=MADERA_OSCURA, stroke_width=3
    )

    lineas_ladrillo = VGroup(*[
        Line([LEFT_X, -0.4 + i*0.2, 0], [RIGHT_X, -0.4 + i*0.2, 0], color=MADERA_OSCURA, stroke_width=1, stroke_opacity=0.5)
        for i, (LEFT_X, RIGHT_X) in enumerate([(-0.4, 0.4), (-0.35, 0.35), (-0.32, 0.32)])
    ])

    techo = Polygon(
        [-0.35, 0.4, 0], [0.35, 0.4, 0], [0, 0.9, 0],
        fill_color=TEJA, fill_opacity=1, stroke_color=MADERA_OSCURA, stroke_width=3
    )
    marco_puerta = RoundedRectangle(corner_radius=0.1, width=0.35, height=0.45, fill_color=MADERA_OSCURA, stroke_width=0).move_to(base_molino.get_bottom() + UP*0.2)
    puerta = RoundedRectangle(corner_radius=0.1, width=0.25, height=0.35, fill_color=MADERA_CLARA, stroke_color=MADERA_OSCURA, stroke_width=1).move_to(marco_puerta.get_bottom() + UP*0.17)
    ventana_marco = Circle(radius=0.1, fill_color=MADERA_OSCURA, stroke_width=0).move_to(base_molino.get_center() + UP*0.1)
    ventana = Circle(radius=0.07, fill_color=TINTA_NEGRA, stroke_width=0).move_to(ventana_marco)
    reja_v = Line(ventana.get_top(), ventana.get_bottom(), color=MADERA_OSCURA, stroke_width=1.5)
    reja_h = Line(ventana.get_left(), ventana.get_right(), color=MADERA_OSCURA, stroke_width=1.5)

    def crear_aspa():
        eje = Line(ORIGIN, RIGHT*0.8, stroke_color=MADERA_OSCURA, stroke_width=3)
        marco = Rectangle(width=0.6, height=0.25, fill_color=PERGAMINO, fill_opacity=0.9, stroke_color=MADERA_OSCURA, stroke_width=2)
        marco.next_to(eje.get_start(), RIGHT, buff=0.15)
        entramado = VGroup(
            Line(marco.get_top(), marco.get_bottom(), color=MADERA_OSCURA, stroke_width=1),
            Line(marco.get_left(), marco.get_right(), color=MADERA_OSCURA, stroke_width=1),
            Line(marco.get_left() + RIGHT*0.15, marco.get_right() + LEFT*0.45, color=MADERA_OSCURA, stroke_width=1),
            Line(marco.get_right() + LEFT*0.15, marco.get_left() + RIGHT*0.45, color=MADERA_OSCURA, stroke_width=1)
        )
        return VGroup(eje, marco, entramado)

    aspas = VGroup(*[crear_aspa().rotate(i * PI/2, about_point=ORIGIN) for i in range(4)])
    centro = Dot(ORIGIN, color=HIERRO, radius=0.08)
    centro_detalle = Dot(ORIGIN, color=ACERO, radius=0.03)
    sistema_aspas = VGroup(aspas, centro, centro_detalle).move_to(techo.get_bottom() + UP*0.15)

    return VGroup(base_molino, lineas_ladrillo, techo, marco_puerta, puerta, ventana_marco, ventana, reja_v, reja_h, sistema_aspas)

def crear_sol_cervantino():
    centro_borde = Circle(radius=0.28, fill_color=TERRACOTA, stroke_width=0)
    centro = Circle(radius=0.25, fill_color=ORO_VIEJO, stroke_color=TERRACOTA, stroke_width=3)
    anillo_interior = Circle(radius=0.20, stroke_color=TERRACOTA, stroke_width=1, stroke_opacity=0.5)

    cara = VGroup(
        Arc(radius=0.1, start_angle=PI, angle=PI, color=TERRACOTA, stroke_width=2).shift(DOWN*0.05),
        Dot(radius=0.02, color=TERRACOTA).shift(LEFT*0.08 + UP*0.05),
        Dot(radius=0.02, color=TERRACOTA).shift(RIGHT*0.08 + UP*0.05)
    )

    rayos = VGroup()
    for i in range(16):
        angle = i * (PI / 8)
        length = 0.5 if i % 2 == 0 else 0.35
        rayo = Polygon(
            [0.28, -0.05, 0], [0.28, 0.05, 0], [length, 0, 0],
            fill_color=ORO_VIEJO if i % 2 == 0 else TERRACOTA, fill_opacity=1, stroke_width=0
        ).rotate(angle, about_point=ORIGIN)
        rayos.add(rayo)

    return VGroup(rayos, centro_borde, centro, anillo_interior, cara)

def crear_estrella():
    puntas = VGroup()
    for i in range(4):
        angle = i * (PI / 2)
        mitad_clara = Polygon([0,0,0], [0.05,0.05,0], [0,0.25,0], fill_color=ORO_VIEJO, fill_opacity=1, stroke_width=0)
        mitad_oscura = Polygon([0,0,0], [-0.05,0.05,0], [0,0.25,0], fill_color=LATON, fill_opacity=1, stroke_width=0)
        punta = VGroup(mitad_clara, mitad_oscura).rotate(angle, about_point=ORIGIN)
        puntas.add(punta)
    centro_brillo = Dot(ORIGIN, radius=0.03, color=BLANCO)
    return VGroup(puntas, centro_brillo)

def crear_tintero_y_pluma():
    cuerpo = Polygon(
        [-0.25, -0.2, 0], [0.25, -0.2, 0], [0.2, 0.15, 0], [-0.2, 0.15, 0],
        fill_color=TINTA_NEGRA, fill_opacity=1, stroke_color=LATON, stroke_width=2
    )
    brillo = Polygon(
        [-0.15, -0.15, 0], [-0.05, -0.15, 0], [-0.05, 0.1, 0], [-0.1, 0.1, 0],
        fill_color=ACERO, fill_opacity=0.4, stroke_width=0
    )
    tapa = Rectangle(width=0.3, height=0.08, fill_color=LATON, stroke_width=0).next_to(cuerpo, UP, buff=0)
    cuello = Rectangle(width=0.2, height=0.06, fill_color=TINTA_NEGRA, stroke_width=0).next_to(tapa, UP, buff=0)
    tintero = VGroup(cuerpo, brillo, tapa, cuello)

    tallo = Line(cuello.get_top(), cuello.get_top() + UP*1.4 + RIGHT*0.8, color=PERGAMINO, stroke_width=3)
    punta_pluma = Triangle(fill_color=TINTA_NEGRA, stroke_width=0).scale(0.08).move_to(tallo.get_start()).rotate(PI)

    pluma_forma = Polygon(
        tallo.get_start() + UP*0.1, tallo.get_start() + UP*0.6 + LEFT*0.3, tallo.get_end() + LEFT*0.1,
        tallo.get_end(), tallo.get_end() + DOWN*0.3 + RIGHT*0.2, tallo.get_start() + RIGHT*0.1 + UP*0.1,
        fill_color=PERGAMINO, fill_opacity=1, stroke_color=MADERA_CLARA, stroke_width=1
    )
    cortes = VGroup(
        Line(tallo.get_start() + UP*0.4 + LEFT*0.15, tallo.get_start() + UP*0.35, color=MADERA_CLARA, stroke_width=2),
        Line(tallo.get_start() + UP*0.6 + LEFT*0.2, tallo.get_start() + UP*0.5, color=MADERA_CLARA, stroke_width=2),
        Line(tallo.get_end() + DOWN*0.4, tallo.get_end() + DOWN*0.3 + RIGHT*0.1, color=MADERA_CLARA, stroke_width=2),
        Line(tallo.get_end() + DOWN*0.6 + LEFT*0.1, tallo.get_end() + DOWN*0.45 + RIGHT*0.05, color=MADERA_CLARA, stroke_width=2)
    )

    gotas = VGroup(
        Dot(radius=0.04, color=TINTA_NEGRA).move_to(cuello.get_right() + RIGHT*0.2 + UP*0.2),
        Dot(radius=0.02, color=TINTA_NEGRA).move_to(cuello.get_right() + RIGHT*0.35 + UP*0.05),
        Dot(radius=0.03, color=TINTA_NEGRA).move_to(cuello.get_left() + LEFT*0.1 + UP*0.1)
    )

    return VGroup(tintero, pluma_forma, cortes, tallo, punta_pluma, gotas)

def crear_pila_libros():
    def crear_libro(ancho, alto, color_cubierta, color_paginas, rotacion, desplazamiento):
        cubierta = RoundedRectangle(corner_radius=0.05, width=ancho, height=alto, fill_color=color_cubierta, fill_opacity=1, stroke_color=TINTA_NEGRA, stroke_width=2)
        brillo_cubierta = Line(cubierta.get_corner(UL) + RIGHT*0.05, cubierta.get_corner(UR) + LEFT*0.05, color=BLANCO, stroke_width=2, stroke_opacity=0.3)

        paginas = Rectangle(width=ancho-0.1, height=alto-0.08, fill_color=color_paginas, stroke_width=0).move_to(cubierta).align_to(cubierta, RIGHT).shift(LEFT*0.02)
        lineas_pag = VGroup(*[Line(paginas.get_left() + UP*(alto/2 - 0.08 - i*0.04), paginas.get_right() + UP*(alto/2 - 0.08 - i*0.04), color=MADERA_CLARA, stroke_width=1) for i in range(int(alto/0.05))])
        lomo_detalle = Line(cubierta.get_left() + UP*(alto/3), cubierta.get_left() + DOWN*(alto/3), color=ORO_VIEJO, stroke_width=2).shift(RIGHT*0.15)

        libro = VGroup(cubierta, brillo_cubierta, paginas, lineas_pag, lomo_detalle).rotate(rotacion).shift(desplazamiento)
        return libro

    libro1 = crear_libro(1.5, 0.35, ROJO_SANGRE, PERGAMINO, 0, ORIGIN)
    libro2 = crear_libro(1.3, 0.3, AZUL_NOCHE, PERGAMINO, 2 * DEGREES, UP*0.32 + LEFT*0.05)
    marcapaginas = Rectangle(width=0.08, height=0.5, fill_color=ORO_VIEJO, stroke_width=1, stroke_color=TINTA_NEGRA).move_to(libro2.get_right() + LEFT*0.3 + DOWN*0.2)
    libro3 = crear_libro(1.2, 0.25, VERDE_BOSQUE, PERGAMINO, -8 * DEGREES, UP*0.58 + RIGHT*0.05)

    return VGroup(libro1, marcapaginas, libro2, libro3)

def crear_lanza():
    astil = Line(DL*2, UR*2, color=MADERA_VIEJA, stroke_width=8)

    punta_base = Polygon(
        UR*2.8, UR*2.2 + UL*0.18, UR*1.8, UR*2.2 + DR*0.18,
        fill_color=METAL_CLARO, fill_opacity=1, stroke_color=BLACK, stroke_width=2
    )
    brillo_punta = Polygon(
        UR*2.8, UR*2.2 + UL*0.18, UR*1.8,
        fill_color=WHITE, fill_opacity=0.2, stroke_width=0
    )
    nervio = Line(UR*1.8, UR*2.8, color=BLACK, stroke_width=1.5)

    estandarte = Polygon(
        UR*1.6, UR*1.6 + RIGHT*1.4, UR*1.3 + RIGHT*0.8, UR*1.0 + RIGHT*1.4, UR*1.0,
        fill_color=TELA_DESLAVADA, fill_opacity=0.9, stroke_color=METAL_OSCURO, stroke_width=2
    )

    empunadura = VGroup(*[
        Line(UL*0.12, DR*0.12, color=CUERO_GASTADO, stroke_width=3)
        .move_to(astil.point_from_proportion(0.3 + i*0.015))
        for i in range(12)
    ])

    return VGroup(astil, empunadura, estandarte, punta_base, brillo_punta, nervio)

def crear_escudo():
    forma_escudo = [
        [-0.8, 0.8, 0],
        [0.8, 0.8, 0],
        [0.7, 0.1, 0],
        [0, -1.2, 0],
        [-0.7, 0.1, 0],
    ]

    fondo = Polygon(*forma_escudo, fill_color=FONDO_ESCUDO, fill_opacity=1, stroke_width=0)
    borde = Polygon(*forma_escudo, fill_color=BLACK, fill_opacity=0, stroke_color=METAL_OSCURO, stroke_width=10)
    borde_interno = Polygon(*forma_escudo, fill_color=BLACK, fill_opacity=0, stroke_color=METAL_CLARO, stroke_width=2).scale(0.88)

    div_v = Line(UP*0.8, DOWN*1.2, color=METAL_OSCURO, stroke_width=4)
    div_h = Line(LEFT*0.77, RIGHT*0.77, color=METAL_OSCURO, stroke_width=4).shift(UP*0.2)

    diamante = Polygon(
        UP*0.35, RIGHT*0.25, DOWN*0.35, LEFT*0.25,
        fill_color=METAL_OSCURO, fill_opacity=1, stroke_color=BLACK, stroke_width=2
    ).shift(UP*0.2)

    brillo_diamante = Polygon(
        UP*0.35, RIGHT*0.25, DOWN*0.35,
        fill_color=WHITE, fill_opacity=0.3, stroke_width=0
    ).shift(UP*0.2)

    escudo_completo = VGroup(fondo, div_v, div_h, borde, borde_interno, diamante, brillo_diamante)
    escudo_completo.rotate(-PI/12)
    return escudo_completo

def crear_escudo_y_lanza():
    lanza = crear_lanza()
    escudo = crear_escudo()

    escudo.move_to(lanza.get_center() + DOWN*0.2 + RIGHT*0.4)
    sombra_escudo = escudo[0].copy().set_color(BLACK).set_opacity(0.5).shift(DR*0.1)

    composicion = VGroup(lanza, sombra_escudo, escudo)
    composicion.move_to(ORIGIN)
    composicion.scale(0.75)

    return composicion

def crear_herradura():
    cuerpo = AnnularSector(
        inner_radius=0.25, outer_radius=0.45, angle=PI*1.4, start_angle=-PI*0.2,
        fill_color=HIERRO, fill_opacity=1, stroke_color=TINTA_NEGRA, stroke_width=2
    ).rotate(-PI/2 - PI*0.2)

    borde_interior = AnnularSector(
        inner_radius=0.32, outer_radius=0.38, angle=PI*1.3, start_angle=-PI*0.15,
        fill_color=TINTA_NEGRA, fill_opacity=0.3, stroke_width=0
    ).rotate(-PI/2 - PI*0.15)
    brillo_metal = AnnularSector(
        inner_radius=0.38, outer_radius=0.42, angle=PI*0.5, start_angle=PI*0.3,
        fill_color=BLANCO, fill_opacity=0.3, stroke_width=0
    ).rotate(-PI/2 - PI*0.2)

    agujeros = VGroup()
    for i in range(7):
        angulo = i * (PI/5) + PI*0.1
        punto = Rectangle(width=0.02, height=0.06, fill_color=TINTA_NEGRA, stroke_width=0).move_to(
            [0.35 * np.cos(angulo), 0.35 * np.sin(angulo), 0]
        ).rotate(angulo + PI/2)
        agujeros.add(punto)

    herradura = VGroup(cuerpo, borde_interior, brillo_metal, agujeros).rotate(-PI*0.2)
    return herradura

def crear_pergamino():
    cuerpo = Polygon(
        [-0.4, 0.6, 0], [0.4, 0.6, 0], [0.35, -0.6, 0], [-0.45, -0.6, 0],
        fill_color=PERGAMINO, fill_opacity=1, stroke_color=MADERA_OSCURA, stroke_width=2
    )

    rollo_sup = RoundedRectangle(corner_radius=0.1, width=1.0, height=0.25, fill_color=PERGAMINO, fill_opacity=1, stroke_color=MADERA_OSCURA, stroke_width=2).move_to(UP*0.6)
    rollo_sup_espiral = Arc(radius=0.08, start_angle=0, angle=PI*1.5, color=MADERA_OSCURA, stroke_width=2).move_to(rollo_sup.get_right() + LEFT*0.1)

    rollo_inf = RoundedRectangle(corner_radius=0.1, width=0.9, height=0.25, fill_color=PERGAMINO, fill_opacity=1, stroke_color=MADERA_OSCURA, stroke_width=2).move_to(DOWN*0.6)
    rollo_inf_espiral = Arc(radius=0.08, start_angle=PI, angle=PI*1.5, color=MADERA_OSCURA, stroke_width=2).move_to(rollo_inf.get_left() + RIGHT*0.1)
    cinta1 = Line(cuerpo.get_bottom() + UP*0.2 + RIGHT*0.2, cuerpo.get_bottom() + DOWN*0.1 + RIGHT*0.1, color=ROJO_SANGRE, stroke_width=4)
    cinta2 = Line(cuerpo.get_bottom() + UP*0.2 + RIGHT*0.2, cuerpo.get_bottom() + DOWN*0.15 + RIGHT*0.3, color=ROJO_SANGRE, stroke_width=4)

    sello_cera = Circle(radius=0.15, fill_color=ROJO_SANGRE, stroke_color=TINTA_NEGRA, stroke_width=1).move_to(cuerpo.get_bottom() + UP*0.2 + RIGHT*0.2)
    sello_detalle = Circle(radius=0.1, stroke_color=TINTA_NEGRA, stroke_width=1).move_to(sello_cera)

    lineas = VGroup()
    anchos = [0.6, 0.7, 0.5, 0.65, 0.4]
    for i, ancho in enumerate(anchos):
        linea = Line(LEFT*(ancho/2), RIGHT*(ancho/2), color=TINTA_NEGRA, stroke_width=2).shift(UP*(0.3 - i*0.15))
        lineas.add(linea)

    return VGroup(cuerpo, rollo_sup, rollo_sup_espiral, rollo_inf, rollo_inf_espiral, lineas, cinta1, cinta2, sello_cera, sello_detalle)

def crear_yelmo_mambrino():
    cuenco = Arc(
        radius=0.5, start_angle=0, angle=PI,
        fill_color=LATON, fill_opacity=1, stroke_color=MADERA_OSCURA, stroke_width=2
    )
    brillo = Arc(
        radius=0.4, start_angle=PI/4, angle=PI/3,
        color=PERGAMINO, stroke_width=6, stroke_opacity=0.7
    )

    borde = RoundedRectangle(
        corner_radius=0.08, width=1.5, height=0.15,
        fill_color=LATON, fill_opacity=1, stroke_color=MADERA_OSCURA, stroke_width=2
    ).next_to(cuenco, DOWN, buff=-0.05)
    remaches_borde = VGroup(*[
        Dot(radius=0.025, color=ORO_VIEJO).move_to(borde.get_left() + RIGHT*0.15 + RIGHT*i*0.2)
        for i in range(7)
    ])
    muesca = Arc(
        radius=0.18, start_angle=0, angle=PI,
        fill_color=NEGRO_SUAVE, fill_opacity=1, stroke_width=2, stroke_color=MADERA_OSCURA
    ).move_to(borde.get_bottom() + UP*0.08)

    return VGroup(cuenco, borde, brillo, remaches_borde, muesca)

def crear_rust_quijote():
    logo = ImageMobject(os.path.join("assets", "quijote_rust.png")).scale(0.3)

    return logo.move_to(ORIGIN)

def crear_rust_sancho():
    imagen_sancho = ImageMobject(os.path.join("assets", "sancho_rust.png")).scale(0.3)

    return imagen_sancho.move_to(ORIGIN)

try:
    from pygments.style import Style
    from pygments.token import Token, Keyword, Name, String, Number, Operator, Punctuation, Comment

    class EstiloCervantino(Style):
        background_color = FONDO_CAJA
        styles = {
            Token:           TINTA_NEGRA,
            Keyword:         f'bold {NARANJA_TERRACOTA}',
            Keyword.Type:    f'bold {MARRON_OSCURO}',
            String:          MARRON_OSCURO,
            Number:          PAPEL_TAN,
            Name.Function:   NARANJA_TERRACOTA,
            Operator:        TINTA_NEGRA,
            Punctuation:     TINTA_NEGRA,
            Comment:         f'italic {CAJA_INFERIOR}',
            Comment.Preproc: PAPEL_TAN,  # atributos de Rust: #[derive], #[pyclass], ...
            String.Doc:      f'italic {MARRON_OSCURO}',  # comentarios /// de los snippets
        }
except ImportError:
    EstiloCervantino = None
