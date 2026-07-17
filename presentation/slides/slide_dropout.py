import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manim import *
from manim_code_blocks import *
import os
from colores import *
from objetos import *


class SlideDropout:
    def slide_dropout(self):

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Dropout",
            palabra_clave="Dropout",
            color_clave=NARANJA_TERRACOTA
        )

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))


        tamaños_capas = [4, 5, 5, 5, 4]
        colores_capas = [BLUE_D, NARANJA_TERRACOTA, NARANJA_TERRACOTA, NARANJA_TERRACOTA, GREEN_D]
        nombres_capas = ["Input", "Dense 1", "Dense 2", "Dense 3", "Output"]

        nodos = VGroup()
        etiquetas = VGroup()

        for size, color, nombre in zip(tamaños_capas, colores_capas, nombres_capas):
            capa = VGroup(*[Dot(radius=0.15, color=color) for _ in range(size)]).arrange(DOWN, buff=0.4)
            nodos.add(capa)
            etiqueta = Text(nombre, font=FUENTE, font_size=16, color=MARRON_OSCURO, weight=BOLD)
            etiquetas.add(etiqueta)

        nodos.arrange(RIGHT, buff=1.8).shift(DOWN * 0.2)

        for i, etiqueta in enumerate(etiquetas):
            etiqueta.next_to(nodos[i], UP, buff=0.3)

        conexiones = VGroup()
        for i in range(len(tamaños_capas) - 1):
            capa_act = nodos[i]
            capa_sig = nodos[i+1]
            grupo_conexiones = VGroup()
            for n1 in capa_act:
                for n2 in capa_sig:
                    grupo_conexiones.add(Line(n1.get_center(), n2.get_center(),
                                             stroke_width=1.5, color=MARRON_OSCURO, stroke_opacity=0.3))
            conexiones.add(grupo_conexiones)

        red_grupo = VGroup(conexiones, nodos, etiquetas)
        self.play(FadeIn(red_grupo))


        txt_problema = Text(
            "Durante el entrenamiento, no todas las neuronas aprenden por igual...",
            font=FUENTE, font_size=20, color=TINTA_NEGRA
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(txt_problema, shift=UP))
        self._siguiente()
        self.play(FadeOut(txt_problema))


        dominantes = {1: [0, 2], 2: [1, 3], 3: [0, 2]}
        ociosas    = {1: [1, 3, 4], 2: [0, 2, 4], 3: [1, 3, 4]}

        anims_dom = []
        for capa_idx, idxs in dominantes.items():
            for idx in idxs:
                anims_dom.append(
                    nodos[capa_idx][idx].animate.set_color(ORO_VIEJO).set_opacity(1.0)
                )
        self.play(*anims_dom, run_time=1.0)

        txt_dominantes = Text(
            "Unas pocas neuronas acaparan la mayor parte de la representación.",
            font=FUENTE, font_size=20, color=TINTA_NEGRA, weight=BOLD
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(txt_dominantes, shift=UP))
        self._siguiente()
        self.play(FadeOut(txt_dominantes))


        anims_oci = []
        for capa_idx, idxs in ociosas.items():
            for idx in idxs:
                anims_oci.append(
                    nodos[capa_idx][idx].animate.set_color(ACERO).set_opacity(0.25)
                )
        self.play(*anims_oci, run_time=1.0)

        txt_ociosas = Text(
            "Mientras tanto, otras apenas participan y quedan relegadas.",
            font=FUENTE, font_size=20, color=TINTA_NEGRA
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(txt_ociosas, shift=UP))
        self._siguiente()
        self.play(FadeOut(txt_ociosas))


        txt_consecuencia = Text(
            "El modelo deja de aprender patrones… y comienza a memorizar.",
            font=FUENTE, font_size=20, color=TINTA_NEGRA, weight=BOLD
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(txt_consecuencia, shift=UP))
        self._siguiente()
        self.play(FadeOut(txt_consecuencia))


        anims_reset = []
        for capa_idx, color in enumerate(colores_capas):
            for nodo in nodos[capa_idx]:
                anims_reset.append(nodo.animate.set_color(color).set_opacity(1.0))
        for grupo in conexiones:
            for linea in grupo:
                anims_reset.append(linea.animate.set_stroke(opacity=0.3))
        self.play(*anims_reset, run_time=0.8)


        txt_solucion = Text(
            "Dropout: forzar a la red a no depender de nadie en particular.",
            font=FUENTE, font_size=20, color=TINTA_NEGRA, weight=BOLD
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(txt_solucion, shift=UP))
        self._siguiente()
        self.play(FadeOut(txt_solucion))


        def aplicar_dropout_capa(indice_capa, indices_apagar, texto_explicativo):
            animaciones = []
            capa = nodos[indice_capa]

            for idx in indices_apagar:
                animaciones.append(capa[idx].animate.set_color(ACERO).set_opacity(0.2))

                if indice_capa > 0:
                    for n1_idx in range(tamaños_capas[indice_capa - 1]):
                        line_idx = n1_idx * tamaños_capas[indice_capa] + idx
                        animaciones.append(conexiones[indice_capa - 1][line_idx].animate.set_stroke(opacity=0.02))

                if indice_capa < len(tamaños_capas) - 1:
                    for n2_idx in range(tamaños_capas[indice_capa + 1]):
                        line_idx = idx * tamaños_capas[indice_capa + 1] + n2_idx
                        animaciones.append(conexiones[indice_capa][line_idx].animate.set_stroke(opacity=0.02))

            texto = Text(texto_explicativo, font=FUENTE, font_size=20, color=TINTA_NEGRA).to_edge(DOWN, buff=0.5)
            self.play(*animaciones, FadeIn(texto, shift=UP), run_time=1.5)
            self.play(FadeOut(texto))

        aplicar_dropout_capa(1, [1, 4],
            "Aplicar dropout en Dense 1"
        )

        aplicar_dropout_capa(2, [0, 2, 3],
            "Aplicar dropout en Dense 2"
        )

        aplicar_dropout_capa(3, [1, 4],
            "Aplicar dropout en Dense 3"
        )

        texto_metafora = Text(
            "\"No levantes un reino sobre un solo guerrero;\n"
            "haz que cada espada sepa luchar por sí misma.\"",
            font=FUENTE, font_size=24, color=TINTA_NEGRA, slant=ITALIC
        ).to_edge(DOWN, buff=0.5)

        self.play(Write(texto_metafora))
        self._siguiente()
        self.limpiar_pantalla()


