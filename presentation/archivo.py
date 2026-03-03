from manim import *
from manim_slides import Slide

class Presentacion(Slide):
    def construct(self):
        self.camera.background_color = WHITE

        HIGHLIGHT_COLOR = "#2E8B57" 
        ALERT_COLOR = "#D32F2F"
        TOKEN_COLOR_1 = "#8E44AD" 
        TOKEN_COLOR_2 = "#E67E22" 
        TOKEN_COLOR_FINAL = "#2980B9"

        # =========================================================
        # SLIDE 7 — Molinete AI y su origen
        # =========================================================
        titulo5 = Text("Molinete AI", font_size=55, color=TOKEN_COLOR_2, weight=BOLD).to_edge(UP)
        linea5 = Underline(titulo5, color=TOKEN_COLOR_2, stroke_width=4)

        descripcion = Text("Un modelo de lenguaje entrenado con El Quijote.", font_size=32, color=DARK_GRAY)
        subtitulo = Text("¿Por qué “Molinete”?", font_size=40, color=BLACK, weight=BOLD)
        
        punto1 = Text("• Hace referencia al universo cervantino.", font_size=28, color=BLACK)
        punto2 = Text("• Si Feste toma su identidad del bufón\n  ingenioso en Twelfth Night...", font_size=28, color=DARK_GRAY, slant=ITALIC)
        punto3 = Text("• ...Molinete alude a los fieros oponentes\n  (molinos) del Hidalgo.", font_size=28, color=BLACK)

        grupo_puntos = VGroup(punto1, punto2, punto3).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        textos_molinete = VGroup(descripcion, subtitulo, grupo_puntos).arrange(DOWN, aligned_edge=LEFT, buff=0.8)
        
        imagen_molino = ImageMobject("assets/quijote_vs_molinos.png").scale(1) 
        
        # CORRECCIÓN AQUÍ: Cambiado VGroup por Group
        contenido_completo = Group(textos_molinete, imagen_molino).arrange(RIGHT, buff=0.5)
        contenido_completo.next_to(linea5, DOWN, buff=0.8)

        self.play(Write(titulo5), Create(linea5))
        self.play(FadeIn(descripcion, shift=UP*0.3))
        
        self.next_slide()
        
        self.play(
            FadeIn(subtitulo, shift=RIGHT*0.3),
            FadeIn(imagen_molino, shift=LEFT*0.3)
        )
        
        self.play(LaggedStart(
            FadeIn(punto1, shift=RIGHT*0.2),
            FadeIn(punto2, shift=RIGHT*0.2),
            FadeIn(punto3, shift=RIGHT*0.2),
            lag_ratio=0.5
        ))

        self.next_slide()
        self.play(FadeOut(Group(titulo5, linea5, contenido_completo)))

        # =========================================================
        # SLIDE 8 — ¿Qué es un Transformer?
        # =========================================================
        titulo6 = Text(
            "La Arquitectura Transformer", 
            font_size=50, 
            color=BLACK, 
            weight=BOLD,
            t2c={"Transformer": TOKEN_COLOR_1}
        ).to_edge(UP)
        linea6 = Underline(titulo6, color=TOKEN_COLOR_1, stroke_width=4)

        punto_t1 = Text("• Es el 'motor' de los LLM modernos\n  (GPT, Llama, Molinete).", font_size=28, color=BLACK)
        punto_t2 = Text("• Mecanismo de Atención:\n  Descubre qué palabras están conectadas\n  entre sí, sin importar la distancia.", font_size=28, color=DARK_GRAY)
        punto_t3 = Text("• Procesamiento en Paralelo:\n  A diferencia de modelos antiguos, lee\n  todo el texto de golpe, no palabra por palabra.", font_size=28, color=BLACK)

        textos_transformer = VGroup(punto_t1, punto_t2, punto_t3).arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        imagen_transformer = ImageMobject("assets/arquitectura_transforme.png").scale(0.5)

        # CORRECCIÓN AQUÍ: Cambiado VGroup por Group
        contenido_transformer = Group(textos_transformer, imagen_transformer).arrange(RIGHT, buff=1.0)
        contenido_transformer.next_to(linea6, DOWN, buff=0.8)

        self.play(Write(titulo6), Create(linea6))
        
        self.next_slide()
        self.play(FadeIn(imagen_transformer, shift=UP*0.3))
        
        self.next_slide()
        self.play(LaggedStart(
            FadeIn(punto_t1, shift=RIGHT*0.2),
            FadeIn(punto_t2, shift=RIGHT*0.2),
            FadeIn(punto_t3, shift=RIGHT*0.2),
            lag_ratio=0.5
        ))

        self.next_slide()
        self.play(FadeOut(Group(titulo6, linea6, contenido_transformer)))