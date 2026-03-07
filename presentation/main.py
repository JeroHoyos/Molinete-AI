from manim import *
from manim_slides import Slide
from manim_code_blocks import *
import numpy as np
import random
import math

# CONFIGURACIÓN GLOBAL
FUENTE = "Goudy Old Style"

# PALETA DE COLORES
MARRON_OSCURO = "#3D3834"
NARANJA_TERRACOTA = "#A36536"
PAPEL_CREMA = "#F2E6D8"
PAPEL_TAN = "#B78B68"
FONDO_CAJA = "#FCF3E4"
CAJA_INFERIOR = "#E0C2A8"
TINTA_NEGRA = "#1A1A1A"

# COMPATIBILIDAD
SOFT_BG = PAPEL_CREMA
RUST_COLOR = NARANJA_TERRACOTA

class Presentacion(Slide):
    def construct(self):
        self.camera.background_color = WHITE

        # self.slide_introduction()
        # self.slide_credits()
        # self.slide_arquitectura_transformer()
        # self.slide_molinete_ai()
        
        # self.slide_por_que_no_python()
        # self.slide_por_que_no_cpp()
        # self.slide_por_que_si_rust()

        self.diapo_codigo_rust()

    # --- FUNCIONES AUXILIARES ---

    def crear_titulo(self, texto, palabra_clave=None, color_clave=NARANJA_TERRACOTA, font_size=35):
        t2c = {palabra_clave: color_clave} if palabra_clave else {}
        titulo = Text(texto, font=FUENTE, font_size=font_size, color=TINTA_NEGRA, t2c=t2c).to_edge(UP)
        linea = Underline(titulo, color=color_clave, stroke_width=4)
        return titulo, linea

    def crear_bloque(self, texto="", color_fondo=FONDO_CAJA, color_texto=TINTA_NEGRA, ancho=0.8, alto=0.8):
        rect = RoundedRectangle(
            corner_radius=0.15, width=ancho, height=alto, 
            fill_color=color_fondo, fill_opacity=1, 
            stroke_color=MARRON_OSCURO, stroke_width=2
        )
        lbl = Text(str(texto), font=FUENTE, font_size=24, color=color_texto).move_to(rect.get_center())
        return VGroup(rect, lbl)

    def limpiar_pantalla(self):
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def diapo_codigo_rust(self):

        codigo_fuente = """let pair_counts: HashMap<(String, String), usize> = tokens
        .par_chunks(chunk_size)
        .enumerate()
        .fold(HashMap::new, |mut local_counts, (chunk_idx, chunk)| {
            // Count pairs within this chunk
            for window in chunk.windows(2) {
                let pair = (window[0].clone(), window[1].clone());
                *local_counts.entry(pair).or_insert(0) += 1;
            }
            // Handle chunk boundaries...
            local_counts
        })
        .reduce(HashMap::new, |mut a, b| {
            // Merge counts from all chunks
            for (pair, count) in b {
                *a.entry(pair).or_insert(0) += count;
            }
            a
        });"""

        bloque_codigo = Code(
            code_string=codigo_fuente,
            language="rust",
            background="rectangle"
        )

        bloque_codigo.scale(0.9)
        bloque_codigo.move_to(ORIGIN)

        self.play(FadeIn(bloque_codigo, shift=UP * 0.3), run_time=2.5)
        self.next_slide()

        self.limpiar_pantalla()

    # --- DIAPOSITIVAS ---

    def slide_introduction(self):

        titulo, linea = self.crear_titulo("Construyendo un Transformer con Rust", palabra_clave="Rust")
        subtitulo = Text("Jerónimo Hoyos Botero", font=FUENTE, font_size=25, color=MARRON_OSCURO)
        
        VGroup(titulo, linea, subtitulo).arrange(DOWN, buff=0.2).to_edge(UP, buff=0.5)

        cajas = VGroup(*[
            Rectangle(
                width=2.6, height=1.8, fill_color=FONDO_CAJA, fill_opacity=1, 
                stroke_color=MARRON_OSCURO, stroke_width=2
            ).shift(UP * i * 0.05 + RIGHT * i * 0.05) for i in range(6)
        ]).move_to(LEFT * 5.5 + DOWN * 0.5) 
        
        txt_molinete = Text("Molinete AI", font=FUENTE, font_size=20, weight=BOLD, color=TINTA_NEGRA).move_to(cajas)
        flecha = Arrow(LEFT, RIGHT, color=TINTA_NEGRA, max_tip_length_to_length_ratio=0.15).scale(0.5).next_to(cajas, RIGHT, buff=0.2)

        pos_start = flecha.get_right() + RIGHT * 4.3 + UP * 1.6 

        def create_probs(target_word):
            dummies = ["hidalgo", "espada", "vino", "capa", "oro", "plaza", "mujer", "caballo"]
            random.shuffle(dummies)
            probs = [random.uniform(0.7, 0.9), random.uniform(0.05, 0.1), random.uniform(0.01, 0.03)]
            
            datos = [(target_word, probs[0], NARANJA_TERRACOTA), (dummies[0], probs[1], PAPEL_TAN), (dummies[1], probs[2], CAJA_INFERIOR)]
            
            txts = VGroup(*[Text(d[0], font=FUENTE, font_size=16, color=TINTA_NEGRA) for d in datos]).arrange(DOWN, aligned_edge=RIGHT, buff=0.15)
            bars = VGroup(*[Rectangle(height=0.15, width=d[1]*2, fill_color=d[2], fill_opacity=1, stroke_width=0) for d in datos]).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            
            for i in range(len(datos)): bars[i].match_y(txts[i])
            
            col = VGroup(txts, bars).arrange(RIGHT, buff=0.1)
            bg = Rectangle(width=col.width+0.4, height=txts[0].height+0.2, fill_color=CAJA_INFERIOR, fill_opacity=0.4, stroke_width=1).move_to(txts[0]).match_x(col)
            
            return VGroup(bg, col).move_to(flecha.get_right() + RIGHT * 0.3, aligned_edge=LEFT)

        self.play(Write(titulo), Create(linea), FadeIn(subtitulo, shift=DOWN))
        self.next_slide()
        self.play(FadeIn(cajas, shift=RIGHT), Write(txt_molinete), GrowArrow(flecha))

        poema = [
            ["retorciendo", "el", "mostacho", "soldadesco,"],
            ["por", "ver", "que", "ya", "su", "bolsa", "le", "repica,"],
            ["a", "un", "corrillo", "llegó", "de", "gente", "rica"],
            ["y", "en", "el", "nombre", "de", "Dios", "pidió", "refresco."],
            ["Den", "voacedes,", "por", "Dios,", "a", "mi", "pobreza,"],
            ["les", "dice;", "donde", "no,", "por", "ocho", "santos"],
            ["que", "haré", "lo", "que", "hacer", "suelo", "sin", "tardanza."]
        ]
        
        curr_y = pos_start[1]
        curr_probs = create_probs("...").set_opacity(0)

        for linea_texto in poema:
            curr_x = pos_start[0]
            for word in linea_texto:
                new_probs = create_probs(word)
                word_mob = Text(word, font=FUENTE, font_size=20, color=TINTA_NEGRA).move_to([curr_x, curr_y, 0], aligned_edge=LEFT)
                
                self.play(ReplacementTransform(curr_probs, new_probs), FadeIn(word_mob, shift=LEFT*0.1), run_time=0.5)
                curr_probs, curr_x = new_probs, curr_x + word_mob.width + 0.15
            curr_y -= 0.45

        self.next_slide()
        self.limpiar_pantalla()

    def slide_credits(self):
        titulo_creditos, linea_creditos = self.crear_titulo("Esta presentación se basa en:", palabra_clave="Rbasa en:t", color_clave=MARRON_OSCURO)
        self.play(Write(titulo_creditos), Create(linea_creditos))

        imagen_creditos = ImageMobject("assets/creditos_guia_original.png")
        imagen_creditos.scale(1.5)
        imagen_creditos.next_to(linea_creditos, DOWN, buff=0.8)

        self.play(FadeIn(imagen_creditos, shift=UP))
        
        self.next_slide()

        self.limpiar_pantalla()

    def slide_arquitectura_transformer(self):
        titulo, linea = self.crear_titulo(
            "Arquitectura Transformer", 
            palabra_clave="Transformer", 
            color_clave=NARANJA_TERRACOTA
        )

        punto_t1 = Text(
            "• Arquitectura fundamental de los LLM modernos\n  (GPT, Llama, Molinete).", 
            font=FUENTE, font_size=26, color=TINTA_NEGRA,
            t2c={"Arquitectura fundamental": NARANJA_TERRACOTA, "Molinete": MARRON_OSCURO}
        )

        punto_t2 = Text(
            "• Mecanismo de Atención:\n  Modela dependencias entre tokens\n  mediante matrices Q, K y V.", 
            font=FUENTE, font_size=26, color=TINTA_NEGRA,
            t2c={"Mecanismo de Atención:": NARANJA_TERRACOTA, "Q, K y V": MARRON_OSCURO}
        )

        punto_t3 = Text(
            "• Computación Paralela:\n  Operaciones matriciales eficientes.", 
            font=FUENTE, font_size=26, color=TINTA_NEGRA,
            t2c={"Computación Paralela:": NARANJA_TERRACOTA, "d_model = 512": MARRON_OSCURO}
        )
        textos_transformer = VGroup(punto_t1, punto_t2, punto_t3).arrange(DOWN, aligned_edge=LEFT, buff=0.7)

        imagen_transformer = ImageMobject("assets/arquitectura_transformer.png")
        imagen_transformer.scale(0.65)

        contenido = Group(textos_transformer, imagen_transformer).arrange(RIGHT, buff=1.0)
        
        textos_transformer.shift(DOWN * 0.4) 
        
        contenido.next_to(linea, DOWN, buff=0.8)
        contenido.center().shift(DOWN * 0.3)
        
        self.play(Write(titulo), Create(linea))
        self.next_slide()

        self.play(FadeIn(imagen_transformer, shift=UP * 0.3))
        self.next_slide()

        for punto in [punto_t1, punto_t2, punto_t3]:
            self.play(FadeIn(punto, shift=RIGHT * 0.3), run_time=0.8)
            self.next_slide() 

        self.limpiar_pantalla()
    
    def slide_molinete_ai(self):
        titulo, linea = self.crear_titulo(
            "Molinete AI", 
            palabra_clave="Molinete", 
            color_clave=NARANJA_TERRACOTA
        )

        subtitulo = Text(
            "Origen del nombre “Molinete”", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            weight=BOLD
        )

        descripcion = Text(
            "Modelo de lenguaje entrenado sobre\nel corpus de El Quijote.", 
            font=FUENTE, font_size=24, color=MARRON_OSCURO
        )

        punto1 = Text(
            "• Inspirado en el imaginario cervantino.",
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"imaginario cervantino": NARANJA_TERRACOTA}
        )

        punto2 = Text(
            "• Nombre que evoca los molinos de viento\n  enfrentados por Don Quijote.",
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"molinos de viento": NARANJA_TERRACOTA}
        )

        punto3 = Text(
            "• Identidad literaria para un modelo\n  de lenguaje contemporáneo.",
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"modelo de lenguaje": MARRON_OSCURO}
        )

        grupo_puntos = VGroup(punto1, punto2, punto3)\
            .arrange(DOWN, aligned_edge=LEFT, buff=0.4) 

        textos_molinete = VGroup(descripcion, subtitulo, grupo_puntos)\
            .arrange(DOWN, aligned_edge=LEFT, buff=0.5) 

        imagen_molino = ImageMobject("assets/quijote_vs_molinos.png")

        imagen_molino.height = 4.5 

        contenido_completo = Group(textos_molinete, imagen_molino)\
            .arrange(RIGHT, buff=0.5) 
        if contenido_completo.width > 12.5:
            contenido_completo.width = 12.5

        contenido_completo.next_to(linea, DOWN, buff=0.75)

        self.play(Write(titulo), Create(linea))
        self.play(FadeIn(descripcion, shift=UP * 0.2))
        self.next_slide()

        self.play(
            FadeIn(subtitulo, shift=RIGHT * 0.4),
            FadeIn(imagen_molino, shift=LEFT * 0.4)
        )
        self.next_slide()
        
        self.play(
            LaggedStart(
                FadeIn(punto1, shift=UP * 0.2),
                FadeIn(punto2, shift=UP * 0.2),
                FadeIn(punto3, shift=UP * 0.2),
                lag_ratio=0.8
            ),
            run_time=2.5
        )
        self.next_slide()

        self.limpiar_pantalla()
    def slide_por_que_no_python(self):
        titulo, linea = self.crear_titulo(
        "Limitaciones de Python en sistemas de inferencia", 
        palabra_clave="Python", 
        color_clave=NARANJA_TERRACOTA
        )

        p1 = Text(
            "• Excelente para investigación y prototipado rápido.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        )
        
        p2 = Text(
            "• Recolector de basura introduce pausas\n  no determinísticas en ejecución.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, 
            t2c={"Recolector de basura": MARRON_OSCURO}
        )
        
        p3 = Text(
            "• Control limitado sobre asignación y liberación\n  de memoria.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, 
            t2c={"memoria": MARRON_OSCURO}
        )
        
        p4 = Text(
            "• En la inferencia de LLM cada ciclo de CPU\n  y cada byte de memoria impactan el rendimiento.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, 
            t2c={"cada ciclo": NARANJA_TERRACOTA}
        )
        
        puntos_python = VGroup(p1, p2, p3, p4).arrange(DOWN, aligned_edge=LEFT, buff=0.4)

        logo_python = ImageMobject("assets/logo_python.png")
        logo_python.height = 3.5

        contenido = Group(puntos_python, logo_python).arrange(RIGHT, buff=0.8)
        
        if contenido.width > 12.5:
            contenido.width = 12.5

        contenido.next_to(linea, DOWN, buff=0.8)

        self.play(Write(titulo), Create(linea))
        self.next_slide()

        self.play(
            FadeIn(logo_python, shift=DOWN * 0.3),
            logo_python.animate.set_opacity(0.8),
            run_time=1
        )
        self.play(logo_python.animate.set_opacity(1))
        
        self.play(
            LaggedStart(
                *[FadeIn(p, shift=RIGHT * 0.2) for p in puntos_python],
                lag_ratio=0.4 
            ),
            run_time=2.0
        )
        self.next_slide()

        self.limpiar_pantalla()

    def slide_por_que_no_cpp(self):
        titulo, linea = self.crear_titulo(
        "C++ como estándar en sistemas de alto rendimiento", 
        palabra_clave="C++", 
        color_clave=MARRON_OSCURO
        )

        p1 = Text(
            "• Lenguaje predominante en sistemas\n  de computación de alto rendimiento.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        )

        p2 = Text(
            "• Permite control directo sobre memoria\n  y recursos del hardware.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        )

        p3 = Text(
            "• Amplio ecosistema de librerías y motores\n  de inferencia ya establecidos.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"ecosistema": MARRON_OSCURO}
        )

        p4 = Text(
            "• Menor incentivo para exploración\n  arquitectónica desde primeros principios.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"primeros principios": NARANJA_TERRACOTA}
        )

        puntos_cpp = VGroup(p1, p2, p3, p4).arrange(DOWN, aligned_edge=LEFT, buff=0.4)

        logo_cpp = ImageMobject("assets/logo_cpp.png")
        logo_cpp.height = 3.5

        contenido = Group(puntos_cpp, logo_cpp).arrange(RIGHT, buff=0.8)
        
        if contenido.width > 12.5:
            contenido.width = 12.5
            
        contenido.next_to(linea, DOWN, buff=0.8)

        self.play(Write(titulo), Create(linea))
        self.next_slide()

        self.play(GrowFromCenter(logo_cpp))
        self.next_slide()

        self.play(
            LaggedStart(
                *[FadeIn(p, shift=UP * 0.2) for p in puntos_cpp],
                lag_ratio=0.4
            ),
            run_time=2.0
        )
        self.next_slide()

        self.limpiar_pantalla()

    def slide_por_que_si_rust(self):
        titulo, linea = self.crear_titulo(
        "Rust como alternativa moderna para sistemas de inferencia", 
        palabra_clave="Rust", 
        color_clave=NARANJA_TERRACOTA
        )

        p1 = Text(
            "• Combina rendimiento cercano a C++\n  con abstracciones modernas.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        )

        p2 = Text(
            "• Zero-cost abstractions: control de memoria\n  sin recolector de basura.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"Zero-cost abstractions": MARRON_OSCURO}
        )

        p3 = Text(
            "• Modelo de ownership garantiza seguridad\n  en concurrencia evitando data races.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"ownership": NARANJA_TERRACOTA}
        )

        p4 = Text(
            "• Ecosistema emergente adecuado para\n  implementar un LLM desde primeros principios.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"LLM": NARANJA_TERRACOTA}
        )

        puntos_rust = VGroup(p1, p2, p3, p4).arrange(DOWN, aligned_edge=LEFT, buff=0.4)

        logo_rust = ImageMobject("assets/logo_rust.png")
        logo_rust.height = 3.5 

        contenido = Group(puntos_rust, logo_rust).arrange(RIGHT, buff=0.8)
        
        if contenido.width > 12.5:
            contenido.width = 12.5
            
        contenido.next_to(linea, DOWN, buff=0.8)
        
        self.play(Write(titulo), Create(linea))
        self.next_slide()

        logo_rust.rotate(-math.pi / 4).scale(0.6)

        self.play(
            FadeIn(logo_rust),
            logo_rust.animate.rotate(math.pi / 4).scale(1.6),
            run_time=1.2
        )
        self.next_slide()

        self.play(
            LaggedStart(
                *[FadeIn(p, shift=RIGHT * 0.2) for p in puntos_rust],
                lag_ratio=0.4
            ),
            run_time=2.0
        )
        self.next_slide()

        self.limpiar_pantalla()

    def diapo_problema_strawberry(self):
        titulo, linea = self.crear_titulo(
            "¿Por qué los LLM no saben 'leer'?", 
            palabra_clave="'leer'?", 
            color_clave=ALERT_COLOR
        )

        def crear_burbuja(texto, color_fondo, color_texto, es_usuario=True, t2c_dict=None):
            txt = Text(texto, font_size=24, color=color_texto, t2c=t2c_dict)
            
            fondo = RoundedRectangle(
                width=txt.width + 0.8, 
                height=txt.height + 0.5, 
                corner_radius=0.2, 
                fill_color=color_fondo, 
                fill_opacity=1, 
                stroke_width=0 if es_usuario else 1,
                stroke_color=DARK_GRAY
            )
            txt.move_to(fondo.get_center())
            burbuja_base = VGroup(fondo, txt)
            
            remitente = Text(
                "Tú" if es_usuario else "Molinete AI", 
                font_size=16, 
                color=DARK_GRAY,
                weight=BOLD
            )
            
            if es_usuario:
                remitente.next_to(burbuja_base, UP, buff=0.2, aligned_edge=RIGHT)
            else:
                remitente.next_to(burbuja_base, UP, buff=0.1, aligned_edge=LEFT)
                
            return VGroup(remitente, burbuja_base)

        burbuja_pregunta = crear_burbuja(
            "¿Cuántas letras 'r' hay en 'strawberry'?", 
            color_fondo=HIGHLIGHT_COLOR, 
            color_texto=WHITE, 
            es_usuario=True
        )
        
        burbuja_respuesta = crear_burbuja(
            "Hay 2 letras 'r' en 'strawberry'.", 
            color_fondo=SOFT_BG, 
            color_texto=BLACK, 
            es_usuario=False,
            t2c_dict={"2": ALERT_COLOR}
        )

        grupo_chat = VGroup(burbuja_pregunta, burbuja_respuesta).arrange(DOWN, buff=0.5)
        
        burbuja_pregunta.shift(RIGHT * 1.5)
        burbuja_respuesta.shift(LEFT * 1.5)

        grupo_chat.next_to(linea, DOWN, buff=0.6)

        texto_explicacion = Text(
            "Para el modelo, la palabra se divide en 'pedazos' (Tokens):", 
            font_size=24, color=DARK_GRAY
        )
        
        token1 = self.crear_bloque("str", SOFT_BG, ancho=1.2)
        token2 = self.crear_bloque("aw", SOFT_BG, ancho=1.2)
        token3 = self.crear_bloque("berry", SOFT_BG, ancho=1.6)
        
        tokens_straw = VGroup(token1, token2, token3).arrange(RIGHT, buff=0.15)
        
        grupo_visual = VGroup(texto_explicacion, tokens_straw).arrange(DOWN, buff=0.5)
        grupo_visual.next_to(grupo_chat, DOWN, buff=1.0)

        self.play(Write(titulo), Create(linea))
        self.next_slide()

        self.play(FadeIn(burbuja_pregunta, shift=UP*0.2, scale=0.9))
        self.wait(0.5)
        
        self.play(FadeIn(burbuja_respuesta, shift=UP*0.2, scale=0.9))
        self.next_slide()

        self.play(FadeIn(texto_explicacion, shift=UP*0.2))
        self.play(
            LaggedStart(
                GrowFromCenter(token1),
                GrowFromCenter(token2),
                GrowFromCenter(token3),
                lag_ratio=0.2
            )
        )
        self.next_slide()

        cruz = Cross(tokens_straw, stroke_color=ALERT_COLOR, stroke_width=6)
        self.play(Create(cruz))
        self.next_slide()

        self.limpiar_pantalla()
    def diapo_tokenizacion(self):
        titulo, linea = self.crear_titulo(
            "La Tokenización", 
            palabra_clave="Tokenización", 
            color_clave=HIGHLIGHT_COLOR
        )

        p1 = Text(
            "1. Por palabra: Vocabulario infinito, muy ineficiente.", 
            font_size=24, color=BLACK,
            t2c={"Por palabra:": DARK_GRAY, "infinito": ALERT_COLOR}
        )
        tokens_palabra = ["En", "un", "lugar", "de", "la", "Mancha"]
        
        ej_palabra = VGroup(*[
            self.crear_bloque(t, color_fondo=SOFT_BG, ancho=max(0.8, len(t) * 0.25)) 
            for t in tokens_palabra
        ]).arrange(RIGHT, buff=0.15)
        
        grupo_palabra = VGroup(p1, ej_palabra).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        p2 = Text(
            "2. Por carácter: Secuencias larguísimas, pierde contexto.", 
            font_size=24, color=BLACK,
            t2c={"Por carácter:": DARK_GRAY, "pierde contexto": ALERT_COLOR}
        )
        frase = "En un lugar de la Mancha"
        tokens_caracter = [c if c != ' ' else '_' for c in frase]
        
        ej_caracter = VGroup(*[
            self.crear_bloque(t, color_fondo=SOFT_BG, ancho=0.3) 
            for t in tokens_caracter
        ]).arrange(RIGHT, buff=0.05)
        
        if ej_caracter.width > 12:
            ej_caracter.scale_to_fit_width(12)
            
        grupo_caracter = VGroup(p2, ej_caracter).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        p3 = Text(
            "3. Sub-palabras (BPE): Algo más equilibrado.", 
            font_size=26, color=BLACK, weight=BOLD, 
            t2c={"Sub-palabras (BPE):": HIGHLIGHT_COLOR, "equilibrio": HIGHLIGHT_COLOR}
        )
        tokens_bpe = ["En", " un", " lugar", " de", " la", " Man", "cha"]
        
        ej_bpe = VGroup(*[
            self.crear_bloque(t, color_fondo=SOFT_BG, ancho=max(0.8, len(t) * 0.25)) 
            for t in tokens_bpe
        ]).arrange(RIGHT, buff=0.15)
        
        grupo_bpe = VGroup(p3, ej_bpe).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        puntos = VGroup(grupo_palabra, grupo_caracter, grupo_bpe).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        puntos.next_to(linea, DOWN, buff=0.5).shift(LEFT * 0.5)

        self.play(Write(titulo), Create(linea))
        self.next_slide()

        self.play(FadeIn(p1, shift=RIGHT * 0.3))
        self.play(LaggedStart(*[GrowFromCenter(b) for b in ej_palabra], lag_ratio=0.1))
        self.next_slide()

        self.play(FadeIn(p2, shift=RIGHT * 0.3))
        self.play(LaggedStart(*[GrowFromCenter(b) for b in ej_caracter], lag_ratio=0.03)) 
        self.next_slide()

        self.play(FadeIn(p3, shift=RIGHT * 0.3))
        self.play(LaggedStart(*[GrowFromCenter(b) for b in ej_bpe], lag_ratio=0.1))
        self.next_slide()

        self.limpiar_pantalla()
    def diapo_bpe_animacion(self):
        # 1. Título
        titulo, linea = self.crear_titulo(
            "Byte Pair Encoding (BPE)", 
            palabra_clave="Byte Pair Encoding (BPE)", 
            color_clave=TOKEN_COLOR_FINAL
        )

        explicacion_bpe = Text(
            "Fusión iterativa de los pares más frecuentes:", 
            font_size=28, color=DARK_GRAY
        ).next_to(linea, DOWN, buff=0.5)

        self.play(Write(titulo), Create(linea), FadeIn(explicacion_bpe))

        def calc_ancho(texto):
            return 0.6 + len(texto) * 0.3

        letras = ["t", "a", "c", "o", "t", "a", "c", "o"]

        fila_actual = VGroup(*[
            self.crear_bloque(letra, color_fondo=SOFT_BG, ancho=calc_ancho(letra)) 
            for letra in letras
        ]).arrange(RIGHT, buff=0.15).center().shift(UP*0.5)
        
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN*0.5) for b in fila_actual], lag_ratio=0.1))
        self.next_slide()

        texto_paso1 = Text(
            "Paso 1: 't' y 'a' son el par más común", 
            font_size=24, color=BLACK
        ).next_to(fila_actual, DOWN, buff=1)
        self.play(FadeIn(texto_paso1))
        
        fila_paso1 = VGroup(
            self.crear_bloque("ta", TOKEN_COLOR_1, ancho=calc_ancho("ta")),
            self.crear_bloque("c", SOFT_BG, ancho=calc_ancho("c")),
            self.crear_bloque("o", SOFT_BG, ancho=calc_ancho("o")),
            self.crear_bloque("ta", TOKEN_COLOR_1, ancho=calc_ancho("ta")),
            self.crear_bloque("c", SOFT_BG, ancho=calc_ancho("c")),
            self.crear_bloque("o", SOFT_BG, ancho=calc_ancho("o"))
        ).arrange(RIGHT, buff=0.15).center().shift(UP*0.5)

        self.play(
            ReplacementTransform(VGroup(fila_actual[0], fila_actual[1]), fila_paso1[0]),
            ReplacementTransform(fila_actual[2], fila_paso1[1]),
            ReplacementTransform(fila_actual[3], fila_paso1[2]),
            ReplacementTransform(VGroup(fila_actual[4], fila_actual[5]), fila_paso1[3]),
            ReplacementTransform(fila_actual[6], fila_paso1[4]),
            ReplacementTransform(fila_actual[7], fila_paso1[5]),
        )
        fila_actual = fila_paso1
        self.next_slide()

        texto_paso2 = Text(
            "Paso 2: 'c' y 'o' son el par más común", 
            font_size=24, color=BLACK
        ).next_to(fila_actual, DOWN, buff=1)
        
        fila_paso2 = VGroup(
            self.crear_bloque("ta", TOKEN_COLOR_1, ancho=calc_ancho("ta")),
            self.crear_bloque("co", TOKEN_COLOR_2, ancho=calc_ancho("co")),
            self.crear_bloque("ta", TOKEN_COLOR_1, ancho=calc_ancho("ta")),
            self.crear_bloque("co", TOKEN_COLOR_2, ancho=calc_ancho("co"))
        ).arrange(RIGHT, buff=0.15).center().shift(UP*0.5)

        self.play(
            ReplacementTransform(texto_paso1, texto_paso2),
            ReplacementTransform(fila_actual[0], fila_paso2[0]),
            ReplacementTransform(VGroup(fila_actual[1], fila_actual[2]), fila_paso2[1]),
            ReplacementTransform(fila_actual[3], fila_paso2[2]),
            ReplacementTransform(VGroup(fila_actual[4], fila_actual[5]), fila_paso2[3]),
        )
        fila_actual = fila_paso2
        self.next_slide()

        texto_paso3 = Text(
            "Paso 3: 'ta' y 'co' forman un nuevo token", 
            font_size=24, color=BLACK
        ).next_to(fila_actual, DOWN, buff=1)

        fila_paso3 = VGroup(
            self.crear_bloque("taco", TOKEN_COLOR_FINAL, ancho=calc_ancho("taco")),
            self.crear_bloque("taco", TOKEN_COLOR_FINAL, ancho=calc_ancho("taco"))
        ).arrange(RIGHT, buff=0.15).center().shift(UP*0.5)

        self.play(
            ReplacementTransform(texto_paso2, texto_paso3),
            ReplacementTransform(VGroup(fila_actual[0], fila_actual[1]), fila_paso3[0]),
            ReplacementTransform(VGroup(fila_actual[2], fila_actual[3]), fila_paso3[1]),
        )
        
        self.next_slide()
        self.limpiar_pantalla()
    def diapo_tamano_vocabulario(self):
        titulo, linea = self.crear_titulo(
            "El Tamaño del Vocabulario", 
            palabra_clave="Vocabulario", 
            color_clave=HIGHLIGHT_COLOR
        )

        encabezados = VGroup(
            Text("Vocabulario", font_size=24, color=DARK_GRAY, weight=BOLD),
            Text("Total Tokens", font_size=24, color=DARK_GRAY, weight=BOLD),
            Text("Compresión", font_size=24, color=HIGHLIGHT_COLOR, weight=BOLD)
        ).arrange(RIGHT, buff=1.5)

        linea_separadora = Line(LEFT, RIGHT, color=DARK_GRAY, stroke_width=2)
        
        def crear_fila(v, t, c, es_final=False):
            color_c = TOKEN_COLOR_FINAL if es_final else BLACK
            peso_c = BOLD if es_final else NORMAL
            return VGroup(
                Text(v, font_size=24, color=BLACK),
                Text(t, font_size=24, color=BLACK),
                Text(c, font_size=24, color=color_c, weight=peso_c)
            )

        fila1 = crear_fila("256", "2,168,312", "1.00x")
        fila2 = crear_fila("1,024", "724,453", "2.99x")
        fila3 = crear_fila("20,534", "460,900", "4.70x", es_final=True)

        for fila in [fila1, fila2, fila3]:
            for i in range(3):
                fila[i].set_x(encabezados[i].get_x())

        filas_datos = VGroup(fila1, fila2, fila3).arrange(DOWN, buff=0.4)
        
        tabla_interna = VGroup(encabezados, linea_separadora, filas_datos).arrange(DOWN, buff=0.3)
        linea_separadora.set_width(tabla_interna.width + 1)
        
        fondo_tabla = RoundedRectangle(
            width=tabla_interna.width + 1.5, 
            height=tabla_interna.height + 0.8, 
            corner_radius=0.2, 
            fill_color=SOFT_BG, 
            fill_opacity=1, 
            stroke_width=0
        )
        
        grupo_tabla = VGroup(fondo_tabla, tabla_interna)
        grupo_tabla.next_to(linea, DOWN, buff=0.6)

        tradeoff_titulo = Text("El Trade-off (Compensación):", font_size=28, color=BLACK, weight=BOLD)
        
        pro_icon = Text("✅", font_size=24)
        pro_text = Text(
            "Más vocabulario = Textos cortos = Inferencia rápida", 
            font_size=24, color=DARK_GRAY, 
            t2c={"Inferencia rápida": TOKEN_COLOR_FINAL}
        )
        pro_group = VGroup(pro_icon, pro_text).arrange(RIGHT, buff=0.2)

        con_icon = Text("❌", font_size=24)
        con_text = Text(
            "Más vocabulario = Matriz gigante = Más VRAM", 
            font_size=24, color=DARK_GRAY, 
            t2c={"Más VRAM": ALERT_COLOR}
        )
        con_group = VGroup(con_icon, con_text).arrange(RIGHT, buff=0.2)
        
        textos_inferiores = VGroup(tradeoff_titulo, pro_group, con_group).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        textos_inferiores.next_to(grupo_tabla, DOWN, buff=0.6)

        self.play(Write(titulo), Create(linea))
        self.next_slide()

        self.play(FadeIn(fondo_tabla), FadeIn(encabezados), Create(linea_separadora))
        self.play(
            LaggedStart(
                FadeIn(fila1, shift=UP*0.2), 
                FadeIn(fila2, shift=UP*0.2), 
                FadeIn(fila3, shift=UP*0.2), 
                lag_ratio=0.3
            )
        )
        self.next_slide()
        
        self.play(FadeIn(tradeoff_titulo, shift=UP*0.2))
        self.play(FadeIn(pro_group, shift=RIGHT*0.2))
        self.play(FadeIn(con_group, shift=RIGHT*0.2))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_que_es_un_tensor(self):
        titulo11, linea11 = self.crear_titulo("¿Qué es un Tensor?", palabra_clave="Tensor?", color_clave=HIGHLIGHT_COLOR)
        self.play(Write(titulo11), Create(linea11))
        
        def_tensor = Text(
            "Un Tensor es un contenedor matemático\npara almacenar datos en múltiples dimensiones.", 
            font_size=32, 
            color=DARK_GRAY,
            t2c={"múltiples dimensiones.": HIGHLIGHT_COLOR}
        ).next_to(linea11, DOWN, buff=0.4)
        
        self.play(Write(def_tensor))
        self.next_slide()
        
        txt_0d = Text("Escalar (0D) - Un solo valor", font_size=28, color=DARK_GRAY).next_to(def_tensor, DOWN, buff=0.8)
        escalar = self.crear_bloque("7", SOFT_BG, BLACK, 0.8, 0.8).next_to(txt_0d, DOWN, buff=0.5)
        
        self.play(Write(txt_0d))
        self.play(GrowFromCenter(escalar))
        self.next_slide()
        self.play(FadeOut(txt_0d), FadeOut(escalar))
        
        txt_1d = Text("Vector (1D) - Lista de valores", font_size=28, color=DARK_GRAY).next_to(def_tensor, DOWN, buff=0.8)
        vector = self.crear_matriz_bloques(1, 4, SOFT_BG, ["1", "5", "9", "2"]).next_to(txt_1d, DOWN, buff=0.5)
        
        self.play(Write(txt_1d))
        self.play(LaggedStart(*[FadeIn(b, shift=UP*0.2) for b in vector[0]], lag_ratio=0.1))
        self.next_slide()
        self.play(FadeOut(txt_1d), FadeOut(vector))
        
        txt_2d = Text("Matriz (2D) - Tabla de valores", font_size=28, color=DARK_GRAY).next_to(def_tensor, DOWN, buff=0.8)
        valores_matriz = ["3","1","4","2", "5","9","2","6", "5","3","5","8"]
        matriz = self.crear_matriz_bloques(3, 4, SOFT_BG, valores_matriz).next_to(txt_2d, DOWN, buff=0.5)
        
        self.play(Write(txt_2d))
        bloques_anim = [FadeIn(b, shift=UP*0.2) for fila in matriz for b in fila]
        self.play(LaggedStart(*bloques_anim, lag_ratio=0.05))
        self.next_slide()
        self.play(FadeOut(txt_2d), FadeOut(matriz))
        
        txt_3d = Text("Tensor (3D+) - Cubo de valores", font_size=28, color=DARK_GRAY).next_to(def_tensor, DOWN, buff=0.5)
        
        matriz_base_3d = self.crear_matriz_bloques(3, 3, SOFT_BG, ["1","2","3","4","5","6","7","8","9"])
        for b in matriz_base_3d.submobjects:
            for sub_b in b.submobjects:
                sub_b[0].set_fill(opacity=0.3).set_stroke(opacity=0.3)

        matriz_medio_3d = self.crear_matriz_bloques(3, 3, SOFT_GREEN, ["9","8","7","6","5","4","3","2","1"]).shift(UP*0.25 + RIGHT*0.25)
        for b in matriz_medio_3d.submobjects:
            for sub_b in b.submobjects:
                sub_b[0].set_fill(opacity=0.6).set_stroke(opacity=0.6)

        matriz_top_3d = self.crear_matriz_bloques(3, 3, HIGHLIGHT_COLOR, ["2","4","6","8","0","2","4","6","8"]).shift(UP*0.5 + RIGHT*0.5)
        
        tensor_3d = VGroup(matriz_base_3d, matriz_medio_3d, matriz_top_3d).next_to(txt_3d, DOWN, buff=0.5).shift(LEFT * 0.25)
        
        self.play(Write(txt_3d))
        self.play(FadeIn(matriz_base_3d, shift=UP*0.3))
        self.play(FadeIn(matriz_medio_3d, shift=UP*0.3))
        self.play(FadeIn(matriz_top_3d, shift=UP*0.3))
        self.next_slide()
        
        self.play(FadeOut(txt_3d), FadeOut(tensor_3d), FadeOut(def_tensor))

        nota_ram = Text("Pero en memoria RAM, todos terminan siendo un arreglo plano...", font_size=26, color=RUST_COLOR).next_to(linea11, DOWN, buff=0.5)
        self.play(Write(nota_ram))
        
        matriz_ram = self.crear_matriz_bloques(3, 4, SOFT_BG, valores_matriz).next_to(nota_ram, DOWN, buff=0.8)
        self.play(FadeIn(matriz_ram))
        self.next_slide()

        bloques_individuales = [bloque for fila in matriz_ram for bloque in fila]
        grupo_plano = VGroup(*bloques_individuales)
        
        self.play(
            grupo_plano.animate.arrange(RIGHT, buff=0.05).scale(0.8).next_to(nota_ram, DOWN, buff=1.5),
            run_time=2
        )
        self.play(Indicate(grupo_plano, color=GHOST_COLOR))
        self.next_slide()

        self.limpiar_pantalla()

    def diapo_matmul(self):

        titulo, linea = self.crear_titulo(
            "Multiplicación: El 'Dot Product'", 
            palabra_clave="'Dot Product'", 
            color_clave=HIGHLIGHT_COLOR
        )
        

        val_A = ["1","2","3",  "4","5","6",  "7","8","9"]
        val_B = ["2","1","0",  "0","2","1",  "1","0","3"]
        val_C = ["","","",  "","","",  "","",""]
        
        mat_A = self.crear_matriz_bloques(3, 3, SOFT_BG, val_A)
        signo_por = Text("×", font_size=40, color=BLACK)
        mat_B = self.crear_matriz_bloques(3, 3, SOFT_BG, val_B)
        signo_igual = Text("=", font_size=40, color=BLACK)
        mat_C = self.crear_matriz_bloques(3, 3, SOFT_BG, val_C)

        grupo_matmul = VGroup(mat_A, signo_por, mat_B, signo_igual, mat_C).arrange(RIGHT, buff=0.4)
        grupo_matmul.shift(UP * 0.5) 
        fila_A = mat_A[0] 
        col_B = VGroup(mat_B[0][2], mat_B[1][2], mat_B[2][2]) 
        celda_C = mat_C[0][2] 

        self.play(Write(titulo), Create(linea))
        self.play(FadeIn(grupo_matmul))
        self.next_slide()

        self.play(
            *[b[0].animate.set_fill(HIGHLIGHT_COLOR, opacity=0.4) for b in fila_A],
            *[b[0].animate.set_fill(HIGHLIGHT_COLOR, opacity=0.4) for b in col_B]
        )
        self.next_slide()

        calculo_texto = Text(
            "(1 × 0) + (2 × 1) + (3 × 3) = 11", 
            font_size=32, color=DARK_GRAY,
            t2c={"11": TOKEN_COLOR_FINAL}
        ).next_to(grupo_matmul, DOWN, buff=1.0)
        
        fila_copia = fila_A.copy()
        col_copia = col_B.copy()

        self.play(
            ReplacementTransform(fila_copia, calculo_texto), 
            ReplacementTransform(col_copia, calculo_texto)
        )
        self.next_slide()


        dot_calc = self.crear_bloque("11", TOKEN_COLOR_FINAL, WHITE, ancho=0.8)
        dot_calc.move_to(calculo_texto.get_center())

        self.play(ReplacementTransform(calculo_texto, dot_calc))
        
        self.play(
            dot_calc.animate.move_to(celda_C.get_center()),
            celda_C[0].animate.set_fill(TOKEN_COLOR_FINAL, opacity=0.2) 
        )
        self.play(Indicate(dot_calc, color=HIGHLIGHT_COLOR, scale_factor=1.2))
        self.next_slide()
        
        def_matmul = Text(
            "Combina cada elemento de la fila con su pareja en la columna.\nSe multiplican y se suman para obtener un único número.", 
            font_size=24, color=BLACK
        ).to_edge(DOWN, buff=1.0)
        
        self.play(Write(def_matmul))
        self.next_slide()
        
        self.limpiar_pantalla()

    def diapo_softmax(self):

            titulo, linea = self.crear_titulo(
                "Softmax", 
                palabra_clave="Probabilidades", 
                color_clave=TOKEN_COLOR_FINAL
            )
            
            formula = MathTex(
                r"\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}", 
                color=BLACK, font_size=38
            ).next_to(linea, DOWN, buff=0.4)

            self.play(Write(titulo), Create(linea))
            self.play(Write(formula))
            self.next_slide()

            ancho_caja = 1.1
            
            col1 = VGroup(
                Text("1. Logits", font_size=18, color=DARK_GRAY, weight=BOLD),
                VGroup(*[self.crear_bloque(v, SOFT_BG, ancho=ancho_caja) for v in ["2.0", "1.0", "0.1"]]).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            flecha1 = Arrow(LEFT, RIGHT, color=BLACK).scale(0.5)
            txt_op1 = MathTex(r"\exp(x)", font_size=22, color=BLACK).next_to(flecha1, UP, buff=0.1)
            conector1 = VGroup(flecha1, txt_op1)

            col2 = VGroup(
                Text("2. Exp", font_size=18, color=DARK_GRAY, weight=BOLD),
                VGroup(*[self.crear_bloque(v, HIGHLIGHT_COLOR, ancho=ancho_caja) for v in ["7.39", "2.72", "1.10"]]).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            flecha2 = Arrow(LEFT, RIGHT, color=BLACK).scale(0.5)
            txt_op2 = MathTex(r"\div \sum", font_size=22, color=BLACK).next_to(flecha2, UP, buff=0.1)
            conector2 = VGroup(flecha2, txt_op2)

            col3 = VGroup(
                Text("3. Prob", font_size=18, color=TOKEN_COLOR_FINAL, weight=BOLD),
                VGroup(*[self.crear_bloque(v, TOKEN_COLOR_FINAL, WHITE, ancho=ancho_caja) for v in ["66%", "24%", "10%"]]).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            acto1_horiz = VGroup(col1, conector1, col2, conector2, col3).arrange(RIGHT, buff=0.5).move_to(DOWN * 0.5)

            self.play(FadeIn(col1, shift=UP*0.2))
            self.play(Write(conector1), ReplacementTransform(col1[1].copy(), col2[1]), FadeIn(col2[0]))
            self.play(Write(conector2), ReplacementTransform(col2[1].copy(), col3[1]), FadeIn(col3[0]))
            self.next_slide()

            self.play(FadeOut(acto1_horiz), FadeOut(formula))

            titulo_error = Text("Problema: Un valor arruina el vector", font_size=32, color=ALERT_COLOR).move_to(titulo)
            self.play(ReplacementTransform(titulo, titulo_error))

            col_err_1 = VGroup(
                Text("Logits", font_size=18, color=DARK_GRAY, weight=BOLD),
                VGroup(
                    self.crear_bloque("800.0", ALERT_COLOR, WHITE, ancho=ancho_caja),
                    self.crear_bloque("2.0", SOFT_BG, ancho=ancho_caja),
                    self.crear_bloque("-1.0", SOFT_BG, ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            f_err = Arrow(LEFT, RIGHT, color=BLACK).scale(0.5)
            t_err = MathTex(r"\exp(x)", font_size=22, color=BLACK).next_to(f_err, UP, buff=0.1)
            conector_err = VGroup(f_err, t_err)

            col_err_2 = VGroup(
                Text("Exp", font_size=18, color=DARK_GRAY, weight=BOLD),
                VGroup(
                    self.crear_bloque("inf", ALERT_COLOR, WHITE, ancho=ancho_caja),
                    self.crear_bloque("7.39", HIGHLIGHT_COLOR, ancho=ancho_caja),
                    self.crear_bloque("0.37", HIGHLIGHT_COLOR, ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            flujo_error = VGroup(col_err_1, conector_err, col_err_2).arrange(RIGHT, buff=0.8).move_to(DOWN * 0.2)

            self.play(FadeIn(col_err_1))
            self.play(Flash(col_err_1[1][0], color=ALERT_COLOR))
            self.play(Write(conector_err))
            self.play(ReplacementTransform(col_err_1[1].copy(), col_err_2[1]), FadeIn(col_err_2[0]))
            self.play(Wiggle(col_err_2[1][0])) 
            
            nota_error = Text("Si intentamos dividir por 'inf', todo el vector se vuelve NaN.", font_size=20, color=DARK_GRAY).next_to(flujo_error, DOWN, buff=0.6)
            self.play(Write(nota_error))
            self.next_slide()

            self.play(FadeOut(flujo_error), FadeOut(nota_error))
            
            titulo_fix = Text("Solución: Restar el Máximo (Shift)", font_size=32, color=TOKEN_COLOR_FINAL).move_to(titulo_error)
            self.play(ReplacementTransform(titulo_error, titulo_fix))

            col_fix_1 = VGroup(
                Text("Logits", font_size=18, color=DARK_GRAY, weight=BOLD),
                VGroup(
                    self.crear_bloque("800.0", SOFT_BG, ancho=ancho_caja),
                    self.crear_bloque("2.0", SOFT_BG, ancho=ancho_caja),
                    self.crear_bloque("-1.0", SOFT_BG, ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            f_shift = Arrow(LEFT, RIGHT, color=BLACK).scale(0.5)
            t_shift = Text("- Max (x)", font_size=16, color=TOKEN_COLOR_FINAL, weight=BOLD).next_to(f_shift, UP, buff=0.1)
            conector_shift = VGroup(f_shift, t_shift)

            col_fix_2 = VGroup(
                Text("Shifted", font_size=18, color=TOKEN_COLOR_FINAL, weight=BOLD),
                VGroup(
                    self.crear_bloque("0.0", TOKEN_COLOR_FINAL, WHITE, ancho=ancho_caja), 
                    self.crear_bloque("-798.0", HIGHLIGHT_COLOR, ancho=ancho_caja),
                    self.crear_bloque("-801.0", HIGHLIGHT_COLOR, ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            f_exp2 = Arrow(LEFT, RIGHT, color=BLACK).scale(0.5)
            t_exp2 = MathTex(r"\exp(x)", font_size=22, color=BLACK).next_to(f_exp2, UP, buff=0.1)
            conector_exp2 = VGroup(f_exp2, t_exp2)

            col_fix_3 = VGroup(
                Text("Exp Seguro", font_size=18, color=DARK_GRAY, weight=BOLD),
                VGroup(
                    self.crear_bloque("1.0", TOKEN_COLOR_FINAL, WHITE, ancho=ancho_caja),
                    self.crear_bloque("0.0", SOFT_BG, ancho=ancho_caja), 
                    self.crear_bloque("0.0", SOFT_BG, ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            flujo_fix = VGroup(col_fix_1, conector_shift, col_fix_2, conector_exp2, col_fix_3).arrange(RIGHT, buff=0.35).move_to(DOWN * 0.2)

            self.play(FadeIn(col_fix_1))
            self.play(Write(conector_shift))
            self.play(ReplacementTransform(col_fix_1[1].copy(), col_fix_2[1]), FadeIn(col_fix_2[0]))
            self.play(Write(conector_exp2))
            self.play(ReplacementTransform(col_fix_2[1].copy(), col_fix_3[1]), FadeIn(col_fix_3[0]))
            
            nota_fix = Text(
                "Los valores gigantes se vuelven 0.0 y los demás se acercan a 0.\n¡El vector es matemáticamente idéntico pero 100% estable!", 
                font_size=18, color=BLACK
            ).next_to(flujo_fix, DOWN, buff=0.5)
            
            self.play(FadeIn(nota_fix, shift=UP*0.2))
            self.next_slide()

            self.limpiar_pantalla()

    def slide_broadcasting(self):

        titulo, linea = self.crear_titulo(
            "Broadcasting: Expansión Virtual", 
            palabra_clave="Expansión Virtual", 
            color_clave=SOFT_GREEN
        )

        val_base = ["1", "2", "3", "4", "5", "6"]
        val_vec = ["10", "20", "30"]
        val_res = ["11", "22", "33", "14", "25", "36"]


        matriz_base = self.crear_matriz_bloques(2, 3, SOFT_BG, val_base)
        signo_mas = Text("+", font_size=40, color=BLACK)
        
        vector_real = VGroup(*[self.crear_bloque(val, SOFT_GREEN) for val in val_vec]).arrange(RIGHT, buff=0.05)
    
        ecuacion = VGroup(matriz_base, signo_mas, vector_real).arrange(RIGHT, buff=0.5).shift(LEFT * 2.5 + UP * 0.5)
        vector_real.align_to(matriz_base, UP)

        self.play(Write(titulo), Create(linea))
        self.play(
            FadeIn(matriz_base, shift=UP*0.2), 
            Write(signo_mas), 
            FadeIn(vector_real, shift=LEFT*0.2)
        )
        self.next_slide()

        vector_fantasma = vector_real.copy()
        
        self.play(vector_fantasma.animate.next_to(vector_real, DOWN, buff=0.05), run_time=1)
        
        self.play(
            *[b[0].animate.set_stroke(opacity=0.4).set_fill(opacity=0.2) for b in vector_fantasma],
            *[b[1].animate.set_opacity(0.4) for b in vector_fantasma],
            run_time=0.8
        )
        
        rectangulo_base = SurroundingRectangle(
            VGroup(vector_real, vector_fantasma), 
            color=SOFT_GREEN, buff=0.1, corner_radius=0.1
        )
        caja_virtual = DashedVMobject(rectangulo_base, num_dashes=35)
        
        texto_virtual = Text("Matriz Virtual", font_size=16, color=DARK_GRAY).next_to(caja_virtual, DOWN, buff=0.15)

        self.play(Create(caja_virtual), Write(texto_virtual))
        self.next_slide()

        flecha_res = Arrow(LEFT, RIGHT, color=BLACK).next_to(caja_virtual, RIGHT, buff=0.5)
        matriz_res = self.crear_matriz_bloques(2, 3, HIGHLIGHT_COLOR, val_res).next_to(flecha_res, RIGHT, buff=0.5)

        self.play(GrowArrow(flecha_res))
        
        self.play(
            ReplacementTransform(matriz_base.copy(), matriz_res),
            ReplacementTransform(VGroup(vector_real.copy(), vector_fantasma.copy()), matriz_res),
        )
        self.next_slide()
        
        def_broad = Text(
            "Se duplica el vector pequeño 'virtualmente' para coincidir\ncon la matriz grande, ahorrando muchísima memoria.", 
            font_size=24, color=BLACK, t2c={"virtualmente": SOFT_GREEN, "ahorrando muchísima memoria": TOKEN_COLOR_FINAL}
        ).to_edge(DOWN, buff=1.0)
        
        self.play(FadeIn(def_broad, shift=UP*0.2))
        self.next_slide()
        self.limpiar_pantalla()
    def slide_strides(self):

        titulo, linea = self.crear_titulo(
            "Strides: Saltando en Memoria 1D", 
            palabra_clave="Strides:", 
            color_clave=HIGHLIGHT_COLOR
        )

        arr_1d = VGroup(*[self.crear_bloque(str(i), SOFT_BG) for i in range(6)])
        arr_1d.arrange(RIGHT, buff=0.1).shift(UP * 1.5)
        
        lbl_1d = Text("Memoria RAM (Física, 1D)", font_size=20, color=DARK_GRAY).next_to(arr_1d, UP, buff=0.3)

        self.play(Write(titulo), Create(linea))
        self.play(FadeIn(arr_1d, shift=UP*0.2), FadeIn(lbl_1d, shift=UP*0.2))
        self.next_slide()

        fila1 = VGroup(*[arr_1d[i].copy() for i in range(3)]).arrange(RIGHT, buff=0.1)
        fila2 = VGroup(*[arr_1d[i].copy() for i in range(3, 6)]).arrange(RIGHT, buff=0.1)
        
        mat_shape = VGroup(fila1, fila2).arrange(DOWN, buff=0.1).shift(DOWN * 0.5)
        lbl_2d = Text("Shape Lógica (2x3)", font_size=20, color=DARK_GRAY).next_to(mat_shape, DOWN, buff=0.3)

        self.play(
            TransformFromCopy(VGroup(*arr_1d[0:3]), fila1),
            TransformFromCopy(VGroup(*arr_1d[3:6]), fila2),
            FadeIn(lbl_2d, shift=DOWN*0.2),
            run_time=1.5
        )
        self.next_slide()

        self.play(
            arr_1d[0][0].animate.set_fill(HIGHLIGHT_COLOR, opacity=0.5),
            arr_1d[3][0].animate.set_fill(HIGHLIGHT_COLOR, opacity=0.5),
            mat_shape[0][0][0].animate.set_fill(HIGHLIGHT_COLOR, opacity=0.5),
            mat_shape[1][0][0].animate.set_fill(HIGHLIGHT_COLOR, opacity=0.5),
        )

        arco_1d = CurvedArrow(arr_1d[0].get_top(), arr_1d[3].get_top(), angle=-PI/2, color=HIGHLIGHT_COLOR)
        txt_stride_1d = Text("Stride = 3 pasos", font_size=20, color=HIGHLIGHT_COLOR).next_to(arco_1d, UP, buff=0.1)
        
        arco_2d = CurvedArrow(mat_shape[0][0].get_left(), mat_shape[1][0].get_left(), angle=PI/2, color=HIGHLIGHT_COLOR).shift(LEFT*0.1)
        txt_stride_2d = Text("+1 Fila", font_size=16, color=HIGHLIGHT_COLOR).next_to(arco_2d, LEFT, buff=0.1)

        self.play(
            Create(arco_1d), Write(txt_stride_1d),
            Create(arco_2d), Write(txt_stride_2d)
        )
        self.next_slide()
        
        def_stride = Text(
            "Como todo es 1D en RAM, los 'strides' dictan\ncuántos casilleros avanzar para encontrar la siguiente fila.", 
            font_size=24, color=BLACK, t2c={"1D en RAM": DARK_GRAY, "'strides'": HIGHLIGHT_COLOR}
        ).to_edge(DOWN, buff=1.0) 
        
        self.play(FadeIn(def_stride, shift=UP*0.2))
        self.next_slide()
        self.limpiar_pantalla()

    def slide_masked_fill(self):

        titulo, linea = self.crear_titulo(
            "Masked Fill: Causalidad", 
            palabra_clave="Causalidad", 
            color_clave=ALERT_COLOR
        )

        val_mask = [
            "4.1", "1.2", "0.5", "2.1", 
            "3.3", "5.0", "1.8", "0.9", 
            "1.1", "2.4", "6.2", "1.5", 
            "0.8", "1.7", "3.0", "7.1"
        ]
        
        matriz_mask = self.crear_matriz_bloques(4, 4, SOFT_BG, val_mask).scale(1.2).shift(DOWN * 0.2)
        
        lbl_matriz = Text("Scores de Atención (Previo al Masking)", font_size=20, color=DARK_GRAY).next_to(matriz_mask, UP, buff=0.3)
        
        self.play(Write(titulo), Create(linea))
        self.play(FadeIn(matriz_mask, shift=UP*0.2), FadeIn(lbl_matriz, shift=UP*0.2))
        self.next_slide()

        animaciones_mask = []
        animaciones_keep = []
        
        for i in range(4):
            for j in range(4):
                bloque = matriz_mask[i][j]
                if j > i: 
                    nuevo_bloque = self.crear_bloque("-∞", DARK_GRAY, WHITE)
                    nuevo_bloque.match_height(bloque).move_to(bloque)
                    animaciones_mask.append(ReplacementTransform(bloque, nuevo_bloque))
                else:
                    animaciones_keep.append(
                        bloque[0].animate.set_stroke(color=SOFT_GREEN, width=2)
                    )

        lbl_mask = Text("Máscara Causal Aplicada (Masked Fill)", font_size=20, color=ALERT_COLOR).move_to(lbl_matriz)

        self.play(
            LaggedStart(*animaciones_mask, lag_ratio=0.15),
            *animaciones_keep,
            ReplacementTransform(lbl_matriz, lbl_mask),
            run_time=2
        )
        self.next_slide()
        
        def_mask = Text(
            "Se tapan los valores del 'futuro' con menos infinito (-∞).\nAl pasar por Softmax, esto se convierte en 0% de probabilidad.", 
            font_size=24, color=BLACK, 
            t2c={"-∞": ALERT_COLOR, "0% de probabilidad": ALERT_COLOR, "'futuro'": DARK_GRAY}
        ).to_edge(DOWN, buff=1.0)
        
        self.play(FadeIn(def_mask, shift=UP*0.2))
        self.next_slide()
        self.limpiar_pantalla()
    def slide_reshape_transpose(self):
        diccionario_t2c = {"Reshaping": HIGHLIGHT_COLOR, "Transposing": RUST_COLOR}
        titulo, linea = self.crear_titulo("Reshaping vs Transposing", t2c_dict=diccionario_t2c)
        
        linea_central = DashedLine(UP*2.2, DOWN*2.5, color=GRAY)

        self.play(Write(titulo), Create(linea), Create(linea_central))
        self.next_slide()

        txt_reshape = Text("Reshape", font_size=28, color=HIGHLIGHT_COLOR).move_to(LEFT * 3.5 + UP * 2.2)
        sub_reshape = Text("(Misma memoria)", font_size=16, color=DARK_GRAY).next_to(txt_reshape, DOWN, buff=0.1)
        
        valores_orig = ["1", "2", "3", "4", "5", "6"]
        
        mat_r_orig = self.crear_matriz_bloques(2, 3, SOFT_BG, valores_orig).scale(0.8).next_to(sub_reshape, DOWN, buff=0.4)
        flecha_r = Arrow(mat_r_orig.get_bottom(), mat_r_orig.get_bottom() + DOWN * 0.8, color=HIGHLIGHT_COLOR)
        mat_r_final = self.crear_matriz_bloques(3, 2, SOFT_GREEN, valores_orig).scale(0.8).next_to(flecha_r, DOWN, buff=0.2)

        self.play(Write(txt_reshape), Write(sub_reshape), FadeIn(mat_r_orig))
        self.play(GrowArrow(flecha_r))
        
        self.play(
            TransformFromCopy(mat_r_orig[0][0], mat_r_final[0][0]), # 1
            TransformFromCopy(mat_r_orig[0][1], mat_r_final[0][1]), # 2
            TransformFromCopy(mat_r_orig[0][2], mat_r_final[1][0]), # 3
            TransformFromCopy(mat_r_orig[1][0], mat_r_final[1][1]), # 4
            TransformFromCopy(mat_r_orig[1][1], mat_r_final[2][0]), # 5
            TransformFromCopy(mat_r_orig[1][2], mat_r_final[2][1]), # 6
            run_time=2
        )
        self.next_slide()

        txt_transpose = Text("Transpose", font_size=28, color=RUST_COLOR).move_to(RIGHT * 3.5 + UP * 2.2)
        sub_transpose = Text("(Reorganización física)", font_size=16, color=DARK_GRAY).next_to(txt_transpose, DOWN, buff=0.1)

        mat_t_orig = self.crear_matriz_bloques(2, 3, SOFT_BG, valores_orig).scale(0.8).next_to(sub_transpose, DOWN, buff=0.4)
        flecha_t = Arrow(mat_t_orig.get_bottom(), mat_t_orig.get_bottom() + DOWN * 0.8, color=RUST_COLOR)
        
        valores_trans = ["1", "4", "2", "5", "3", "6"]
        mat_t_final = self.crear_matriz_bloques(3, 2, RUST_COLOR, valores_trans).scale(0.8).next_to(flecha_t, DOWN, buff=0.2)

        self.play(Write(txt_transpose), Write(sub_transpose), FadeIn(mat_t_orig))
        self.play(GrowArrow(flecha_t))

        self.play(
            TransformFromCopy(mat_t_orig[0][0], mat_t_final[0][0]),                
            TransformFromCopy(mat_t_orig[0][1], mat_t_final[1][0], path_arc=PI/3),  
            TransformFromCopy(mat_t_orig[0][2], mat_t_final[2][0], path_arc=PI/3),  
            TransformFromCopy(mat_t_orig[1][0], mat_t_final[0][1], path_arc=-PI/3), 
            TransformFromCopy(mat_t_orig[1][1], mat_t_final[1][1], path_arc=-PI/3), 
            TransformFromCopy(mat_t_orig[1][2], mat_t_final[2][1]),                 
            run_time=2.5
        )
        self.next_slide()

        def_res_trans = Text(
            "Reshape solo cambia cómo leemos la lista.\nTranspose requiere copiar y cambiar el orden físico.", 
            font_size=22, color=BLACK, 
            t2c={"Reshape": HIGHLIGHT_COLOR, "Transpose": RUST_COLOR}
        ).to_edge(DOWN, buff=0.5)
        
        self.play(FadeIn(def_res_trans, shift=UP*0.2))
        self.next_slide()
        self.limpiar_pantalla()
    def slide_forward_pass(self):

        titulo, linea = self.crear_titulo("Arquitectura: El Forward Pass", color_clave=BLUE)
        self.play(Write(titulo), Create(linea))
        
        frase_completa = VGroup(
            Text('"En un lugar de la', font_size=42, color=DARK_GRAY),
            Text(' Mancha"', font_size=42, color="#D9534F")
        ).arrange(RIGHT, buff=0.1).move_to(UP * 1.8)

        frase_base = frase_completa[0]
        posicion_destino_mancha = frase_completa[1].get_center()
        
        self.play(Write(frase_base))
        self.next_slide()

        BG_CAJA = "#F8F9FA"
        BORDE_CAJA = "#343A40"
        TXT_PRINCIPAL = BLACK
        TXT_SECUNDARIO = BLUE_D 

        def crear_caja_nodo(label_sup, valor_array, label_inf, is_stack=False, highlight_last=False):
            grupo = VGroup()
            
            txt_sup = Text(label_sup, font_size=18, font="Monospace", weight=BOLD, color=TXT_SECUNDARIO)
            
            ANCHO = 2.65
            ALTO = 1.7
            RADIO = 0.15
            
            sombra = RoundedRectangle(corner_radius=RADIO, width=ANCHO, height=ALTO)
            sombra.set_fill(BLACK, opacity=0.15).set_stroke(width=0)
            sombra.shift(RIGHT * 0.06 + DOWN * 0.06)
            
            caja = RoundedRectangle(corner_radius=RADIO, width=ANCHO, height=ALTO)
            caja.set_fill(color=BG_CAJA, opacity=1).set_stroke(color=BORDE_CAJA, width=2.5)
            
            color_array = "#D9534F" if highlight_last else TXT_PRINCIPAL
            txt_arr = Text(valor_array, font_size=18, font="Monospace", weight=BOLD, color=color_array) 
            txt_inf = Text(label_inf, font_size=16, color=TXT_PRINCIPAL) 
            
            if txt_arr.width > (ANCHO - 0.3):
                txt_arr.scale_to_fit_width(ANCHO - 0.3)
                
            Textos_caja = VGroup(txt_arr, txt_inf).arrange(DOWN, buff=0.25)
            
            if is_stack:
                caja_fondo1 = caja.copy().set_stroke(width=1.5).shift(RIGHT * 0.1 + UP * 0.1)
                caja_fondo2 = caja.copy().set_stroke(width=1.5).shift(RIGHT * 0.05 + UP * 0.05)
                fondo = VGroup(sombra, caja_fondo1, caja_fondo2, caja)
            else:
                fondo = VGroup(sombra, caja)
                
            caja_y_textos = VGroup(fondo, Textos_caja)
            Textos_caja.move_to(caja.get_center())
            
            txt_sup.next_to(caja_y_textos, UP, buff=0.25)
            grupo.add(txt_sup, caja_y_textos)
            
            grupo.caja_principal = caja 
            return grupo

        nodo_1 = crear_caja_nodo("token_ids", "[145, 892...]", "Tokens: En|un...")
        nodo_2 = crear_caja_nodo("tok+pos_emb", "[0.81, -0.2...]", "Vectores (768d)")
        nodo_3 = crear_caja_nodo("blocks", "[0.55, 0.9...]", "Atención (x12)", is_stack=True)
        nodo_4 = crear_caja_nodo("ln_f", "[0.12, -0.4...]", "Norm (768d)")
        nodo_5 = crear_caja_nodo("lm_head", "[..., 25.4...]", "Score Máximo:\n\" Mancha\"", highlight_last=True)

        pipeline = VGroup(nodo_1, nodo_2, nodo_3, nodo_4, nodo_5)
        pipeline.arrange(RIGHT, buff=0.25).scale(0.76).move_to(DOWN * 0.8)

        flechas = VGroup()
        for i in range(len(pipeline) - 1):
            flecha = Arrow(
                pipeline[i].caja_principal.get_right(), 
                pipeline[i+1].caja_principal.get_left(), 
                buff=0.08, color=BLUE_D, stroke_width=5, max_tip_length_to_length_ratio=0.15
            )
            flechas.add(flecha)

        self.play(FadeIn(pipeline, shift=UP*0.2), FadeIn(flechas, shift=UP*0.2), run_time=1.5)
        self.next_slide()

        tokens_viajeros = frase_base.copy()
        centro_primera_caja = pipeline[0].caja_principal.get_center()
        punto_flujo = Dot(color=BLUE, radius=0.18).move_to(centro_primera_caja)
        
        self.play(ReplacementTransform(tokens_viajeros, punto_flujo, path_arc=-PI/3), run_time=1.2)
        self.play(Indicate(pipeline[0].caja_principal, color=BLUE, scale_factor=1.05))
        
        for i in range(len(flechas)):
            centro_siguiente_caja = pipeline[i+1].caja_principal.get_center()
            self.play(
                punto_flujo.animate.move_to(centro_siguiente_caja), 
                run_time=0.6, 
                rate_func=linear
            )
            self.play(Indicate(pipeline[i+1].caja_principal, color=BLUE, scale_factor=1.05), run_time=0.4)

        prediccion_txt = Text(' Mancha"', font_size=42, color="#D9534F").move_to(pipeline[-1].caja_principal.get_center())
        self.play(ReplacementTransform(punto_flujo, prediccion_txt), run_time=0.8)
        self.next_slide()

        self.play(
            prediccion_txt.animate.move_to(posicion_destino_mancha),
            run_time=1.5,
            path_arc=-PI/3
        )
        
        frase_final = VGroup(frase_base, prediccion_txt)
        self.play(Circumscribe(frase_final, color=BLUE, time_width=2, stroke_width=4))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_embeddings(self):
        titulo_p1 = Text("Embeddings: ", font_size=42, weight=BOLD, color=BLACK)
        titulo_p2 = Text("De IDs a Vectores", font_size=42, weight=BOLD, color=BLUE_D)
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 5, RIGHT * 5, color=BLUE_D).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)
        
        self.play(Write(titulo_completo), Create(linea))

        ejes = Axes(
            x_range=[-4, 4, 1], y_range=[-3, 4, 1],
            x_length=8, y_length=5.0, 
            axis_config={"color": BLACK, "stroke_width": 2, "include_ticks": False}
        ).shift(DOWN * 0.1) 
        
        self.play(Create(ejes))

        v_hombre = np.array([2, -1, 0])
        v_caballero = np.array([2, 2, 0])
        v_mujer = np.array([-2, -1, 0])
        v_dama = np.array([-2, 2, 0])

        def crear_vector_2d(coord, color, label_text, direccion):
            p = ejes.c2p(*coord)
            flecha = Arrow(start=ejes.c2p(0,0), end=p, color=color, buff=0, stroke_width=5)
            lbl = Text(label_text, font_size=24, weight=BOLD, color=color)
            lbl.set_background_stroke(color=WHITE, width=4)
            lbl.next_to(p, direccion, buff=0.2)
            return VGroup(flecha, lbl)

        caballero = crear_vector_2d(v_caballero, BLUE_D, "Caballero", UR)
        hombre = crear_vector_2d(v_hombre, BLUE_C, "Hombre", DR)
        mujer = crear_vector_2d(v_mujer, RED_B, "Mujer", DL)
        dama = crear_vector_2d(v_dama, GOLD, "Dama", UL)

        self.play(Create(caballero), Create(hombre))
        
        v_relacion = DashedLine(ejes.c2p(*v_hombre), ejes.c2p(*v_caballero), color=GRAY).set_stroke(width=4)
        v_relacion.add_tip()
        
        lbl_relacion = Text("+ Nobleza", font_size=20, color=BLACK, weight=BOLD)
        lbl_relacion.set_background_stroke(color=WHITE, width=4)
        lbl_relacion.next_to(v_relacion, RIGHT, buff=0.2)
        
        self.play(Create(v_relacion), Write(lbl_relacion))
        self.next_slide()

        self.play(Create(mujer))
        
        v_relacion_movida = v_relacion.copy().shift(ejes.c2p(*v_mujer) - ejes.c2p(*v_hombre))
        lbl_relacion_movida = lbl_relacion.copy().next_to(v_relacion_movida, LEFT, buff=0.2)
        
        self.play(
            Transform(v_relacion, v_relacion_movida),
            Transform(lbl_relacion, lbl_relacion_movida)
        )
        
        punto_llegada = Dot(ejes.c2p(*v_dama), color=GOLD, radius=0.08)
        self.play(FadeIn(punto_llegada))
        
        self.play(ReplacementTransform(punto_llegada, dama))
        self.play(Indicate(dama))

        formula = MathTex(
            "\\vec{Mujer} + (\\vec{Caballero} - \\vec{Hombre}) \\approx \\vec{Dama}",
            font_size=38, color=BLACK
        ).to_edge(DOWN, buff=1.0) 
        
        self.play(Write(formula))
        self.next_slide()

        self.play(FadeOut(ejes, caballero, hombre, mujer, dama, v_relacion, lbl_relacion, formula))

        input_word = Text('"lugar"', font_size=40, color=BLACK, weight=BOLD).move_to(LEFT * 4.5 + UP * 1.5)
        
        id_token = Text("ID: 301", font_size=36, weight=BOLD, color=WHITE)
        bg_id = RoundedRectangle(corner_radius=0.2, width=2.5, height=1, color=BLUE_D).set_fill(BLUE_D, 1)
        grupo_id = VGroup(bg_id, id_token).move_to(LEFT * 4.5 + DOWN * 0.2)

        self.play(FadeIn(input_word))
        self.next_slide()
        
        self.play(ReplacementTransform(input_word, grupo_id))
        self.next_slide()

        matriz_v = VGroup()
        filas, cols = 8, 10
        COLOR_VECTOR = RED_D
        for i in range(filas):
            fila = VGroup()
            for j in range(cols):
                cuadro = RoundedRectangle(corner_radius=0.05, width=0.35, height=0.35).set_stroke(GRAY, opacity=0.4)
                if i == 4:
                    cuadro.set_fill(COLOR_VECTOR, opacity=0.15).set_stroke(COLOR_VECTOR, opacity=0.6)
                fila.add(cuadro)
            matriz_v.add(fila.arrange(RIGHT, buff=0.08))
            
        matriz_v.arrange(DOWN, buff=0.08).move_to(RIGHT * 3 + DOWN * 0)
        lbl_matriz = Text("Matriz de Embeddings", font_size=20, weight=BOLD, color=GRAY).next_to(matriz_v, UP, buff=0.4)

        self.play(FadeIn(matriz_v), Write(lbl_matriz))
        self.next_slide()

        flecha_busqueda = Arrow(grupo_id.get_right(), matriz_v[4].get_left(), color=BLUE_C, buff=0.2)
        self.play(GrowArrow(flecha_busqueda))
        
        fila_sel = matriz_v[4].copy().set_fill(RED_D, 0.8).set_stroke(RED_D, 2)
        self.play(Transform(matriz_v[4], fila_sel))
        self.next_slide()

        vector_final = Text("[  0.12,  -0.45,  0.88,  ...  ]", font_size=36, font="Monospace", color=RED_D)
        vector_final.move_to(DOWN * 2.8)
        
        lbl_final = Text("Vector Semántico", font_size=24, weight=BOLD, color=BLACK)
        lbl_size = Text("(768 dimensiones)", font_size=18, color=GRAY)
        
        grupo_lbl_final = VGroup(lbl_final, lbl_size).arrange(DOWN, buff=0.1).next_to(vector_final, UP, buff=0.2)

        self.play(
            ReplacementTransform(matriz_v[4].copy(), vector_final),
            Write(grupo_lbl_final),
            run_time=1.5
        )
        self.play(Indicate(vector_final, color=RED_B))
        
        self.next_slide()
        self.limpiar_pantalla()
    def slide_position_embeddings(self):
        # --- 1. TÍTULO CON ÉNFASIS ---
        titulo_p1 = Text("Embeddings de ", font_size=42, weight=BOLD, color=BLACK)
        titulo_p2 = Text("Posición", font_size=42, weight=BOLD, color=BLUE_D) 
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 4, RIGHT * 4, color=GRAY).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)
        
        self.play(Write(titulo_completo), Create(linea))
        self.next_slide()

        # --- 2. EL PROBLEMA: PROCESAMIENTO EN PARALELO ---
        # Volvemos a Cervantes
        frase = Text("En un lugar de la Mancha", font_size=40, color=BLACK)
        nota_problema = Text("Los Transformers procesan todo en paralelo.\n¡Se pierde el orden de las palabras!", font_size=20, color=RED_D).next_to(frase, DOWN, buff=0.5)
        
        self.play(FadeIn(frase))
        self.play(Write(nota_problema))
        self.next_slide()

        palabras = ["En", "un", "lugar", "de", "la", "Mancha"]
        tokens = VGroup(*[Text(w, font_size=32, color=BLACK) for w in palabras]).arrange(RIGHT, buff=0.4)
        posiciones = VGroup(*[Text(f"Pos: {i}", font_size=18, color=GRAY) for i in range(len(palabras))])

        for t, p in zip(tokens, posiciones):
            p.next_to(t, DOWN, buff=0.2)

        grupo_tokens = VGroup(tokens, posiciones).move_to(UP * 1.5)
        
        self.play(
            FadeOut(nota_problema),
            ReplacementTransform(frase, grupo_tokens)
        )
        self.next_slide()

        # --- 3. AISLAMIENTO Y DIMENSIONES (768) ---
        self.play(
            *[FadeOut(t) for t in tokens[1:]],
            *[FadeOut(p) for p in posiciones[1:]],
            tokens[0].animate.move_to(LEFT * 4.5 + UP * 0.8),
            posiciones[0].animate.move_to(LEFT * 4.5 + DOWN * 0.8)
        )

        # Vector Semántico (d=768)
        vec_token = Text("[ 0.10, -0.30, 0.20, ... ]", font_size=26, font="Monospace", color=BLACK)
        lbl_token = Text("Token Embedding (Significado)", font_size=16, weight=BOLD, color=GRAY).next_to(vec_token, UP, buff=0.1)
        grupo_vec_token = VGroup(lbl_token, vec_token).move_to(RIGHT * 1 + UP * 1.2)

        # Vector de Posición (d=768) - Valores exactos del texto
        vec_pos = Text("[ 0.05, -0.02, 0.01, ... ]", font_size=26, font="Monospace", color=BLACK)
        lbl_pos = Text("Position Embedding (Orden)", font_size=16, weight=BOLD, color=GRAY).next_to(vec_pos, UP, buff=0.1)
        grupo_vec_pos = VGroup(lbl_pos, vec_pos).move_to(RIGHT * 1 + DOWN * 0.4)

        signo_mas = Text("+", font_size=36, weight=BOLD, color=BLACK).move_to(RIGHT * 1 + UP * 0.4)
        nota_suma = Text("Suma elemento a elemento (Ambos de 768 dimensiones)", font_size=16, color=BLUE_D).next_to(vec_pos, DOWN, buff=0.5)

        flecha_token = Arrow(tokens[0].get_right(), grupo_vec_token.get_left(), color=GRAY, buff=0.2)
        flecha_pos = Arrow(posiciones[0].get_right(), grupo_vec_pos.get_left(), color=GRAY, buff=0.2)

        self.play(
            GrowArrow(flecha_token), FadeIn(grupo_vec_token),
            GrowArrow(flecha_pos), FadeIn(grupo_vec_pos),
            Write(signo_mas), FadeIn(nota_suma)
        )
        self.next_slide()

        # --- 4. RESULTADO FINAL DE LA SUMA ---
        linea_suma = Line(LEFT * 2, RIGHT * 4, color=GRAY).next_to(nota_suma, DOWN, buff=0.2)
        vec_comb = Text("[ 0.15, -0.32, 0.21, ... ]", font_size=26, font="Monospace", color=RED_D)
        lbl_comb = Text("Combined Vector (Listo para el Transformer)", font_size=18, weight=BOLD, color=GRAY).next_to(vec_comb, UP, buff=0.1)
        grupo_vec_comb = VGroup(lbl_comb, vec_comb).next_to(linea_suma, DOWN, buff=0.3)

        self.play(Create(linea_suma))
        self.play(
            ReplacementTransform(VGroup(vec_token.copy(), vec_pos.copy()), vec_comb),
            Write(lbl_comb)
        )
        self.play(Indicate(vec_comb, color=RED_B))
        self.next_slide()

        # --- 5. NOTA TÉCNICA FINAL: LA MATRIZ Y EL CONTEXT WINDOW ---
        # Limpiamos todo excepto el título
        elementos_a_borrar = [grupo_vec_token, grupo_vec_pos, signo_mas, nota_suma, linea_suma, grupo_vec_comb, tokens[0], posiciones[0], flecha_token, flecha_pos]
        self.play(*[FadeOut(el) for el in elementos_a_borrar])

        # Construimos la ilustración de la matriz
        titulo_matriz = Text("Tabla de Position Embeddings (GPT-2)", font_size=28, weight=BOLD, color=BLACK).move_to(UP * 2)
        
        # Filas de la matriz
        fila_0 = Text("Pos 0:    [ 0.05, -0.02, 0.01, ... (768) ]", font_size=22, font="Monospace", color=GRAY)
        fila_1 = Text("Pos 1:    [ 0.12,  0.45, -0.3, ... (768) ]", font_size=22, font="Monospace", color=GRAY)
        puntos = Text("...", font_size=30, color=BLACK).rotate(PI/2) # Puntos suspensivos verticales
        fila_n = Text("Pos 1023: [ -0.8,  0.11,  0.9, ... (768) ]", font_size=22, font="Monospace", color=GRAY)

        matriz = VGroup(fila_0, fila_1, puntos, fila_n).arrange(DOWN, buff=0.3).next_to(titulo_matriz, DOWN, buff=0.5)
        
        # Agregamos una llave (Brace) para indicar el block size
        llave = Brace(matriz, direction=LEFT, color=BLUE_D)
        texto_llave = Text("1024 filas\n(block_size)", font_size=20, color=BLUE_D).next_to(llave, LEFT, buff=0.2)

        # Nota final sobre el Context Window
        nota_context = Text("¡Este límite físico es el Context Window del modelo!", font_size=26, weight=BOLD, color=RED_D).next_to(matriz, DOWN, buff=0.6)

        self.play(Write(titulo_matriz))
        self.play(
            FadeIn(fila_0, shift=UP),
            FadeIn(fila_1, shift=UP)
        )
        self.play(Write(puntos))
        self.play(FadeIn(fila_n, shift=UP))
        self.play(GrowFromCenter(llave), Write(texto_llave))
        self.next_slide()

        self.play(Write(nota_context))
        self.play(Indicate(nota_context, color=RED_B, scale_factor=1.1))
        self.next_slide()

        self.limpiar_pantalla()
    def slide_layer_normalization(self):
        # --- 1. TÍTULO CON ÉNFASIS ---
        titulo_p1 = Text("Layer ", font_size=42, weight=BOLD, color=BLACK)
        titulo_p2 = Text("Normalization", font_size=42, weight=BOLD, color=BLUE_D) 
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 4, RIGHT * 4, color=GRAY).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)
        
        self.play(Write(titulo_completo), Create(linea))
        self.next_slide()

        # --- 2. EL PROBLEMA: INESTABILIDAD ---
        nota_caos = Text("Sin normalización: Los valores explotan o desaparecen", font_size=24, color=RED_D).move_to(UP * 1.5)
        
        vec_inestable = Text("[ 1045.2,  0.0001, -532.8, ... ]", font_size=32, font="Monospace", color=BLACK)
        
        self.play(Write(nota_caos))
        self.play(FadeIn(vec_inestable, shift=UP))
        self.play(Indicate(vec_inestable, color=RED, scale_factor=1.2))
        self.next_slide()

        # --- 3. LA SOLUCIÓN: ESTANDARIZACIÓN ---
        nota_estable = Text("Con LayerNorm: Se fuerza una Media=0 y Varianza=1", font_size=24, color=BLUE_D).move_to(UP * 1.5)
        
        vec_estable = Text("[  1.23,  -0.45,   0.89, ... ]", font_size=32, font="Monospace", color=BLUE_E)
        
        self.play(
            ReplacementTransform(nota_caos, nota_estable),
            ReplacementTransform(vec_inestable, vec_estable)
        )
        self.next_slide()

        # --- 4. LA FÓRMULA MATEMÁTICA ---
        self.play(FadeOut(nota_estable), FadeOut(vec_estable))
        
        # SOLUCIÓN EPSILON: Usamos substrings_to_isolate para encontrar las partes exactas
        formula = MathTex(
            r"\text{output} = \frac{\text{input} - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta",
            substrings_to_isolate=[r"\epsilon", r"\times \gamma + \beta"],
            color=BLACK
        ).scale(1.2).move_to(UP * 0.5)

        lbl_formula = Text("La Fórmula (por cada token):", font_size=20, weight=BOLD, color=GRAY).next_to(formula, UP, buff=0.5)

        self.play(Write(lbl_formula), FadeIn(formula))
        self.next_slide()

        # --- 5. EXPLICACIÓN DE LOS COMPONENTES ---
        # Resaltando Epsilon (Buscamos la parte exacta por su texto)
        parte_eps = formula.get_part_by_tex(r"\epsilon")
        caja_eps = SurroundingRectangle(parte_eps, color=RED_B, buff=0.05)
        nota_eps = Text("eps: Evita dividir por cero (ej. 0.00001)", font_size=18, color=RED_D).next_to(caja_eps, DOWN, buff=0.5)
        
        self.play(Create(caja_eps), FadeIn(nota_eps, shift=UP))
        self.next_slide()

        # Resaltando Gamma y Beta
        self.play(FadeOut(caja_eps), FadeOut(nota_eps))
        
        parte_params = formula.get_part_by_tex(r"\times \gamma + \beta")
        caja_params = SurroundingRectangle(parte_params, color=GREEN_D, buff=0.1)
        nota_params = Text("gamma / beta: Parámetros aprendidos\n¡Le dan flexibilidad al modelo!", font_size=18, color=GREEN_D).next_to(caja_params, DOWN, buff=0.5)

        self.play(Create(caja_params), FadeIn(nota_params, shift=UP))
        self.next_slide()

        # --- 6. CONTEXTO EN LA ARQUITECTURA (DIAGRAMA SIMPLIFICADO) ---
        self.play(
            *[FadeOut(m) for m in [lbl_formula, formula, caja_params, nota_params]]
        )

        # Textos a la izquierda
        nota_final = Text("Se aplica 2 veces por capa:", font_size=28, weight=BOLD, color=BLACK)
        paso_1 = Text("1. Antes de Attention", font_size=24, color=BLUE_D)
        paso_2 = Text("2. Antes de FFN (MLP)", font_size=24, color=BLUE_D)
        
        textos_izq = VGroup(nota_final, paso_1, paso_2).arrange(DOWN, aligned_edge=LEFT, buff=0.4).to_edge(LEFT, buff=1).shift(UP * 0.5)

        # Función auxiliar para crear bloques
        def crear_cajita(texto, bg_color, borde_color="#137B85", w=2.6, h=0.7):
            caja = RoundedRectangle(corner_radius=0.1, width=w, height=h, 
                                    fill_color=bg_color, fill_opacity=1, 
                                    stroke_color=borde_color, stroke_width=2)
            lbl = Text(texto, font_size=20, color=BLACK).move_to(caja.get_center())
            return VGroup(caja, lbl)

        # Construcción de los bloques
        b_in = crear_cajita("Input", "#D4EAF2")
        b_ln1 = crear_cajita("LayerNorm 1", "#EBEBEB")
        b_attn = crear_cajita("Attention", "#FFF566", "#F28C00")
        b_ln2 = crear_cajita("LayerNorm 2", "#EBEBEB")
        b_mlp = crear_cajita("MLP (FFN)", "#9DF28F")
        b_out = crear_cajita("Output", "#D4EAF2")

        bloques = VGroup(b_in, b_ln1, b_attn, b_ln2, b_mlp, b_out).arrange(DOWN, buff=0.4)
        
        flechas = VGroup(*[
            Arrow(bloques[i].get_bottom(), bloques[i+1].get_top(), buff=0.1, 
                  max_tip_length_to_length_ratio=0.15, color=GRAY) 
            for i in range(len(bloques)-1)
        ])

        diagrama_simplificado = VGroup(bloques, flechas)

        # SOLUCIÓN DIAGRAMA: Lo hacemos más pequeño (75%) y lo empujamos más a la izquierda (buff=3.5)
        diagrama_simplificado.scale(0.75).to_edge(RIGHT, buff=3.5).shift(DOWN * 0.2)

        # Animación de la estructura base
        self.play(Write(nota_final), FadeIn(diagrama_simplificado, shift=LEFT))
        self.next_slide()

        # Animación del Paso 1
        resalto_1 = SurroundingRectangle(b_ln1, color=RED, stroke_width=4, buff=0.05)
        self.play(
            FadeIn(paso_1, shift=RIGHT), 
            Create(resalto_1), 
            b_ln1[0].animate.set_fill("#FFD1D1")
        )
        self.next_slide()

        # Animación del Paso 2
        resalto_2 = SurroundingRectangle(b_ln2, color=RED, stroke_width=4, buff=0.05)
        self.play(
            FadeIn(paso_2, shift=RIGHT), 
            Create(resalto_2), 
            b_ln2[0].animate.set_fill("#FFD1D1")
        )
        self.next_slide()

        self.limpiar_pantalla()
    def slide_multihead_attention(self):
        # =========================================================================
        # --- TÍTULO PRINCIPAL (Se mantiene fijo en toda la diapositiva) ---
        # =========================================================================
        titulo = Text("Multi-Head Self-Attention", font_size=42, weight=BOLD, color=BLACK).to_edge(UP)
        subtitulo = Text("El corazón del Transformer", font_size=24, color=DARK_GRAY).next_to(titulo, DOWN)
        self.play(Write(titulo), FadeIn(subtitulo, shift=UP))
        self.next_slide()

        # Función auxiliar para limpiar la pantalla sin borrar los títulos
        def limpiar_escena():
            mobs_a_borrar = [m for m in self.mobjects if m not in [titulo, subtitulo]]
            if mobs_a_borrar:
                self.play(*[FadeOut(m) for m in mobs_a_borrar])

        # =========================================================================
        # --- ACTO 1: LA INTUICIÓN (Q, K, V) - ESPACIADO EXPANDIDO ---
        # =========================================================================
        nota_qkv = Text("Por cada token, el modelo genera 3 matrices:", 
                        font_size=24, color=GRAY_D).move_to(UP * 1.5)
        self.play(FadeIn(nota_qkv, shift=DOWN))

        def crear_tarjeta_premium(letra, nombre, pregunta, color_base):
            # Reducimos un poco el ancho (de 3.2 a 2.8) para ganar más espacio entre ellas
            caja = RoundedRectangle(
                corner_radius=0.3, width=2.8, height=2.2, 
                fill_color=color_base, fill_opacity=0.05, 
                stroke_color=color_base, stroke_width=2
            )
            header = Rectangle(
                width=2.8, height=0.6, fill_color=color_base, 
                fill_opacity=0.8, stroke_width=0
            ).move_to(caja.get_top(), aligned_edge=UP)
            
            # Recorte para que el header siga la redondez de la caja
            header = Intersection(header, caja, fill_opacity=0.8, color=color_base, stroke_width=0)
            
            txt_letra = Text(letra, font_size=32, weight=BOLD, color=WHITE).move_to(header.get_center())
            txt_nombre = Text(nombre, font_size=20, weight=BOLD, color=color_base).next_to(header, DOWN, buff=0.3)
            txt_pregunta = Text(pregunta, font_size=16, slant=ITALIC, color=BLACK).next_to(txt_nombre, DOWN, buff=0.2)
            
            return VGroup(caja, header, txt_letra, txt_nombre, txt_pregunta)

        tarjeta_q = crear_tarjeta_premium("Q", "Query", "¿Qué busco?", RED_D)
        tarjeta_k = crear_tarjeta_premium("K", "Key", "¿Qué tengo?", GREEN_D)
        tarjeta_v = crear_tarjeta_premium("V", "Value", "Contenido", BLUE_D)

        # --- CAMBIO CLAVE: buff=1.5 para máxima separación ---
        tarjetas_qkv = VGroup(tarjeta_q, tarjeta_k, tarjeta_v).arrange(RIGHT, buff=1.5).next_to(nota_qkv, DOWN, buff=0.8)

        self.play(
            LaggedStart(
                *[FadeIn(t, shift=UP) for t in tarjetas_qkv],
                lag_ratio=0.2,
                run_time=1.5
            )
        )
        self.next_slide()

        # Conexión elegante Q -> K con la distancia ampliada
        linea_conexion = Line(
            tarjeta_q.get_right(), 
            tarjeta_k.get_left(), 
            color=ORANGE, 
            stroke_width=2
        ).set_z_index(-1)

        dot_product_label = MathTex(r"Q \cdot K^T", color=ORANGE, font_size=30).next_to(linea_conexion, UP, buff=0.2)
        
        self.play(Create(linea_conexion), Write(dot_product_label))
        
        # Efecto de pulso
        self.play(
            linea_conexion.animate.set_stroke(width=5),
            Flash(linea_conexion.get_center(), color=ORANGE, flash_radius=0.5),
            Indicate(tarjeta_q[0], color=ORANGE, scale_factor=1.05),
            Indicate(tarjeta_k[0], color=ORANGE, scale_factor=1.05),
            run_time=1
        )
        self.play(linea_conexion.animate.set_stroke(width=2))
        
        self.next_slide()
        limpiar_escena()

        # =========================================================================
        # --- ACTO 2: LA FÓRMULA MATEMÁTICA PASO A PASO ---
        # =========================================================================
        lbl_formula = Text("La Matemática:", font_size=28, weight=BOLD, color=BLUE_E).move_to(UP * 1.2)
        
        # TRUCO: Usamos {V} al final de la ecuación para que Manim la diferencie de la primera V
        formula = MathTex(
            r"\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d}} \right) {V}",
            substrings_to_isolate=[r"Q K^T", r"\sqrt{d}", r"\text{softmax}", r"{V}"],
            color=BLACK, font_size=52
        ).next_to(lbl_formula, DOWN, buff=1)

        self.play(Write(lbl_formula), FadeIn(formula, shift=UP))
        self.next_slide()

        partes = [
            (r"Q K^T", "1. Similitud: ¿Qué tanto se parecen Q y K?", RED_D),
            (r"\sqrt{d}", "2. Escalado: Evita que los números exploten", ORANGE),
            (r"\text{softmax}", "3. Softmax: Convierte a porcentajes (0 a 100%)", PURPLE_D),
            (r"{V}", "4. Valor: Se aplica el % al contenido real", BLUE_D)
        ]

        caja_actual = None
        texto_actual = None

        for tex_string, texto, color in partes:
            parte_tex = formula.get_part_by_tex(tex_string)
            
            nueva_caja = SurroundingRectangle(parte_tex, color=color, buff=0.1, stroke_width=4)
            nuevo_texto = Text(texto, font_size=24, weight=BOLD, color=color).next_to(formula, DOWN, buff=1.5)
            
            if caja_actual is not None:
                self.play(FadeOut(caja_actual), FadeOut(texto_actual, shift=UP))
            self.play(Create(nueva_caja), FadeIn(nuevo_texto, shift=UP))
            self.next_slide()
            
            caja_actual = nueva_caja
            texto_actual = nuevo_texto

        limpiar_escena()

        # =========================================================================
        # --- ACTO 3: EJEMPLO PASO A PASO CON MATRICES ---
        # =========================================================================
        
        # --- PASO 1: Presentación de Q, K, V ---
        explicacion = Text("Paso 1: Tenemos las matrices Q, K y V (Ej. 2 tokens, 3 dimensiones)", font_size=24, color=DARK_GRAY).move_to(UP * 1.5)
        self.play(FadeIn(explicacion, shift=DOWN))

        # Valores inventados sencillos para que la matemática cuadre fácil
        val_q = ["1", "0", "2",  "0", "1", "0"]
        val_k = ["1", "1", "0",  "0", "1", "1"]
        val_v = ["3", "0", "0",  "0", "3", "0"]

        # Construcción visual usando tus funciones y colores
        mat_q = self.crear_matriz_bloques(2, 3, color_fondo=RUST_COLOR, valores=val_q)
        mat_k = self.crear_matriz_bloques(2, 3, color_fondo=SOFT_GREEN, valores=val_k)
        mat_v = self.crear_matriz_bloques(2, 3, color_fondo=HIGHLIGHT_COLOR, valores=val_v)

        lbl_q = MathTex("Q", color=BLACK).next_to(mat_q, UP)
        lbl_k = MathTex("K", color=BLACK).next_to(mat_k, UP)
        lbl_v = MathTex("V", color=BLACK).next_to(mat_v, UP)

        grupo_q = VGroup(lbl_q, mat_q)
        grupo_k = VGroup(lbl_k, mat_k)
        grupo_v = VGroup(lbl_v, mat_v)

        # Ubicamos Q, K, V alineados horizontalmente
        matrices_iniciales = VGroup(grupo_q, grupo_k, grupo_v).arrange(RIGHT, buff=1.2).next_to(explicacion, DOWN, buff=1)

        self.play(LaggedStart(FadeIn(grupo_q), FadeIn(grupo_k), FadeIn(grupo_v), lag_ratio=0.3))
        self.next_slide()

        # --- PASO 2: Q * K^T (Scores) ---
        explicacion_2 = Text("Paso 2: Multiplicamos Q por K transpuesta (Scores de Similitud)", font_size=24, color=DARK_GRAY).move_to(explicacion)
        
        # K transpuesta (ahora es 3x2)
        val_kt = ["1", "0",  "1", "1",  "0", "1"]
        mat_kt = self.crear_matriz_bloques(3, 2, color_fondo=SOFT_GREEN, valores=val_kt)
        lbl_kt = MathTex("K^T", color=BLACK).next_to(mat_kt, UP)
        grupo_kt = VGroup(lbl_kt, mat_kt)

        # Resultado de Q * K^T (Matriz 2x2)
        val_scores = ["1", "2",  "1", "1"]
        mat_scores = self.crear_matriz_bloques(2, 2, color_fondo=GHOST_COLOR, valores=val_scores)
        lbl_scores = MathTex(r"Q \cdot K^T", color=BLACK).next_to(mat_scores, UP)
        grupo_scores = VGroup(lbl_scores, mat_scores)

        # Ecuación visual: Q x K^T = Scores
        ec_paso2 = VGroup(grupo_q, MathTex(r"\times", color=BLACK).scale(1.2), grupo_kt, MathTex("=", color=BLACK).scale(1.2), grupo_scores)
        ec_paso2.arrange(RIGHT, buff=0.5).next_to(explicacion_2, DOWN, buff=0.8)

        self.play(
            Transform(explicacion, explicacion_2),
            FadeOut(grupo_k), FadeOut(grupo_v),
            grupo_q.animate.move_to(ec_paso2[0]),
            FadeIn(ec_paso2[1]),
            FadeIn(ec_paso2[2], shift=LEFT),
            FadeIn(ec_paso2[3]),
            FadeIn(ec_paso2[4], shift=LEFT)
        )
        self.next_slide()

        # --- PASO 3: Softmax ---
        explicacion_3 = Text("Paso 3: Escalamos y aplicamos Softmax (Probabilidades %)", font_size=24, color=DARK_GRAY).move_to(explicacion)

        # Valores aplicando Softmax (aproximados para 1,2 y 1,1)
        val_soft = ["0.3", "0.7",  "0.5", "0.5"]
        mat_soft = self.crear_matriz_bloques(2, 2, color_fondo=TOKEN_COLOR_2, valores=val_soft)
        lbl_soft = MathTex("Softmax", color=BLACK).next_to(mat_soft, UP)
        grupo_soft = VGroup(lbl_soft, mat_soft)

        # Animamos la transformación de Scores a Softmax
        self.play(
            Transform(explicacion, explicacion_3),
            FadeOut(ec_paso2[0:4]), # Ocultamos Q, x, K^T, =
            grupo_scores.animate.move_to(LEFT * 2.5 + DOWN * 0.5)
        )
        
        flecha_soft = Arrow(grupo_scores.get_right(), grupo_scores.get_right() + RIGHT * 1.5, color=DARK_GRAY, buff=0.2)
        grupo_soft.next_to(flecha_soft, RIGHT, buff=0.2).align_to(grupo_scores, DOWN)

        self.play(Create(flecha_soft), TransformFromCopy(grupo_scores, grupo_soft))
        self.next_slide()

        # --- PASO 4: Multiplicar por V ---
        explicacion_4 = Text("Paso 4: Multiplicamos por V (Matriz de Contexto Final)", font_size=24, color=DARK_GRAY).move_to(explicacion)

        # Resultado final (Matriz 2x3): Softmax * V
        val_final = ["0.9", "2.1", "0",  "1.5", "1.5", "0"]
        mat_final = self.crear_matriz_bloques(2, 3, color_fondo=TOKEN_COLOR_FINAL, valores=val_final)
        lbl_final = MathTex("Z", color=BLACK).next_to(mat_final, UP)
        grupo_final = VGroup(lbl_final, mat_final)

        # Recreamos V para la ecuación final
        grupo_v_paso4 = VGroup(MathTex("V", color=BLACK), self.crear_matriz_bloques(2, 3, color_fondo=HIGHLIGHT_COLOR, valores=val_v))
        grupo_v_paso4[1].next_to(grupo_v_paso4[0], DOWN)

        # Ecuación visual: Softmax x V = Z
        grupo_soft.generate_target()
        ec_paso4 = VGroup(grupo_soft.target, MathTex(r"\times", color=BLACK).scale(1.2), grupo_v_paso4, MathTex("=", color=BLACK).scale(1.2), grupo_final)
        ec_paso4.arrange(RIGHT, buff=0.4).next_to(explicacion_4, DOWN, buff=1)

        self.play(
            Transform(explicacion, explicacion_4),
            FadeOut(grupo_scores), FadeOut(flecha_soft),
            MoveToTarget(grupo_soft),
            FadeIn(ec_paso4[1]),
            FadeIn(ec_paso4[2], shift=LEFT),
            FadeIn(ec_paso4[3]),
            FadeIn(ec_paso4[4], shift=LEFT)
        )
        self.next_slide()

        # Limpiamos antes de entrar al Acto 4 (Multi-head)
        limpiar_escena()

        # =========================================================================
        # --- ACTO 4: MULTI-HEAD (EL PARALELISMO) ---
        # =========================================================================
        desc_mh = Text("El modelo se divide en 'cabezas' para analizar múltiples contextos a la vez.", font_size=24, color=BLACK).move_to(UP * 1.5)
        self.play(Write(desc_mh))

        caja_768 = Rectangle(width=10, height=1, fill_color=LIGHT_GRAY, fill_opacity=0.6, stroke_color=BLACK, stroke_width=2)
        txt_768 = Text("Vectores Originales (Ej. 768 dimensiones)", font_size=26, color=BLACK).move_to(caja_768.get_center())
        grupo_768 = VGroup(caja_768, txt_768).next_to(desc_mh, DOWN, buff=1.2)

        self.play(FadeIn(grupo_768, shift=DOWN))
        self.next_slide()

        colores_cabezas = [
            RED_E, BLUE_E, GREEN_E, TEAL_E, 
            ORANGE, PURPLE_E, MAROON_E, GOLD_E, 
            PINK, LIGHT_PINK, DARK_BROWN, GRAY_BROWN
        ]
        
        ancho_cabeza = 10 / 12
        
        cabezas = VGroup(*[
            Rectangle(width=ancho_cabeza, height=1.5, fill_color=colores_cabezas[i], fill_opacity=0.8, stroke_color=BLACK, stroke_width=1.5) 
            for i in range(12)
        ]).arrange(RIGHT, buff=0.05).move_to(grupo_768.get_center())
        
        txt_cabezas = Text("12 Cabezas de Atención (64 dims c/u)", font_size=24, weight=BOLD, color=BLUE_E).next_to(cabezas, DOWN, buff=0.6)

        self.play(
            ReplacementTransform(caja_768, cabezas),
            FadeOut(txt_768),
            FadeIn(txt_cabezas, shift=UP)
        )
        self.next_slide()

        self.play(
            LaggedStart(
                *[Indicate(cabeza, scale_factor=1.2, color=BLACK) for cabeza in cabezas],
                lag_ratio=0.05,
                run_time=1.5
            )
        )
        
        nota_final = Text("Cada cabeza hace su propia multiplicación de matrices Q, K, V", font_size=24, color=DARK_GRAY).next_to(txt_cabezas, DOWN, buff=0.4)
        self.play(Write(nota_final))
        self.next_slide()

        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects])

    def slide_causal_masking(self):
        # =========================================================================
        # --- TÍTULO Y LIMPIEZA INICIAL ---
        # =========================================================================
        titulo, linea = self.crear_titulo("Causal Masking", palabra_clave="Causal Masking", color_clave=HIGHLIGHT_COLOR)
        self.play(Write(titulo), Create(linea))
        
        subtitulo = Text("El 'Triángulo de No Mirar' (No Peeking)", font_size=32, color=DARK_GRAY)
        frase = VGroup(
            Text("Secuencia: ", font_size=32, color=BLACK, weight=BOLD),
            Text("[better] [a] [witty] [fool]", font_size=32, color=TOKEN_COLOR_FINAL)
        ).arrange(RIGHT)

        grupo_intro = VGroup(subtitulo, frase).arrange(DOWN, buff=0.8).move_to(ORIGIN)

        self.play(FadeIn(grupo_intro, shift=UP))
        self.next_slide()
        
        self.play(FadeOut(grupo_intro))

        # =========================================================================
        # --- PASO 1: LA MATRIZ DE ATENCIÓN Y EL PROBLEMA ---
        # =========================================================================
        texto_explicativo = Text("Matriz de Atención: ¡El modelo ve el futuro!", font_size=24, color=ALERT_COLOR).next_to(linea, DOWN, buff=0.3)
        self.play(FadeIn(texto_explicativo))

        val_scores = [
            "2.1", "1.5", "0.8", "-0.5",
            "0.5", "3.0", "1.2", "0.9",
            "-1.0", "0.2", "2.5", "1.8",
            "0.1", "-0.2", "0.4", "2.2"
        ]
        mat_scores = self.crear_matriz_bloques(4, 4, color_fondo=SOFT_BG, valores=val_scores)
        
        tokens = ["better", "a", "witty", "fool"]
        etiquetas_filas = VGroup(*[Text(t, font_size=18, color=DARK_GRAY) for t in tokens]).arrange(DOWN, buff=0.6).next_to(mat_scores, LEFT, buff=0.4)
        etiquetas_cols = VGroup(*[Text(t, font_size=18, color=DARK_GRAY).rotate(PI/4) for t in tokens]).arrange(RIGHT, buff=0.4).next_to(mat_scores, UP, buff=0.2)

        # Matriz estática en el centro exacto
        grupo_atencion = VGroup(mat_scores, etiquetas_filas, etiquetas_cols).move_to(ORIGIN)
        
        self.play(FadeIn(grupo_atencion, shift=UP))
        self.next_slide()

        # Resaltamos directamente la "trampa" en rojo (el triángulo superior)
        animaciones_trampa = []
        for i in range(4):
            for j in range(4):
                if j > i: 
                    bloque = mat_scores[i][j]
                    animaciones_trampa.append(bloque[0].animate.set_fill(ALERT_COLOR, opacity=0.3))
                    animaciones_trampa.append(bloque[0].animate.set_stroke(ALERT_COLOR, width=3))

        self.play(*animaciones_trampa, run_time=1.5)
        self.next_slide()

        # =========================================================================
        # --- PASO 2: APLICANDO -INFINITO (Sin mover la matriz) ---
        # =========================================================================
        # Actualizamos el texto para indicar qué estamos haciendo con esa zona roja
        texto_infinito = Text("Solución: Enmascarar las posiciones futuras con -Infinito", font_size=24, color=RUST_COLOR).move_to(texto_explicativo)
        self.play(Transform(texto_explicativo, texto_infinito))

        animaciones_infinito = []
        for i in range(4):
            for j in range(4):
                if j > i:
                    bloque_score = mat_scores[i][j]
                    nuevo_texto = MathTex(r"-\infty", font_size=32, color=WHITE).move_to(bloque_score.get_center())
                    # Rellenamos sólido y cambiamos el texto en el mismo lugar
                    animaciones_infinito.append(bloque_score[0].animate.set_fill(RUST_COLOR, opacity=1).set_stroke(BLACK, width=1))
                    animaciones_infinito.append(Transform(bloque_score[1], nuevo_texto))

        # Reproducimos el castigo matemático sin mover nada
        self.play(*animaciones_infinito, run_time=1.5)
        self.next_slide()

        # =========================================================================
        # --- PASO 3: EL EFECTO DEL SOFTMAX ---
        # =========================================================================
        texto_softmax = Text("Después del Softmax: exp(-inf) = 0", font_size=24, color=TOKEN_COLOR_FINAL).move_to(texto_explicativo)
        self.play(Transform(texto_explicativo, texto_softmax))

        animaciones_ceros = []
        for i in range(4):
            for j in range(4):
                if j > i:
                    bloque_score = mat_scores[i][j]
                    cero_texto = Text("0", font_size=24, color=DARK_GRAY).move_to(bloque_score.get_center())
                    # Apagamos las celdas prohibidas
                    animaciones_ceros.append(bloque_score[0].animate.set_fill(DARK_GRAY, opacity=0.2).set_stroke(DARK_GRAY, width=1))
                    animaciones_ceros.append(Transform(bloque_score[1], cero_texto))
                else:
                    bloque_score = mat_scores[i][j]
                    bloque_score[0].generate_target()
                    # Iluminamos las celdas válidas simulando el Softmax
                    bloque_score[0].target.set_fill(TOKEN_COLOR_FINAL, opacity=0.6)
                    animaciones_ceros.append(MoveToTarget(bloque_score[0]))

        self.play(*animaciones_ceros, run_time=1.5)
        
        # Nota final asegurada con buen margen inferior
        nota_final = Text("Cada posición solo atiende al pasado.", font_size=24, weight=BOLD, color=BLACK).next_to(grupo_atencion, DOWN, buff=0.5)
        self.play(FadeIn(nota_final, shift=UP))
        self.next_slide()

        # Limpiamos todo
        self.limpiar_pantalla()


    def slide_mlp_layer(self):

        # =========================================================================
        # --- TÍTULO ---
        # =========================================================================
        titulo, linea = self.crear_titulo("La Capa MLP", palabra_clave="MLP", color_clave=HIGHLIGHT_COLOR)
        self.play(Write(titulo), Create(linea))
        
        # =========================================================================
        # --- ACTO 1: DIAGRAMA DE RED NEURONAL (MÁS PROFUNDO) ---
        # =========================================================================
        texto_red = Text("1. Arquitectura: Expansión (3072) y Compresión (768)", font_size=26, weight=BOLD, color=BLACK).next_to(linea, DOWN, buff=0.3)
        self.play(FadeIn(texto_red))

        # Configuración de los nodos
        radio_nodo = 0.15
        color_linea = "#ECF0F1" # Gris muy claro para no saturar

        # Creamos las columnas de nodos
        capa_in = VGroup(*[Circle(radius=radio_nodo, fill_color=SOFT_BG, fill_opacity=1, stroke_color=DARK_GRAY, stroke_width=2) for _ in range(4)]).arrange(DOWN, buff=0.2)
        hid_1 = VGroup(*[Circle(radius=radio_nodo, fill_color=HIGHLIGHT_COLOR, fill_opacity=1, stroke_color=DARK_GRAY, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)
        hid_2 = VGroup(*[Circle(radius=radio_nodo, fill_color=HIGHLIGHT_COLOR, fill_opacity=1, stroke_color=DARK_GRAY, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)
        hid_3 = VGroup(*[Circle(radius=radio_nodo, fill_color=HIGHLIGHT_COLOR, fill_opacity=1, stroke_color=DARK_GRAY, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)
        capa_out = VGroup(*[Circle(radius=radio_nodo, fill_color=TOKEN_COLOR_FINAL, fill_opacity=1, stroke_color=DARK_GRAY, stroke_width=2) for _ in range(4)]).arrange(DOWN, buff=0.2)

        # Distribuimos la red horizontalmente
        red = VGroup(capa_in, hid_1, hid_2, hid_3, capa_out).arrange(RIGHT, buff=0.8).move_to(DOWN * 0.5)

        # Dibujamos las conexiones
        con_in_h1 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=color_linea) for n1 in capa_in for n2 in hid_1])
        con_h1_h2 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=color_linea) for n1 in hid_1 for n2 in hid_2])
        con_h2_h3 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=color_linea) for n1 in hid_2 for n2 in hid_3])
        con_h3_out = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=color_linea) for n1 in hid_3 for n2 in capa_out])

        # Etiquetas
        lbl_in = Text("Entrada\n(768)", font_size=16, color=BLACK).next_to(capa_in, DOWN, buff=0.2)
        lbl_hid = Text("Capa Oculta Expansiva\n(3072 dimensiones)", font_size=16, color=HIGHLIGHT_COLOR, weight=BOLD).next_to(hid_2, DOWN, buff=0.2)
        lbl_out = Text("Salida\n(768)", font_size=16, color=BLACK).next_to(capa_out, DOWN, buff=0.2)

        # Agrupamos todo para facilitar la limpieza
        grupo_red_completo = VGroup(con_in_h1, con_h1_h2, con_h2_h3, con_h3_out, red, lbl_in, lbl_hid, lbl_out)

        # Animación secuencial
        self.play(FadeIn(capa_in), FadeIn(lbl_in))
        self.play(Create(con_in_h1), FadeIn(hid_1))
        self.play(Create(con_h1_h2), FadeIn(hid_2), FadeIn(lbl_hid))
        self.play(Create(con_h2_h3), FadeIn(hid_3))
        self.play(Create(con_h3_out), FadeIn(capa_out), FadeIn(lbl_out))
        self.next_slide()

        # Limpiamos Acto 1
        self.play(FadeOut(grupo_red_completo), FadeOut(texto_red))

        # =========================================================================
        # --- ACTO 2: REPRESENTACIÓN MATRICIAL (MEJORADA) ---
        # =========================================================================
        texto_matriz = Text("2. El Flujo Matricial (Multiplicación de Bloques)", font_size=26, weight=BOLD, color=BLACK).move_to(texto_red)
        self.play(FadeIn(texto_matriz))

        # Paso 1: Expansión
        eq_exp = MathTex(r"X \cdot W_1 = H", color=BLACK, font_size=36).move_to(UP * 1.2)
        
        # Bloques proporcionales visualmente
        # X: [Tokens x 768] (Alto, estrecho)
        bloque_x = self.crear_bloque("X\n[L x 768]", SOFT_BG, BLACK, 1.0, 1.5)
        signo_por = MathTex(r"\times", color=BLACK).scale(1.2)
        # W1: [768 x 3072] (Bajo, muy ancho) - El alto encaja con el ancho de X
        bloque_w1 = self.crear_bloque("W_1\n[768 x 3072]", HIGHLIGHT_COLOR, WHITE, 3.5, 1.0)
        signo_igual = MathTex(r"=", color=BLACK).scale(1.2)
        # H: [L x 3072] (Alto y ancho)
        bloque_h = self.crear_bloque("H\n[L x 3072]", SOFT_BG, BLACK, 3.5, 1.5)

        flujo_expansion = VGroup(bloque_x, signo_por, bloque_w1, signo_igual, bloque_h).arrange(RIGHT, buff=0.3).next_to(eq_exp, DOWN, buff=0.5)

        self.play(Write(eq_exp))
        self.play(FadeIn(flujo_expansion, shift=UP))
        self.next_slide()

        # Transición a Compresión
        eq_comp = MathTex(r"H_{act} \cdot W_2 = Y", color=BLACK, font_size=36).move_to(eq_exp)
        
        # H_act: [L x 3072]
        bloque_h_act = self.crear_bloque("H_{act}\n[L x 3072]", SOFT_BG, BLACK, 3.5, 1.5)
        # W2: [3072 x 768] (Alto, estrecho)
        bloque_w2 = self.crear_bloque("W_2\n[3072 x 768]", HIGHLIGHT_COLOR, WHITE, 1.0, 3.5)
        # Y: [L x 768] (Regresa a tamaño original)
        bloque_y = self.crear_bloque("Y\n[L x 768]", TOKEN_COLOR_FINAL, WHITE, 1.0, 1.5)

        flujo_compresion = VGroup(bloque_h_act, signo_por.copy(), bloque_w2, signo_igual.copy(), bloque_y).arrange(RIGHT, buff=0.3).next_to(eq_comp, DOWN, buff=0.5)

        self.play(
            Transform(eq_exp, eq_comp),
            Transform(flujo_expansion, flujo_compresion)
        )
        self.next_slide()

        # Limpiamos Acto 2
        self.play(FadeOut(eq_exp), FadeOut(flujo_expansion), FadeOut(texto_matriz))

        # =========================================================================
        # --- ACTO 3: FÓRMULA GELU Y GRÁFICA (AJUSTADA) ---
        # =========================================================================
        texto_formula = Text("3. Activación GELU: Matemáticas del Suavizado", font_size=26, weight=BOLD, color=BLACK).move_to(texto_red)
        self.play(FadeIn(texto_formula))

        # Fórmula
        formula_gelu = MathTex(
            r"\text{GELU}(x) = 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3)\right)\right)",
            color=BLACK, font_size=32
        ).next_to(texto_formula, DOWN, buff=0.4)

        self.play(Write(formula_gelu))

        # Ejes
        ejes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-1, 4, 1],
            x_length=7,
            y_length=4,
            axis_config={"color": DARK_GRAY, "include_numbers": True, "font_size": 16}
        ).move_to(DOWN * 1)

        # Curvas
        curva_gelu = ejes.plot(lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))), color=RUST_COLOR, stroke_width=4)
        curva_lineal = ejes.plot(lambda x: x, color=DARK_GRAY, stroke_width=2).set_opacity(0.3)
        linea_cero = ejes.get_horizontal_line(ejes.c2p(4, 0), color=DARK_GRAY, line_func=DashedLine)

        self.play(Create(ejes), Create(linea_cero))
        self.play(Create(curva_lineal), run_time=1)
        self.play(Create(curva_gelu), run_time=2)

        # Texto "Suavizado" reposicionado para que no se corte (Arriba a la izquierda)
        # Lo ponemos apuntando a la curva negativa pero desde arriba
        flecha_negativa = Arrow(start=LEFT * 3.5 + UP * 0.5, end=ejes.c2p(-2, -0.1), color=ALERT_COLOR, buff=0.1)
        texto_negativo = Text("Se suaviza\nhacia 0", font_size=18, color=ALERT_COLOR).next_to(flecha_negativa, UP, buff=0.1)

        # Agregamos texto para el lado positivo también, para balancear
        flecha_positiva = Arrow(start=RIGHT * 2 + UP * 1.5, end=ejes.c2p(2, 2.1), color=TOKEN_COLOR_FINAL, buff=0.1)
        texto_positivo = Text("Comportamiento\ncasi lineal", font_size=18, color=TOKEN_COLOR_FINAL).next_to(flecha_positiva, UP, buff=0.1)

        self.play(
            GrowArrow(flecha_negativa), FadeIn(texto_negativo),
            GrowArrow(flecha_positiva), FadeIn(texto_positivo)
        )
        self.next_slide()

        # Limpieza total
        self.limpiar_pantalla()

    def slide_capa_transformer(self):
        # =========================================================================
        # --- CONFIGURACIÓN DE COLORES Y ESCALA GLOBAL ---
        # =========================================================================
        COLOR_LINEA = "#138D90"
        COLOR_IN_OUT = "#D0E8EC"
        COLOR_LN = "#EEEEEE"
        COLOR_ATTN_BG = "#FFF23D"
        COLOR_ATTN_BORD = "#F28C00"
        COLOR_MLP_BG = "#85EA7E"
        COLOR_MLP_BORD = "#28A745"
        
        ALERT_COLOR = RED 
        RUST_COLOR = MAROON
        
        escala = 0.65 

        def crear_nodo(texto, bg_color, borde_color, ancho=2.5 * escala, alto=0.8 * escala):
            caja = RoundedRectangle(corner_radius=0.15 * escala, width=ancho, height=alto, 
                                    fill_color=bg_color, fill_opacity=1, 
                                    stroke_color=borde_color, stroke_width=2.5 * escala)
            txt = Text(texto, font_size=20 * escala, color=BLACK)
            return VGroup(caja, txt)

        # =========================================================================
        # --- TÍTULO ---
        # =========================================================================
        titulo, linea = self.crear_titulo("Arquitectura: Transformer Layer", palabra_clave="Transformer Layer", color_clave=COLOR_LINEA)
        self.play(Write(titulo), Create(linea))
        
        # CORRECCIÓN 1: Movimos la base de las explicaciones más a la izquierda.
        # Antes estaba en RIGHT * 3.5. Ahora en RIGHT * 0.5 (puedes ajustarlo entre 0 y 1.5 según prefieras).
        pos_explicacion = RIGHT * 0.5 + UP * 1.5 

        # =========================================================================
        # --- COLUMNA IZQUIERDA: CONSTRUCCIÓN DEL DIAGRAMA ---
        # =========================================================================
        nodo_input = crear_nodo("Input", COLOR_IN_OUT, COLOR_LINEA).move_to(UP * 2.5 * escala + LEFT * 3.5)
        
        add_1 = VGroup(Circle(radius=0.25 * escala, fill_color=WHITE, fill_opacity=1, stroke_color=BLACK, stroke_width=2 * escala),
                       Text("+", font_size=24 * escala, color=BLACK, weight=BOLD)).move_to(UP * 0.2 * escala + LEFT * 3.5)
        
        add_2 = VGroup(Circle(radius=0.25 * escala, fill_color=WHITE, fill_opacity=1, stroke_color=BLACK, stroke_width=2 * escala),
                       Text("+", font_size=24 * escala, color=BLACK, weight=BOLD)).move_to(DOWN * 2.2 * escala + LEFT * 3.5)
        
        nodo_output = crear_nodo("Output", COLOR_IN_OUT, COLOR_LINEA).move_to(DOWN * 3.3 * escala + LEFT * 3.5)

        X_BLOQUES = 2.5 * escala
        X_LATERAL = LEFT * 3.5 + RIGHT * X_BLOQUES

        nodo_ln1 = crear_nodo("ln_1", COLOR_LN, COLOR_LINEA).move_to(UP * 1.4 * escala + X_LATERAL)
        nodo_attn = crear_nodo("attn", COLOR_ATTN_BG, COLOR_ATTN_BORD).move_to(UP * 0.2 * escala + X_LATERAL)
        
        nodo_ln2 = crear_nodo("ln_2", COLOR_LN, COLOR_LINEA).move_to(DOWN * 1.0 * escala + X_LATERAL)
        nodo_mlp = crear_nodo("mlp", COLOR_MLP_BG, COLOR_MLP_BORD).move_to(DOWN * 2.2 * escala + X_LATERAL)

        # --- RUTAS ORTOGONALES ---
        S_WIDTH = 2.5 * escala

        linea_baja_input = Line(nodo_input.get_bottom(), nodo_input.get_bottom() + DOWN * 0.3 * escala, stroke_color=COLOR_LINEA, stroke_width=S_WIDTH)
        punto_split_1 = linea_baja_input.get_end()
        flecha_res_1 = Arrow(punto_split_1, add_1.get_top(), buff=0, color=COLOR_LINEA, stroke_width=S_WIDTH, max_tip_length_to_length_ratio=0.15)
        txt_res_1 = Text("Residual Stream", font_size=12 * escala, color=COLOR_LINEA, slant=ITALIC).rotate(PI/2).next_to(flecha_res_1, LEFT, buff=0.1)

        esquina_1 = [nodo_ln1.get_center()[0], punto_split_1[1], 0]
        linea_der_1 = Line(punto_split_1, esquina_1, stroke_color=COLOR_LINEA, stroke_width=S_WIDTH)
        flecha_ln1 = Arrow(esquina_1, nodo_ln1.get_top(), buff=0, color=COLOR_LINEA, stroke_width=S_WIDTH)
        flecha_attn = Arrow(nodo_ln1.get_bottom(), nodo_attn.get_top(), buff=0, color=COLOR_LINEA, stroke_width=S_WIDTH)
        flecha_retorno_1 = Arrow(nodo_attn.get_left(), add_1.get_right(), buff=0, color=COLOR_LINEA, stroke_width=S_WIDTH)

        linea_baja_add1 = Line(add_1.get_bottom(), add_1.get_bottom() + DOWN * 0.3 * escala, stroke_color=COLOR_LINEA, stroke_width=S_WIDTH)
        punto_split_2 = linea_baja_add1.get_end()
        flecha_res_2 = Arrow(punto_split_2, add_2.get_top(), buff=0, color=COLOR_LINEA, stroke_width=S_WIDTH, max_tip_length_to_length_ratio=0.15)
        txt_res_2 = Text("Residual Stream", font_size=12 * escala, color=COLOR_LINEA, slant=ITALIC).rotate(PI/2).next_to(flecha_res_2, LEFT, buff=0.1)

        esquina_2 = [nodo_ln2.get_center()[0], punto_split_2[1], 0]
        linea_der_2 = Line(punto_split_2, esquina_2, stroke_color=COLOR_LINEA, stroke_width=S_WIDTH)
        flecha_ln2 = Arrow(esquina_2, nodo_ln2.get_top(), buff=0, color=COLOR_LINEA, stroke_width=S_WIDTH)
        flecha_mlp = Arrow(nodo_ln2.get_bottom(), nodo_mlp.get_top(), buff=0, color=COLOR_LINEA, stroke_width=S_WIDTH)
        flecha_retorno_2 = Arrow(nodo_mlp.get_left(), add_2.get_right(), buff=0, color=COLOR_LINEA, stroke_width=S_WIDTH)

        flecha_out = Arrow(add_2.get_bottom(), nodo_output.get_top(), buff=0, color=COLOR_LINEA, stroke_width=S_WIDTH)

        linea_punteada_1 = DashedLine(start=UP*1.6*escala + X_LATERAL + RIGHT*1.8*escala, end=DOWN*0.2*escala + X_LATERAL + RIGHT*1.8*escala, color=DARK_GRAY, stroke_width=2*escala)
        txt_bloque_attn = Text("Attention\nBlock", font_size=16 * escala, color=BLACK, weight=BOLD).next_to(linea_punteada_1, RIGHT, buff=0.2*escala)
        
        linea_punteada_2 = DashedLine(start=DOWN*0.8*escala + X_LATERAL + RIGHT*1.8*escala, end=DOWN*2.6*escala + X_LATERAL + RIGHT*1.8*escala, color=DARK_GRAY, stroke_width=2*escala)
        txt_bloque_mlp = Text("MLP\nBlock", font_size=16 * escala, color=BLACK, weight=BOLD).next_to(linea_punteada_2, RIGHT, buff=0.2*escala)

        # --- ANIMACIÓN SECUENCIAL DEL DIAGRAMA FIJO ---
        self.play(FadeIn(nodo_input, shift=DOWN * escala))
        self.play(Create(linea_baja_input))
        self.play(Create(linea_der_1), Create(flecha_ln1))
        self.play(FadeIn(nodo_ln1, shift=DOWN * escala))
        self.play(Create(flecha_attn), FadeIn(nodo_attn, shift=DOWN * escala))
        self.play(Create(flecha_res_1), FadeIn(txt_res_1))
        self.play(Create(flecha_retorno_1), FadeIn(add_1, scale=0.5))
        self.play(Create(linea_punteada_1), FadeIn(txt_bloque_attn, shift=LEFT * escala))
        
        self.play(Create(linea_baja_add1))
        self.play(Create(linea_der_2), Create(flecha_ln2))
        self.play(FadeIn(nodo_ln2, shift=DOWN * escala))
        self.play(Create(flecha_mlp), FadeIn(nodo_mlp, shift=DOWN * escala))
        self.play(Create(flecha_res_2), FadeIn(txt_res_2))
        self.play(Create(flecha_retorno_2), FadeIn(add_2, scale=0.5))
        self.play(Create(linea_punteada_2), FadeIn(txt_bloque_mlp, shift=LEFT * escala))
        
        self.play(Create(flecha_out), FadeIn(nodo_output, shift=UP * escala))
        self.next_slide()

        # =========================================================================
        # --- COLUMNA DERECHA: EXPLICACIÓN MODULAR ---
        # =========================================================================
        
        # --- CONCEPTO 1: Residual Stream ---
        # CORRECCIÓN 2: Envolvemos toda la columna izquierda (flechas, nodos add y texto) para un bloque limpio.
        grupo_residual = VGroup(flecha_res_1, txt_res_1, add_1, linea_baja_add1, flecha_res_2, txt_res_2, add_2)
        caja1 = SurroundingRectangle(grupo_residual, color=ALERT_COLOR, buff=0.2)
        
        head1 = Text("Residual Stream (Autopista):", font_size=30, weight=BOLD, color=COLOR_LINEA).move_to(pos_explicacion).align_to(pos_explicacion, LEFT)
        list1 = BulletedList(
            "Mantiene la información original intacta.",
            "Permite entrenar redes Nx12 sin perder gradiente.",
            "Los bloques suman (+) info, no sobreescriben.",
            font_size=24, color=DARK_GRAY
        ).next_to(head1, DOWN, aligned_edge=LEFT)

        self.play(Create(caja1), FadeIn(head1), FadeIn(list1))
        self.next_slide()
        self.play(FadeOut(caja1), FadeOut(head1), FadeOut(list1))

        # --- CONCEPTO 2: Pre-Layer Normalization ---
        # CORRECCIÓN 3: Dos rectángulos separados agrupados para no comerse el nodo Attention que está en medio.
        caja2_a = SurroundingRectangle(nodo_ln1, color=DARK_GRAY, buff=0.1)
        caja2_b = SurroundingRectangle(nodo_ln2, color=DARK_GRAY, buff=0.1)
        caja2 = VGroup(caja2_a, caja2_b)
        
        head2 = Text("Pre-Layer Norm (LN):", font_size=30, weight=BOLD, color=BLACK).move_to(pos_explicacion).align_to(pos_explicacion, LEFT)
        list2 = BulletedList(
            "Normaliza antes de entrar al bloque.",
            "Esencial para estabilidad en modelos profundos.",
            "Estándar en Transformers modernos (como GPT).",
            font_size=24, color=DARK_GRAY
        ).next_to(head2, DOWN, aligned_edge=LEFT)

        self.play(Create(caja2), FadeIn(head2), FadeIn(list2))
        self.next_slide()
        self.play(FadeOut(caja2), FadeOut(head2), FadeOut(list2))

        # --- CONCEPTO 3: Los Bloques ---
        # CORRECCIÓN 4: Igual que antes, rectángulos separados para los bloques principales.
        caja3_a = SurroundingRectangle(nodo_attn, color=ALERT_COLOR, buff=0.15)
        caja3_b = SurroundingRectangle(nodo_mlp, color=ALERT_COLOR, buff=0.15)
        caja3 = VGroup(caja3_a, caja3_b)
        
        head3 = Text("Bloques de Cómputo:", font_size=30, weight=BOLD, color=BLACK).move_to(pos_explicacion).align_to(pos_explicacion, LEFT)
        list3 = BulletedList(
            "Attention: Relaciona tokens entre sí.",
            "MLP: Procesa tokens individualmente.",
            font_size=24, color=RUST_COLOR
        ).next_to(head3, DOWN, aligned_edge=LEFT)

        self.play(Create(caja3), FadeIn(head3), FadeIn(list3))
        self.next_slide()
        self.play(FadeOut(caja3), FadeOut(head3), FadeOut(list3))

        # =========================================================================
        # --- LIMPIEZA TOTAL ---
        # =========================================================================
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def slide_entrenamiento(self):
        """
        Escena premium del entrenamiento (El Quijote).
        Muestra la arquitectura, el cálculo de Loss detallado y la matemática 
        visual de cómo los gradientes actualizan los pesos del modelo.
        """
        # =========================================================================
        # --- PREPARACIÓN Y TÍTULO ---
        # =========================================================================
        titulo, linea = self.crear_titulo(
            texto="Entrenamiento: El Quijote en Acción", 
            palabra_clave="Entrenamiento:", 
            color_clave=HIGHLIGHT_COLOR, 
            font_size=36
        )
        self.play(Write(titulo), GrowFromCenter(linea))

        # Indicador de estado superior (estilo UI premium)
        estado_ui = self.crear_bloque("FASE: Inicializando...", color_fondo=SOFT_BG, color_texto=BLACK, ancho=4, alto=0.5)
        estado_ui[0].set_stroke(width=1)
        estado_ui[1].set_font_size(16)
        estado_ui.to_edge(UP, buff=1.2)
        
        self.play(FadeIn(estado_ui, shift=DOWN))

        # =========================================================================
        # --- FASE 1: FORWARD PASS (EL INTENTO ERRÓNEO) ---
        # =========================================================================
        self.play(estado_ui[1].animate.set_text("FASE 1: Forward Pass (Predicción)"), estado_ui[0].animate.set_stroke(HIGHLIGHT_COLOR))

        # 1. Datos de Entrada
        lbl_in = Text("Contexto", font_size=14, color=DARK_GRAY, weight=BOLD).move_to(LEFT * 5 + UP * 1)
        caja_in = self.crear_bloque("En un lugar\nde la...", color_fondo=SOFT_BG, ancho=2.2, alto=1.2).next_to(lbl_in, DOWN, buff=0.1)
        grupo_in = VGroup(lbl_in, caja_in)

        # 2. Arquitectura del Modelo (Estilo "Procesador")
        modelo_bg = RoundedRectangle(corner_radius=0.2, width=3.5, height=3, fill_color=SOFT_BG, fill_opacity=0.3, stroke_color=HIGHLIGHT_COLOR, stroke_width=2).move_to(LEFT * 0.5 + DOWN * 0.5)
        lbl_modelo = Text("Red Neuronal", font_size=16, color=HIGHLIGHT_COLOR, weight=BOLD).next_to(modelo_bg.get_top(), DOWN, buff=0.2)
        
        # Matriz de pesos iniciales
        lbl_pesos = Text("Pesos Desajustados:", font_size=12, color=DARK_GRAY).next_to(lbl_modelo, DOWN, buff=0.3)
        pesos_iniciales = self.crear_matriz_bloques(2, 2, color_fondo=WHITE, valores=["0.1", "-0.8", "0.4", "0.2"]).scale(0.7).next_to(lbl_pesos, DOWN, buff=0.2)
        grupo_modelo = VGroup(modelo_bg, lbl_modelo, lbl_pesos, pesos_iniciales)

        # 3. Salida (Predicción)
        lbl_out = Text("Predicción", font_size=14, color=DARK_GRAY, weight=BOLD).move_to(RIGHT * 4 + UP * 1)
        caja_out = self.crear_bloque("playa?", color_fondo=SOFT_BG, ancho=2, alto=1).next_to(lbl_out, DOWN, buff=0.1)
        grupo_out = VGroup(lbl_out, caja_out)

        # Conexiones Forward
        flecha_fw1 = Arrow(caja_in.get_right(), modelo_bg.get_left(), color=BLACK, buff=0.1)
        flecha_fw2 = Arrow(modelo_bg.get_right(), caja_out.get_left(), color=BLACK, buff=0.1)

        # Animación coreografiada del Forward Pass
        self.play(FadeIn(grupo_in, shift=RIGHT))
        self.play(GrowArrow(flecha_fw1))
        self.play(FadeIn(modelo_bg, lbl_modelo), Write(lbl_pesos), FadeIn(pesos_iniciales, shift=UP))
        
        # Pulso de procesamiento
        self.play(Indicate(pesos_iniciales, color=GHOST_COLOR, scale_factor=1.1))
        
        self.play(GrowArrow(flecha_fw2))
        self.play(FadeIn(grupo_out, shift=RIGHT))
        self.next_slide()

        # =========================================================================
        # --- FASE 2: CÁLCULO DE LA PÉRDIDA (LOSS) ---
        # =========================================================================
        self.play(estado_ui[1].animate.set_text("FASE 2: Función de Pérdida (Loss)"), estado_ui[0].animate.set_stroke(ALERT_COLOR))

        # Target (Lo que debió decir)
        caja_real = self.crear_bloque("Mancha", color_fondo=SOFT_GREEN, color_texto=WHITE, ancho=2, alto=1).next_to(caja_out, DOWN, buff=1.2)
        lbl_real = Text("Target (Real)", font_size=14, color=SOFT_GREEN, weight=BOLD).next_to(caja_real, UP, buff=0.1)
        grupo_real = VGroup(lbl_real, caja_real)

        self.play(FadeIn(grupo_real, shift=UP))

        # Nodo de comparación matemática y Error
        nodo_diff = MathTex(r"\Delta", font_size=36, color=ALERT_COLOR).move_to(RIGHT * 4 + DOWN * 0.7)
        caja_error = self.crear_bloque("Loss: Alto", color_fondo=ALERT_COLOR, color_texto=WHITE, ancho=2, alto=0.8).next_to(nodo_diff, RIGHT, buff=0.5)
        
        flecha_err1 = Arrow(caja_out.get_bottom(), nodo_diff.get_top(), color=ALERT_COLOR, buff=0.1)
        flecha_err2 = Arrow(caja_real.get_top(), nodo_diff.get_bottom(), color=ALERT_COLOR, buff=0.1)
        flecha_err_out = Arrow(nodo_diff.get_right(), caja_error.get_left(), color=ALERT_COLOR, buff=0.1)

        self.play(GrowArrow(flecha_err1), GrowArrow(flecha_err2))
        self.play(Write(nodo_diff))
        self.play(GrowArrow(flecha_err_out), FadeIn(caja_error, shift=RIGHT))
        
        # Vibración de alerta
        self.play(Wiggle(caja_error, scale_value=1.1, rotation_angle=0.05))
        
        
        self.next_slide()

        # =========================================================================
        # --- FASE 3: BACKWARD PASS (AJUSTE DETALLADO) ---
        # =========================================================================
        self.play(estado_ui[1].animate.set_text("FASE 3: Backward Pass & Optimización"), estado_ui[0].animate.set_stroke(HIGHLIGHT_COLOR))

        # Animación del viaje del Gradiente
        flecha_back = CurvedArrow(caja_error.get_bottom(), modelo_bg.get_bottom(), angle=-TAU/4, color=HIGHLIGHT_COLOR, stroke_width=4)
        lbl_gradiente_viaje = Text("Gradientes (Error)", font_size=14, color=HIGHLIGHT_COLOR, weight=BOLD).next_to(flecha_back, DOWN)

        self.play(Create(flecha_back), Write(lbl_gradiente_viaje), run_time=1.5)

        # --- El Detalle Premium: Mostramos la matemática de la actualización ---
        # 1. Reducimos los pesos iniciales y los movemos a la izquierda
        self.play(
            pesos_iniciales.animate.scale(0.8).shift(LEFT * 0.8),
            lbl_pesos.animate.set_text("Actualizando Pesos...").set_color(HIGHLIGHT_COLOR)
        )

        # 2. Aparece el símbolo menos y la matriz de gradientes
        signo_menos = MathTex("-", font_size=36, color=BLACK).next_to(pesos_iniciales, RIGHT, buff=0.2)
        matriz_gradientes = self.crear_matriz_bloques(2, 2, color_fondo=SOFT_BG, valores=["-0.7", "0.9", "-0.5", "-0.4"]).scale(0.56).next_to(signo_menos, RIGHT, buff=0.2)
        
        # Coloreamos los gradientes de rojo
        for fila in matriz_gradientes:
            for bloque in fila:
                bloque[1].set_color(ALERT_COLOR)

        lbl_grad_matriz = Text("Gradientes", font_size=10, color=ALERT_COLOR).next_to(matriz_gradientes, UP, buff=0.1)
        grupo_resta = VGroup(signo_menos, matriz_gradientes, lbl_grad_matriz)

        self.play(FadeIn(grupo_resta, shift=LEFT))
        self.next_slide()

        # 3. Fusión: Los pesos originales y gradientes se convierten en los NUEVOS pesos
        pesos_nuevos = self.crear_matriz_bloques(2, 2, color_fondo=WHITE, valores=["0.8", "0.1", "0.9", "-0.2"]).scale(0.7).move_to(LEFT * 0.5 + DOWN * 0.8) # Centro del modelo
        
        for fila in pesos_nuevos:
            for bloque in fila:
                bloque[0].set_stroke(SOFT_GREEN, width=3)
                bloque[1].set_color(SOFT_GREEN)

        self.play(
            FadeOut(grupo_resta, pesos_iniciales),
            FadeIn(pesos_nuevos, shift=UP),
            modelo_bg.animate.set_stroke(SOFT_GREEN, width=4),
            lbl_pesos.animate.set_text("Pesos Optimizados").set_color(SOFT_GREEN)
        )
        self.play(Indicate(pesos_nuevos, color=SOFT_GREEN, scale_factor=1.1))
        
        
        self.next_slide()

        # =========================================================================
        # --- FASE 4: EL RESULTADO EXITOSO ---
        # =========================================================================
        self.play(estado_ui[1].animate.set_text("FASE 4: Nuevo Forward Pass (Aprendizaje Exitoso)"), estado_ui[0].animate.set_stroke(SOFT_GREEN))

        # Limpiamos todo el rastro de error
        self.play(
            FadeOut(grupo_out, grupo_real, nodo_diff, caja_error, flecha_err1, flecha_err2, flecha_err_out, flecha_back, lbl_gradiente_viaje, flecha_fw2)
        )

        # El modelo vuelve a procesar, pero ahora con los pesos en verde
        self.play(Indicate(pesos_nuevos, color=SOFT_GREEN, scale_factor=1.1))

        # Nueva salida (Correcta)
        lbl_out_ok = Text("Predicción", font_size=14, color=DARK_GRAY, weight=BOLD).move_to(RIGHT * 4 + UP * 1)
        caja_out_ok = self.crear_bloque("Mancha", color_fondo=TOKEN_COLOR_FINAL, color_texto=WHITE, ancho=2, alto=1).next_to(lbl_out_ok, DOWN, buff=0.1)
        grupo_out_ok = VGroup(lbl_out_ok, caja_out_ok)
        
        flecha_fw2_ok = Arrow(modelo_bg.get_right(), caja_out_ok.get_left(), color=TOKEN_COLOR_FINAL, buff=0.1)

        self.play(GrowArrow(flecha_fw2_ok))
        self.play(FadeIn(grupo_out_ok, shift=RIGHT))
        
        # Celebración final
        circulo_exito = Circle(radius=1.5, color=SOFT_GREEN, stroke_width=4).move_to(caja_out_ok.get_center())
        self.play(Create(circulo_exito))
        self.play(FadeOut(circulo_exito, scale=1.5))

        self.next_slide()
        self.limpiar_pantalla()
    def slide_linear_gradient(self):
        """
        Escena simplificada y elegante que muestra directamente 
        las derivadas (gradientes) de una capa lineal, todo centrado.
        """
        # =========================================================================
        # --- TÍTULO ---
        # =========================================================================
        titulo = Text("Linear Gradient", font_size=42, weight=BOLD, color=HIGHLIGHT_COLOR).to_edge(UP, buff=1)
        linea = Line(LEFT, RIGHT).set_width(8).next_to(titulo, DOWN, buff=0.2).set_color(DARK_GRAY)
        
        self.play(Write(titulo), GrowFromCenter(linea))

        # =========================================================================
        # --- ECUACIÓN ORIGINAL (FORWARD PASS) - CENTRADO ---
        # =========================================================================
        lbl_original = Text("Forward Pass", font_size=24, color=DARK_GRAY)
        eq_original = MathTex(r"y = x \cdot W + b", font_size=48)
        
        # Agrupamos y centramos verticalmente
        grupo_original = VGroup(lbl_original, eq_original).arrange(DOWN, buff=0.3).move_to(UP * 1.5)
        
        self.play(FadeIn(grupo_original, shift=DOWN))
        self.play(Indicate(eq_original, color=BLUE))
        self.next_slide()

        # =========================================================================
        # --- LAS DERIVADAS (BACKWARD PASS) - CENTRADO ---
        # =========================================================================
        lbl_derivadas = Text("Backward Pass (Gradientes)", font_size=24, color=DARK_GRAY)
        
        # Fórmulas de los gradientes matemáticos formales
        grad_W = MathTex(r"\frac{\partial L}{\partial W} = x^T \cdot \nabla y", font_size=42, color=SOFT_GREEN)
        grad_b = MathTex(r"\frac{\partial L}{\partial b} = \sum \nabla y", font_size=42, color=SOFT_GREEN)
        grad_x = MathTex(r"\frac{\partial L}{\partial x} = \nabla y \cdot W^T", font_size=42, color=BLUE)

        # Agrupamos las ecuaciones centradas entre sí
        grupo_grads_eqs = VGroup(grad_W, grad_b, grad_x).arrange(DOWN, buff=0)
        
        # Agrupamos el título del backward y las ecuaciones, y lo ponemos debajo del forward
        grupo_derivadas = VGroup(lbl_derivadas, grupo_grads_eqs).arrange(DOWN, buff=0.5).next_to(grupo_original, DOWN, buff=0.8)

        self.play(FadeIn(lbl_derivadas, shift=UP))
        
        # Aparecen una por una en el centro
        self.play(Write(grad_W))
        self.play(Write(grad_b))
        self.play(Write(grad_x))

        # Recuadro elegante centrado para enmarcar el resultado matemático
        caja_grads = SurroundingRectangle(grupo_grads_eqs, color=DARK_GRAY, corner_radius=0.2, buff=0.4)
        self.play(Create(caja_grads))
        
        self.next_slide()
        # Asegúrate de tener tu función limpiar_pantalla() o usa self.play(FadeOut(*self.mobjects))
        self.limpiar_pantalla()

    def slide_layer_norm_gradient(self):
        """
        Escena que muestra las derivadas de Layer Normalization, 
        destacando las 3 vías de dependencia en el gradiente de X.
        """
        # =========================================================================
        # --- TÍTULO ---
        # =========================================================================
        titulo = Text("Layer Norm Gradient", font_size=42, weight=BOLD, color=HIGHLIGHT_COLOR).to_edge(UP, buff=1)
        linea = Line(LEFT, RIGHT).set_width(8).next_to(titulo, DOWN, buff=0.2).set_color(DARK_GRAY)
        
        self.play(Write(titulo), GrowFromCenter(linea))

        # =========================================================================
        # --- GRADIENTES DE LOS PARÁMETROS (GAMMA Y BETA) - CENTRADO ---
        # =========================================================================
        lbl_params = Text("Parámetros de Escala y Desplazamiento", font_size=24, color=DARK_GRAY)
        
        # Gamma (pesos) y Beta (sesgos)
        grad_gamma = MathTex(r"\frac{\partial L}{\partial \gamma} = \sum (\nabla y \cdot \hat{x})", font_size=38, color=SOFT_GREEN)
        grad_beta = MathTex(r"\frac{\partial L}{\partial \beta} = \sum \nabla y", font_size=38, color=SOFT_GREEN)
        
        grupo_params_eqs = VGroup(grad_gamma, grad_beta).arrange(DOWN, buff=0.4)
        grupo_params = VGroup(lbl_params, grupo_params_eqs).arrange(DOWN, buff=0.4).move_to(UP * 1.2)
        
        self.play(FadeIn(grupo_params, shift=DOWN))
        self.next_slide()

        # =========================================================================
        # --- GRADIENTE DE LA ENTRADA (X) - LAS 3 VÍAS ---
        # =========================================================================
        lbl_x = Text("Gradiente de la Entrada (Dependencias Múltiples)", font_size=24, color=DARK_GRAY)
        
        # Fórmula con las 3 vías: Directa, Media y Varianza
        # Usamos \hat{x} para x_norm y \nabla \hat{x} para grad_x_norm
        formula_x = r"\frac{\partial L}{\partial x} = \frac{1}{\sigma} \Big( \nabla \hat{x} - \text{mean}(\nabla \hat{x}) - \hat{x} \cdot \text{mean}(\nabla \hat{x} \cdot \hat{x}) \Big)"
        grad_x = MathTex(formula_x, font_size=36, color=BLUE)
        
        grupo_x = VGroup(lbl_x, grad_x).arrange(DOWN, buff=0.5).next_to(grupo_params, DOWN, buff=0.8)
        
        self.play(FadeIn(lbl_x, shift=UP))
        self.play(Write(grad_x))
        
        # Etiquetas explicativas para los 3 términos de la ecuación
        lbl_term1 = Text("1. Vía Directa", font_size=14, color=DARK_GRAY).next_to(grad_x, DOWN, buff=0.4).shift(LEFT * 2.8)
        lbl_term2 = Text("2. Resta de la Media", font_size=14, color=DARK_GRAY).next_to(grad_x, DOWN, buff=0.4)
        lbl_term3 = Text("3. Resta de la Varianza", font_size=14, color=DARK_GRAY).next_to(grad_x, DOWN, buff=0.4).shift(RIGHT * 3.2)
        
        grupo_labels = VGroup(lbl_term1, lbl_term2, lbl_term3)
        self.play(FadeIn(grupo_labels, shift=UP))

        # Recuadro elegante centrado para enmarcar la fórmula compleja
        caja_x = SurroundingRectangle(VGroup(grupo_x, grupo_labels), color=DARK_GRAY, corner_radius=0.2, buff=0.3)
        self.play(Create(caja_x))
        
        self.next_slide()
        self.limpiar_pantalla()

    def slide_attention_backward(self):
        """
        Escena que muestra el Backward Pass de Attention,
        destacando el gradiente del Softmax y la suma de las proyecciones Q, K, V.
        """
        # =========================================================================
        # --- TÍTULO ---
        # =========================================================================
        titulo = Text("Attention Backward Pass", font_size=42, weight=BOLD, color=HIGHLIGHT_COLOR).to_edge(UP, buff=1)
        linea = Line(LEFT, RIGHT).set_width(8).next_to(titulo, DOWN, buff=0.2).set_color(DARK_GRAY)
        
        self.play(Write(titulo), GrowFromCenter(linea))

        # =========================================================================
        # --- 1. EL CUELLO DE BOTELLA: GRADIENTE DEL SOFTMAX ---
        # =========================================================================
        lbl_softmax = Text("1. Gradiente del Softmax (Dependencia de Fila)", font_size=24, color=DARK_GRAY)
        
        # A = Attention Weights, S = Scores (antes del softmax)
        # La fórmula refleja el código: grad_score = attn[j] * (grad_attn[j] - dot_product)
        formula_softmax = r"\frac{\partial L}{\partial S_{ij}} = A_{ij} \left( \nabla A_{ij} - \sum_{k} (A_{ik} \cdot \nabla A_{ik}) \right)"
        grad_softmax = MathTex(formula_softmax, font_size=38, color=ALERT_COLOR)
        
        grupo_softmax = VGroup(lbl_softmax, grad_softmax).arrange(DOWN, buff=0.4).move_to(UP * 1.2)
        
        self.play(FadeIn(lbl_softmax, shift=DOWN))
        self.play(Write(grad_softmax))
        
        # Pequeña nota explicativa
        lbl_nota_soft = Text("Cada score depende de todos los gradientes de su fila", font_size=14, slant=ITALIC, color=DARK_GRAY).next_to(grad_softmax, DOWN, buff=0.3)
        self.play(FadeIn(lbl_nota_soft))
        
        self.next_slide()

        # =========================================================================
        # --- 2. EL ENSAMBLAJE FINAL: GRADIENTE DE ENTRADA ---
        # =========================================================================
        lbl_input = Text("2. Gradiente de Entrada Final (Suma de Rutas)", font_size=24, color=DARK_GRAY)
        
        # El gradiente de entrada es la suma de los gradientes que fluyen por Q, K y V
        formula_input = r"\nabla x = \nabla x_Q + \nabla x_K + \nabla x_V"
        grad_input = MathTex(formula_input, font_size=42, color=BLUE)
        
        grupo_input = VGroup(lbl_input, grad_input).arrange(DOWN, buff=0.4).next_to(grupo_softmax, DOWN, buff=1)
        
        self.play(FadeIn(lbl_input, shift=UP))
        self.play(Write(grad_input))
        
        # Pequeña nota explicativa sobre el caché
        lbl_nota_input = Text("Requiere un caché masivo del Forward Pass", font_size=14, slant=ITALIC, color=BLUE).next_to(grad_input, DOWN, buff=0.3)
        self.play(FadeIn(lbl_nota_input))

        # Recuadro elegante general
        caja_total = SurroundingRectangle(VGroup(grupo_softmax, lbl_nota_soft, grupo_input, lbl_nota_input), color=DARK_GRAY, corner_radius=0.2, buff=0.4)
        self.play(Create(caja_total))
        
        self.next_slide()
        self.limpiar_pantalla()
    
    def slide_residual_connections(self):
        """
        Escena que explica el Backward Pass en las Conexiones Residuales,
        destacando la acumulación de gradientes y la 'autopista' de información.
        """
        # =========================================================================
        # --- TÍTULO ---
        # =========================================================================
        titulo = Text("Residual Connections", font_size=42, weight=BOLD, color=HIGHLIGHT_COLOR).to_edge(UP, buff=1)
        linea = Line(LEFT, RIGHT).set_width(8).next_to(titulo, DOWN, buff=0.2).set_color(DARK_GRAY)
        
        self.play(Write(titulo), GrowFromCenter(linea))

        # =========================================================================
        # --- 1. FORWARD PASS (LA BIFURCACIÓN) - CENTRADO ---
        # =========================================================================
        lbl_forward = Text("Forward Pass (Bifurcación)", font_size=24, color=DARK_GRAY)
        
        # y = x + Sublayer(x)
        eq_forward = MathTex(r"y = x + \text{Sublayer}(x)", font_size=42)
        eq_forward[0][0].set_color(HIGHLIGHT_COLOR)   # y
        eq_forward[0][2].set_color(BLUE)              # x (skip)
        
        grupo_forward = VGroup(lbl_forward, eq_forward).arrange(DOWN, buff=0.3).move_to(UP * 1.5)
        
        self.play(FadeIn(grupo_forward, shift=DOWN))
        self.next_slide()

        # =========================================================================
        # --- 2. BACKWARD PASS (LA ACUMULACIÓN) - CENTRADO ---
        # =========================================================================
        lbl_backward = Text("Backward Pass (Acumulación)", font_size=24, color=DARK_GRAY)
        
        # grad_x = grad_y + grad_sublayer
        # Usamos \nabla y para el gradiente que entra, y \nabla F(x) para el de la subcapa
        formula_backward = r"\nabla x = \nabla y + \nabla x_{\text{sublayer}}"
        eq_backward = MathTex(formula_backward, font_size=48)
        
        eq_backward[0][0:2].set_color(BLUE)            # \nabla x (salida final hacia atrás)
        eq_backward[0][3:5].set_color(HIGHLIGHT_COLOR) # \nabla y (el que pasa directo)
        eq_backward[0][6:].set_color(SOFT_GREEN)       # \nabla x_sublayer (el transformado)
        
        grupo_backward = VGroup(lbl_backward, eq_backward).arrange(DOWN, buff=0.4).next_to(grupo_forward, DOWN, buff=1)
        
        self.play(FadeIn(lbl_backward, shift=UP))
        self.play(Write(eq_backward))
        
        # =========================================================================
        # --- 3. EL CONCEPTO CLAVE: LA AUTOPISTA ---
        # =========================================================================
        # Etiquetas explicativas debajo de la ecuación
        lbl_skip = Text("Pasa directo (Autopista)", font_size=16, color=HIGHLIGHT_COLOR).next_to(eq_backward[0][3:5], DOWN, buff=0.4)
        lbl_trans = Text("Pasa por la capa", font_size=16, color=SOFT_GREEN).next_to(eq_backward[0][6:], DOWN, buff=0.4)
        
        self.play(FadeIn(lbl_skip, shift=UP), FadeIn(lbl_trans, shift=UP))

        # Mensaje final de impacto
        lbl_impacto = Text("Evita el desvanecimiento de gradientes en redes profundas.", font_size=18, slant=ITALIC, color=DARK_GRAY)
        lbl_impacto.next_to(grupo_backward, DOWN, buff=1.5)
        
        self.play(Write(lbl_impacto))

        # Recuadro elegante
        caja_total = SurroundingRectangle(VGroup(grupo_backward, lbl_skip, lbl_trans), color=DARK_GRAY, corner_radius=0.2, buff=0.4)
        self.play(Create(caja_total))
        
        self.next_slide()
        self.limpiar_pantalla()
    
    def slide_training_techniques(self):
        """
        Escena secuencial que explica las 3 técnicas de estabilización y
        optimización: Gradient Clipping, Dropout y AdamW.
        """
        # =========================================================================
        # --- TÍTULO GLOBAL ---
        # =========================================================================
        titulo = Text("Training Stabilizers & Optimization", font_size=42, weight=BOLD, color=HIGHLIGHT_COLOR).to_edge(UP, buff=1)
        linea = Line(LEFT, RIGHT).set_width(8).next_to(titulo, DOWN, buff=0.2).set_color(DARK_GRAY)
        
        self.play(Write(titulo), GrowFromCenter(linea))

        # =========================================================================
        # --- ACTO 1: GRADIENT CLIPPING ---
        # =========================================================================
        lbl_clip = Text("1. Gradient Clipping (Evitar Explosiones)", font_size=28, color=DARK_GRAY).move_to(UP * 1.5)
        
        # Fórmula matemática del clipping basado en norma L2
        formula_clip = r"g \leftarrow g \cdot \min\left(1, \frac{\text{max\_norm}}{||g||_2}\right)"
        eq_clip = MathTex(formula_clip, font_size=48, color=ALERT_COLOR).next_to(lbl_clip, DOWN, buff=0.8)
        
        nota_clip = Text("Escala el gradiente proporcionalmente si supera el umbral, manteniendo su dirección.", font_size=16, slant=ITALIC, color=DARK_GRAY).next_to(eq_clip, DOWN, buff=0.8)

        grupo_clip = VGroup(lbl_clip, eq_clip, nota_clip)
        
        self.play(FadeIn(lbl_clip, shift=DOWN))
        self.play(Write(eq_clip))
        self.play(FadeIn(nota_clip))
        self.next_slide()
        
        self.play(FadeOut(grupo_clip, shift=UP))

        # =========================================================================
        # --- ACTO 2: DROPOUT REGULARIZATION ---
        # =========================================================================
        lbl_drop = Text("2. Dropout (Prevenir Memorización / Overfitting)", font_size=28, color=DARK_GRAY).move_to(UP * 1.5)
        
        # Representación matemática de la máscara de Dropout
        formula_drop = r"y = x \odot M, \quad M \sim \text{Bernoulli}(1 - p)"
        eq_drop = MathTex(formula_drop, font_size=48, color=BLUE).next_to(lbl_drop, DOWN, buff=0.8)
        
        nota_drop = Text("Apaga aleatoriamente el 10% de las neuronas (p=0.1) para forzar redundancia.", font_size=16, slant=ITALIC, color=DARK_GRAY).next_to(eq_drop, DOWN, buff=0.8)

        grupo_drop = VGroup(lbl_drop, eq_drop, nota_drop)

        self.play(FadeIn(lbl_drop, shift=DOWN))
        self.play(Write(eq_drop))
        self.play(FadeIn(nota_drop))
        self.next_slide()
        
        self.play(FadeOut(grupo_drop, shift=UP))

        # =========================================================================
        # --- ACTO 3: ADAMW OPTIMIZER ---
        # =========================================================================
        lbl_adamw = Text("3. AdamW (Optimización con Decoupled Weight Decay)", font_size=28, color=DARK_GRAY).move_to(UP * 2)
        
        # Ecuaciones separadas para mostrar el paso a paso
        # 1. Weight Decay
        eq_wd = MathTex(r"\theta_{t} = \theta_{t-1}(1 - \eta \lambda)", font_size=36, color=SOFT_GREEN)
        # 2. Momentum y Varianza
        eq_mv = MathTex(r"m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad | \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2", font_size=36)
        # 3. Actualización final
        eq_update = MathTex(r"\theta_{t} \leftarrow \theta_{t} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}", font_size=42, color=BLUE)

        # Agrupamos las ecuaciones de AdamW
        grupo_eqs_adam = VGroup(eq_wd, eq_mv, eq_update).arrange(DOWN, buff=0.6).next_to(lbl_adamw, DOWN, buff=0.6)
        
        nota_adamw = Text("Separa la regularización (Weight Decay) del historial de gradientes (Momentum/Varianza).", font_size=16, slant=ITALIC, color=DARK_GRAY).next_to(grupo_eqs_adam, DOWN, buff=0.8)

        grupo_adamw = VGroup(lbl_adamw, grupo_eqs_adam, nota_adamw)

        self.play(FadeIn(lbl_adamw, shift=DOWN))
        self.play(Write(eq_wd))      # Muestra el decaimiento de pesos primero
        self.play(Write(eq_mv))      # Luego el momento y varianza
        self.play(Write(eq_update))  # Finalmente la actualización
        self.play(FadeIn(nota_adamw))
        
        # Recuadro final para AdamW, ya que es la estrella
        caja_adamw = SurroundingRectangle(grupo_eqs_adam, color=DARK_GRAY, corner_radius=0.2, buff=0.3)
        self.play(Create(caja_adamw))

        self.next_slide()
        self.limpiar_pantalla()

    def slide_training_metrics(self):
        """
        Escena en 3 actos que muestra las métricas de entrenamiento,
        el Learning Rate y la evolución espectacular del texto generado.
        """
        # =========================================================================
        # --- TÍTULO GLOBAL ---
        # =========================================================================
        titulo = Text("Training Metrics & Evolution", font_size=42, weight=BOLD, color=HIGHLIGHT_COLOR).to_edge(UP, buff=1)
        linea = Line(LEFT, RIGHT).set_width(8).next_to(titulo, DOWN, buff=0.2).set_color(DARK_GRAY)
        
        self.play(Write(titulo), GrowFromCenter(linea))

        # =========================================================================
        # --- ACTO 1: LOSS Y PERPLEXITY ---
        # =========================================================================
        lbl_metrics = Text("1. Loss & Perplexity", font_size=28, color=DARK_GRAY).move_to(UP * 1.5)
        
        # Fórmula de la perplejidad
        eq_ppl = MathTex(r"\text{Perplexity} = e^{\text{Loss}}", font_size=48, color=BLUE)
        
        nota_loss = Text("Loss: Error de predicción (Cross-Entropy).", font_size=20, color=DARK_GRAY)
        nota_ppl = Text("Perplexity: Nivel de incertidumbre (Ej: 50 = dudar entre 50 tokens).", font_size=20, color=DARK_GRAY)
        
        grupo_metrics = VGroup(lbl_metrics, eq_ppl, nota_loss, nota_ppl).arrange(DOWN, buff=0.6).move_to(UP * 0.5)
        
        self.play(FadeIn(lbl_metrics, shift=DOWN))
        self.play(Write(eq_ppl))
        self.play(FadeIn(nota_loss), FadeIn(nota_ppl))
        
        self.next_slide()
        self.play(FadeOut(grupo_metrics, shift=UP))

        # =========================================================================
        # --- ACTO 2: LEARNING RATE SCHEDULE ---
        # =========================================================================
        lbl_lr = Text("2. Learning Rate Schedule", font_size=28, color=DARK_GRAY).move_to(UP * 1.5)
        
        # Representación visual simulada del Warmup + Decay
        ejes = Axes(
            x_range=[0, 10, 1], y_range=[0, 5, 1],
            x_length=6, y_length=3,
            axis_config={"color": DARK_GRAY, "include_ticks": False}
        ).next_to(lbl_lr, DOWN, buff=0.5)
        
        # Curva: sube rápido (warmup), baja lento (decay)
        curva_lr = ejes.plot(lambda x: 4 * (x/2) if x < 2 else 4 * (2/x)**0.5, color=SOFT_GREEN)
        
        # Usamos ejes.coords_to_point para anclar los textos correctamente
        lbl_warmup = Text("Warmup", font_size=16, color=SOFT_GREEN).next_to(ejes.coords_to_point(1, 2), LEFT)
        lbl_decay = Text("Decay", font_size=16, color=SOFT_GREEN).next_to(ejes.coords_to_point(6, 2), RIGHT)
        
        grupo_lr = VGroup(lbl_lr, ejes, curva_lr, lbl_warmup, lbl_decay)
        
        self.play(FadeIn(lbl_lr, shift=DOWN))
        self.play(Create(ejes), Create(curva_lr))
        self.play(FadeIn(lbl_warmup), FadeIn(lbl_decay))
        
        self.next_slide()
        self.play(FadeOut(grupo_lr, shift=UP))

        lbl_evo = Text("3. Generación de Texto: De Ruido a Shakespeare", font_size=28, color=DARK_GRAY).move_to(UP * 2.5)
        self.play(FadeIn(lbl_evo, shift=DOWN))

        def mostrar_paso(step, loss, ppl, texto_gen, color_texto):
            stats = Text(f"Step {step} | Loss {loss} | Perplexity {ppl}", font_size=20, weight=BOLD, color=HIGHLIGHT_COLOR)
            ejemplo = Text(texto_gen, font_size=22, slant=ITALIC, color=color_texto)
            grupo = VGroup(stats, ejemplo).arrange(DOWN, buff=0.3).move_to(ORIGIN)
            return grupo

        # Paso 0: Ruido puro
        paso_0 = mostrar_paso("0", "7.75", "2313", '"To be, or not to beI knowdodaydegentsidESAR."', DARK_GRAY)
        self.play(FadeIn(paso_0, shift=UP))
        self.next_slide()
        self.play(FadeOut(paso_0, shift=UP))

        # Paso 4000: Palabras reales
        paso_4k = mostrar_paso("4000", "4.19", "66", '"To be, or not to be; and for the your sone,\nshall he spar,\nBut heaven"', BLUE)
        self.play(FadeIn(paso_4k, shift=UP))
        self.next_slide()
        self.play(FadeOut(paso_4k, shift=UP))

        paso_16k = mostrar_paso("16000", "3.02", "20", '"To be, or not to become you; if I must\nIf Caesar hath a wrong, thou k"', SOFT_GREEN)
        self.play(FadeIn(paso_16k, shift=UP))
        
        nota_final = Text("El modelo ha aprendido métrica, vocabulario y estructura sin reglas gramaticales.", font_size=18, color=DARK_GRAY).next_to(paso_16k, DOWN, buff=1)
        self.play(FadeIn(nota_final))

        caja_final = SurroundingRectangle(VGroup(paso_16k, nota_final), color=DARK_GRAY, corner_radius=0.2, buff=0.4)
        self.play(Create(caja_final))

        self.next_slide()
        self.limpiar_pantalla()

