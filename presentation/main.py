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

        self.slide_introduction()
        self.slide_credits()
        self.slide_arquitectura_transformer()
        self.slide_molinete_ai()
        
        self.slide_por_que_no_python()
        self.slide_por_que_no_cpp()
        self.slide_por_que_si_rust()

        self.diapo_problema_strawberry()
        self.diapo_tokenizacion()
        self.diapo_byte_pair_encoding()
        self.diapo_tamano_vocabulario()

        self.slide_que_es_un_tensor()
        self.diapo_matmul()
        self.diapo_softmax()
        self.slide_broadcasting()
        self.slide_strides()
        self.slide_masked_fill()
        self.slide_reshape_transpose()

        self.slide_forward_pass()
        self.slide_embeddings()
        self.slide_position_embeddings()
        self.slide_layer_normalization()

        self.slide_mha_acto1_intuicion()
        self.slide_mha_acto2_formula()
        self.slide_mha_acto3_calculo()
        self.slide_mha_acto4_multihead()

        self.slide_causal_masking()

        # self.diapo_codigo_rust()

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
    
    def crear_matriz_bloques(self, filas, columnas, color_fondo=FONDO_CAJA, color_texto=TINTA_NEGRA, valores=None, ancho=0.8, alto=0.8):
        if valores is None:
            valores = [""] * (filas * columnas)
            
        matriz = VGroup()
        idx = 0
        
        for i in range(filas):
            fila_bloques = VGroup()
            for j in range(columnas):
                texto = valores[idx] if idx < len(valores) else ""

                bloque = self.crear_bloque(
                    texto=texto, 
                    color_fondo=color_fondo, 
                    color_texto=color_texto, 
                    ancho=ancho, 
                    alto=alto
                )
                fila_bloques.add(bloque)
                idx += 1
                
            fila_bloques.arrange(RIGHT, buff=0.05)
            matriz.add(fila_bloques)
            
        matriz.arrange(DOWN, buff=0.05)
        
        return matriz

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
            color_clave=NARANJA_TERRACOTA
        )

        def crear_burbuja(texto, color_fondo, color_texto, es_usuario=True, t2c_dict=None):
            txt = Text(texto, font=FUENTE, font_size=24, color=color_texto, t2c=t2c_dict)
            
            fondo = RoundedRectangle(
                width=txt.width + 0.8, 
                height=txt.height + 0.5, 
                corner_radius=0.2, 
                fill_color=color_fondo, 
                fill_opacity=1, 
                stroke_width=0 if es_usuario else 1.5,
                stroke_color=MARRON_OSCURO
            )
            txt.move_to(fondo.get_center())
            burbuja_base = VGroup(fondo, txt)
            
            remitente = Text(
                "Tú" if es_usuario else "Molinete AI", 
                font=FUENTE,
                font_size=16, 
                color=MARRON_OSCURO,
                weight=BOLD
            )
            
            if es_usuario:
                remitente.next_to(burbuja_base, UP, buff=0.2, aligned_edge=RIGHT)
            else:
                remitente.next_to(burbuja_base, UP, buff=0.1, aligned_edge=LEFT)
                
            return VGroup(remitente, burbuja_base)

        burbuja_pregunta = crear_burbuja(
            "¿Cuántas letras 'r' hay en 'strawberry'?", 
            color_fondo=MARRON_OSCURO, 
            color_texto=PAPEL_CREMA, 
            es_usuario=True
        )
        
        burbuja_respuesta = crear_burbuja(
            "Hay 2 letras 'r' en 'strawberry'.", 
            color_fondo=FONDO_CAJA, 
            color_texto=TINTA_NEGRA, 
            es_usuario=False,
            t2c_dict={"2": NARANJA_TERRACOTA}
        )

        grupo_chat = VGroup(burbuja_pregunta, burbuja_respuesta).arrange(DOWN, buff=0.5)
        
        burbuja_pregunta.shift(RIGHT * 1.5)
        burbuja_respuesta.shift(LEFT * 1.5)

        grupo_chat.next_to(linea, DOWN, buff=0.6)

        texto_explicacion = Text(
            "Para el modelo, la palabra se divide en 'pedazos' (Tokens):", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        )
        
        token1 = self.crear_bloque("str", ancho=1.2)
        token2 = self.crear_bloque("aw", ancho=1.2)
        token3 = self.crear_bloque("berry", ancho=1.6)
        
        tokens_straw = VGroup(token1, token2, token3).arrange(RIGHT, buff=0.15)
        
        grupo_visual = VGroup(texto_explicacion, tokens_straw).arrange(DOWN, buff=0.5)
        grupo_visual.next_to(grupo_chat, DOWN, buff=1.0)

        # 4. Animaciones
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

        cruz = Cross(tokens_straw, stroke_color=NARANJA_TERRACOTA, stroke_width=6)
        self.play(Create(cruz))
        self.next_slide()
        
        self.limpiar_pantalla()

    def diapo_tokenizacion(self):
        titulo, linea = self.crear_titulo(
            "La Tokenización", 
            palabra_clave="Tokenización", 
            color_clave=NARANJA_TERRACOTA
        )

        frase = "Confía en el tiempo, que suele dar dulces salidas a muchas amargas dificultades"

        p1 = Text(
            "1. Por palabra: Vocabulario infinito, muy ineficiente.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"Por palabra:": MARRON_OSCURO, "infinito": NARANJA_TERRACOTA}
        )
        tokens_palabra = frase.split(" ")
        
        ej_palabra = VGroup(*[
            self.crear_bloque(t, ancho=max(0.8, len(t) * 0.25)) 
            for t in tokens_palabra
        ]).arrange(RIGHT, buff=0.15)
        
        if ej_palabra.width > 12.5:
            ej_palabra.scale_to_fit_width(12.5)
            
        grupo_palabra = VGroup(p1, ej_palabra).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        p2 = Text(
            "2. Por carácter: Secuencias larguísimas, pierde contexto.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA,
            t2c={"Por carácter:": MARRON_OSCURO, "pierde contexto": NARANJA_TERRACOTA}
        )
        tokens_caracter = [c if c != ' ' else '_' for c in frase]
        
        ej_caracter = VGroup(*[
            self.crear_bloque(t, ancho=0.3) 
            for t in tokens_caracter
        ]).arrange(RIGHT, buff=0.05)
        

        if ej_caracter.width > 12.5:
            ej_caracter.scale_to_fit_width(12.5)
            
        grupo_caracter = VGroup(p2, ej_caracter).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

 
        p3 = Text(
            "3. Sub-palabras (BPE): Algo más equilibrado.", 
            font=FUENTE, font_size=26, color=TINTA_NEGRA, weight=BOLD, 
            t2c={"Sub-palabras (BPE):": NARANJA_TERRACOTA, "equilibrado.": NARANJA_TERRACOTA}
        )
    
        tokens_bpe = [
            "Con", "f", "ía", "en", "el", "_tiem", "po", ",", "qu", "e", 
            "su", "ele", "d", "ar", "dul", "ces", "sali", "das", "a", 
            "mu", "chas", "amar", "gas", "di", "fic", "ul", "ta", "des"
        ]
        
        ej_bpe = VGroup(*[
            self.crear_bloque(t, ancho=max(0.8, len(t) * 0.25)) 
            for t in tokens_bpe
        ]).arrange(RIGHT, buff=0.15)
        
        if ej_bpe.width > 12.5:
            ej_bpe.scale_to_fit_width(12.5)
            
        grupo_bpe = VGroup(p3, ej_bpe).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        puntos = VGroup(grupo_palabra, grupo_caracter, grupo_bpe).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        puntos.next_to(linea, DOWN, buff=0.5).shift(LEFT * 0.5)

        self.play(Write(titulo), Create(linea))
        self.next_slide()

        self.play(FadeIn(p1, shift=RIGHT * 0.3))
        self.play(LaggedStart(*[GrowFromCenter(b) for b in ej_palabra], lag_ratio=0.05))
        self.next_slide()

        self.play(FadeIn(p2, shift=RIGHT * 0.3))
        self.play(LaggedStart(*[GrowFromCenter(b) for b in ej_caracter], lag_ratio=0.01)) 
        self.next_slide()

        self.play(FadeIn(p3, shift=RIGHT * 0.3))
        self.play(LaggedStart(*[GrowFromCenter(b) for b in ej_bpe], lag_ratio=0.05))
        self.next_slide()

        self.limpiar_pantalla()

    def diapo_byte_pair_encoding(self):
        titulo, linea = self.crear_titulo(
            "Byte Pair Encoding (BPE)", 
            palabra_clave="Byte Pair Encoding (BPE)", 
            color_clave=NARANJA_TERRACOTA
        )

        explicacion_bpe = Text(
            "Fusión iterativa de los pares más frecuentes:", 
            font=FUENTE, font_size=28, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.5)

        self.play(Write(titulo), Create(linea), FadeIn(explicacion_bpe))

        def calc_ancho(texto):
            return 0.6 + len(texto) * 0.3

        letras = ["t", "a", "c", "o", "t", "a", "c", "o"]

        fila_actual = VGroup(*[
            self.crear_bloque(letra, ancho=calc_ancho(letra)) 
            for letra in letras
        ]).arrange(RIGHT, buff=0.15).center().shift(UP*0.5)
        
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN*0.5) for b in fila_actual], lag_ratio=0.1))
        self.next_slide()

        texto_paso1 = Text(
            "Paso 1: 't' y 'a' son el par más común", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        ).next_to(fila_actual, DOWN, buff=1)
        self.play(FadeIn(texto_paso1))
        
        fila_paso1 = VGroup(
            self.crear_bloque("ta", color_fondo=PAPEL_TAN, ancho=calc_ancho("ta")),
            self.crear_bloque("c", ancho=calc_ancho("c")),
            self.crear_bloque("o", ancho=calc_ancho("o")),
            self.crear_bloque("ta", color_fondo=PAPEL_TAN, ancho=calc_ancho("ta")),
            self.crear_bloque("c", ancho=calc_ancho("c")),
            self.crear_bloque("o", ancho=calc_ancho("o"))
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
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        ).next_to(fila_actual, DOWN, buff=1)
        
        fila_paso2 = VGroup(
            self.crear_bloque("ta", color_fondo=PAPEL_TAN, ancho=calc_ancho("ta")),
            self.crear_bloque("co", color_fondo=CAJA_INFERIOR, ancho=calc_ancho("co")),
            self.crear_bloque("ta", color_fondo=PAPEL_TAN, ancho=calc_ancho("ta")),
            self.crear_bloque("co", color_fondo=CAJA_INFERIOR, ancho=calc_ancho("co"))
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
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        ).next_to(fila_actual, DOWN, buff=1)

        fila_paso3 = VGroup(
            self.crear_bloque("taco", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=calc_ancho("taco")),
            self.crear_bloque("taco", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=calc_ancho("taco"))
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
            color_clave=NARANJA_TERRACOTA
        )

        encabezados = VGroup(
            Text("Vocabulario", font=FUENTE, font_size=24, color=MARRON_OSCURO, weight=BOLD),
            Text("Total Tokens", font=FUENTE, font_size=24, color=MARRON_OSCURO, weight=BOLD),
            Text("Compresión", font=FUENTE, font_size=24, color=NARANJA_TERRACOTA, weight=BOLD)
        ).arrange(RIGHT, buff=1.5)

        linea_separadora = Line(LEFT, RIGHT, color=MARRON_OSCURO, stroke_width=2)
        
        def crear_fila(v, t, c, es_final=False):
            color_c = NARANJA_TERRACOTA if es_final else TINTA_NEGRA
            peso_c = BOLD if es_final else NORMAL
            return VGroup(
                Text(v, font=FUENTE, font_size=24, color=TINTA_NEGRA),
                Text(t, font=FUENTE, font_size=24, color=TINTA_NEGRA),
                Text(c, font=FUENTE, font_size=24, color=color_c, weight=peso_c)
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
            corner_radius=0.15, 
            fill_color=FONDO_CAJA, 
            fill_opacity=1, 
            stroke_width=2,
            stroke_color=MARRON_OSCURO
        )
        
        grupo_tabla = VGroup(fondo_tabla, tabla_interna)
        grupo_tabla.next_to(linea, DOWN, buff=0.6)

        tradeoff_titulo = Text("El Trade-off (Compensación):", font=FUENTE, font_size=28, color=TINTA_NEGRA, weight=BOLD)
        
        pro_icon = Text("✅", font=FUENTE, font_size=24)
        pro_text = Text(
            "Más vocabulario = Textos cortos = Inferencia rápida", 
            font=FUENTE, font_size=24, color=MARRON_OSCURO, 
            t2c={"Inferencia rápida": NARANJA_TERRACOTA}
        )
        pro_group = VGroup(pro_icon, pro_text).arrange(RIGHT, buff=0.2)

        con_icon = Text("❌", font=FUENTE, font_size=24)
        con_text = Text(
            "Más vocabulario = Matriz gigante = Más VRAM", 
            font=FUENTE, font_size=24, color=MARRON_OSCURO, 
            t2c={"Más VRAM": NARANJA_TERRACOTA}
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
        titulo11, linea11 = self.crear_titulo("¿Qué es un Tensor?", palabra_clave="Tensor?", color_clave=NARANJA_TERRACOTA)
        self.play(Write(titulo11), Create(linea11))
        
        def_tensor = Text(
            "Un Tensor es un contenedor matemático\npara almacenar datos en múltiples dimensiones.", 
            font=FUENTE, font_size=32, 
            color=MARRON_OSCURO,
            t2c={"múltiples dimensiones.": NARANJA_TERRACOTA}
        ).next_to(linea11, DOWN, buff=0.4)
        
        self.play(Write(def_tensor))
        self.next_slide()
        
        txt_0d = Text("Escalar (0D) - Un solo valor", font=FUENTE, font_size=28, color=MARRON_OSCURO).next_to(def_tensor, DOWN, buff=0.8)
        escalar = self.crear_bloque("7", ancho=0.8, alto=0.8).next_to(txt_0d, DOWN, buff=0.5)
        
        self.play(Write(txt_0d))
        self.play(GrowFromCenter(escalar))
        self.next_slide()
        self.play(FadeOut(txt_0d), FadeOut(escalar))
        
        txt_1d = Text("Vector (1D) - Lista de valores", font=FUENTE, font_size=28, color=MARRON_OSCURO).next_to(def_tensor, DOWN, buff=0.8)
        vector = self.crear_matriz_bloques(1, 4, valores=["1", "5", "9", "2"]).next_to(txt_1d, DOWN, buff=0.5)
        
        self.play(Write(txt_1d))
        self.play(LaggedStart(*[FadeIn(b, shift=UP*0.2) for b in vector[0]], lag_ratio=0.1))
        self.next_slide()
        self.play(FadeOut(txt_1d), FadeOut(vector))
        
        txt_2d = Text("Matriz (2D) - Tabla de valores", font=FUENTE, font_size=28, color=MARRON_OSCURO).next_to(def_tensor, DOWN, buff=0.8)
        valores_matriz = ["3","1","4","2", "5","9","2","6", "5","3","5","8"]
        matriz = self.crear_matriz_bloques(3, 4, valores=valores_matriz).next_to(txt_2d, DOWN, buff=0.5)
        
        self.play(Write(txt_2d))
        bloques_anim = [FadeIn(b, shift=UP*0.2) for fila in matriz for b in fila]
        self.play(LaggedStart(*bloques_anim, lag_ratio=0.05))
        self.next_slide()
        self.play(FadeOut(txt_2d), FadeOut(matriz))
        
        txt_3d = Text("Tensor (3D+) - Cubo de valores", font=FUENTE, font_size=28, color=MARRON_OSCURO).next_to(def_tensor, DOWN, buff=0.5)
        
        matriz_base_3d = self.crear_matriz_bloques(3, 3, valores=["1","2","3","4","5","6","7","8","9"])
        for b in matriz_base_3d.submobjects:
            for sub_b in b.submobjects:
                sub_b[0].set_fill(opacity=0.3).set_stroke(opacity=0.3)

        matriz_medio_3d = self.crear_matriz_bloques(3, 3, color_fondo=PAPEL_TAN, valores=["9","8","7","6","5","4","3","2","1"]).shift(UP*0.25 + RIGHT*0.25)
        for b in matriz_medio_3d.submobjects:
            for sub_b in b.submobjects:
                sub_b[0].set_fill(opacity=0.6).set_stroke(opacity=0.6)

        matriz_top_3d = self.crear_matriz_bloques(3, 3, color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, valores=["2","4","6","8","0","2","4","6","8"]).shift(UP*0.5 + RIGHT*0.5)
        
        tensor_3d = VGroup(matriz_base_3d, matriz_medio_3d, matriz_top_3d).next_to(txt_3d, DOWN, buff=0.5).shift(LEFT * 0.25)
        
        self.play(Write(txt_3d))
        self.play(FadeIn(matriz_base_3d, shift=UP*0.3))
        self.play(FadeIn(matriz_medio_3d, shift=UP*0.3))
        self.play(FadeIn(matriz_top_3d, shift=UP*0.3))
        self.next_slide()
        
        self.play(FadeOut(txt_3d), FadeOut(tensor_3d), FadeOut(def_tensor))

        nota_ram = Text("Pero en memoria RAM, todos terminan siendo un arreglo plano...", font=FUENTE, font_size=26, color=NARANJA_TERRACOTA).next_to(linea11, DOWN, buff=0.5)
        self.play(Write(nota_ram))
        
        matriz_ram = self.crear_matriz_bloques(3, 4, valores=valores_matriz).next_to(nota_ram, DOWN, buff=0.8)
        self.play(FadeIn(matriz_ram))
        self.next_slide()

        bloques_individuales = [bloque for fila in matriz_ram for bloque in fila]
        grupo_plano = VGroup(*bloques_individuales)

        self.play(
            grupo_plano.animate.arrange(RIGHT, buff=0.05).scale(0.8).next_to(nota_ram, DOWN, buff=1.5),
            run_time=2
        )
        self.play(Indicate(grupo_plano, color=PAPEL_TAN))
        self.next_slide()

        self.limpiar_pantalla()

    def diapo_matmul(self):
        titulo, linea = self.crear_titulo(
            "Multiplicación: El 'Dot Product'", 
            palabra_clave="'Dot Product'", 
            color_clave=NARANJA_TERRACOTA
        )

        val_A = ["1","2","3",  "4","5","6",  "7","8","9"]
        val_B = ["2","1","0",  "0","2","1",  "1","0","3"]
        val_C = ["","","",  "","","",  "","",""]

        mat_A = self.crear_matriz_bloques(3, 3, valores=val_A)
        signo_por = Text("×", font=FUENTE, font_size=40, color=TINTA_NEGRA)
        mat_B = self.crear_matriz_bloques(3, 3, valores=val_B)
        signo_igual = Text("=", font=FUENTE, font_size=40, color=TINTA_NEGRA)
        mat_C = self.crear_matriz_bloques(3, 3, valores=val_C)

        grupo_matmul = VGroup(mat_A, signo_por, mat_B, signo_igual, mat_C).arrange(RIGHT, buff=0.4)
        grupo_matmul.shift(UP * 0.5) 
        
        fila_A = mat_A[0] 
        col_B = VGroup(mat_B[0][2], mat_B[1][2], mat_B[2][2]) 
        celda_C = mat_C[0][2] 

        self.play(Write(titulo), Create(linea))
        self.play(FadeIn(grupo_matmul))
        self.next_slide()

        self.play(
            *[b[0].animate.set_fill(PAPEL_TAN, opacity=0.8) for b in fila_A],
            *[b[0].animate.set_fill(PAPEL_TAN, opacity=0.8) for b in col_B]
        )
        self.next_slide()

        calculo_texto = Text(
            "(1 × 0) + (2 × 1) + (3 × 3) = 11", 
            font=FUENTE, font_size=32, color=MARRON_OSCURO,
            t2c={"11": NARANJA_TERRACOTA}
        ).next_to(grupo_matmul, DOWN, buff=1.0)
        
        fila_copia = fila_A.copy()
        col_copia = col_B.copy()

        self.play(
            ReplacementTransform(fila_copia, calculo_texto), 
            ReplacementTransform(col_copia, calculo_texto)
        )
        self.next_slide()

        dot_calc = self.crear_bloque("11", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=0.8)
        dot_calc.move_to(calculo_texto.get_center())

        self.play(ReplacementTransform(calculo_texto, dot_calc))
        
        self.play(
            dot_calc.animate.move_to(celda_C.get_center()),
            celda_C[0].animate.set_fill(PAPEL_TAN, opacity=0.4) 
        )
        self.play(Indicate(dot_calc, color=PAPEL_TAN, scale_factor=1.2))
        self.next_slide()
        
        def_matmul = Text(
            "Combina cada elemento de la fila con su pareja en la columna.\nSe multiplican y se suman para obtener un único número.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        ).to_edge(DOWN, buff=1.0)
        
        self.play(Write(def_matmul))
        self.next_slide()
        
        self.limpiar_pantalla()

    def diapo_softmax(self):
            titulo, linea = self.crear_titulo(
                "Softmax", 
                palabra_clave="Probabilidades", 
                color_clave=NARANJA_TERRACOTA
            )
            
            formula = MathTex(
                r"\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}", 
                color=TINTA_NEGRA, font_size=38
            ).next_to(linea, DOWN, buff=0.4)

            self.play(Write(titulo), Create(linea))
            self.play(Write(formula))
            self.next_slide()

            ancho_caja = 1.1
            
            col1 = VGroup(
                Text("1. Logits", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(*[self.crear_bloque(v, ancho=ancho_caja) for v in ["2.0", "1.0", "0.1"]]).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            flecha1 = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).scale(0.5)
            txt_op1 = MathTex(r"\exp(x)", font_size=22, color=TINTA_NEGRA).next_to(flecha1, UP, buff=0.1)
            conector1 = VGroup(flecha1, txt_op1)

            col2 = VGroup(
                Text("2. Exp", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(*[self.crear_bloque(v, color_fondo=PAPEL_TAN, ancho=ancho_caja) for v in ["7.39", "2.72", "1.10"]]).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            flecha2 = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).scale(0.5)
            txt_op2 = MathTex(r"\div \sum", font_size=22, color=TINTA_NEGRA).next_to(flecha2, UP, buff=0.1)
            conector2 = VGroup(flecha2, txt_op2)

            col3 = VGroup(
                Text("3. Prob", font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD),
                VGroup(*[self.crear_bloque(v, color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=ancho_caja) for v in ["66%", "24%", "10%"]]).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            acto1_horiz = VGroup(col1, conector1, col2, conector2, col3).arrange(RIGHT, buff=0.5).move_to(DOWN * 0.5)

            self.play(FadeIn(col1, shift=UP*0.2))
            self.play(Write(conector1), ReplacementTransform(col1[1].copy(), col2[1]), FadeIn(col2[0]))
            self.play(Write(conector2), ReplacementTransform(col2[1].copy(), col3[1]), FadeIn(col3[0]))
            self.next_slide()

            self.play(FadeOut(acto1_horiz), FadeOut(formula))

            titulo_error = Text("Problema: Un valor arruina el vector", font=FUENTE, font_size=32, color=NARANJA_TERRACOTA).move_to(titulo)
            self.play(ReplacementTransform(titulo, titulo_error))

            col_err_1 = VGroup(
                Text("Logits", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(
                    self.crear_bloque("800.0", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=ancho_caja),
                    self.crear_bloque("2.0", ancho=ancho_caja),
                    self.crear_bloque("-1.0", ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            f_err = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).scale(0.5)
            t_err = MathTex(r"\exp(x)", font_size=22, color=TINTA_NEGRA).next_to(f_err, UP, buff=0.1)
            conector_err = VGroup(f_err, t_err)

            col_err_2 = VGroup(
                Text("Exp", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(
                    self.crear_bloque("inf", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=ancho_caja),
                    self.crear_bloque("7.39", color_fondo=PAPEL_TAN, ancho=ancho_caja),
                    self.crear_bloque("0.37", color_fondo=PAPEL_TAN, ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            flujo_error = VGroup(col_err_1, conector_err, col_err_2).arrange(RIGHT, buff=0.8).move_to(DOWN * 0.2)

            self.play(FadeIn(col_err_1))
            self.play(Flash(col_err_1[1][0], color=NARANJA_TERRACOTA))
            self.play(Write(conector_err))
            self.play(ReplacementTransform(col_err_1[1].copy(), col_err_2[1]), FadeIn(col_err_2[0]))
            self.play(Wiggle(col_err_2[1][0])) 
            
            nota_error = Text("Si intentamos dividir por 'inf', todo el vector se vuelve NaN.", font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(flujo_error, DOWN, buff=0.6)
            self.play(Write(nota_error))
            self.next_slide()

            self.play(FadeOut(flujo_error), FadeOut(nota_error))
            
            # --- ACTO 3: LA SOLUCIÓN (SHIFT) ---
            titulo_fix = Text("Solución: Restar el Máximo (Shift)", font=FUENTE, font_size=32, color=NARANJA_TERRACOTA).move_to(titulo_error)
            self.play(ReplacementTransform(titulo_error, titulo_fix))

            col_fix_1 = VGroup(
                Text("Logits", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(
                    self.crear_bloque("800.0", ancho=ancho_caja),
                    self.crear_bloque("2.0", ancho=ancho_caja),
                    self.crear_bloque("-1.0", ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            f_shift = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).scale(0.5)
            t_shift = Text("- Max (x)", font=FUENTE, font_size=16, color=NARANJA_TERRACOTA, weight=BOLD).next_to(f_shift, UP, buff=0.1)
            conector_shift = VGroup(f_shift, t_shift)

            col_fix_2 = VGroup(
                Text("Shifted", font=FUENTE, font_size=18, color=NARANJA_TERRACOTA, weight=BOLD),
                VGroup(
                    self.crear_bloque("0.0", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=ancho_caja), 
                    self.crear_bloque("-798.0", color_fondo=PAPEL_TAN, ancho=ancho_caja),
                    self.crear_bloque("-801.0", color_fondo=PAPEL_TAN, ancho=ancho_caja)
                ).arrange(DOWN, buff=0.1)
            ).arrange(DOWN, buff=0.2)

            f_exp2 = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).scale(0.5)
            t_exp2 = MathTex(r"\exp(x)", font_size=22, color=TINTA_NEGRA).next_to(f_exp2, UP, buff=0.1)
            conector_exp2 = VGroup(f_exp2, t_exp2)

            col_fix_3 = VGroup(
                Text("Exp Seguro", font=FUENTE, font_size=18, color=MARRON_OSCURO, weight=BOLD),
                VGroup(
                    self.crear_bloque("1.0", color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, ancho=ancho_caja),
                    self.crear_bloque("0.0", ancho=ancho_caja), 
                    self.crear_bloque("0.0", ancho=ancho_caja)
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
                font=FUENTE, font_size=18, color=TINTA_NEGRA
            ).next_to(flujo_fix, DOWN, buff=0.5)
            
            self.play(FadeIn(nota_fix, shift=UP*0.2))
            self.next_slide()

            self.limpiar_pantalla()

    def slide_broadcasting(self):
        titulo, linea = self.crear_titulo(
            "Broadcasting: Expansión Virtual", 
            palabra_clave="Expansión Virtual", 
            color_clave=PAPEL_TAN
        )

        val_base = ["1", "2", "3", "4", "5", "6"]
        val_vec = ["10", "20", "30"]
        val_res = ["11", "22", "33", "14", "25", "36"]

        matriz_base = self.crear_matriz_bloques(2, 3, valores=val_base)
        signo_mas = Text("+", font=FUENTE, font_size=40, color=TINTA_NEGRA)
        
        vector_real = VGroup(*[self.crear_bloque(val, color_fondo=PAPEL_TAN) for val in val_vec]).arrange(RIGHT, buff=0.05)
    
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
            color=MARRON_OSCURO, buff=0.1, corner_radius=0.1
        )
        caja_virtual = DashedVMobject(rectangulo_base, num_dashes=35)
        
        texto_virtual = Text("Matriz Virtual", font=FUENTE, font_size=16, color=MARRON_OSCURO).next_to(caja_virtual, DOWN, buff=0.15)

        self.play(Create(caja_virtual), Write(texto_virtual))
        self.next_slide()

        flecha_res = Arrow(LEFT, RIGHT, color=TINTA_NEGRA).next_to(caja_virtual, RIGHT, buff=0.5)

        matriz_res = self.crear_matriz_bloques(2, 3, color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, valores=val_res).next_to(flecha_res, RIGHT, buff=0.5)

        self.play(GrowArrow(flecha_res))
        
        self.play(
            ReplacementTransform(matriz_base.copy(), matriz_res),
            ReplacementTransform(VGroup(vector_real.copy(), vector_fantasma.copy()), matriz_res),
        )
        self.next_slide()
        
        def_broad = Text(
            "Se duplica el vector pequeño 'virtualmente' para coincidir\ncon la matriz grande, ahorrando muchísima memoria.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, 
            t2c={"virtualmente": PAPEL_TAN, "ahorrando muchísima memoria": NARANJA_TERRACOTA}
        ).to_edge(DOWN, buff=1.0)
        
        self.play(FadeIn(def_broad, shift=UP*0.2))
        self.next_slide()
        self.limpiar_pantalla()

    def slide_strides(self):
        titulo, linea = self.crear_titulo(
            "Strides: Saltando en Memoria 1D", 
            palabra_clave="Strides:", 
            color_clave=NARANJA_TERRACOTA
        )

        arr_1d = VGroup(*[self.crear_bloque(str(i)) for i in range(6)])
        arr_1d.arrange(RIGHT, buff=0.1).shift(UP * 1.5)
        
        lbl_1d = Text("Memoria RAM (Física, 1D)", font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(arr_1d, UP, buff=0.3)

        self.play(Write(titulo), Create(linea))
        self.play(FadeIn(arr_1d, shift=UP*0.2), FadeIn(lbl_1d, shift=UP*0.2))
        self.next_slide()

        fila1 = VGroup(*[arr_1d[i].copy() for i in range(3)]).arrange(RIGHT, buff=0.1)
        fila2 = VGroup(*[arr_1d[i].copy() for i in range(3, 6)]).arrange(RIGHT, buff=0.1)
        
        mat_shape = VGroup(fila1, fila2).arrange(DOWN, buff=0.1).shift(DOWN * 0.5)
        lbl_2d = Text("Shape Lógica (2x3)", font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(mat_shape, DOWN, buff=0.3)

        self.play(
            TransformFromCopy(VGroup(*arr_1d[0:3]), fila1),
            TransformFromCopy(VGroup(*arr_1d[3:6]), fila2),
            FadeIn(lbl_2d, shift=DOWN*0.2),
            run_time=1.5
        )
        self.next_slide()

        self.play(
            arr_1d[0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            arr_1d[3][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            mat_shape[0][0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
            mat_shape[1][0][0].animate.set_fill(PAPEL_TAN, opacity=0.8),
        )

        arco_1d = CurvedArrow(arr_1d[0].get_top(), arr_1d[3].get_top(), angle=-PI/2, color=NARANJA_TERRACOTA)
        txt_stride_1d = Text("Stride = 3 pasos", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA).next_to(arco_1d, UP, buff=0.1)
        
        arco_2d = CurvedArrow(mat_shape[0][0].get_left(), mat_shape[1][0].get_left(), angle=PI/2, color=NARANJA_TERRACOTA).shift(LEFT*0.1)
        txt_stride_2d = Text("+1 Fila", font=FUENTE, font_size=16, color=NARANJA_TERRACOTA).next_to(arco_2d, LEFT, buff=0.1)

        self.play(
            Create(arco_1d), Write(txt_stride_1d),
            Create(arco_2d), Write(txt_stride_2d)
        )
        self.next_slide()
        
        def_stride = Text(
            "Como todo es 1D en RAM, los 'strides' dictan\ncuántos casilleros avanzar para encontrar la siguiente fila.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, 
            t2c={"1D en RAM": MARRON_OSCURO, "'strides'": NARANJA_TERRACOTA}
        ).to_edge(DOWN, buff=1.0) 
        
        self.play(FadeIn(def_stride, shift=UP*0.2))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_masked_fill(self):
        titulo, linea = self.crear_titulo(
            "Masked Fill: Causalidad", 
            palabra_clave="Causalidad", 
            color_clave=NARANJA_TERRACOTA
        )

        val_mask = [
            "4.1", "1.2", "0.5", "2.1", 
            "3.3", "5.0", "1.8", "0.9", 
            "1.1", "2.4", "6.2", "1.5", 
            "0.8", "1.7", "3.0", "7.1"
        ]
        
        matriz_mask = self.crear_matriz_bloques(4, 4, valores=val_mask).scale(1.2).shift(DOWN * 0.2)
        
        lbl_matriz = Text("Scores de Atención (Previo al Masking)", font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(matriz_mask, UP, buff=0.3)
        
        self.play(Write(titulo), Create(linea))
        self.play(FadeIn(matriz_mask, shift=UP*0.2), FadeIn(lbl_matriz, shift=UP*0.2))
        self.next_slide()

        animaciones_mask = []
        animaciones_keep = []
        
        for i in range(4):
            for j in range(4):
                bloque = matriz_mask[i][j]
                if j > i: 
                    nuevo_bloque = self.crear_bloque("-∞", color_fondo=MARRON_OSCURO, color_texto=PAPEL_CREMA)
                    nuevo_bloque.match_height(bloque).move_to(bloque)
                    animaciones_mask.append(ReplacementTransform(bloque, nuevo_bloque))
                else:
                    animaciones_keep.append(
                        bloque[0].animate.set_stroke(color=PAPEL_TAN, width=3)
                    )

        lbl_mask = Text("Máscara Causal Aplicada (Masked Fill)", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA).move_to(lbl_matriz)

        self.play(
            LaggedStart(*animaciones_mask, lag_ratio=0.15),
            *animaciones_keep,
            ReplacementTransform(lbl_matriz, lbl_mask),
            run_time=2
        )
        self.next_slide()

        def_mask = Text(
            "Se tapan los valores del 'futuro' con menos infinito (-∞).\nAl pasar por Softmax, esto se convierte en 0% de probabilidad.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, 
            t2c={"-∞": NARANJA_TERRACOTA, "0% de probabilidad": NARANJA_TERRACOTA, "'futuro'": MARRON_OSCURO}
        ).to_edge(DOWN, buff=1.0)
        
        self.play(FadeIn(def_mask, shift=UP*0.2))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_reshape_transpose(self):
        titulo, linea = self.crear_titulo(
            "Reshaping vs Transposing", 
            palabra_clave="vs", 
            color_clave=MARRON_OSCURO
        )
        
        titulo.set_color_by_t2c({"Reshaping": MARRON_OSCURO, "Transposing": NARANJA_TERRACOTA})
        
        linea_central = DashedLine(UP*2.2, DOWN*2.5, color=MARRON_OSCURO)

        self.play(Write(titulo), Create(linea), Create(linea_central))
        self.next_slide()

        txt_reshape = Text("Reshape", font=FUENTE, font_size=28, color=MARRON_OSCURO).move_to(LEFT * 3.5 + UP * 2.2)
        sub_reshape = Text("(Misma memoria)", font=FUENTE, font_size=16, color=TINTA_NEGRA).next_to(txt_reshape, DOWN, buff=0.1)
        
        valores_orig = ["1", "2", "3", "4", "5", "6"]
        
        mat_r_orig = self.crear_matriz_bloques(2, 3, valores=valores_orig).scale(0.8).next_to(sub_reshape, DOWN, buff=0.4)
        flecha_r = Arrow(mat_r_orig.get_bottom(), mat_r_orig.get_bottom() + DOWN * 0.8, color=MARRON_OSCURO)
        mat_r_final = self.crear_matriz_bloques(3, 2, color_fondo=PAPEL_TAN, valores=valores_orig).scale(0.8).next_to(flecha_r, DOWN, buff=0.2)

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

        txt_transpose = Text("Transpose", font=FUENTE, font_size=28, color=NARANJA_TERRACOTA).move_to(RIGHT * 3.5 + UP * 2.2)
        sub_transpose = Text("(Reorganización física)", font=FUENTE, font_size=16, color=TINTA_NEGRA).next_to(txt_transpose, DOWN, buff=0.1)

        mat_t_orig = self.crear_matriz_bloques(2, 3, valores=valores_orig).scale(0.8).next_to(sub_transpose, DOWN, buff=0.4)
        flecha_t = Arrow(mat_t_orig.get_bottom(), mat_t_orig.get_bottom() + DOWN * 0.8, color=NARANJA_TERRACOTA)
        
        valores_trans = ["1", "4", "2", "5", "3", "6"]
        mat_t_final = self.crear_matriz_bloques(3, 2, color_fondo=NARANJA_TERRACOTA, color_texto=PAPEL_CREMA, valores=valores_trans).scale(0.8).next_to(flecha_t, DOWN, buff=0.2)

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
            font=FUENTE, font_size=22, color=TINTA_NEGRA, 
            t2c={"Reshape": MARRON_OSCURO, "Transpose": NARANJA_TERRACOTA}
        ).to_edge(DOWN, buff=0.5)
        
        self.play(FadeIn(def_res_trans, shift=UP*0.2))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_forward_pass(self):
        titulo, linea = self.crear_titulo("Arquitectura: El Forward Pass", color_clave=NARANJA_TERRACOTA)
        self.play(Write(titulo), Create(linea))
        
        frase_completa = VGroup(
            Text('"En un lugar de la', font=FUENTE, font_size=42, color=TINTA_NEGRA),
            Text(' Mancha"', font=FUENTE, font_size=42, color=NARANJA_TERRACOTA)
        ).arrange(RIGHT, buff=0.1).move_to(UP * 1.8)

        frase_base = frase_completa[0]
        posicion_destino_mancha = frase_completa[1].get_center()
        
        self.play(Write(frase_base))
        self.next_slide()

        BG_CAJA = PAPEL_CREMA
        BORDE_CAJA = MARRON_OSCURO
        TXT_PRINCIPAL = TINTA_NEGRA
        TXT_SECUNDARIO = MARRON_OSCURO 

        def crear_caja_nodo(label_sup, valor_array, label_inf, is_stack=False, highlight_last=False):
            grupo = VGroup()
            
            txt_sup = Text(label_sup, font="Monospace", font_size=18, weight=BOLD, color=TXT_SECUNDARIO)
            
            ANCHO = 2.65
            ALTO = 1.7
            RADIO = 0.15
            
            sombra = RoundedRectangle(corner_radius=RADIO, width=ANCHO, height=ALTO)
            sombra.set_fill(MARRON_OSCURO, opacity=0.15).set_stroke(width=0)
            sombra.shift(RIGHT * 0.06 + DOWN * 0.06)
            
            caja = RoundedRectangle(corner_radius=RADIO, width=ANCHO, height=ALTO)
            caja.set_fill(color=BG_CAJA, opacity=1).set_stroke(color=BORDE_CAJA, width=2.5)
            
            color_array = NARANJA_TERRACOTA if highlight_last else TXT_PRINCIPAL
            txt_arr = Text(valor_array, font="Monospace", font_size=18, weight=BOLD, color=color_array) 
            txt_inf = Text(label_inf, font=FUENTE, font_size=16, color=TXT_PRINCIPAL) 
            
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
                buff=0.08, color=MARRON_OSCURO, stroke_width=5, max_tip_length_to_length_ratio=0.15
            )
            flechas.add(flecha)

        self.play(FadeIn(pipeline, shift=UP*0.2), FadeIn(flechas, shift=UP*0.2), run_time=1.5)
        self.next_slide()

        tokens_viajeros = frase_base.copy()
        centro_primera_caja = pipeline[0].caja_principal.get_center()
        punto_flujo = Dot(color=PAPEL_TAN, radius=0.18).move_to(centro_primera_caja)
        
        self.play(ReplacementTransform(tokens_viajeros, punto_flujo, path_arc=-PI/3), run_time=1.2)
        self.play(Indicate(pipeline[0].caja_principal, color=PAPEL_TAN, scale_factor=1.05))
        
        for i in range(len(flechas)):
            centro_siguiente_caja = pipeline[i+1].caja_principal.get_center()
            self.play(
                punto_flujo.animate.move_to(centro_siguiente_caja), 
                run_time=0.6, 
                rate_func=linear
            )
            self.play(Indicate(pipeline[i+1].caja_principal, color=PAPEL_TAN, scale_factor=1.05), run_time=0.4)

        prediccion_txt = Text(' Mancha"', font=FUENTE, font_size=42, color=NARANJA_TERRACOTA).move_to(pipeline[-1].caja_principal.get_center())
        self.play(ReplacementTransform(punto_flujo, prediccion_txt), run_time=0.8)
        self.next_slide()

        self.play(
            prediccion_txt.animate.move_to(posicion_destino_mancha),
            run_time=1.5,
            path_arc=-PI/3
        )
        
        frase_final = VGroup(frase_base, prediccion_txt)
        self.play(Circumscribe(frase_final, color=NARANJA_TERRACOTA, time_width=2, stroke_width=4))
        self.next_slide()

        self.limpiar_pantalla()
    def slide_embeddings(self):
        titulo_p1 = Text("Embeddings: ", font_size=42, weight=BOLD, color=TINTA_NEGRA)
        titulo_p2 = Text("De IDs a Vectores", font_size=42, weight=BOLD, color=NARANJA_TERRACOTA)
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 5, RIGHT * 5, color=MARRON_OSCURO).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)
        
        self.play(Write(titulo_completo), Create(linea))

        ejes = Axes(
            x_range=[-4, 4, 1], y_range=[-3, 4, 1],
            x_length=8, y_length=5.0, 
            axis_config={"color": TINTA_NEGRA, "stroke_width": 2, "include_ticks": False}
        ).shift(DOWN * 0.1) 
        
        self.play(Create(ejes))

        v_feliz = np.array([2, -1.5, 0])
        v_triste = np.array([-2, -1.5, 0])
        v_caballero = np.array([2, 1.5, 0])
        v_quijote = np.array([-2, 1.5, 0])

        def crear_vector_2d(coord, color, label_text, direccion):
            p = ejes.c2p(*coord)
            flecha = Arrow(start=ejes.c2p(0,0), end=p, color=color, buff=0, stroke_width=5)
            lbl = Text(label_text, font_size=24, weight=BOLD, color=color)
            lbl.set_background_stroke(color=PAPEL_CREMA, width=4)
            lbl.next_to(p, direccion, buff=0.2)
            return VGroup(flecha, lbl)

        feliz = crear_vector_2d(v_feliz, MARRON_OSCURO, "Feliz", DR)
        triste = crear_vector_2d(v_triste, PAPEL_TAN, "Triste", DL)
        caballero = crear_vector_2d(v_caballero, MARRON_OSCURO, "Caballero", UR)
        quijote = crear_vector_2d(v_quijote, NARANJA_TERRACOTA, "Don Quijote", UL)

        self.play(Create(feliz), Create(triste))
        
        v_relacion = DashedLine(ejes.c2p(*v_feliz), ejes.c2p(*v_triste), color=CAJA_INFERIOR).set_stroke(width=4)
        v_relacion.add_tip()
        
        lbl_relacion = Text("- Felicidad", font_size=20, color=TINTA_NEGRA, weight=BOLD)
        lbl_relacion.set_background_stroke(color=PAPEL_CREMA, width=4)
        lbl_relacion.next_to(v_relacion, DOWN, buff=0.2)
        
        self.play(Create(v_relacion), Write(lbl_relacion))
        self.next_slide()

        self.play(Create(caballero))
        
        v_relacion_movida = v_relacion.copy().shift(ejes.c2p(*v_quijote) - ejes.c2p(*v_triste))
        lbl_relacion_movida = lbl_relacion.copy().next_to(v_relacion_movida, UP, buff=0.2)
        
        self.play(
            Transform(v_relacion, v_relacion_movida),
            Transform(lbl_relacion, lbl_relacion_movida)
        )
        
        punto_llegada = Dot(ejes.c2p(*v_quijote), color=NARANJA_TERRACOTA, radius=0.08)
        self.play(FadeIn(punto_llegada))
        
        self.play(ReplacementTransform(punto_llegada, quijote))
        self.play(Indicate(quijote, color=NARANJA_TERRACOTA))

        formula = MathTex(
            "\\vec{Triste} - \\vec{Feliz} + \\vec{Caballero} \\approx \\vec{Don\\ Quijote}",
            font_size=38, color=TINTA_NEGRA
        ).to_edge(DOWN, buff=1.0) 
        
        self.play(Write(formula))
        self.next_slide()

        self.play(FadeOut(ejes, feliz, triste, caballero, quijote, v_relacion, lbl_relacion, formula))

        input_word = Text('"quijote"', font_size=40, color=TINTA_NEGRA, weight=BOLD).move_to(LEFT * 4.5 + UP * 1.5)
        
        id_token = Text("ID: 1605", font_size=36, weight=BOLD, color=FONDO_CAJA) 
        bg_id = RoundedRectangle(corner_radius=0.2, width=2.5, height=1, color=MARRON_OSCURO).set_fill(MARRON_OSCURO, 1)
        grupo_id = VGroup(bg_id, id_token).move_to(LEFT * 4.5 + DOWN * 0.2)

        self.play(FadeIn(input_word))
        self.next_slide()
        
        self.play(ReplacementTransform(input_word, grupo_id))
        self.next_slide()

        matriz_v = VGroup()
        filas, cols = 8, 10
        COLOR_VECTOR = NARANJA_TERRACOTA
        for i in range(filas):
            fila = VGroup()
            for j in range(cols):
                cuadro = RoundedRectangle(corner_radius=0.05, width=0.35, height=0.35).set_stroke(CAJA_INFERIOR, opacity=0.6)
                if i == 4:
                    cuadro.set_fill(COLOR_VECTOR, opacity=0.2).set_stroke(COLOR_VECTOR, opacity=0.8)
                fila.add(cuadro)
            matriz_v.add(fila.arrange(RIGHT, buff=0.08))
            
        matriz_v.arrange(DOWN, buff=0.08).move_to(RIGHT * 3 + DOWN * 0)
        lbl_matriz = Text("Matriz de Embeddings", font_size=20, weight=BOLD, color=PAPEL_TAN).next_to(matriz_v, UP, buff=0.4)

        self.play(FadeIn(matriz_v), Write(lbl_matriz))
        self.next_slide()

        flecha_busqueda = Arrow(grupo_id.get_right(), matriz_v[4].get_left(), color=MARRON_OSCURO, buff=0.2)
        self.play(GrowArrow(flecha_busqueda))
        
        fila_sel = matriz_v[4].copy().set_fill(COLOR_VECTOR, 0.9).set_stroke(MARRON_OSCURO, 2)
        self.play(Transform(matriz_v[4], fila_sel))
        self.next_slide()

        vector_final = Text("[  0.12,  -0.45,  0.88,  ...  ]", font_size=36, font="Monospace", color=NARANJA_TERRACOTA)
        vector_final.move_to(DOWN * 2.8)
        
        lbl_final = Text("Vector Semántico", font_size=24, weight=BOLD, color=TINTA_NEGRA)
        lbl_size = Text("(768 dimensiones)", font_size=18, color=PAPEL_TAN)
        
        grupo_lbl_final = VGroup(lbl_final, lbl_size).arrange(DOWN, buff=0.1).next_to(vector_final, UP, buff=0.2)

        self.play(
            ReplacementTransform(matriz_v[4].copy(), vector_final),
            Write(grupo_lbl_final),
            run_time=1.5
        )
        self.play(Indicate(vector_final, color=NARANJA_TERRACOTA))
        
        self.next_slide()
        self.limpiar_pantalla()

    def slide_position_embeddings(self):
        titulo_p1 = Text("Embeddings de ", font_size=42, weight=BOLD, color=TINTA_NEGRA)
        titulo_p2 = Text("Posición", font_size=42, weight=BOLD, color=NARANJA_TERRACOTA) 
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 4, RIGHT * 4, color=MARRON_OSCURO).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)
        
        self.play(Write(titulo_completo), Create(linea))
        self.next_slide()

        frase_l1 = Text("La virtud más es perseguida de los malos", font_size=32, color=TINTA_NEGRA)
        frase_l2 = Text("que amada de los buenos.", font_size=32, color=TINTA_NEGRA)
        frase = VGroup(frase_l1, frase_l2).arrange(DOWN, buff=0.15)
        
        nota_problema = Text("Los Transformers procesan todo en paralelo.\n¡Se pierde el orden de las palabras!", font_size=20, color=NARANJA_TERRACOTA).next_to(frase, DOWN, buff=0.5)
        
        self.play(FadeIn(frase))
        self.play(Write(nota_problema))
        self.next_slide()

        palabras = ["La", "virtud", "más", "es", "perseguida", "..."]
        tokens = VGroup(*[Text(w, font_size=28, color=TINTA_NEGRA) for w in palabras]).arrange(RIGHT, buff=0.3)
        
        posiciones = VGroup()
        for i, w in enumerate(palabras):
            texto_pos = f"Pos: {i}" if w != "..." else ""
            pos_token = Text(texto_pos, font_size=16, color=PAPEL_TAN)
            pos_token.next_to(tokens[i], DOWN, buff=0.2)
            posiciones.add(pos_token)

        grupo_tokens = VGroup(tokens, posiciones).move_to(UP * 1.5)
        
        self.play(
            FadeOut(nota_problema),
            ReplacementTransform(frase, grupo_tokens)
        )
        self.next_slide()

        self.play(
            *[FadeOut(t) for t in tokens[1:]],
            *[FadeOut(p) for p in posiciones[1:]],
            tokens[0].animate.move_to(LEFT * 4.5 + UP * 0.8),
            posiciones[0].animate.move_to(LEFT * 4.5 + DOWN * 0.8)
        )

        vec_token = Text("[ 0.10, -0.30,  0.20, ... ]", font_size=26, font="Monospace", color=TINTA_NEGRA)
        lbl_token = Text("Token Embedding (Significado)", font_size=16, weight=BOLD, color=PAPEL_TAN).next_to(vec_token, UP, buff=0.1)
        grupo_vec_token = VGroup(lbl_token, vec_token).move_to(RIGHT * 1 + UP * 1.2)

        vec_pos = Text("[ 0.05, -0.02,  0.01, ... ]", font_size=26, font="Monospace", color=TINTA_NEGRA)
        lbl_pos = Text("Position Embedding (Orden)", font_size=16, weight=BOLD, color=PAPEL_TAN).next_to(vec_pos, UP, buff=0.1)
        grupo_vec_pos = VGroup(lbl_pos, vec_pos).move_to(RIGHT * 1 + DOWN * 0.4)

        signo_mas = Text("+", font_size=36, weight=BOLD, color=TINTA_NEGRA).move_to(RIGHT * 1 + UP * 0.4)
        nota_suma = Text("Suma elemento a elemento (Ambos de 768 dimensiones)", font_size=16, color=MARRON_OSCURO).next_to(vec_pos, DOWN, buff=0.5)

        flecha_token = Arrow(tokens[0].get_right(), grupo_vec_token.get_left(), color=PAPEL_TAN, buff=0.2)
        flecha_pos = Arrow(posiciones[0].get_right(), grupo_vec_pos.get_left(), color=PAPEL_TAN, buff=0.2)

        self.play(
            GrowArrow(flecha_token), FadeIn(grupo_vec_token),
            GrowArrow(flecha_pos), FadeIn(grupo_vec_pos),
            Write(signo_mas), FadeIn(nota_suma)
        )
        self.next_slide()

        linea_suma = Line(LEFT * 2, RIGHT * 4, color=CAJA_INFERIOR).next_to(nota_suma, DOWN, buff=0.2)
        vec_comb = Text("[ 0.15, -0.32,  0.21, ... ]", font_size=26, font="Monospace", color=NARANJA_TERRACOTA)
        lbl_comb = Text("Combined Vector (Listo para el Transformer)", font_size=18, weight=BOLD, color=MARRON_OSCURO).next_to(vec_comb, UP, buff=0.1)
        grupo_vec_comb = VGroup(lbl_comb, vec_comb).next_to(linea_suma, DOWN, buff=0.3)

        self.play(Create(linea_suma))
        self.play(
            ReplacementTransform(VGroup(vec_token.copy(), vec_pos.copy()), vec_comb),
            Write(lbl_comb)
        )
        self.play(Indicate(vec_comb, color=NARANJA_TERRACOTA))
        self.next_slide()

        elementos_a_borrar = [grupo_vec_token, grupo_vec_pos, signo_mas, nota_suma, linea_suma, grupo_vec_comb, tokens[0], posiciones[0], flecha_token, flecha_pos]
        self.play(*[FadeOut(el) for el in elementos_a_borrar])

        titulo_matriz = Text("Tabla de Position Embeddings (GPT-2)", font_size=28, weight=BOLD, color=TINTA_NEGRA).move_to(UP * 2)

        fila_0 = Text("Pos 0:    [  0.05, -0.02,  0.01, ... (768) ]", font_size=22, font="Monospace", color=PAPEL_TAN)
        fila_1 = Text("Pos 1:    [  0.12,  0.45, -0.30, ... (768) ]", font_size=22, font="Monospace", color=PAPEL_TAN)
        puntos = Text("...", font_size=30, color=TINTA_NEGRA).rotate(PI/2) 
        fila_n = Text("Pos 1023: [ -0.80,  0.11,  0.90, ... (768) ]", font_size=22, font="Monospace", color=PAPEL_TAN)

        matriz = VGroup(fila_0, fila_1, puntos, fila_n).arrange(DOWN, buff=0.3).next_to(titulo_matriz, DOWN, buff=0.5)

        llave = Brace(matriz, direction=LEFT, color=MARRON_OSCURO)
        texto_llave = Text("1024 filas\n(block_size)", font_size=20, color=MARRON_OSCURO).next_to(llave, LEFT, buff=0.2)

        nota_context = Text("¡Este límite físico es el Context Window del modelo!", font_size=26, weight=BOLD, color=NARANJA_TERRACOTA).next_to(matriz, DOWN, buff=0.6)

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
        self.play(Indicate(nota_context, color=NARANJA_TERRACOTA, scale_factor=1.1))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_layer_normalization(self):
        titulo_p1 = Text("Layer ", font_size=42, weight=BOLD, color=TINTA_NEGRA)
        titulo_p2 = Text("Normalization", font_size=42, weight=BOLD, color=NARANJA_TERRACOTA) 
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 4, RIGHT * 4, color=MARRON_OSCURO).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)
        
        self.play(Write(titulo_completo), Create(linea))
        self.next_slide()

        nota_caos = Text("Sin normalización: Los valores explotan o desaparecen", font_size=24, color=NARANJA_TERRACOTA).move_to(UP * 1.5)
        
        vec_inestable = Text("[ 1045.2,  0.0001, -532.8, ... ]", font_size=32, font="Monospace", color=TINTA_NEGRA)
        
        self.play(Write(nota_caos))
        self.play(FadeIn(vec_inestable, shift=UP))
        self.play(Indicate(vec_inestable, color=NARANJA_TERRACOTA, scale_factor=1.2))
        self.next_slide()

        nota_estable = Text("Con LayerNorm: Se fuerza una Media=0 y Varianza=1", font_size=24, color=MARRON_OSCURO).move_to(UP * 1.5)
        
        vec_estable = Text("[  1.23,  -0.45,   0.89, ... ]", font_size=32, font="Monospace", color=TINTA_NEGRA)
        
        self.play(
            ReplacementTransform(nota_caos, nota_estable),
            ReplacementTransform(vec_inestable, vec_estable)
        )
        self.next_slide()

        self.play(FadeOut(nota_estable), FadeOut(vec_estable))
        
        formula = MathTex(
            r"\text{output} = \frac{\text{input} - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta",
            substrings_to_isolate=[r"\epsilon", r"\times \gamma + \beta"],
            color=TINTA_NEGRA
        ).scale(1.2).move_to(UP * 0.5)

        lbl_formula = Text("La Fórmula (por cada token):", font_size=20, weight=BOLD, color=MARRON_OSCURO).next_to(formula, UP, buff=0.5)

        self.play(Write(lbl_formula), FadeIn(formula))
        self.next_slide()

        parte_eps = formula.get_part_by_tex(r"\epsilon")
        caja_eps = SurroundingRectangle(parte_eps, color=NARANJA_TERRACOTA, buff=0.05)
        nota_eps = Text("eps: Evita dividir por cero (ej. 0.00001)", font_size=18, color=NARANJA_TERRACOTA).next_to(caja_eps, DOWN, buff=0.5)
        
        self.play(Create(caja_eps), FadeIn(nota_eps, shift=UP))
        self.next_slide()

        self.play(FadeOut(caja_eps), FadeOut(nota_eps))
        
        parte_params = formula.get_part_by_tex(r"\times \gamma + \beta")
        caja_params = SurroundingRectangle(parte_params, color=MARRON_OSCURO, buff=0.1)
        nota_params = Text("gamma / beta: Parámetros aprendidos\n¡Le dan flexibilidad al modelo!", font_size=18, color=MARRON_OSCURO).next_to(caja_params, DOWN, buff=0.5)

        self.play(Create(caja_params), FadeIn(nota_params, shift=UP))
        self.next_slide()

        self.play(
            *[FadeOut(m) for m in [lbl_formula, formula, caja_params, nota_params]]
        )

        nota_final = Text("Se aplica 2 veces por capa:", font_size=28, weight=BOLD, color=TINTA_NEGRA)
        paso_1 = Text("1. Antes de Attention", font_size=24, color=MARRON_OSCURO)
        paso_2 = Text("2. Antes de FFN (MLP)", font_size=24, color=MARRON_OSCURO)
        
        textos_izq = VGroup(nota_final, paso_1, paso_2).arrange(DOWN, aligned_edge=LEFT, buff=0.4).to_edge(LEFT, buff=1).shift(UP * 0.5)

        def crear_cajita(texto, bg_color, borde_color=MARRON_OSCURO, w=2.6, h=0.7):
            caja = RoundedRectangle(corner_radius=0.1, width=w, height=h, 
                                    fill_color=bg_color, fill_opacity=1, 
                                    stroke_color=borde_color, stroke_width=2)
            lbl = Text(texto, font_size=20, color=TINTA_NEGRA).move_to(caja.get_center())
            return VGroup(caja, lbl)

        b_in = crear_cajita("Input", "#E8DCC4")       
        b_ln1 = crear_cajita("LayerNorm 1", "#D9C8AA")  
        b_attn = crear_cajita("Attention", "#E6A87C", borde_color="#C0573E")
        b_ln2 = crear_cajita("LayerNorm 2", "#D9C8AA") 
        b_mlp = crear_cajita("MLP (FFN)", "#C2B280")    
        b_out = crear_cajita("Output", "#E8DCC4")      

        bloques = VGroup(b_in, b_ln1, b_attn, b_ln2, b_mlp, b_out).arrange(DOWN, buff=0.4)
        
        flechas = VGroup(*[
            Arrow(bloques[i].get_bottom(), bloques[i+1].get_top(), buff=0.1, 
                  max_tip_length_to_length_ratio=0.15, color=MARRON_OSCURO) 
            for i in range(len(bloques)-1)
        ])

        diagrama_simplificado = VGroup(bloques, flechas)

        diagrama_simplificado.scale(0.75).to_edge(RIGHT, buff=3.5).shift(DOWN * 0.2)

        self.play(Write(nota_final), FadeIn(diagrama_simplificado, shift=LEFT))
        self.next_slide()

        resalto_1 = SurroundingRectangle(b_ln1, color=NARANJA_TERRACOTA, stroke_width=4, buff=0.05)
        self.play(
            FadeIn(paso_1, shift=RIGHT), 
            Create(resalto_1), 
            b_ln1[0].animate.set_fill("#C0573E") 
        )
        self.next_slide()

        resalto_2 = SurroundingRectangle(b_ln2, color=NARANJA_TERRACOTA, stroke_width=4, buff=0.05)
        self.play(
            FadeIn(paso_2, shift=RIGHT), 
            Create(resalto_2), 
            b_ln2[0].animate.set_fill("#C0573E") 
        )
        self.next_slide()

        self.limpiar_pantalla()

    def slide_mha_acto1_intuicion(self):

        titulo, linea = self.crear_titulo(
            "Multi-Head Self-Attention",
            palabra_clave="Attention",
            color_clave=NARANJA_TERRACOTA
        )

        subtitulo = Text(
            "La Intuición (Q, K, V)",
            font=FUENTE, font_size=24, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.5)

        self.play(Write(titulo), Create(linea), FadeIn(subtitulo, shift=DOWN))
        self.next_slide()

        nota_qkv = Text(
            "Para entender su contexto, cada palabra genera tres vectores:",
            font=FUENTE, font_size=24, color=MARRON_OSCURO
        ).next_to(subtitulo, DOWN, buff=0.8)

        self.play(FadeIn(nota_qkv, shift=UP))
        
        def crear_tarjeta(letra, nombre, pregunta, color):
            caja = RoundedRectangle(
                corner_radius=0.2, width=3, height=2.2,
                fill_color=FONDO_CAJA, fill_opacity=1,
                stroke_color=color, stroke_width=2
            )
            header_rect = Rectangle(
                width=3, height=0.7, fill_color=color, fill_opacity=1, stroke_width=0
            ).move_to(caja.get_top(), aligned_edge=UP)
            
            header = Intersection(header_rect, caja, color=color, fill_opacity=1, stroke_width=0)

            letra_txt = Text(letra, font=FUENTE, font_size=36, weight=BOLD, color=FONDO_CAJA).move_to(header.get_center())
            nombre_txt = Text(nombre, font=FUENTE, font_size=22, weight=BOLD, color=color).next_to(header, DOWN, buff=0.3)
            pregunta_txt = Text(pregunta, font=FUENTE, font_size=18, slant=ITALIC, color=TINTA_NEGRA).next_to(nombre_txt, DOWN, buff=0.15)

            return VGroup(caja, header, letra_txt, nombre_txt, pregunta_txt)

        tarjetas_qkv = VGroup(
            crear_tarjeta("Q", "Query", "¿Qué busco?", NARANJA_TERRACOTA),
            crear_tarjeta("K", "Key", "¿Qué ofrezco?", MARRON_OSCURO),
            crear_tarjeta("V", "Value", "Contenido", PAPEL_TAN)
        ).arrange(RIGHT, buff=0.6).next_to(nota_qkv, DOWN, buff=0.8)

        self.play(LaggedStart(*[FadeIn(t, shift=UP) for t in tarjetas_qkv], lag_ratio=0.2), run_time=1.5)
        self.next_slide()

        self.play(
            FadeOut(nota_qkv),
            FadeOut(tarjetas_qkv),
            FadeOut(subtitulo)
        )

        palabras = ["El", "hidalgo", "vio", "al", "gigante,", "pero", "este", "era", "un", "molino."]
        oracion = VGroup(*[Text(p, font=FUENTE, font_size=28, color=TINTA_NEGRA) for p in palabras]).arrange(RIGHT, buff=0.2)

        oracion.set_z_index(1)
        oracion.shift(DOWN * 1.0)

        nota_ejemplo = Text(
            "Ejemplo:",
            font=FUENTE, font_size=24, weight=BOLD, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.8)

        self.play(Write(nota_ejemplo), FadeIn(oracion, shift=UP))
        self.next_slide()

        idx_query = 6        # "este"
        idx_key_fuerte = 4   # "gigante,"
        idx_key_debil = 1    # "hidalgo"

        caja_query = SurroundingRectangle(oracion[idx_query], color=NARANJA_TERRACOTA, corner_radius=0.1, buff=0.1)
        caja_query.set_z_index(1)
        label_q = Text("Query (busca antecedente)", font=FUENTE, font_size=16, color=NARANJA_TERRACOTA).next_to(caja_query, DOWN, buff=0.2)

        self.play(Create(caja_query), Write(label_q), oracion[idx_query].animate.set_color(NARANJA_TERRACOTA))
        self.next_slide()

        punto_inicio = oracion[idx_query].get_center() + UP * 0.4
        punto_fin_fuerte = oracion[idx_key_fuerte].get_center() + UP * 0.4
        punto_fin_debil = oracion[idx_key_debil].get_center() + UP * 0.4

        label_k_fuerte = Text("Key", font=FUENTE, font_size=14, color=NARANJA_TERRACOTA).next_to(oracion[idx_key_fuerte], UP, buff=0.1)
        label_k_debil = Text("Key", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(oracion[idx_key_debil], UP, buff=0.1).set_opacity(0.5)

        flecha_fuerte = CurvedArrow(
            start_point=punto_inicio, end_point=punto_fin_fuerte,
            angle=PI/1.2, color=NARANJA_TERRACOTA, stroke_width=3
        ).set_z_index(-1) 
        
        peso_fuerte = Text("0.85", font=FUENTE, font_size=22, color=NARANJA_TERRACOTA).next_to(flecha_fuerte.point_from_proportion(0.5), UP, buff=0.1).set_z_index(-1)

        flecha_debil = CurvedArrow(
            start_point=punto_inicio, end_point=punto_fin_debil,
            angle=PI/2, color=MARRON_OSCURO, stroke_width=3
        ).set_z_index(-1)
        
        flecha_debil.set_stroke(opacity=0.4)
        flecha_debil.get_tip().set_fill(opacity=0.4)
        flecha_debil.get_tip().set_stroke(opacity=0.4)
        
        peso_debil = Text("0.10", font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(flecha_debil.point_from_proportion(0.5), UP, buff=0.1).set_opacity(0.4).set_z_index(-1)

        self.play(
            FadeIn(label_k_debil, shift=DOWN*0.2), oracion[idx_key_debil].animate.set_opacity(0.5),
            FadeIn(label_k_fuerte, shift=DOWN*0.2), oracion[idx_key_fuerte].animate.set_color(NARANJA_TERRACOTA),
            run_time=1
        )
        
        self.play(
            Create(flecha_debil), 
            Create(flecha_fuerte),
            run_time=1.2
        )
        
        self.play(
            FadeIn(peso_debil, shift=DOWN*0.1),
            FadeIn(peso_fuerte, shift=DOWN*0.1),
            run_time=0.8
        )
        self.next_slide()

        nota_final = Text(
            "Value: 'este' absorbe el significado de 'gigante'",
            font=FUENTE, font_size=22, color=MARRON_OSCURO
        ).next_to(label_q, DOWN, buff=0.8)
        
        self.play(Write(nota_final))

        particulas = VGroup(*[
            Dot(point=oracion[idx_key_fuerte].get_center() + np.random.uniform(-0.2, 0.2, 3), radius=0.04, color=NARANJA_TERRACOTA).set_z_index(2)
            for _ in range(20)
        ])

        self.play(FadeIn(particulas, lag_ratio=0.1), run_time=0.5)
        self.play(
            LaggedStart(*[p.animate.move_to(oracion[idx_query].get_center() + np.random.uniform(-0.1, 0.1, 3)) for p in particulas], lag_ratio=0.03),
            oracion[idx_query].animate.scale(1.15).set_color(NARANJA_TERRACOTA),
            run_time=1.5
        )
        self.play(
            FadeOut(particulas, shift=DOWN*0.2), 
            oracion[idx_query].animate.scale(1/1.15)
        )
        self.next_slide()

        elementos_escena = VGroup(
            oracion, caja_query, label_q, flecha_fuerte, flecha_debil,
            peso_fuerte, peso_debil, nota_ejemplo, nota_final, label_k_fuerte, label_k_debil
        )
        self.play(FadeOut(elementos_escena))
        
    def slide_mha_acto2_formula(self):
        titulo, linea = self.crear_titulo("Multi-Head Self-Attention", palabra_clave="Attention", color_clave=NARANJA_TERRACOTA)
        subtitulo = Text("La Ecuación de Atención", font=FUENTE, font_size=24, color=MARRON_OSCURO).next_to(linea, DOWN)
        
        self.play(Write(titulo), Create(linea), FadeIn(subtitulo, shift=DOWN))
        self.next_slide()
        
        formula = MathTex(
            r"\text{Attention}(Q, K, V) = ", # Índice 0
            r"\text{softmax}",               # Índice 1
            r"\left( \frac{",                # Índice 2
            r"Q K^T",                        # Índice 3
            r"}{",                           # Índice 4
            r"\sqrt{d_k}",                   # Índice 5
            r"} \right) ",                   # Índice 6
            r"V",                            # Índice 7
            color=TINTA_NEGRA, font_size=48
        ).move_to(UP * 0.5)

        self.play(FadeIn(formula, shift=UP))
        self.next_slide()

        partes_explicacion = [
            (3, "Similitud: Medimos qué tanto se relacionan los tokens.", NARANJA_TERRACOTA),
            (5, "Escalado: Estabilizamos gradientes.", MARRON_OSCURO),
            (1, "Normalización: Convertimos scores a probabilidades.", TINTA_NEGRA),
            (7, "Contexto: Extraemos la información ponderada.", PAPEL_TAN)
        ]

        caja_enfoque = None
        txt_enfoque = None

        for idx, desc, col in partes_explicacion:
            parte_formula = formula[idx]
            nueva_caja = SurroundingRectangle(parte_formula, color=col, buff=0.1)
            
            nuevo_txt = Text(desc, font=FUENTE, font_size=22, color=col).next_to(formula, DOWN, buff=1.5)
            
            if caja_enfoque:
                self.play(ReplacementTransform(caja_enfoque, nueva_caja), ReplacementTransform(txt_enfoque, nuevo_txt))
            else:
                self.play(Create(nueva_caja), FadeIn(nuevo_txt, shift=UP))
            
            caja_enfoque, txt_enfoque = nueva_caja, nuevo_txt
            self.next_slide()

        self.limpiar_pantalla()

    def slide_mha_acto3_calculo(self):
        titulo, linea = self.crear_titulo("Multi-Head Self-Attention", palabra_clave="Attention", color_clave=NARANJA_TERRACOTA)
        subtitulo = Text("Visualizando el Flujo de Datos", font=FUENTE, font_size=24, color=MARRON_OSCURO).next_to(linea, DOWN)
        
        self.play(Write(titulo), Create(linea), FadeIn(subtitulo, shift=DOWN))
        self.next_slide()

        escala_mat = 0.75 
        buff_espacio = 0.3

        txt_paso1 = Text("1. Calculamos los 'Scores' crudos (Similitud)", font=FUENTE, font_size=24, color=MARRON_OSCURO).move_to(UP * 1.5)
        
        m_q = self.crear_matriz_bloques(3, 3, color_fondo=FONDO_CAJA, valores=["1","1","0", "0","1","1", "1","0","1"]).scale(escala_mat)
        m_kt = self.crear_matriz_bloques(3, 3, color_fondo=CAJA_INFERIOR, valores=["1","0","1", "1","1","0", "0","1","1"]).scale(escala_mat)
        m_scores = self.crear_matriz_bloques(3, 3, color_fondo=PAPEL_TAN, valores=["2","1","1", "1","2","1", "1","1","2"]).scale(escala_mat)
        
        q_lbl = MathTex("Q", color=TINTA_NEGRA).next_to(m_q, UP, buff=0.2)
        kt_lbl = MathTex("K^T", color=TINTA_NEGRA).next_to(m_kt, UP, buff=0.2)
        scores_lbl = MathTex("Scores", color=TINTA_NEGRA).next_to(m_scores, UP, buff=0.2)

        grupo_q = VGroup(q_lbl, m_q)
        grupo_kt = VGroup(kt_lbl, m_kt)
        grupo_scores = VGroup(scores_lbl, m_scores)

        signo_por_1 = MathTex(r"\times", color=TINTA_NEGRA).scale(1.2)
        signo_igual_1 = MathTex("=", color=TINTA_NEGRA).scale(1.2)

        grupo_p1 = VGroup(grupo_q, signo_por_1, grupo_kt, signo_igual_1, grupo_scores)
        grupo_p1.arrange(RIGHT, buff=buff_espacio).move_to(DOWN * 0.5) 

        self.play(FadeIn(txt_paso1, shift=DOWN))
        self.play(FadeIn(VGroup(grupo_q, signo_por_1, grupo_kt), lag_ratio=0.2)) 
        self.next_slide()
        
        self.play(Write(signo_igual_1), TransformFromCopy(VGroup(m_q, m_kt), m_scores), FadeIn(scores_lbl, shift=DOWN)) 
        self.next_slide()

        txt_paso2 = Text("2. Aplicamos Softmax (Convertir a Porcentajes)", font=FUENTE, font_size=24, color=MARRON_OSCURO).move_to(UP * 1.5)
        
        m_soft = self.crear_matriz_bloques(3, 3, color_fondo=FONDO_CAJA, valores=["0.6","0.2","0.2", "0.2","0.6","0.2", "0.2","0.2","0.6"]).scale(escala_mat)
        soft_lbl = MathTex("Softmax", color=TINTA_NEGRA).next_to(m_soft, UP, buff=0.2)
        grupo_soft = VGroup(soft_lbl, m_soft)
        
        flecha_soft = Arrow(LEFT, RIGHT, buff=0.2, color=MARRON_OSCURO)

        grupo_p2 = VGroup(grupo_scores.copy(), flecha_soft, grupo_soft)
        grupo_p2.arrange(RIGHT, buff=0.8).move_to(DOWN * 0.5)

        self.play(
            ReplacementTransform(txt_paso1, txt_paso2),
            FadeOut(VGroup(grupo_q, signo_por_1, grupo_kt, signo_igual_1)), 
            grupo_scores.animate.move_to(grupo_p2[0].get_center()) 
        )

        self.play(GrowArrow(flecha_soft))
        self.play(TransformFromCopy(m_scores, m_soft), FadeIn(soft_lbl, shift=DOWN))
        self.next_slide()

        txt_paso3 = Text("3. Multiplicamos por V (Extracción de Contexto)", font=FUENTE, font_size=24, color=MARRON_OSCURO).move_to(UP * 1.5)
        
        m_v = self.crear_matriz_bloques(3, 3, color_fondo=CAJA_INFERIOR, valores=["10","0","0", "0","10","0", "0","0","10"]).scale(escala_mat)
        v_lbl = MathTex("V", color=TINTA_NEGRA).next_to(m_v, UP, buff=0.2)
        grupo_v = VGroup(v_lbl, m_v)

        m_out = self.crear_matriz_bloques(3, 3, color_fondo=NARANJA_TERRACOTA, color_texto=FONDO_CAJA, valores=["6","2","2", "2","6","2", "2","2","6"]).scale(escala_mat)
        out_lbl = MathTex("Output", color=TINTA_NEGRA).next_to(m_out, UP, buff=0.2)
        grupo_out = VGroup(out_lbl, m_out)

        signo_por_2 = MathTex(r"\times", color=TINTA_NEGRA).scale(1.2)
        signo_igual_2 = MathTex("=", color=TINTA_NEGRA).scale(1.2)

        grupo_p3 = VGroup(grupo_soft.copy(), signo_por_2, grupo_v, signo_igual_2, grupo_out)
        grupo_p3.arrange(RIGHT, buff=buff_espacio).move_to(DOWN * 0.5)

        self.play(
            ReplacementTransform(txt_paso2, txt_paso3), 
            FadeOut(grupo_scores), 
            FadeOut(flecha_soft),
            grupo_soft.animate.move_to(grupo_p3[0].get_center())
        )
        
        self.play(FadeIn(signo_por_2), FadeIn(grupo_v, shift=LEFT))
        self.next_slide()
        
        self.play(Write(signo_igual_2), TransformFromCopy(VGroup(m_soft, m_v), m_out), FadeIn(out_lbl, shift=DOWN))
        self.next_slide()
        
        elementos_finales = VGroup(txt_paso3, grupo_soft, signo_por_2, grupo_v, signo_igual_2, grupo_out)
        self.play(FadeOut(elementos_finales))
        self.limpiar_pantalla()

    def slide_mha_acto4_multihead(self):
        titulo, linea = self.crear_titulo("Multi-Head Self-Attention", palabra_clave="Attention", color_clave=NARANJA_TERRACOTA)
        subtitulo = Text("¿Por qué 'Multi-Head'?", font=FUENTE, font_size=24, color=MARRON_OSCURO).next_to(linea, DOWN)
        
        self.play(Write(titulo), Create(linea), FadeIn(subtitulo, shift=DOWN))
        self.next_slide()

        vector_completo = Rectangle(width=10, height=0.8, fill_color=FONDO_CAJA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2)
        txt_dim = Text("Vector de Incrustación (ej. 768 dimensiones)", font=FUENTE, font_size=20, color=TINTA_NEGRA).move_to(vector_completo)
        
        self.play(FadeIn(vector_completo), FadeIn(txt_dim))
        self.next_slide()

        colores_h = [NARANJA_TERRACOTA, MARRON_OSCURO, PAPEL_TAN, CAJA_INFERIOR] * 3 
        
        cabezas = VGroup(*[
            Rectangle(width=10/12, height=1.2, fill_color=colores_h[i], fill_opacity=0.9, stroke_color=FONDO_CAJA, stroke_width=1)
            for i in range(12)
        ]).arrange(RIGHT, buff=0.02).move_to(vector_completo)

        txt_mh = Text("12 Cabezas de Atención (64 dims c/u)", 
                      font=FUENTE, font_size=22, color=MARRON_OSCURO).next_to(cabezas, DOWN, buff=1)

        self.play(
            ReplacementTransform(vector_completo, cabezas),
            FadeOut(txt_dim),
            FadeIn(txt_mh, shift=UP)
        )
        
        self.play(LaggedStart(*[Indicate(c, scale_factor=1.1, color=PAPEL_CREMA) for c in cabezas], lag_ratio=0.1))
        self.next_slide()

        final_msg = Text("Cada cabeza hace su propia multiplicación de matrices Q, K, V", 
                         font=FUENTE, font_size=20, color=TINTA_NEGRA).next_to(txt_mh, DOWN, buff=0.5)
        self.play(Write(final_msg))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_causal_masking(self):
        titulo, linea = self.crear_titulo("Causal Masking", palabra_clave="Masking", color_clave=NARANJA_TERRACOTA)
        self.play(Write(titulo), Create(linea))
        
        subtitulo = Text("El 'Triángulo de No Mirar' (No Peeking)", font=FUENTE, font_size=26, color=MARRON_OSCURO)
        
        tokens = ["El", "amor", "nunca", "hizo", "ningún", "cobarde"]
        frase_display = VGroup(
            Text("Secuencia: ", font=FUENTE, font_size=24, color=TINTA_NEGRA, weight=BOLD),
            Text(" ".join([f"[{t}]" for t in tokens]), font=FUENTE, font_size=24, color=NARANJA_TERRACOTA)
        ).arrange(RIGHT)

        grupo_intro = VGroup(subtitulo, frase_display).arrange(DOWN, buff=0.5).next_to(linea, DOWN, buff=0.6)

        self.play(FadeIn(grupo_intro, shift=UP))
        self.next_slide()
        self.play(FadeOut(grupo_intro))

        texto_explicativo = Text("Sin máscara: 'El' ya sabe que la frase termina en 'cobarde'", 
                                 font=FUENTE, font_size=20, weight=BOLD, color=NARANJA_TERRACOTA).next_to(linea, DOWN, buff=0.4)
        self.play(FadeIn(texto_explicativo))

        val_scores = [f"{random.uniform(0.1, 3.5):.1f}" for _ in range(36)]
        
        

        mat_scores = self.crear_matriz_bloques(6, 6, color_fondo=FONDO_CAJA, valores=val_scores)
        
        etiquetas_filas = VGroup(*[Text(t, font=FUENTE, font_size=14, color=MARRON_OSCURO) for t in tokens]).arrange(DOWN, buff=0.42).next_to(mat_scores, LEFT, buff=0.25)
        etiquetas_cols = VGroup(*[Text(t, font=FUENTE, font_size=14, color=MARRON_OSCURO).rotate(PI/4) for t in tokens]).arrange(RIGHT, buff=0.28).next_to(mat_scores, UP, buff=0.15)

        grupo_atencion = VGroup(mat_scores, etiquetas_filas, etiquetas_cols).scale(0.85).move_to(DOWN * 0.6)
        
        self.play(FadeIn(grupo_atencion, shift=UP))
        self.next_slide()

        animaciones_trampa = []
        for i in range(6):
            for j in range(6):
                if j > i: 
                    bloque = mat_scores[i][j]
                    animaciones_trampa.append(bloque[0].animate.set_fill(NARANJA_TERRACOTA, opacity=0.3))
                    animaciones_trampa.append(bloque[0].animate.set_stroke(NARANJA_TERRACOTA, width=2))

        self.play(*animaciones_trampa, run_time=1.2)
        self.next_slide()

        texto_infinito = Text("Solución: Forzar el valor -Infinito en el triángulo superior", 
                              font=FUENTE, font_size=20, weight=BOLD, color=MARRON_OSCURO).move_to(texto_explicativo)
        self.play(Transform(texto_explicativo, texto_infinito))

        animaciones_infinito = []
        textos_a_reemplazar = []
        
        for i in range(6):
            for j in range(6):
                if j > i:
                    bloque_score = mat_scores[i][j]
                    inf_text = MathTex(r"-\infty", font_size=24, color=FONDO_CAJA).move_to(bloque_score.get_center())
                    
                    animaciones_infinito.append(bloque_score[0].animate.set_fill(MARRON_OSCURO, opacity=1))
                    animaciones_infinito.append(FadeIn(inf_text))
                    textos_a_reemplazar.append(bloque_score[1])
                    bloque_score[1] = inf_text

        self.play(*[FadeOut(t) for t in textos_a_reemplazar], *animaciones_infinito, run_time=1.5)
        self.next_slide()

        texto_softmax = Text("Resultado: Probabilidad 0 de 'mirar' hacia adelante", 
                             font=FUENTE, font_size=20, weight=BOLD, color=TINTA_NEGRA).move_to(texto_explicativo)
        self.play(Transform(texto_explicativo, texto_softmax))

        animaciones_finales = []
        textos_viejos = []

        for i in range(6):
            for j in range(6):
                bloque = mat_scores[i][j]
                if j > i:
                    cero = Text("0", font=FUENTE, font_size=18, color=MARRON_OSCURO).move_to(bloque.get_center())
                    animaciones_finales.append(bloque[0].animate.set_fill(TINTA_NEGRA, opacity=0.05).set_stroke(MARRON_OSCURO, width=1))
                    animaciones_finales.append(FadeIn(cero))
                    textos_viejos.append(bloque[1])
                else:
                    animaciones_finales.append(bloque[0].animate.set_fill(PAPEL_TAN, opacity=1))

        self.play(*[FadeOut(t) for t in textos_viejos], *animaciones_finales, run_time=1.5)
        
        nota_final = Text("Para predecir 'cobarde', el modelo solo puede usar 'El amor nunca hizo ningún'.", 
                          font=FUENTE, font_size=18, color=TINTA_NEGRA).to_edge(DOWN, buff=0.6)
        
        self.play(FadeIn(nota_final, shift=UP))
        self.next_slide()

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

