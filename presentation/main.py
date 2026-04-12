from manim import *
from manim_slides import Slide
from manim_code_blocks import *
import numpy as np
import random
import math


FUENTE = "Goudy Old Style"

MARRON_OSCURO      = "#3D3834"
NARANJA_TERRACOTA  = "#A36536"
PAPEL_CREMA        = "#F2E6D8"
PAPEL_TAN          = "#B78B68"
FONDO_CAJA         = "#FCF3E4"
CAJA_INFERIOR      = "#E0C2A8"
TINTA_NEGRA        = "#1A1A1A"
NEGRO_SUAVE        = "#1E1E1E"
BLANCO             = "#FFFFFF"
MADERA_OSCURA      = "#3E2723"
MADERA_CLARA       = "#5D4037"
LADRILLO           = "#8D6E63"
TERRACOTA          = "#BF360C"
TEJA               = "#D84315"
HIERRO             = "#424242"
ACERO              = "#78909C"
PERGAMINO          = "#F4E4BC"
ORO_VIEJO          = "#D4AF37"
LATON              = "#B5A642"
ROJO_SANGRE        = "#8B0000"
AZUL_NOCHE         = "#000080"
VERDE_BOSQUE       = "#228B22"
TIERRA_MANCHEGA    = "#D4B872"
ARENA_MANCHEGA     = "#E6D3A8"
BARRO_MANCHEGO     = "#8B5A2B"
ROJO_TOMATE        = "#E24A4A"
VERDE_OLIVA        = "#6B8E23"
LAVANDA            = "#C4C4FF"
SALMON_CLARO       = "#F2D5CE"
CREMA_CALIDA       = "#E8DCC4"
BEIGE_MEDIO        = "#D9C8AA"
LADRILLO_VIVO      = "#C0573E"
SALMON_ATENCION    = "#E6A87C"
ARENA_DORADA       = "#C2B280"
NARANJA_CLARO      = "#FFCC99"
AMARILLO_PALIDO    = "#FFFFCC"
MENTA_PALIDA       = "#CCFFCC"
PERGAMINO_CLARO    = "#F4EBD0"
ROJO_MAC           = "#FF5F56"
AMARILLO_MAC       = "#FFBD2E"
VERDE_MAC          = "#27C93F"
OCRE_CERVANTINO    = "#C9A84C"
MARRON_QUIJOTE     = "#8B2500"
ROJO_CONTRA        = "#B33A3A"


RUST_SNIPPETS = {
    "tensor.rs": """pub struct Tensor {
    pub datos: Vec<f32>,      // Arreglo plano de valores en memoria lineal
    pub forma: Vec<usize>,    // Dimensiones (ej. [lote, secuencia, dim_embd])
    pub saltos: Vec<usize>,   // Saltos para indexar la memoria 1D como N-D
}""",

    "matmul_base.rs": """// Versión secuencial — se usa cuando m*n*k < 1_000 operaciones
// (evita el overhead de lanzar hilos para trabajo pequeño)
let mut resultado = vec![0.0; m * n];
for i in 0..m {
    for j in 0..n {
        let mut suma = 0.0;
        for l in 0..k {
            // Producto punto clásico: filas de self × columnas de other
            suma += self.datos[i * k + l] * other.datos[l * n + j];
        }
        resultado[i * n + j] = suma;
    }
}
Tensor::new(resultado, vec![m, n])""",

    "cache_blocking.rs": """// Tamaño de bloque 8×8 = 256 bytes → cabe en la caché L1 (32-64 KB)
const TAM_BLOQUE: usize = 8;

resultado
    .par_chunks_mut(TAM_BLOQUE * n)   // cada hilo recibe TAM_BLOQUE filas
    .enumerate()
    .for_each(|(i_bloque, bloque_resultado)| {
        let i_inicio = i_bloque * TAM_BLOQUE;
        for j_inicio in (0..n).step_by(TAM_BLOQUE) {
            for k_inicio in (0..k).step_by(TAM_BLOQUE) {
                // Bucle interno: acceso secuencial → alta tasa de aciertos de caché
                for i in i_inicio..(i_inicio + TAM_BLOQUE).min(m) {
                    for k_idx in k_inicio..(k_inicio + TAM_BLOQUE).min(k) {
                        let val_a = self.datos[i * k + k_idx];
                        Self::matmul_interno_simd(
                            val_a,
                            &other.datos[k_idx * n + j_inicio..],
                            &mut bloque_resultado[..],
                        );
                    }
                }
            }
        }
    })""",

    "parallel_rayon.rs": """// par_chunks_mut divide el buffer de resultado en franjas de TAM_BLOQUE filas.
// Rayon asigna cada franja a un hilo del pool automáticamente.
resultado
    .par_chunks_mut(TAM_BLOQUE * n)
    .enumerate()
    .for_each(|(i_bloque, bloque_resultado)| {
        // i_bloque → qué grupo de filas somos
        // bloque_resultado → porción exclusiva del buffer de salida (sin locks)
        let i_inicio = i_bloque * TAM_BLOQUE;
        let i_fin = (i_inicio + TAM_BLOQUE).min(m);
        // ... cálculo independiente para estas filas ...
    });""",

    "batched_matmul.rs": """// Tensor 4D: [lote, num_cabezas, sec, dim_cabeza]
// Cada par (lote, cabeza) ejecuta su propia matmul 2D de forma independiente.
resultado
    .par_chunks_mut(sec1 * sec2)   // un slab por (lote, cabeza)
    .enumerate()
    .for_each(|(idx_lc, bloque)| {
        let b = idx_lc / num_cabezas;   // índice de lote
        let h = idx_lc % num_cabezas;   // índice de cabeza
        // matmul 2D aislada para este (b, h) — escribe en bloque exclusivo
        for i in 0..sec1 {
            for j in 0..sec2 {
                let mut suma = 0.0;
                for l in 0..dim_interna {
                    let idx_self  = ((b * num_cabezas + h) * sec1 + i) * dim_interna + l;
                    let idx_other = ((b * num_cabezas + h) * dim_interna + l) * sec2 + j;
                    suma += self.datos[idx_self] * other.datos[idx_other];
                }
                bloque[i * sec2 + j] = suma;
            }
        }
    });""",

    "simd_vectorization.rs": """// Bucle más interno del cache-blocked matmul.
// iter_mut().zip() elimina el bounds-checking en runtime.
// Sin esas ramas extra, LLVM auto-vectoriza con AVX2 (8×f32) o NEON (4×f32).
#[inline(always)]
fn matmul_interno_simd(val_a: f32, b: &[f32], resultado: &mut [f32]) {
    for (r, &val_b) in resultado.iter_mut().zip(b.iter()) {
        *r += val_a * val_b;
        // Una instrucción SIMD procesa 4-8 de estos pares en paralelo
    }
}""",

    "BDPtokenizer.rs": """#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenizadorBPE {
    /// Asocia cadenas de texto de tokens a sus IDs enteros
    /// Siempre comienza con 256 tokens base (un token por cada byte 0-255)
    vocabulario: HashMap<String, usize>,

    /// Reglas de fusión aprendidas durante el entrenamiento.
    /// Se aplican en orden durante la codificación.
    fusiones: Vec<(String, String)>,
}""",

    "pair_counts.rs": """let conteo_pares: HashMap<(String, String), usize> = tokens
    // par_chunks divide el vector entre los hilos disponibles
    .par_chunks(tam_fragmento)
    .enumerate()
    // fold: cada hilo acumula sus propios conteos locales (sin Mutex → lock-free)
    .fold(HashMap::new, |mut conteos_locales, (idx_fragmento, fragmento)| {
        for ventana in fragmento.windows(2) {
            let par = (ventana[0].clone(), ventana[1].clone());
            *conteos_locales.entry(par).or_insert(0) += 1;
        }
        // Manejo de fronteras: el par entre el último token del fragmento N
        // y el primero del fragmento N+1 quedaría excluido sin este bloque
        if let (Some(ultimo), Some(siguiente)) = (
            fragmento.last(),
            tokens.get(idx_fragmento * tam_fragmento + fragmento.len()),
        ) {
            *conteos_locales.entry((ultimo.clone(), siguiente.clone())).or_insert(0) += 1;
        }
        conteos_locales
    })
    // reduce: fusiona los HashMaps locales en uno global (patrón MapReduce)
    .reduce(HashMap::new, |mut a, b| {
        for (par, conteo) in b { *a.entry(par).or_insert(0) += conteo; }
        a
    });""",

    "embedding.rs": """pub fn forward(&self, ids_tokens: &[Vec<usize>]) -> Tensor {
    // === 1. Embeddings de tokens (lookup table) ===
    let mut x = self.embedding_tokens.forward(ids_tokens);

    // === 2. Embeddings de posición ===
    // Creamos índices [0, 1, 2, ..., long_sec-1] y los sumamos al tensor
    let posiciones: Vec<Vec<usize>> = vec![(0..long_sec).collect()];
    let emb_pos = self.embedding_posicion.forward(&posiciones);
    // broadcast: [1, long_sec, n_embd] → [lote, long_sec, n_embd]
    for l in 0..tamano_lote {
        for s in 0..long_sec {
            for e in 0..self.configuracion.n_embd {
                x.datos[(l * long_sec + s) * self.configuracion.n_embd + e]
                    += emb_pos.datos[s * self.configuracion.n_embd + e];
            }
        }
    }
    // === 3-5. Bloques transformer → LayerNorm final → proyección al vocabulario ===
    for bloque in &self.bloques { x = bloque.forward(&x); }
    x = self.ln_final.forward(&x);
    self.cabeza_lm.forward(&x)
}""",

    "normalization.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheNormCapa) {
    // 1. Media y varianza a lo largo de la última dimensión
    let media    = x.mean(-1, true);
    let varianza = x.var(-1, true);
    let desv_est = varianza.add_scalar(self.eps).sqrt(); // eps evita división por cero

    // 2. Normalizar: (x - media) / desv_est → media 0, varianza 1
    let x_centrado = x.sub(&media);
    let x_norm     = x_centrado.div(&desv_est);

    // 3. Escala y desplazamiento aprendibles: y = γ * x_norm + β
    let y = x_norm.mul(&self.gamma).add(&self.beta);

    let cache = CacheNormCapa { x: x.clone(), x_norm, media, desv_est };
    (y, cache)
}""",

    "attention.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheAtencion) {
    // 1. Proyecciones Q, K, V separadas
    let (q, cache_q) = self.proy_q.forward(x);
    let (k, cache_k) = self.proy_k.forward(x);
    let (v, cache_v) = self.proy_v.forward(x);

    // 2. Puntuaciones escaladas: (Q @ K^T) / √d_k
    let escala = (self.n_embd as f32).sqrt();
    let puntuaciones = q.matmul(&k.transpose(-2, -1)).mul_scalar(1.0 / escala);

    // 3. Máscara causal: posiciones futuras → -1e9 (≈ -∞ antes del softmax)
    let puntuaciones_enmascaradas = puntuaciones.masked_fill(&tensor_mascara, -1e9);

    // 4. Softmax → pesos de atención → multiplicar por V
    let pesos_atencion = puntuaciones_enmascaradas.softmax(-1);
    let salida_atencion = pesos_atencion.matmul(&v);

    // 5. Proyección de salida + dropout residual
    let (y_proy, cache_salida) = self.proy_salida.forward(&salida_atencion);
    let (y, cache_dropout_resid) = self.dropout_resid.forward(&y_proy);
    (y, /* cache ... */)
}""",

    "gelu.rs": """pub fn gelu_forward(x: &Tensor) -> Tensor {
    // Aproximación con tanh (más rápida que la CDF exacta, suficientemente precisa)
    // GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
    let resultado = x
        .datos
        .par_iter()   // Rayon: paraleliza elemento a elemento entre núcleos
        .map(|&valor| {
            0.5 * valor
                * (1.0
                    + ((2.0 / std::f32::consts::PI).sqrt()
                        * (valor + 0.044715 * valor.powi(3)))
                    .tanh())
        })
        .collect();
    Tensor::new(resultado, x.forma.clone())
}""",

    "softmax.rs": """// Softmax 2D por fila — caso común para las puntuaciones de atención
let resultado: Vec<f32> = (0..filas)
    .into_par_iter()  // Rayon: cada fila se calcula en paralelo
    .flat_map_iter(|i| {
        let fila = &self.datos[i * columnas..(i + 1) * columnas];

        // 1. Restamos el máximo para evitar overflow en exp() (truco de estabilidad)
        let maximo = fila.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // 2. exp(x - max)
        let valores_exp: Vec<f32> = fila.iter().map(|&x| (x - maximo).exp()).collect();

        // 3. Normalizar → la fila suma exactamente 1.0
        let suma: f32 = valores_exp.iter().sum();
        valores_exp.into_iter().map(move |val| val / suma)
    })
    .collect();""",

    "mlp_forward.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheMLP) {
    // 1. Expansión 4×: n_embd → 4·n_embd
    //    Le da al modelo "espacio" para representar transformaciones complejas
    let (h, cache_fc1) = self.fc1.forward(x);

    // 2. Activación GELU: no-linealidad suave (mejor que ReLU en transformers)
    let h_activada = gelu_forward(&h);

    // 3. Compresión: 4·n_embd → n_embd (vuelve a la dimensionalidad del transformer)
    let (y_proy, cache_fc2) = self.fc2.forward(&h_activada);

    // 4. Dropout residual: regularización para evitar sobreajuste (overfitting)
    let (y, cache_dropout_resid) = self.dropout_resid.forward(&y_proy);

    // Guardamos la pre-activación (h) porque gelu_backward la necesitará
    let cache = CacheMLP { cache_fc1, h, h_activada, cache_fc2, cache_dropout_resid };
    (y, cache)
}""",

    "compute_loss.rs": """pub fn calcular_perdida(&self, logits: &Tensor, objetivos: &[usize]) -> f32 {
    let mut perdida_total = 0.0;
    for (i, &objetivo) in objetivos.iter().enumerate() {
        let inicio = i * self.config.vocab_size;
        let slice_logits = &logits.datos[inicio..inicio + self.config.vocab_size];

        // Truco de estabilidad: restamos el máximo para evitar overflow en exp()
        let logit_max = slice_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let suma_exp: f32 = slice_logits.iter().map(|&x| (x - logit_max).exp()).sum();

        // Cross-Entropy = -log P(token_correcto)
        // log P = logit_objetivo - logit_max - log(Σ exp)
        let log_prob = (slice_logits[objetivo] - logit_max) - suma_exp.ln();
        perdida_total -= log_prob;
    }
    // Promediamos a lo largo de la secuencia → pérdida por token
    perdida_total / objetivos.len() as f32
}""",

    "linear_backward.rs": """pub fn backward(&self, grad_salida: &Tensor, cache: &CacheLineal) -> GradientesLineales {
    // grad_W = x^T @ grad_salida
    // "¿cuánto cambia la pérdida si cambiamos cada peso?"
    let grad_peso = cache.x.transpose(-2, -1).matmul(grad_salida);

    // grad_b = Σ grad_salida  (suma sobre todas las posiciones de la secuencia)
    let datos_grad_sesgo: Vec<f32> = (0..self.sesgo.datos.len())
        .map(|i| (0..grad_salida.forma[0])
            .map(|fila| grad_salida.datos[fila * grad_salida.forma[1] + i])
            .sum())
        .collect();
    let grad_sesgo = Tensor::new(datos_grad_sesgo, self.sesgo.forma.clone());

    // grad_x = grad_salida @ W^T  → el gradiente sigue bajando a la capa anterior
    let grad_x = grad_salida.matmul(&self.peso.transpose(-2, -1));

    GradientesLineales { peso: grad_peso, sesgo: grad_sesgo, x: grad_x }
}""",

    "block_backward.rs": """pub fn backward(&self, grad_salida: &Tensor, cache: &CacheBloque) -> GradientesBloque {
    // La conexión residual bifurca el gradiente: fluye por AMBOS caminos a la vez
    let mut grad_x_despues_atencion = grad_salida.clone(); // camino directo (skip)

    // --- Camino del MLP (segunda conexión residual) ---
    let grads_mlp = self.mlp.backward(grad_salida, &cache.cache_mlp);
    let grads_ln2 = self.ln2.backward(&grads_mlp.x, &cache.cache_ln2);
    // Acumulamos: el gradiente del skip + el del camino MLP
    for i in 0..grad_x_despues_atencion.datos.len() {
        grad_x_despues_atencion.datos[i] += grads_ln2.x.datos[i];
    }

    // --- Camino de Atención (primera conexión residual) ---
    let grads_atencion = self.atencion.backward(&grad_x_despues_atencion, &cache.cache_atencion);
    let grads_ln1 = self.ln1.backward(&grads_atencion.x, &cache.cache_ln1);
    // Acumulamos de nuevo: skip + camino atención
    let mut grad_x = grad_x_despues_atencion;
    for i in 0..grad_x.datos.len() {
        grad_x.datos[i] += grads_ln1.x.datos[i];
    }
    GradientesBloque { ln1_gamma: grads_ln1.gamma, ln1_beta: grads_ln1.beta,
                       atencion: grads_atencion, ln2_gamma: grads_ln2.gamma,
                       ln2_beta: grads_ln2.beta, mlp: grads_mlp, x: grad_x }
}""",

    "adamw_update.rs": """pub fn actualizar_adamw(model: &mut GPT2Entrenable, grads: &GradientesGPT2,
                         optimizer: &mut OptimizadorAdamW, lr: f32, weight_decay: f32) {
    optimizer.step += 1;
    let correcc1 = 1.0 - optimizer.beta1.powf(optimizer.step as f32); // corrige sesgo de m
    let correcc2 = 1.0 - optimizer.beta2.powf(optimizer.step as f32); // corrige sesgo de v

    // Paralelizamos con Rayon para tensores grandes (>1000 elementos)
    param.datos.par_iter_mut()
        .zip(grad.datos.par_iter())
        .zip(m.datos.par_iter_mut().zip(v.datos.par_iter_mut()))
        .for_each(|((param_val, &grad_val), (m_val, v_val))| {
            // 1. Decaimiento de pesos desacoplado (solo en matrices 2D, no en sesgos/LayerNorm)
            if aplicar_decay { *param_val *= 1.0 - lr * weight_decay; }

            // 2. Primer momento (momentum): promedio móvil exponencial del gradiente
            *m_val = optimizer.beta1 * *m_val + (1.0 - optimizer.beta1) * grad_val;

            // 3. Segundo momento (varianza): promedio móvil del gradiente al cuadrado
            *v_val = optimizer.beta2 * *v_val + (1.0 - optimizer.beta2) * grad_val * grad_val;

            // 4. Corrección de sesgo + actualización del parámetro
            let m_hat = *m_val / correcc1;
            let v_hat = *v_val / correcc2;
            *param_val -= lr * m_hat / (v_hat.sqrt() + optimizer.epsilon);
        });
}""",

    "dropout.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheDropout) {
    // Durante inferencia (o tasa = 0) → pasamos los datos sin tocar
    if !self.entrenando || self.tasa == 0.0 {
        return (x.clone(), CacheDropout { mascara: None, escala: 1.0 });
    }

    // Inverted dropout: escalamos los valores supervivientes en (1 / (1 - tasa))
    // para mantener la misma esperanza matemática que sin dropout
    let escala = 1.0 / (1.0 - self.tasa);
    let mut mascara = Vec::with_capacity(x.datos.len());
    let mut salida  = Tensor::ceros(x.forma.clone());

    for i in 0..x.datos.len() {
        let mantener = rand::random::<f32>() > self.tasa; // ¿sobrevive esta neurona?
        mascara.push(mantener);
        if mantener {
            salida.datos[i] = x.datos[i] * escala; // boost al superviviente
        }
        // else: la neurona queda a 0 (apagada)
    }
    // Guardamos la MISMA máscara: el backward la reutilizará para bloquear
    // los gradientes de las neuronas que fueron apagadas en el forward
    let cache = CacheDropout { mascara: Some(mascara), escala };
    (salida, cache)
}""",

    "temperature.rs": """pub fn generar(&self, prompt: &[usize], max_tokens: usize, temperatura: f32) -> Vec<usize> {
    let mut tokens = prompt.to_vec();

    // Bucle autoregresivo: cada token generado se convierte en la entrada del siguiente
    for _ in 0..max_tokens {
        let (logits, _) = self.forward(&tokens);

        // 1. Tomamos solo los logits de la última posición (el próximo token a predecir)
        let inicio_ultima = (tokens.len() - 1) * self.config.vocab_size;
        let ultimos_logits = &logits.datos[inicio_ultima..inicio_ultima + self.config.vocab_size];

        // 2. Temperatura: dividimos por T antes del softmax
        //    T < 1.0 → distribución más concentrada (respuestas conservadoras)
        //    T > 1.0 → distribución más plana (respuestas más creativas/aleatorias)
        let logits_escalados: Vec<f32> = ultimos_logits.iter().map(|&x| x / temperatura).collect();

        // 3. Softmax con estabilidad numérica → probabilidades
        let logit_max = logits_escalados.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let valores_exp: Vec<f32> = logits_escalados.iter().map(|&x| (x - logit_max).exp()).collect();
        let suma: f32 = valores_exp.iter().sum();
        let probs: Vec<f32> = valores_exp.iter().map(|&x| x / suma).collect();

        // 4. Muestreo estocástico: tiramos los dados según la distribución
        tokens.push(muestrear_de_probs(&probs));
        if tokens.len() >= self.config.block_size { break; }
    }
    tokens
}"""
}


def crear_llanuras_manchegas():
    """Fondo de colinas manchegas (tres elipses semitransparentes).

    Se coloca en z_index=-10 para que quede siempre detrás de todo.

    Returns:
        VGroup con las tres colinas.
    """
    colina_fondo = Ellipse(width=18, height=6, fill_color=TIERRA_MANCHEGA, fill_opacity=0.06, stroke_width=0)
    colina_fondo.move_to(DOWN * 3.5 + LEFT * 3)
    
    colina_media = Ellipse(width=16, height=4, fill_color=ARENA_MANCHEGA, fill_opacity=0.08, stroke_width=0)
    colina_media.move_to(DOWN * 3.8 + RIGHT * 3)
    
    colina_frente = Ellipse(width=20, height=3.5, fill_color=BARRO_MANCHEGO, fill_opacity=0.05, stroke_width=0)
    colina_frente.move_to(DOWN * 4)
    
    llanuras = VGroup(colina_fondo, colina_media, colina_frente).set_z_index(-10)
    return llanuras

def crear_molino():
    """Molino de viento cervantino con aspas animables.

    Las aspas se exponen como último elemento del VGroup,
    por lo que puedes animarlas con::

        molino[-1].add_updater(lambda m, dt: m.rotate(-dt * 0.8))

    Returns:
        VGroup(base, ladrillo, techo, puerta, ventana, aspas).
    """
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
    """Sol decorativo con cara y 16 rayos alternos.

    Returns:
        VGroup(rayos, cuerpo, cara).
    """
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
    """Estrella dorada de 4 puntas con brillo central.

    Pensada para girar con un updater::

        estrella.add_updater(lambda m, dt: m.rotate(dt * 0.5))

    Returns:
        VGroup(puntas, brillo).
    """
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
    """Tintero negro con pluma de ave y gotas de tinta.

    Returns:
        VGroup(tintero, pluma, cortes, tallo, punta, gotas).
    """
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
    """Pila de tres libros con marcapáginas dorado.

    Returns:
        VGroup(libro1, marcapaginas, libro2, libro3).
    """
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

def crear_escudo_y_lanza():
    """Escudo redondo con cruz roja y lanza de caballero.

    Returns:
        VGroup(astil, empunadura, punta, escudo).
    """
    astil = Line(DOWN*1.2 + LEFT*0.6, UP*1.2 + RIGHT*0.6, color=MADERA_OSCURA, stroke_width=6)
    vector_dir = astil.get_unit_vector()
    angulo_astil = astil.get_angle()

    centro_empunadura = astil.get_center() + DOWN*0.4
    empunadura = VGroup(*[
        Line(LEFT*0.12, RIGHT*0.12, color=TINTA_NEGRA, stroke_width=4)
        .rotate(angulo_astil + PI/6)
        .move_to(centro_empunadura + vector_dir * (i * 0.04))
        for i in range(-5, 6)
    ])

    punta_base = Polygon(
        UP*0.4, RIGHT*0.12, DOWN*0.1, LEFT*0.12,
        fill_color=ACERO, fill_opacity=1, stroke_color=HIERRO, stroke_width=1.5
    )
    brillo_punta = Polygon(
        UP*0.4, RIGHT*0.12, DOWN*0.1,
        fill_color=WHITE, fill_opacity=0.4, stroke_width=0
    )
    punta = VGroup(punta_base, brillo_punta)
    punta.rotate(angulo_astil - PI/2)
    punta.move_to(astil.get_end() + vector_dir * 0.15)

    centro_escudo = astil.get_center() + DOWN*0.2 + RIGHT*0.2
    
    sombra_escudo = Circle(radius=0.55, fill_color=BLACK, fill_opacity=0.3, stroke_width=0).shift(DR*0.06)
    escudo_borde = Circle(radius=0.55, fill_color=HIERRO, stroke_color=ACERO, stroke_width=3)
    escudo_fondo = Circle(radius=0.48, fill_color=MADERA_CLARA, stroke_color=MADERA_OSCURA, stroke_width=1.5)
    
    cruz_v = Rectangle(width=0.18, height=0.96, fill_color=TERRACOTA, fill_opacity=0.9, stroke_width=0)
    cruz_h = Rectangle(width=0.96, height=0.18, fill_color=TERRACOTA, fill_opacity=0.9, stroke_width=0)
    
    umbo_base = Circle(radius=0.15, fill_color=ACERO, stroke_color=HIERRO, stroke_width=2)
    umbo_brillo = Arc(radius=0.1, start_angle=PI/4, angle=PI/2, color=WHITE, stroke_width=2, stroke_opacity=0.5)
    
    brillo_escudo = Arc(radius=0.49, start_angle=PI/3, angle=PI/2.5, color=WHITE, stroke_width=3, stroke_opacity=0.4)
    
    remaches = VGroup(*[
        VGroup(
            Dot(radius=0.025, color=HIERRO),
            Dot(radius=0.01, color=WHITE).shift(UL*0.008)
        ).move_to(escudo_borde.point_at_angle(i * PI/6))
        for i in range(12)
    ])
    
    escudo = VGroup(
        sombra_escudo, escudo_borde, escudo_fondo, 
        cruz_v, cruz_h, umbo_base, umbo_brillo, 
        brillo_escudo, remaches
    ).move_to(centro_escudo)
    
    return VGroup(astil, empunadura, punta, escudo)

def crear_herradura():
    """Herradura metálica con agujeros para clavos.

    Returns:
        VGroup(cuerpo, borde_interior, brillo, agujeros).
    """
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

def crear_rueda_carreta():
    """Rueda de carreta con 12 radios y aro de hierro.

    Returns:
        VGroup(radios, aros, centro, eje).
    """
    aro_hierro = Circle(radius=0.6, stroke_color=HIERRO, stroke_width=8)
    aro_madera = Circle(radius=0.54, stroke_color=MADERA_CLARA, stroke_width=12)
    aro_madera_interior = Circle(radius=0.48, stroke_color=MADERA_OSCURA, stroke_width=1, stroke_opacity=0.6)
    
    centro_madera = Circle(radius=0.15, fill_color=MADERA_OSCURA, stroke_color=HIERRO, stroke_width=3)
    eje = Dot(radius=0.05, color=HIERRO)
    
    radios = VGroup(*[
        Polygon([-0.03, 0.15, 0], [0.03, 0.15, 0], [0.02, 0.5, 0], [-0.02, 0.5, 0], fill_color=MADERA_CLARA, stroke_color=MADERA_OSCURA, stroke_width=1).rotate(i * PI/6, about_point=ORIGIN)
        for i in range(12)
    ])
    
    remaches = VGroup(*[Dot(radius=0.02, color=ACERO).move_to(aro_hierro.point_at_angle(i * PI/6)) for i in range(12)])
    
    return VGroup(radios, aro_madera, aro_madera_interior, aro_hierro, remaches, centro_madera, eje)

def crear_pergamino():
    """Pergamino enrollado con líneas de texto y sello de cera.

    Returns:
        VGroup(cuerpo, rollos, lineas, cintas, sello).
    """
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
    """Yelmo de Mambrino (la famosa bacía de barbero).

    Returns:
        VGroup(cuenco, borde, brillo, remaches, muesca).
    """
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
    """Carga la imagen assets/quijote_rust.png centrada en ORIGIN.

    Returns:
        ImageMobject escalado a 0.3.
    """
    logo = ImageMobject(r"assets\quijote_rust.png").scale(0.3)
    
    return logo.move_to(ORIGIN)

def crear_rust_sancho():
    """Carga la imagen assets/sancho_rust.png centrada en ORIGIN.

    Returns:
        ImageMobject escalado a 0.3.
    """
    imagen_sancho = ImageMobject(r"assets\sancho_rust.png").scale(0.3)
    
    return imagen_sancho.move_to(ORIGIN)


class Presentacion(Slide):

    def construct(self):
        """Punto de entrada de Manim. Define el orden de las diapositivas.

        Las líneas comentadas están disponibles pero desactivadas.
        Descomenta las que quieras incluir en la presentación.
        """
        self.camera.background_color = WHITE
        

        self.slide_pronto_iniciamos()
        self.slide_introduction()
        self.slide_credits()
        self.slide_que_es_transformer()
        self.slide_molinete_ai()
        self.slide_por_que_rust()

        self.slide_roadmap()

        self.slide_que_es_un_tensor()
        self.mostrar_snippet("tensor.rs")
        
        self.slide_strides()
        
        self.slide_matmul()
        self.mostrar_snippet("matmul_base.rs")

        self.slide_intro_matmul_optimizacion()

        self.slide_simd()
        self.mostrar_snippet("simd_vectorization.rs")

        self.slide_cache_blocking()
        self.mostrar_snippet("cache_blocking.rs")

        self.slide_parallel_rayon()
        self.mostrar_snippet("parallel_rayon.rs")

        self.slide_batched_matmul()
        self.mostrar_snippet("batched_matmul.rs")

        self.slide_softmax()
        self.mostrar_snippet("softmax.rs")

        self.slide_forward_pass()
        self.slide_problema_strawberry()
        
        self.slide_tokenizacion()
        self.mostrar_snippet("BDPtokenizer.rs")
        
        self.slide_byte_pair_encoding()
        self.mostrar_snippet("pair_counts.rs")
        
        self.slide_tamano_vocabulario()
        
        self.slide_embeddings()
        self.slide_position_embeddings()
        self.mostrar_snippet("embedding.rs")

        self.slide_layer_normalization()
        self.mostrar_snippet("normalization.rs")

        self.slide_mha_acto1_intuicion()
        self.slide_mha_acto2_formula()
        self.slide_mha_acto3_calculo()
        self.slide_mha_acto4_multihead()
        self.mostrar_snippet("attention.rs")

        self.slide_arquitectura_neurona()
        self.mostrar_snippet("mlp_forward.rs")
        
        self.slide_zoom_neurona()
        self.slide_activacion()
        self.mostrar_snippet("gelu.rs")
        
        self.slide_capa_transformer()

        self.slide_entrenamiento()
        self.mostrar_snippet("compute_loss.rs")
        
        self.slide_descenso_gradiente()
        self.mostrar_snippet("linear_backward.rs")
        
        self.slide_backpropagation()
        
        self.slide_residual()
        self.mostrar_snippet("block_backward.rs")
        
        self.slide_adam()
        self.mostrar_snippet("adamw_update.rs")
        
        self.slide_dropout()
        self.mostrar_snippet("dropout.rs")
        
        self.slide_training_metrics()
        
        self.slide_temperature()
        self.mostrar_snippet("temperature.rs")
        
        self.slide_model_in_action()
        self.slide_final()
        
        
    def mostrar_snippet(self, titulo_archivo):
        """Muestra el fragmento de código Rust asociado a `titulo_archivo`
        usando la diapositiva de código y limpia la pantalla al terminar.

        Args:
            titulo_archivo: Clave en RUST_SNIPPETS (ej. "attention.rs").
        """
        self.diapo_codigo(
            codigo_fuente=RUST_SNIPPETS[titulo_archivo],
            titulo_archivo=titulo_archivo
        )
        self.limpiar_pantalla()

    def crear_titulo(self, texto, palabra_clave=None, color_clave=NARANJA_TERRACOTA, font_size=35):
        """Crea un título con subrayado de color y resaltado opcional.

        Args:
            texto:         Texto completo del título.
            palabra_clave: Palabra a resaltar con `color_clave`.
            color_clave:   Color hex del resaltado (default: naranja).
            font_size:     Tamaño de fuente (default: 35).

        Returns:
            Tuple[Text, Underline]
        """
        t2c = {palabra_clave: color_clave} if palabra_clave else {}
        titulo = Text(texto, font=FUENTE, font_size=font_size, color=TINTA_NEGRA, t2c=t2c).to_edge(UP)
        linea = Underline(titulo, color=color_clave, stroke_width=4)
        return titulo, linea

    def crear_bloque(self, texto="", color_fondo=FONDO_CAJA, color_texto=TINTA_NEGRA, ancho=0.8, alto=0.8):
        """Crea una celda rectangular redondeada con texto centrado.

        Útil para representar tokens, celdas de matriz, etc.

        Args:
            texto:       Contenido de la celda.
            color_fondo: Color de relleno del rectángulo.
            color_texto: Color del texto.
            ancho:       Ancho en unidades Manim.
            alto:        Alto en unidades Manim.

        Returns:
            VGroup(rect, label).
        """
        rect = RoundedRectangle(
            corner_radius=0.15, width=ancho, height=alto, 
            fill_color=color_fondo, fill_opacity=1, 
            stroke_color=MARRON_OSCURO, stroke_width=2
        )
        lbl = Text(str(texto), font=FUENTE, font_size=24, color=color_texto).move_to(rect.get_center())
        return VGroup(rect, lbl)
    
    def crear_matriz_bloques(self, filas, columnas, color_fondo=FONDO_CAJA, color_texto=TINTA_NEGRA, valores=None, ancho=0.8, alto=0.8):
        """Crea una cuadrícula de celdas (filas × columnas).

        Args:
            filas, columnas: Dimensiones de la matriz.
            color_fondo:     Color de relleno de cada celda.
            color_texto:     Color del texto de cada celda.
            valores:         Lista plana de textos (fila por fila).
                             Si es None, todas las celdas quedan vacías.
            ancho, alto:     Tamaño de cada celda.

        Returns:
            VGroup de VGroups (una fila por subgrupo).
        """
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
        """Desvanece todos los Mobjects visibles en pantalla.

        Se llama al final de cada diapositiva para dejar la escena limpia
        antes de comenzar la siguiente.
        """
        if self.mobjects:
            self.play(*[FadeOut(mob) for mob in self.mobjects])

    def _crear_adornos_esquinas(self, escala=0.6, buff=0.8):
        """Crea el set estándar de cuatro adornos cervantinos en las esquinas.

        Orden: DL=molino, UR=sol, DR=tintero, UL=escudo.

        Args:
            escala: Factor de escala uniforme para los cuatro adornos.
            buff:   Distancia al borde de pantalla.

        Returns:
            VGroup(molino, sol, tintero, escudo).
        """
        molino  = crear_molino().scale(escala).to_corner(DL, buff=buff)
        sol     = crear_sol_cervantino().scale(escala).to_corner(UR, buff=buff)
        tintero = crear_tintero_y_pluma().scale(escala).to_corner(DR, buff=buff)
        escudo  = crear_escudo_y_lanza().scale(escala).to_corner(UL, buff=buff)
        return VGroup(molino, sol, tintero, escudo)

    def _animar_entrada_slide(self, titulo, linea, adornos=None, fondo=None):
        """Anima la entrada estándar de una diapositiva.

        Siempre sigue el mismo orden:
            1. Título + línea subrayada (Write / GrowFromCenter).
            2. Decoraciones de esquina, si se pasan (LaggedStart FadeIn).
            3. Fondo temático, si se pasa (FadeIn simultáneo con el título).

        Todo ocurre en un solo ``self.play()`` para mantener la consistencia
        visual entre diapositivas.

        Args:
            titulo:  Mobject de texto del título.
            linea:   Underline del título.
            adornos: VGroup con los adornos de esquina (opcional).
            fondo:   Mobject de fondo decorativo (opcional).
        """
        animaciones = [Write(titulo), GrowFromCenter(linea)]

        if fondo is not None:
            animaciones.append(FadeIn(fondo, shift=UP * 0.3))

        if adornos is not None:
            animaciones.append(
                LaggedStart(
                    *[FadeIn(a, scale=0.5) for a in adornos],
                    lag_ratio=0.2
                )
            )

        self.play(*animaciones)

    def _crear_burbuja_chat(self, texto, color_fondo, color_texto, es_usuario=True, t2c_dict=None):
        """Crea una burbuja de chat con sombra y etiqueta de remitente.

        Args:
            texto:       Contenido de la burbuja.
            color_fondo: Color de relleno del globo.
            color_texto: Color del texto interior.
            es_usuario:  True → alineada a la derecha con etiqueta "Tú".
            t2c_dict:    Diccionario text-to-color opcional.

        Returns:
            VGroup(remitente, VGroup(sombra, fondo, txt)).
        """
        txt = Text(texto, font=FUENTE, font_size=24, color=color_texto, t2c=t2c_dict)
        fondo = RoundedRectangle(
            width=txt.width + 0.8,
            height=txt.height + 0.5,
            corner_radius=0.2,
            fill_color=color_fondo, fill_opacity=1,
            stroke_width=0 if es_usuario else 1.5,
            stroke_color=MARRON_OSCURO
        )
        sombra = fondo.copy().set_fill(MARRON_OSCURO, 0.1).set_stroke(width=0).shift(RIGHT * 0.05 + DOWN * 0.05)
        txt.move_to(fondo.get_center())
        burbuja_base = VGroup(sombra, fondo, txt)

        remitente = Text(
            "Tú" if es_usuario else "Molinete AI",
            font=FUENTE, font_size=16,
            color=MARRON_OSCURO, weight=BOLD
        )
        if es_usuario:
            remitente.next_to(burbuja_base, UP, buff=0.2, aligned_edge=RIGHT)
        else:
            remitente.next_to(burbuja_base, UP, buff=0.1, aligned_edge=LEFT)
        return VGroup(remitente, burbuja_base)

    def _crear_fresa(self):
        """Dibuja una fresa decorativa para el slide del problema Strawberry.

        Returns:
            VGroup(cuerpo, hojas) escalado a 0.85.
        """
        cuerpo = Polygon(
            [0, 0.3, 0], [-0.25, 0.1, 0], [-0.2, -0.2, 0],
            [0, -0.4, 0], [0.2, -0.2, 0], [0.25, 0.1, 0],
            fill_color=ROJO_TOMATE, fill_opacity=1,
            stroke_width=1.5, stroke_color=MARRON_OSCURO
        )
        hojas = Polygon(
            [0, 0.2, 0], [-0.2, 0.4, 0], [-0.1, 0.25, 0],
            [0, 0.45, 0], [0.1, 0.25, 0], [0.2, 0.4, 0],
            fill_color=VERDE_OLIVA, fill_opacity=1,
            stroke_width=1.5, stroke_color=MARRON_OSCURO
        )
        return VGroup(cuerpo, hojas).scale(0.85)

    def _crear_burbuja_transformer(self, texto, es_usuario=True):
        """Crea una burbuja de chat simple para el slide ¿Qué es un Transformer?

        Versión más ligera (sin sombra ni remitente) para uso en diálogos rápidos.

        Args:
            texto:      Contenido de la burbuja.
            es_usuario: True → fondo CAJA_INFERIOR con borde MARRON; False → FONDO_CAJA/NARANJA.

        Returns:
            VGroup(burbuja, txt).
        """
        color_fondo = CAJA_INFERIOR if es_usuario else FONDO_CAJA
        color_borde = MARRON_OSCURO if es_usuario else NARANJA_TERRACOTA
        txt = Text(texto, font=FUENTE, font_size=22, color=TINTA_NEGRA)
        burbuja = RoundedRectangle(
            width=max(txt.width + 0.6, 1.8), height=txt.height + 0.5, corner_radius=0.2,
            fill_color=color_fondo, fill_opacity=1, stroke_color=color_borde, stroke_width=2
        )
        txt.move_to(burbuja.get_center())
        return VGroup(burbuja, txt)

    def _animar_pensamiento_transformer(self, texto_ganador, texto_perdedor, pct_ganador, pct_perdedor, obj_referencia):
        """Anima el menú de probabilidades que muestra el razonamiento del modelo.

        Muestra dos opciones junto al objeto de referencia, destaca la ganadora
        y luego desvanece el menú.

        Args:
            texto_ganador:   Palabra/texto de la opción más probable.
            texto_perdedor:  Palabra/texto de la opción menos probable.
            pct_ganador:     Porcentaje de probabilidad de la opción ganadora (str).
            pct_perdedor:    Porcentaje de probabilidad de la opción perdedora (str).
            obj_referencia:  Mobject junto al cual se posiciona el menú.
        """
        opcion_ganadora  = Text(f"{texto_ganador} ({pct_ganador})",  font=FUENTE, font_size=20, weight=BOLD, color=NARANJA_TERRACOTA)
        opcion_perdedora = Text(f"{texto_perdedor} ({pct_perdedor})", font=FUENTE, font_size=16, color=GRAY)
        textos_menu = VGroup(opcion_ganadora, opcion_perdedora).arrange(DOWN, buff=0.15)
        caja_menu = SurroundingRectangle(
            textos_menu, color=NARANJA_TERRACOTA, stroke_width=2,
            fill_color=WHITE, fill_opacity=0.95, corner_radius=0.1, buff=0.25
        )
        menu_final = VGroup(caja_menu, textos_menu).next_to(obj_referencia, RIGHT, buff=0.5)

        self.play(Wiggle(obj_referencia))
        self.play(FadeIn(menu_final, shift=RIGHT * 0.3))
        self.play(Indicate(opcion_ganadora, scale_factor=1.2, color=NARANJA_TERRACOTA))
        self.play(FadeOut(menu_final))

    def _crear_item_utilidad(self, titulo_item, descripcion):
        """Crea un ítem de lista con marcador vertical y dos líneas de texto.

        Usado en la sección "¿Para qué sirven?" de slide_que_es_transformer.

        Args:
            titulo_item:  Texto de la primera línea (destacado en naranja).
            descripcion:  Texto de la segunda línea (color tinta normal).

        Returns:
            VGroup(marcador, VGroup(t_titulo, t_desc)).
        """
        t_titulo = Text(titulo_item, font=FUENTE, font_size=26, color=NARANJA_TERRACOTA, weight=BOLD)
        t_desc   = Text(descripcion,  font=FUENTE, font_size=22, color=TINTA_NEGRA)
        grupo_texto = VGroup(t_titulo, t_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        marcador = Rectangle(width=0.08, height=grupo_texto.height, fill_color=MARRON_OSCURO, fill_opacity=1, stroke_width=0)
        return VGroup(marcador, grupo_texto).arrange(RIGHT, buff=0.25)

    def _crear_panel_probabilidades(self, palabra_objetivo):
        """Genera el panel de barras de probabilidad de la slide de introducción.

        Muestra la palabra objetivo con alta probabilidad y dos palabras
        cervantinas aleatorias con probabilidades menores.

        Args:
            palabra_objetivo: Palabra a destacar como la más probable.

        Returns:
            VGroup(bg, VGroup(textos, barras)) centrado en LEFT*0.5 + DOWN*0.4.
        """
        dummies = ["hidalgo", "espada", "vino", "capa", "oro", "plaza", "mujer", "caballo"]
        random.shuffle(dummies)
        probs = [random.uniform(0.7, 0.9), random.uniform(0.05, 0.1), random.uniform(0.01, 0.03)]

        datos = [
            (palabra_objetivo, probs[0], NARANJA_TERRACOTA),
            (dummies[0],       probs[1], PAPEL_TAN),
            (dummies[1],       probs[2], CAJA_INFERIOR),
        ]
        txts = VGroup(*[
            Text(d[0], font=FUENTE, font_size=16, color=TINTA_NEGRA) for d in datos
        ]).arrange(DOWN, aligned_edge=RIGHT, buff=0.15)

        bars = VGroup(*[
            RoundedRectangle(
                corner_radius=0.05, height=0.15,
                width=max(0.1, d[1] * 2.5),
                fill_color=d[2], fill_opacity=1, stroke_width=0
            ) for d in datos
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        for i in range(len(datos)):
            bars[i].match_y(txts[i])

        col = VGroup(txts, bars).arrange(RIGHT, buff=0.15)
        bg = RoundedRectangle(
            corner_radius=0.1, width=col.width + 0.5,
            height=txts[0].height * 3 + 0.6,
            fill_color=CAJA_INFERIOR, fill_opacity=0.3,
            stroke_color=MARRON_OSCURO, stroke_width=1
        ).move_to(col)
        return VGroup(bg, col).move_to(LEFT * 0.5 + DOWN * 0.4)

    def _crear_fila_vocabulario(self, vocab, total_tokens, compresion, es_final=False):
        """Crea una fila de la tabla de vocabulario en slide_tamano_vocabulario.

        Args:
            vocab:        Texto de la columna Vocabulario.
            total_tokens: Texto de la columna Total Tokens.
            compresion:   Texto de la columna Compresión.
            es_final:     Si True, resalta la compresión en naranja con BOLD.

        Returns:
            VGroup con tres Text, uno por columna.
        """
        color_c = NARANJA_TERRACOTA if es_final else TINTA_NEGRA
        peso_c  = BOLD              if es_final else NORMAL
        return VGroup(
            Text(vocab,        font=FUENTE, font_size=24, color=TINTA_NEGRA),
            Text(total_tokens, font=FUENTE, font_size=24, color=TINTA_NEGRA),
            Text(compresion,   font=FUENTE, font_size=24, color=color_c, weight=peso_c),
        )

    def _crear_corazon(self, color, escala=1.0):
        """Dibuja un corazón usando una función paramétrica continua.

        Args:
            color:  Color de trazo y relleno.
            escala: Factor multiplicador del tamaño base.

        Returns:
            ParametricFunction con relleno semitransparente.
        """
        corazon = ParametricFunction(
            lambda t: np.array([
                16 * np.sin(t) ** 3,
                13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t),
                0
            ]),
            t_range=[0, TAU],
            color=color,
            stroke_width=2
        ).scale(0.03 * escala)
        corazon.set_fill(color, opacity=0.7)
        return corazon


    def slide_pronto_iniciamos(self):

        """Diapositiva: Slide Pronto Iniciamos."""

        gato_caballero = ImageMobject(r"assets\gato_armadura.png").scale(0.5)
        texto_inicio = Text("Pronto iniciamos", font=FUENTE, font_size=50, weight=BOLD, color=MARRON_OSCURO)

        cat_and_text = Group(gato_caballero, texto_inicio).arrange(DOWN, buff=0.8)
        cat_and_text.move_to(ORIGIN)

        pantalla_completa = Group(cat_and_text)

        self.play(FadeIn(pantalla_completa, shift=UP))

        self.next_slide()
        self.limpiar_pantalla()

    def slide_introduction(self):
        """Diapositiva: Slide Introduction."""

        titulo, linea = self.crear_titulo("Construyendo un Transformer con Rust", palabra_clave="Rust")
        subtitulo = Text("Jerónimo Hoyos Botero", font=FUENTE, font_size=25, color=MARRON_OSCURO)
        VGroup(titulo, linea, subtitulo).arrange(DOWN, buff=0.2).to_edge(UP, buff=0.5)

        llanuras_fondo = crear_llanuras_manchegas()

        escritorio_decor = crear_tintero_y_pluma().scale(0.8).next_to(titulo, RIGHT, buff=0.4).shift(DOWN*0.2)
        sol_decor = crear_sol_cervantino().scale(0.7).to_corner(UR, buff=0.5)
        
        molino_grande = crear_molino().scale(1.1).to_corner(DR, buff=0.5).shift(UP*0.1 + LEFT*0.2)
        molino_pequeño = crear_molino().scale(0.75).next_to(molino_grande, LEFT, buff=0.1).shift(DOWN*0.3 + RIGHT*0.2)
        molinos_paisaje = VGroup(molino_pequeño, molino_grande)
        
        molino_grande[-1].add_updater(lambda m, dt: m.rotate(-dt * 0.8))
        molino_pequeño[-1].add_updater(lambda m, dt: m.rotate(-dt * 0.5))
        
        libros_decor = crear_pila_libros().scale(0.9).to_corner(DL, buff=0.6).shift(UP*0.5)
        
        armas_decor = crear_escudo_y_lanza().scale(0.8).to_corner(UL, buff=0.5)

        quijote = crear_rust_quijote().to_edge(DOWN, buff=0.2).shift(LEFT * 3)
        sancho = crear_rust_sancho().next_to(quijote, RIGHT, buff=0.5).shift(UP*0.1)

        star1 = crear_estrella().move_to(armas_decor.get_top() + UP*0.3 + LEFT*0.2)
        star2 = crear_estrella().scale(0.8).move_to(libros_decor.get_bottom() + DOWN*0.2 + RIGHT*0.3)
        star3 = crear_estrella().scale(0.6).move_to(quijote.get_top() + UP*0.4 + RIGHT*0.5)
        estrellas = VGroup(star1, star2, star3) 

        cajas = VGroup(*[
            RoundedRectangle(
                corner_radius=0.1, width=2.6, height=1.8, fill_color=FONDO_CAJA, fill_opacity=1, 
                stroke_color=MARRON_OSCURO, stroke_width=2
            ).shift(UP * i * 0.06 + RIGHT * i * 0.06) for i in range(5)
        ]).move_to(LEFT * 5.0 + DOWN * 0.5) 
        
        txt_molinete = Text("Molinete AI", font=FUENTE, font_size=20, weight=BOLD, color=TINTA_NEGRA).move_to(cajas)
        flecha = Arrow(LEFT, RIGHT, color=TINTA_NEGRA, stroke_width=4).scale(0.5).next_to(cajas, RIGHT, buff=0.3)

        pos_start = RIGHT * 1.5 + UP * 1.5 

        def create_probs(target_word):
            return self._crear_panel_probabilidades(target_word)
        
        self.play(
            FadeIn(llanuras_fondo),
            Write(titulo), GrowFromCenter(linea), FadeIn(subtitulo, shift=DOWN),
            DrawBorderThenFill(escritorio_decor, run_time=1.5),
            SpinInFromNothing(sol_decor),
            GrowFromCenter(molinos_paisaje),
            FadeIn(libros_decor, shift=UP*0.5),
            DrawBorderThenFill(armas_decor, run_time=1.2),
            Create(estrellas, lag_ratio=0.2),
            FadeIn(quijote, shift=UP), 
            FadeIn(sancho, shift=UP),
            FadeIn(cajas, shift=RIGHT*0.5),
            Write(txt_molinete), 
            GrowArrow(flecha)
        )
        
        for estrella in estrellas:
            estrella.add_updater(lambda m, dt: m.rotate(dt * 0.5))

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
        curr_probs = Mobject() 

        for linea_texto in poema:
            curr_x = pos_start[0]
            for word in linea_texto:
                new_probs = create_probs(word)
                word_mob = Text(word, font=FUENTE, font_size=18, color=TINTA_NEGRA).move_to([curr_x, curr_y, 0], aligned_edge=LEFT)
                
                if not curr_probs.submobjects:
                    self.play(FadeIn(new_probs), FadeIn(word_mob, shift=LEFT*0.1), run_time=0.45)
                else:
                    self.play(ReplacementTransform(curr_probs, new_probs), FadeIn(word_mob, shift=LEFT*0.1), run_time=0.45)
                
                curr_probs = new_probs
                curr_x += word_mob.width + 0.15
            curr_y -= 0.45

        molino_grande[-1].clear_updaters()
        molino_pequeño[-1].clear_updaters()
        for estrella in estrellas:
            estrella.clear_updaters()

        self.next_slide()
        self.limpiar_pantalla()

    def slide_credits(self):
        """Diapositiva final: Créditos y agradecimientos con corazones elegantes."""

        
        llanuras = crear_llanuras_manchegas()
        
        titulo_creditos, linea_creditos = self.crear_titulo(
            "Esta presentación se basa en:", 
            palabra_clave="basa en:", 
            color_clave=MARRON_OSCURO 
        )
        
        imagen_creditos = ImageMobject("assets/creditos_guia_original.png")
        imagen_creditos.scale(1.5).next_to(linea_creditos, DOWN, buff=0.8)

        escudo = crear_escudo_y_lanza().scale(0.8).to_corner(UL).shift(DOWN * 0.2 + RIGHT * 0.5)
        
        sol = crear_sol_cervantino().to_corner(UR).shift(DOWN * 0.5 + LEFT * 0.5)
        molino = crear_molino().to_corner(DL).shift(UP * 0.5 + RIGHT * 0.5)
        libros = crear_pila_libros().to_corner(DR).shift(UP * 0.5 + LEFT * 0.5)
        tintero = crear_tintero_y_pluma().next_to(libros, LEFT, buff=0.8)
        
        estrellas = VGroup(
            crear_estrella().move_to(UP * 1.5 + LEFT * 4),
            crear_estrella().move_to(DOWN * 1.5 + LEFT * 5),
            crear_estrella().move_to(UP * 0.5 + RIGHT * 5)
        )


        self.play(FadeIn(llanuras, run_time=1.5))

        self._animar_entrada_slide(
            titulo_creditos, linea_creditos,
            fondo=Group(imagen_creditos, molino, escudo, libros, tintero, estrellas)
        )
        
        self.next_slide()


        num_corazones = 20
        animaciones_corazones = []
        grupo_corazones = VGroup().set_z_index(-1)

        for _ in range(num_corazones):
            x_ini = np.random.uniform(-7.5, 7.5)
            y_ini = np.random.uniform(-5.0, -1.0)
            color_elegido = np.random.choice([NARANJA_TERRACOTA, MARRON_OSCURO, "#D4A373", "#E63946"])
            escala_aleatoria = np.random.uniform(0.5, 1.3)

            corazon = self._crear_corazon(color_elegido, escala_aleatoria).move_to([x_ini, y_ini, 0])
            grupo_corazones.add(corazon)
            
            destino = corazon.get_center() + UP * np.random.uniform(3.5, 7.0) + RIGHT * np.random.uniform(-1.5, 1.5)
            
            animaciones_corazones.append(
                Succession(
                    FadeIn(corazon, scale=0.3, run_time=1),
                    corazon.animate(run_time=np.random.uniform(3.5, 6.0), rate_func=rate_functions.ease_in_out_sine)
                           .move_to(destino)
                           .set_opacity(0)
                           .rotate(np.random.uniform(-PI/5, PI/5))
                )
            )

        self.add(grupo_corazones)
        
        self.play(
            LaggedStart(*animaciones_corazones, lag_ratio=0.15),
            run_time=6.0
        )
        
        self.next_slide()

        elementos_en_pantalla = Group(*self.mobjects)
        self.play(FadeOut(elementos_en_pantalla, scale=0.9), run_time=1.5)
        
        self.limpiar_pantalla()

    def slide_que_es_transformer(self):
        """Diapositiva: Slide Que Es Transformer."""

        titulo, linea = self.crear_titulo("¿Qué es un Transformer?", palabra_clave="Transformer", color_clave=NARANJA_TERRACOTA)
        adornos = self._crear_adornos_esquinas(escala=0.8)

        explicacion_1 = Text(
            "TRANSFORMA la secuencia, observa todo el contexto\n"
            "para predecir qué palabra tiene más sentido a continuación.",
            font=FUENTE, font_size=26, color=TINTA_NEGRA, line_spacing=1.2,
            t2c={"TRANSFORMA": NARANJA_TERRACOTA, "todo el contexto": MARRON_OSCURO, "tiene más sentido": NARANJA_TERRACOTA}
        ).move_to(UP * 2.2)

        caja_maq = RoundedRectangle(corner_radius=0.2, width=2.5, height=1.5, fill_color=MARRON_OSCURO, fill_opacity=0.9, stroke_color=NARANJA_TERRACOTA, stroke_width=3)
        txt_maq = Text("Modelo IA", font=FUENTE, font_size=24, color=FONDO_CAJA, weight=BOLD).move_to(caja_maq)
        maquina = VGroup(caja_maq, txt_maq).to_edge(LEFT, buff=1.0).shift(DOWN * 0.5)

        chat_x_right = RIGHT * 5.2
        chat_x_left  = RIGHT * 1.5

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)
        self.play(FadeIn(explicacion_1, shift=UP * 0.2), FadeIn(maquina, shift=RIGHT * 0.5))
        self.next_slide()

        msg1_usuario = self._crear_burbuja_transformer("¿Quién es?", es_usuario=True).move_to(chat_x_right + UP * 0.8).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg1_usuario, shift=UP * 0.2))
        self._animar_pensamiento_transformer("Soy yo", "El cartero", "98.5%", "1.5%", caja_maq)
        msg2_ia = self._crear_burbuja_transformer("Soy yo", es_usuario=False).next_to(msg1_usuario, DOWN, buff=0.3).align_to(chat_x_left, LEFT)
        self.play(FadeIn(msg2_ia, shift=UP * 0.2))
        self.next_slide()

        msg3_usuario = self._crear_burbuja_transformer("¿Qué vienes a buscar?", es_usuario=True).next_to(msg2_ia, DOWN, buff=0.3).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg3_usuario, shift=UP * 0.2))
        self._animar_pensamiento_transformer("A ti", "Nada", "99.9%", "0.05%", caja_maq)
        msg_final_ia = self._crear_burbuja_transformer("A ti", es_usuario=False).next_to(msg3_usuario, DOWN, buff=0.3).align_to(chat_x_left, LEFT)
        self.play(FadeIn(msg_final_ia, shift=UP * 0.2))
        self.next_slide()

        shift_up = UP * (msg1_usuario.get_center()[1] - msg3_usuario.get_center()[1])
        
        self.play(
            FadeOut(msg1_usuario, shift=shift_up),
            FadeOut(msg2_ia, shift=shift_up),
            msg3_usuario.animate.shift(shift_up),
            msg_final_ia.animate.shift(shift_up),
            run_time=1.2
        )

        msg5_usuario = self._crear_burbuja_transformer("Ya es tarde", es_usuario=True).next_to(msg_final_ia, DOWN, buff=0.3).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg5_usuario, shift=UP * 0.2))
        self._animar_pensamiento_transformer("¿Por qué?", "Vete", "95.0%", "5.0%", caja_maq)
        msg6_ia = self._crear_burbuja_transformer("¿Por qué?", es_usuario=False).next_to(msg5_usuario, DOWN, buff=0.3).align_to(chat_x_left, LEFT)
        self.play(FadeIn(msg6_ia, shift=UP * 0.2))
        self.next_slide()

        shift_up_2 = UP * (msg3_usuario.get_center()[1] - msg5_usuario.get_center()[1])
        self.play(
            FadeOut(msg3_usuario, shift=shift_up_2),
            FadeOut(msg_final_ia, shift=shift_up_2),
            msg5_usuario.animate.shift(shift_up_2),
            msg6_ia.animate.shift(shift_up_2),
            run_time=1.2
        )

        texto_despecho = "Porque ahora soy yo la que\nquiere estar sin ti"
        msg7_usuario = self._crear_burbuja_transformer(texto_despecho, es_usuario=True).next_to(msg6_ia, DOWN, buff=0.3).align_to(chat_x_right, RIGHT)
        self.play(FadeIn(msg7_usuario, shift=UP * 0.2))
        self.next_slide()

        titulo_utilidades = Text("¿Para qué sirven?", font=FUENTE, font_size=32, color=MARRON_OSCURO, weight=BOLD).move_to(UP * 1.5)
        lista_utilidades = VGroup(
            self._crear_item_utilidad("1. Chatbots Asistentes", "Conversación natural fluida"),
            self._crear_item_utilidad("2. Agentes Autónomos", "Resolución de tareas complejas"),
            self._crear_item_utilidad("3. Predicción Avanzada", "Código, traducción y análisis")
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.6).move_to(DOWN * 0.5)

        chat_visible_final = VGroup(msg5_usuario, msg6_ia, msg7_usuario)
        self.play(FadeOut(chat_visible_final), FadeOut(maquina), FadeOut(explicacion_1), run_time=1)
        self.play(FadeIn(titulo_utilidades, shift=UP * 0.2))
        self.play(AnimationGroup(*[FadeIn(item, shift=RIGHT * 0.3) for item in lista_utilidades], lag_ratio=0.3))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_molinete_ai(self):
        """Diapositiva: Slide Molinete Ai."""

        titulo, linea = self.crear_titulo(
            "Molinete AI", 
            palabra_clave="Molinete", 
            color_clave=NARANJA_TERRACOTA
        )

        adornos = self._crear_adornos_esquinas(escala=0.8)

        subtitulo = Text(
            "Origen del nombre “Molinete”", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, weight=BOLD
        )

        descripcion = Text(
            "Modelo de lenguaje entrenado sobre\nel corpus de El Quijote.", 
            font=FUENTE, font_size=24, color=MARRON_OSCURO
        )

        punto1 = Text(
            "• Inspirado en el boss de dark souls.",
            font=FUENTE, font_size=22, color=TINTA_NEGRA, 
            t2c={"imaginario cervantino": NARANJA_TERRACOTA}
        )

        punto2 = Text(
            "• Nombre que evoca los molinos de viento\n  enfrentados por Don Quijote.",
            font=FUENTE, font_size=22, color=TINTA_NEGRA,
            t2c={"molinos de viento": NARANJA_TERRACOTA}
        )

        punto3 = Text(
            "• Para darle identidad propia y un toque de humor\n  dentro del mundo de los LLMs.",
            font=FUENTE, font_size=22, color=TINTA_NEGRA,
            t2c={"modelo de lenguaje": MARRON_OSCURO}
        )

        grupo_puntos = VGroup(punto1, punto2, punto3).arrange(DOWN, aligned_edge=LEFT, buff=0.4) 
        textos_molinete = VGroup(descripcion, subtitulo, grupo_puntos).arrange(DOWN, aligned_edge=LEFT, buff=0.5) 

        imagen_molino = ImageMobject("assets/quijote_vs_molinos.png")
        imagen_molino.height = 5.5 
        contenido_principal = Group(textos_molinete, imagen_molino).arrange(RIGHT, buff=0.8)

        contenido_principal.next_to(linea, DOWN, buff=0.5)

        if contenido_principal.width > 12.5:
            contenido_principal.width = 12.5
            contenido_principal.next_to(linea, DOWN, buff=0.5) 

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)
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

    def slide_problema_strawberry(self):
        """Diapositiva: Slide Problema Strawberry."""

        titulo, linea = self.crear_titulo(
            "¿Por qué los LLM no saben 'leer'?",
            palabra_clave="'leer'?",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN * 0.2 + LEFT * 0.2)

        burbuja_pregunta = self._crear_burbuja_chat(
            "¿Cuántas letras 'r' hay en 'strawberry'?",
            color_fondo=MARRON_OSCURO,
            color_texto=PAPEL_CREMA, es_usuario=True
        )

        burbuja_respuesta = self._crear_burbuja_chat(
            "Hay 2 letras 'r' en 'strawberry'.",
            color_fondo=FONDO_CAJA,
            color_texto=TINTA_NEGRA, es_usuario=False,
            t2c_dict={"2": NARANJA_TERRACOTA}
        )

        grupo_chat = VGroup(burbuja_pregunta, burbuja_respuesta).arrange(DOWN, buff=0.5)
        burbuja_pregunta.shift(RIGHT * 1.5)
        burbuja_respuesta.shift(LEFT * 1.5)
        grupo_chat.next_to(linea, DOWN, buff=0.6)

        fresa_der = self._crear_fresa().to_corner(DR).shift(UP * 0.3 + LEFT * 0.3)
        fresa_izq = self._crear_fresa().to_corner(DL).shift(UP * 0.3 + RIGHT * 0.3)

        token1 = self.crear_bloque("str", ancho=1.2)
        token2 = self.crear_bloque("aw", ancho=1.2)
        token3 = self.crear_bloque("berry", ancho=1.6)
        tokens_straw = VGroup(token1, token2, token3).arrange(RIGHT, buff=0.15)

        texto_explicacion = Text(
            "Para el modelo, la palabra se divide en 'pedazos' (Tokens):",
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        )
        grupo_visual = VGroup(texto_explicacion, tokens_straw).arrange(DOWN, buff=0.5)
        grupo_visual.next_to(grupo_chat, DOWN, buff=1.0)

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo))
        self.next_slide()

        self.add(fresa_der, fresa_izq)
        self.play(FadeIn(burbuja_pregunta, shift=UP * 0.2, scale=0.9))
        self.wait(0.5)
        self.play(FadeIn(burbuja_respuesta, shift=UP * 0.2, scale=0.9))
        self.next_slide()

        self.play(FadeIn(texto_explicacion, shift=UP * 0.2))
        self.play(LaggedStart(
            GrowFromCenter(token1),
            GrowFromCenter(token2),
            GrowFromCenter(token3),
            lag_ratio=0.2
        ))
        self.next_slide()

        cruz = Cross(tokens_straw, stroke_color=NARANJA_TERRACOTA, stroke_width=6)
        self.play(Create(cruz))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_byte_pair_encoding(self):
        """Diapositiva de BPE: Fusión dinámica iterativa sobre un trabalenguas (Animación fluida)."""

        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        
        # ── FIX DEFINITIVO DEL MOLINO ─────────────────────────────────────────
        # Para que solo giren las aspas, debemos apuntar a ese submobject exacto.
        # Por lo general, si construiste el molino añadiendo primero la base y 
        # luego las aspas, las aspas son el ÚLTIMO elemento (índice -1 o 1).
        
        molino = adornos[0]
        
        # Si tienes dudas de qué es cada pieza, descomenta esta línea para ver la estructura:
        # print(f"Submobjects del molino: {len(molino.submobjects)}")
        
        if len(molino) > 1:
            # Asumimos que el índice [-1] (el último añadido) son las aspas.
            aspas = molino[-1] 
            # El secreto: forzar about_point para que giren sobre su propio eje central
            centro_aspas = aspas.get_center()
            aspas.add_updater(lambda m, dt: m.rotate(-dt * 0.3, about_point=centro_aspas))
        else:
            # Si cae aquí, tu método _crear_adornos_esquinas está devolviendo 
            # el molino como un solo objeto (quizás un SVG acoplado).
            molino.add_updater(lambda m, dt: m.rotate(-dt * 0.3))

        titulo, linea = self.crear_titulo(
            "Byte Pair Encoding (BPE)",
            palabra_clave="BPE",
            color_clave=NARANJA_TERRACOTA
        )
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo, adornos=adornos)

        subtitulo = Text(
            "El algoritmo aprende a fusionar los pares más frecuentes",
            font=FUENTE, font_size=21, color=MARRON_OSCURO, slant=ITALIC
        ).next_to(linea, DOWN, buff=0.25)
        self.play(FadeIn(subtitulo, shift=DOWN * 0.2))
        self.next_slide()

        # ── Panel lateral derecho: vocabulario (REDUCIDO) ────────────────────
        panel_rect = RoundedRectangle(
            corner_radius=0.18, width=2.4, height=4.6,  
            fill_color=FONDO_CAJA, fill_opacity=0.97,
            stroke_color=NARANJA_TERRACOTA, stroke_width=2.5
        ).to_edge(RIGHT, buff=0.35).shift(DOWN * 0.2)

        panel_titulo = Text("Vocabulario", font=FUENTE, font_size=16, 
                            color=NARANJA_TERRACOTA, weight=BOLD)
        panel_titulo.next_to(panel_rect.get_top(), DOWN, buff=0.2)

        separador_panel = Line(
            panel_rect.get_left() + RIGHT * 0.2,
            panel_rect.get_right() + LEFT * 0.2,
            color=NARANJA_TERRACOTA, stroke_width=1.5
        ).next_to(panel_titulo, DOWN, buff=0.15)

        self.play(
            FadeIn(panel_rect, shift=LEFT * 0.3),
            Write(panel_titulo),
            Create(separador_panel)
        )

        # ── Contador de iteraciones (Mejor posicionado) ──────────────────────
        iter_label_bg = RoundedRectangle(
            corner_radius=0.12, width=2.6, height=0.55,
            fill_color=MARRON_OSCURO, fill_opacity=0.9,
            stroke_width=0
        ).to_edge(LEFT, buff=0.7).shift(UP * 2.3)

        iter_label = Text("Iteración  1 / 4", font=FUENTE, font_size=20,
                          color=PAPEL_CREMA, weight=BOLD)
        iter_label.move_to(iter_label_bg)

        self.play(FadeIn(iter_label_bg), Write(iter_label))

        # ── Etiqueta de fusión activa (Alineada con el contador) ─────────────
        fusion_bg = RoundedRectangle(
            corner_radius=0.14, width=4.5, height=0.62,
            fill_color=MARRON_OSCURO, fill_opacity=0.0,
            stroke_color=NARANJA_TERRACOTA, stroke_width=0
        ).next_to(iter_label_bg, DOWN, buff=0.25).align_to(iter_label_bg, LEFT)

        fusion_txt = Text("", font=FUENTE, font_size=19, color=NARANJA_TERRACOTA, weight=BOLD)
        fusion_txt.move_to(fusion_bg)

        self.add(fusion_bg, fusion_txt)

        # ── Área principal: frases tokenizadas (Centradas) ───────────────────
        def calc_ancho(texto):
            return max(0.38, 0.28 + len(texto) * 0.18)

        token_colors = {
            "_":    PERGAMINO_CLARO,
            "tr":   SALMON_CLARO,
            "es":   MENTA_PALIDA,
            "tri":  LAVANDA,
            "trig": AMARILLO_PALIDO,
        }

        frases_label = ["Frase 1:", "Frase 2:", "Frase 3:"]
        label_colors = [LADRILLO_VIVO, VERDE_OLIVA, OCRE_CERVANTINO]

        def crear_grid(estado_actual):
            filas = VGroup()
            for idx_row, row in enumerate(estado_actual):
                etiqueta = Text(
                    frases_label[idx_row], font=FUENTE, font_size=17,
                    color=label_colors[idx_row], weight=BOLD
                )
                tokens_row = VGroup(*[
                    self.crear_bloque(
                        s,
                        color_fondo=token_colors.get(s, CREMA_CALIDA),
                        ancho=calc_ancho(s),
                        alto=0.52
                    )
                    for s in row
                ]).arrange(RIGHT, buff=0.06)
                fila = VGroup(etiqueta, tokens_row).arrange(RIGHT, buff=0.22)
                filas.add(fila)

            filas.arrange(DOWN, buff=0.32)
            filas.move_to(LEFT * 0.8 + DOWN * 0.4) 
            return filas

        def get_new_state_and_indices(estado, char1, char2, new_char):
            new_estado = []
            fusions_indices = []
            for r_idx, row in enumerate(estado):
                new_row = []
                c_idx = 0
                while c_idx < len(row):
                    if (c_idx < len(row) - 1
                            and row[c_idx] == char1
                            and row[c_idx + 1] == char2):
                        fusions_indices.append((r_idx, c_idx, c_idx + 1))
                        new_row.append(new_char)
                        c_idx += 2
                    else:
                        new_row.append(row[c_idx])
                        c_idx += 1
                new_estado.append(new_row)
            return new_estado, fusions_indices

        estado_actual = [
            ["t","r","e","s","_","t","r","i","s","t","e","s"],
            ["t","i","g","r","e","s","_","t","r","a","g","a","n"],
            ["t","r","i","g","o","_","t","r","i","g","a","l"],
        ]

        grid_actual = crear_grid(estado_actual)
        self.play(FadeIn(grid_actual, shift=UP * 0.2), run_time=1.2)
        self.wait(0.3)

        # ── Entradas del vocabulario lateral (Proporcionales) ────────────────
        vocab_entries = VGroup()
        vocab_anchor = separador_panel.get_bottom() + DOWN * 0.2

        def agregar_vocab_entry(token, color_tok, color_bg):
            dot = Dot(radius=0.06, color=color_tok)
            tok_rect = RoundedRectangle(
                corner_radius=0.08, width=0.6, height=0.32, 
                fill_color=color_bg, fill_opacity=1,
                stroke_color=color_tok, stroke_width=1.8
            )
            tok_txt = Text(token, font=FUENTE, font_size=15, 
                           color=TINTA_NEGRA, weight=BOLD).move_to(tok_rect)
            entry = VGroup(dot, VGroup(tok_rect, tok_txt)).arrange(RIGHT, buff=0.1)
            
            if len(vocab_entries) == 0:
                entry.next_to(vocab_anchor, DOWN, buff=0.1).align_to(panel_rect, LEFT).shift(RIGHT * 0.25)
            else:
                entry.next_to(vocab_entries[-1], DOWN, buff=0.15).align_to(vocab_entries[-1], LEFT)
            vocab_entries.add(entry)
            return entry

        pasos = [
            ("t",   "r",   "tr",   LADRILLO_VIVO,   SALMON_CLARO,   "1 / 4"),
            ("e",   "s",   "es",   VERDE_OLIVA,     MENTA_PALIDA,   "2 / 4"),
            ("tr",  "i",   "tri",  AZUL_NOCHE,      LAVANDA,        "3 / 4"),
            ("tri", "g",   "trig", OCRE_CERVANTINO, AMARILLO_PALIDO,"4 / 4"),
        ]

        for n_paso, (c1, c2, nuevo, color_resalte, color_bg, iter_str) in enumerate(pasos):

            nueva_iter = Text(f"Iteración  {iter_str}", font=FUENTE, font_size=20,
                              color=PAPEL_CREMA, weight=BOLD).move_to(iter_label_bg)
            self.play(ReplacementTransform(iter_label, nueva_iter), run_time=0.4)
            iter_label = nueva_iter

            nueva_fusion = Text(
                f'Fusionar: "{c1}" + "{c2}"  \u2192  "{nuevo}"',
                font=FUENTE, font_size=19, color=NARANJA_TERRACOTA, weight=BOLD,
                t2c={f'"{nuevo}"': color_resalte}
            ).move_to(fusion_bg) 

            self.play(
                FadeIn(fusion_bg.set_stroke(color=color_resalte, width=2.5)
                               .set_fill(opacity=0.08),
                       scale=0.95),
                ReplacementTransform(fusion_txt, nueva_fusion),
                run_time=0.5
            )
            fusion_txt = nueva_fusion

            new_estado, fusions = get_new_state_and_indices(estado_actual, c1, c2, nuevo)
            new_grid = crear_grid(new_estado)

            pulsos = VGroup()
            indicates = []
            for r, ci1, ci2 in fusions:
                b1 = grid_actual[r][1][ci1]
                b2 = grid_actual[r][1][ci2]
                pulso = SurroundingRectangle(
                    VGroup(b1, b2), color=color_resalte,
                    buff=0.06, stroke_width=3, corner_radius=0.07
                )
                pulsos.add(pulso)
                indicates += [
                    Indicate(b1, color=color_resalte, scale_factor=1.08),
                    Indicate(b2, color=color_resalte, scale_factor=1.08),
                ]

            self.play(
                FadeIn(pulsos),
                LaggedStart(*indicates, lag_ratio=0.04),
                run_time=1.0
            )
            self.play(
                ReplacementTransform(grid_actual, new_grid),
                FadeOut(pulsos),
                run_time=1.1
            )

            entry = agregar_vocab_entry(nuevo, color_resalte, color_bg)
            self.play(FadeIn(entry, scale=0.8), run_time=0.5)

            estado_actual = new_estado
            grid_actual = new_grid
            self.wait(0.25)

        conclusion = Text(
            "¡Vocabulario aprendido iterativamente!",
            font=FUENTE, font_size=24, color=ORO_VIEJO, weight=BOLD
        ).to_edge(DOWN, buff=0.55)

        self.play(Write(conclusion), run_time=0.8)

        for fila in grid_actual:
            for bloque in fila[1]:
                if bloque[1].text == "trig":
                    self.play(
                        Flash(bloque, color=ORO_VIEJO, line_length=0.35, num_lines=8),
                        Indicate(bloque, color=ORO_VIEJO, scale_factor=1.18),
                        run_time=0.6
                    )

        self.wait(0.8)
        self.next_slide()
        self.limpiar_pantalla()
    
    def slide_tamano_vocabulario(self):
        """Diapositiva: Tamaño del Vocabulario — por qué importa y cuál es el trade-off."""

        titulo, linea = self.crear_titulo(
            "El Tamaño del Vocabulario",
            palabra_clave="Vocabulario",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        # ── ACTO 1: ¿QUÉ ES EL VOCABULARIO? ─────────────────────────────────────
        # Mostrar la palabra "embeddings" tokenizada de dos formas distintas
        pregunta = Text("¿Cómo parte el modelo el texto en trozos?",
                        font=FUENTE, font_size=26, weight=BOLD, color=TINTA_NEGRA)
        pregunta.move_to(UP * 2.4)

        palabra_raw = Text('"embeddings"', font="Monospace", font_size=34,
                           color=MARRON_OSCURO, weight=BOLD).move_to(UP * 1.3)

        self.play(Write(pregunta))
        self.play(FadeIn(palabra_raw, shift=DOWN * 0.2))
        self.next_slide()

        # Vocabulario pequeño → muchos trozos
        lbl_chico = Text("Vocab pequeño (256)  →  muchos tokens",
                         font=FUENTE, font_size=21, color=PAPEL_TAN, weight=BOLD).move_to(UP * 0.3)

        def hacer_token_box(texto, color_fondo, color_borde=MARRON_OSCURO):
            rect = RoundedRectangle(corner_radius=0.1, width=len(texto) * 0.22 + 0.4, height=0.52,
                                    fill_color=color_fondo, fill_opacity=0.9,
                                    stroke_color=color_borde, stroke_width=2)
            txt = Text(texto, font="Monospace", font_size=19, color=TINTA_NEGRA)
            txt.move_to(rect.get_center())
            return VGroup(rect, txt)

        tokens_chicos = ["em", "bed", "d", "in", "gs"]
        fila_chica = VGroup(*[hacer_token_box(t, PAPEL_TAN) for t in tokens_chicos])
        fila_chica.arrange(RIGHT, buff=0.15).move_to(DOWN * 0.45)

        # Vocabulario grande → pocos tokens
        lbl_grande = Text("Vocab grande (20 534)  →  pocos tokens",
                          font=FUENTE, font_size=21, color=NARANJA_TERRACOTA, weight=BOLD).move_to(DOWN * 1.4)

        tokens_grandes = ["embed", "dings"]
        fila_grande = VGroup(*[hacer_token_box(t, SALMON_CLARO, NARANJA_TERRACOTA) for t in tokens_grandes])
        fila_grande.arrange(RIGHT, buff=0.15).move_to(DOWN * 2.1)

        self.play(Write(lbl_chico))
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.15) for b in fila_chica], lag_ratio=0.15))
        self.next_slide()

        self.play(Write(lbl_grande))
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.15) for b in fila_grande], lag_ratio=0.2))
        self.next_slide()

        self.play(
            FadeOut(pregunta), FadeOut(palabra_raw),
            FadeOut(lbl_chico), FadeOut(fila_chica),
            FadeOut(lbl_grande), FadeOut(fila_grande)
        )

        # ── ACTO 2: EL COSTO EN MEMORIA ──────────────────────────────────────────
        lbl_costo = Text("Cada palabra del vocabulario ocupa una fila en la matriz de embeddings",
                         font=FUENTE, font_size=22, weight=BOLD, color=TINTA_NEGRA)
        lbl_costo.move_to(UP * 2.4)

        # Diagrama: matriz vocab_size × d_model
        cols = 6
        rows_small = 4   # representación visual pequeña
        cell_w, cell_h = 0.55, 0.38

        def hacer_matriz(num_filas, color_celda, color_borde, label_filas, label_cols, col_offset=0):
            celdas = VGroup()
            for r in range(num_filas):
                for c in range(cols):
                    rect = Rectangle(width=cell_w, height=cell_h,
                                     fill_color=color_celda, fill_opacity=0.6,
                                     stroke_color=color_borde, stroke_width=1.2)
                    rect.move_to(RIGHT * c * (cell_w + 0.06) + DOWN * r * (cell_h + 0.06))
                    celdas.add(rect)
            puntos_v = Text("⋮", font_size=22, color=color_borde).next_to(celdas[-cols:], DOWN, buff=0.12)
            grupo = VGroup(celdas, puntos_v)
            brace_f = Brace(VGroup(celdas, puntos_v), direction=LEFT, color=color_borde)
            txt_f = Text(label_filas, font=FUENTE, font_size=16, color=color_borde).next_to(brace_f, LEFT, buff=0.15)
            brace_c = Brace(celdas[:cols], direction=UP, color=MARRON_OSCURO)
            txt_c = Text(label_cols, font=FUENTE, font_size=16, color=MARRON_OSCURO).next_to(brace_c, UP, buff=0.1)
            return VGroup(grupo, brace_f, txt_f, brace_c, txt_c).shift(RIGHT * col_offset)

        mat_chica = hacer_matriz(rows_small, PAPEL_TAN, MARRON_OSCURO,
                                 "256 palabras", "d=768", col_offset=-3.2).move_to(LEFT * 3.2 + DOWN * 0.1)
        mat_grande = hacer_matriz(rows_small, SALMON_CLARO, NARANJA_TERRACOTA,
                                  "20 534 palabras", "d=768", col_offset=0).move_to(RIGHT * 2.0 + DOWN * 0.1)

        lbl_mat_chica = Text("Vocab 256", font=FUENTE, font_size=20, weight=BOLD,
                             color=MARRON_OSCURO).next_to(mat_chica, DOWN, buff=0.4)
        lbl_mat_grande = Text("Vocab 20 534", font=FUENTE, font_size=20, weight=BOLD,
                              color=NARANJA_TERRACOTA).next_to(mat_grande, DOWN, buff=0.4)

        # Tamaños en MB
        mb_chica  = Text("≈ 1.5 MB", font=FUENTE, font_size=19, color=VERDE_OLIVA).next_to(lbl_mat_chica, DOWN, buff=0.15)
        mb_grande = Text("≈ 120 MB", font=FUENTE, font_size=19, weight=BOLD, color=ROJO_TOMATE).next_to(lbl_mat_grande, DOWN, buff=0.15)

        self.play(Write(lbl_costo))
        self.play(FadeIn(mat_chica, shift=RIGHT * 0.2), Write(lbl_mat_chica))
        self.play(FadeIn(mat_grande, shift=LEFT * 0.2), Write(lbl_mat_grande))
        self.play(FadeIn(mb_chica), FadeIn(mb_grande))
        self.next_slide()

        self.play(*[FadeOut(m) for m in [lbl_costo, mat_chica, mat_grande,
                                          lbl_mat_chica, lbl_mat_grande, mb_chica, mb_grande]])

        # ── ACTO 3: TABLA DE COMPRESIÓN ──────────────────────────────────────────
        encabezados = VGroup(
            Text("Vocabulario", font=FUENTE, font_size=22, color=MARRON_OSCURO, weight=BOLD),
            Text("Total tokens", font=FUENTE, font_size=22, color=MARRON_OSCURO, weight=BOLD),
            Text("Compresión",   font=FUENTE, font_size=22, color=NARANJA_TERRACOTA, weight=BOLD),
        ).arrange(RIGHT, buff=1.8)

        sep = Line(LEFT, RIGHT, color=MARRON_OSCURO, stroke_width=2)

        datos = [
            ("256",    "2 168 312", "1.00×", False),
            ("1 024",  "724 453",   "2.99×", False),
            ("20 534", "460 900",   "4.70×", True),
        ]

        filas = VGroup()
        for vocab, tokens, comp, es_final in datos:
            color_c = NARANJA_TERRACOTA if es_final else TINTA_NEGRA
            peso_c  = BOLD if es_final else NORMAL
            fila = VGroup(
                Text(vocab,  font=FUENTE, font_size=22, color=TINTA_NEGRA),
                Text(tokens, font=FUENTE, font_size=22, color=TINTA_NEGRA),
                Text(comp,   font=FUENTE, font_size=22, color=color_c, weight=peso_c),
            )
            for i in range(3):
                fila[i].set_x(encabezados[i].get_x())
            filas.add(fila)

        filas.arrange(DOWN, buff=0.4)
        tabla_interna = VGroup(encabezados, sep, filas).arrange(DOWN, buff=0.3)
        sep.set_width(tabla_interna.width + 0.8)

        fondo_tabla = RoundedRectangle(
            width=tabla_interna.width + 1.2, height=tabla_interna.height + 0.8,
            corner_radius=0.15, fill_color=FONDO_CAJA, fill_opacity=1,
            stroke_color=MARRON_OSCURO, stroke_width=2
        )
        grupo_tabla = VGroup(fondo_tabla, tabla_interna).move_to(UP * 0.3)

        lbl_tabla = Text("Mismo corpus — menos tokens con vocab más grande",
                         font=FUENTE, font_size=22, weight=BOLD, color=TINTA_NEGRA).next_to(grupo_tabla, UP, buff=0.4)

        self.play(Write(lbl_tabla))
        self.play(FadeIn(fondo_tabla), FadeIn(encabezados), Create(sep))
        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.2) for f in filas], lag_ratio=0.3))
        self.next_slide()

        # Resaltar la fila ganadora
        resalto = SurroundingRectangle(filas[2], color=NARANJA_TERRACOTA,
                                       stroke_width=3, buff=0.08, corner_radius=0.06)
        self.play(Create(resalto), Indicate(filas[2][2], color=NARANJA_TERRACOTA, scale_factor=1.15))
        self.next_slide()

        self.play(FadeOut(lbl_tabla), FadeOut(grupo_tabla), FadeOut(resalto))

        # ── ACTO 4: EL TRADE-OFF ─────────────────────────────────────────────────
        titulo_tradeoff = Text("El trade-off: velocidad vs. memoria",
                               font=FUENTE, font_size=28, weight=BOLD, color=TINTA_NEGRA)
        titulo_tradeoff.move_to(UP * 2.6)

        # Lado PROS (vocab grande)
        lado_pro = VGroup(
            Text("Vocab GRANDE", font=FUENTE, font_size=22, weight=BOLD, color=VERDE_OLIVA),
            Text("✔  Secuencias más cortas", font=FUENTE, font_size=20, color=TINTA_NEGRA),
            Text("✔  Inferencia más rápida", font=FUENTE, font_size=20, color=TINTA_NEGRA),
            Text("✔  Menos pasos de atención", font=FUENTE, font_size=20, color=TINTA_NEGRA),
            Text("✘  Más VRAM para la matriz", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to(LEFT * 2.8 + DOWN * 0.2)

        # Lado CONS (vocab pequeño)
        lado_con = VGroup(
            Text("Vocab PEQUEÑO", font=FUENTE, font_size=22, weight=BOLD, color=ROJO_TOMATE),
            Text("✔  Matriz de embeddings ligera", font=FUENTE, font_size=20, color=TINTA_NEGRA),
            Text("✘  Secuencias muy largas", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA),
            Text("✘  Más lento en inferencia", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA),
            Text("✘  Peor manejo de palabras raras", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to(RIGHT * 2.8 + DOWN * 0.2)

        divisor = DashedLine(UP * 1.5, DOWN * 2.0, color=MARRON_OSCURO,
                             dash_length=0.15, dashed_ratio=0.5).move_to(ORIGIN + DOWN * 0.2)

        self.play(Write(titulo_tradeoff))
        self.play(Create(divisor))
        self.play(LaggedStart(
            FadeIn(lado_pro, shift=RIGHT * 0.2),
            FadeIn(lado_con, shift=LEFT * 0.2),
            lag_ratio=0.2
        ))
        self.next_slide()

        # Conclusión práctica: GPT-2 usa 50 257
        conclusion = Text(
            "GPT-2 eligió 50 257  →  balance empírico entre los dos extremos",
            font=FUENTE, font_size=21, weight=BOLD, color=NARANJA_TERRACOTA
        ).next_to(lado_pro, DOWN, buff=0.5).set_x(0)

        self.play(Write(conclusion))
        self.play(Indicate(conclusion, color=ORO_VIEJO, scale_factor=1.05))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_que_es_un_tensor(self):
        """Diapositiva: Slide Que Es Un Tensor."""

        titulo, linea = self.crear_titulo("¿Qué es un Tensor?", palabra_clave="Tensor?", color_clave=NARANJA_TERRACOTA)
        
        llanuras_fondo = crear_llanuras_manchegas()

        adornos = self._crear_adornos_esquinas()
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.15))

        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)
        
        def_tensor = Text(
            "Un Tensor es un contenedor matemático\npara almacenar datos en múltiples dimensiones.", 
            font=FUENTE, font_size=32, 
            color=MARRON_OSCURO,
            t2c={"múltiples dimensiones.": NARANJA_TERRACOTA}
        ).next_to(linea, DOWN, buff=0.4)
        
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

        nota_ram = Text("Pero en memoria RAM, todos terminan siendo un arreglo plano...", font=FUENTE, font_size=26, color=NARANJA_TERRACOTA).next_to(linea, DOWN, buff=0.5)
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

        adornos[1].clear_updaters()
        self.limpiar_pantalla()

    def slide_softmax(self):
        """Diapositiva: Slide Softmax."""

        titulo, linea = self.crear_titulo(
            "Softmax", 
            palabra_clave="Probabilidades", 
            color_clave=NARANJA_TERRACOTA
        )
            
        formula = MathTex(
            r"\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}", 
            color=TINTA_NEGRA, font_size=38
        ).next_to(linea, DOWN, buff=0.4)

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)
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
        """Diapositiva: Slide Broadcasting."""

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

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)
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
        """Diapositiva: Slide Strides."""

        titulo, linea = self.crear_titulo(
            "Strides: Saltando en Memoria 1D", 
            palabra_clave="Strides:", 
            color_clave=NARANJA_TERRACOTA
        )

        camino_mancha = FunctionGraph(lambda x: 0.5 * math.sin(x) - 0.5, color=MARRON_OSCURO).set_opacity(0.15)
        camino_punteado = DashedVMobject(camino_mancha, num_dashes=45, dashed_ratio=0.5)
        
        lanza_fondo = Line(LEFT * 7 + DOWN * 2, RIGHT * 7 + UP * 2, color=NARANJA_TERRACOTA, stroke_width=2).set_opacity(0.15)
        
        decoracion_quijote = VGroup(camino_punteado, lanza_fondo).set_z_index(-2)

        adornos = self._crear_adornos_esquinas()
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.15))

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=VGroup(llanuras_fondo, decoracion_quijote))

        arr_1d = VGroup(*[self.crear_bloque(str(i)) for i in range(6)])
        arr_1d.arrange(RIGHT, buff=0.1).shift(UP * 1.5)
        
        lbl_1d = Text("Memoria RAM (Física, 1D)", font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(arr_1d, UP, buff=0.3)

        self.play(FadeIn(arr_1d, shift=UP*0.2), FadeIn(lbl_1d, shift=UP*0.2))
        self.next_slide()

        fila1 = VGroup(*[arr_1d[i].copy() for i in range(3)]).arrange(RIGHT, buff=0.1)
        fila2 = VGroup(*[arr_1d[i].copy() for i in range(3, 6)]).arrange(RIGHT, buff=0.1)
        
        mat_shape = VGroup(fila1, fila2).arrange(DOWN, buff=0.1).shift(DOWN * 0.2)
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
        ).to_edge(DOWN, buff=0.6) 
        
        self.play(FadeIn(def_stride, shift=UP*0.2))
        self.next_slide()
        
        adornos[1].clear_updaters()
        self.limpiar_pantalla()

    def slide_masked_fill(self):
        """Diapositiva: Slide Masked Fill."""

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
        
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)
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
        """Diapositiva: Slide Reshape Transpose."""

        titulo, linea = self.crear_titulo(
            "Reshaping vs Transposing", 
            palabra_clave="vs", 
            color_clave=MARRON_OSCURO
        )
        
        titulo.set_color_by_t2c({"Reshaping": MARRON_OSCURO, "Transposing": NARANJA_TERRACOTA})
        
        linea_central = DashedLine(UP*2.2, DOWN*2.5, color=MARRON_OSCURO)

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)
        self.play(Create(linea_central))
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
            TransformFromCopy(mat_r_orig[0][0], mat_r_final[0][0]),
            TransformFromCopy(mat_r_orig[0][1], mat_r_final[0][1]),
            TransformFromCopy(mat_r_orig[0][2], mat_r_final[1][0]),
            TransformFromCopy(mat_r_orig[1][0], mat_r_final[1][1]),
            TransformFromCopy(mat_r_orig[1][1], mat_r_final[2][0]),
            TransformFromCopy(mat_r_orig[1][2], mat_r_final[2][1]),
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
        """Diapositiva: Slide Forward Pass."""

        titulo, linea = self.crear_titulo("Arquitectura: El Forward Pass", color_clave=NARANJA_TERRACOTA)
        
        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        
        llanuras_fondo = crear_llanuras_manchegas()
        
        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))
        
        frase_completa = VGroup(
            Text('"En un lugar de la', font=FUENTE, font_size=42, color=TINTA_NEGRA),
            Text(' Mancha"', font=FUENTE, font_size=42, color=NARANJA_TERRACOTA)
        ).arrange(RIGHT, buff=0.1).move_to(UP * 1.8) 

        frase_base = frase_completa[0]
        posicion_destino_mancha = frase_completa[1].get_center()
        
        self.play(Write(frase_base))
        self.next_slide()

        def crear_caja_nodo(label_sup, valor_array, label_inf, funcion_adorno=None, is_stack=False, highlight_last=False):
            grupo = VGroup()
            txt_sup = Text(label_sup, font="Monospace", font_size=16, weight=BOLD, color=MARRON_OSCURO)
            
            ancho_caja = 2.5 
            alto_caja = 1.6
            radio_caja = 0.2
            
            sombra = RoundedRectangle(corner_radius=radio_caja, width=ancho_caja, height=alto_caja)
            sombra.set_fill(MARRON_OSCURO, opacity=0.12).set_stroke(width=0)
            sombra.shift(RIGHT * 0.08 + DOWN * 0.08)
            
            caja = RoundedRectangle(corner_radius=radio_caja, width=ancho_caja, height=alto_caja)
            caja.set_fill(color=PAPEL_CREMA, opacity=1).set_stroke(color=MARRON_OSCURO, width=2.5)
            
            color_array = NARANJA_TERRACOTA if highlight_last else TINTA_NEGRA
            txt_arr = Text(valor_array, font="Monospace", font_size=16, weight=BOLD, color=color_array) 
            txt_inf = Text(label_inf, font=FUENTE, font_size=15, color=TINTA_NEGRA) 
            
            if txt_arr.width > (ancho_caja - 0.4):
                txt_arr.scale_to_fit_width(ancho_caja - 0.4)
            if txt_inf.width > (ancho_caja - 0.4):
                txt_inf.scale_to_fit_width(ancho_caja - 0.4)
                
            Textos_caja = VGroup(txt_arr, txt_inf).arrange(DOWN, buff=0.25)
            
            if is_stack:
                caja_fondo1 = caja.copy().set_stroke(width=1.5, opacity=0.5).shift(RIGHT * 0.12 + UP * 0.12)
                caja_fondo2 = caja.copy().set_stroke(width=1.5, opacity=0.8).shift(RIGHT * 0.06 + UP * 0.06)
                fondo = VGroup(sombra, caja_fondo1, caja_fondo2, caja)
            else:
                fondo = VGroup(sombra, caja)
                
            caja_y_textos = VGroup(fondo, Textos_caja)
            Textos_caja.move_to(caja.get_center()) 
            

            if funcion_adorno:
                adorno = funcion_adorno().scale(0.65) 
                adorno.next_to(caja_y_textos, UP, buff=0.15)
                txt_sup.next_to(adorno, UP, buff=0.15)
                grupo.add(adorno)
                grupo.adorno = adorno 
            else:
                txt_sup.next_to(caja_y_textos, UP, buff=0.2)
                grupo.adorno = VGroup()
                
            grupo.add(txt_sup, caja_y_textos)
            grupo.caja_principal = caja 
            return grupo

        nodo_1 = crear_caja_nodo("1. Token_IDs", "[145, 892...]", "Tokens: En|un...", funcion_adorno=crear_tintero_y_pluma)
        nodo_2 = crear_caja_nodo("2. Embeddings", "[0.81, -0.2...]", "Vectores (768d)", funcion_adorno=crear_rueda_carreta)
        nodo_3 = crear_caja_nodo("3. Blocks", "[0.55, 0.9...]", "Atención (x12)", funcion_adorno=crear_molino, is_stack=True)
        nodo_4 = crear_caja_nodo("4. LayerNorm", "[0.12, -0.4...]", "Norm (768d)", funcion_adorno=crear_escudo_y_lanza)
        nodo_5 = crear_caja_nodo("5. LM_Head", "[..., 25.4...]", "Score Máximo", funcion_adorno=crear_pila_libros, highlight_last=True)

        pipeline = VGroup(nodo_1, nodo_2, nodo_3, nodo_4, nodo_5)
        pipeline.arrange(RIGHT, buff=0.4).scale(0.62).move_to(DOWN * 0.2)

        flechas = VGroup()
        for i in range(len(pipeline) - 1):
            flecha = Arrow(
                pipeline[i].caja_principal.get_right(), 
                pipeline[i+1].caja_principal.get_left(), 
                buff=0.1, color=MARRON_OSCURO, stroke_width=4, max_tip_length_to_length_ratio=0.2
            )
            flechas.add(flecha)


        self.play(
            AnimationGroup(*[FadeIn(nodo, shift=UP*0.3) for nodo in pipeline], lag_ratio=0.15),
            FadeIn(flechas, shift=UP*0.2),
            run_time=2.5
        )
        self.next_slide()

        tokens_viajeros = frase_base.copy()
        centro_primera_caja = pipeline[0].caja_principal.get_center()
        
        nucleo = Dot(color=WHITE, radius=0.08)
        aura = Dot(color=NARANJA_TERRACOTA, radius=0.2).set_opacity(0.6)
        punto_flujo = VGroup(aura, nucleo).move_to(centro_primera_caja)
        
        estela = TracedPath(punto_flujo.get_center, stroke_color=NARANJA_TERRACOTA, stroke_width=6, dissipating_time=0.4)
        
        panel_estado = RoundedRectangle(
            corner_radius=0.15, height=0.7, width=10, 
            fill_color=MARRON_OSCURO, fill_opacity=0.05, stroke_color=NARANJA_TERRACOTA, stroke_width=1
        )
        panel_estado.move_to(DOWN * 3.2)
        
        texto_estado = Text("Traduciendo a números...", font=FUENTE, font_size=24, color=NARANJA_TERRACOTA, weight=BOLD)
        texto_estado.move_to(panel_estado.get_center())

        self.play(
            ReplacementTransform(tokens_viajeros, punto_flujo, path_arc=-PI/4), 
            FadeIn(panel_estado, shift=UP*0.2),
            FadeIn(texto_estado, shift=UP*0.2),
            run_time=1.2
        )
        self.add(estela)
        
        self.play(
            pipeline[0].caja_principal.animate.scale(1.05).set_color(NARANJA_TERRACOTA),
            pipeline[0].adorno.animate.shift(UP*0.1), 
            rate_func=there_and_back, run_time=0.5
        )

        descripciones = [
            "Convirtiendo en vectores espaciales...",
            "Calculando atención y contexto (12 capas)...",
            "Estabilizando los datos...",
            "Calculando probabilidad de la siguiente palabra..."
        ]

        for i in range(len(flechas)):
            centro_siguiente_caja = pipeline[i+1].caja_principal.get_center()
            nuevo_texto_estado = Text(descripciones[i], font=FUENTE, font_size=24, color=NARANJA_TERRACOTA, weight=BOLD)
            nuevo_texto_estado.move_to(panel_estado.get_center())

            self.play(flechas[i].animate.set_color(NARANJA_TERRACOTA), run_time=0.2)
            

            self.play(
                punto_flujo.animate.move_to(centro_siguiente_caja), 
                ReplacementTransform(texto_estado, nuevo_texto_estado),
                run_time=0.8, 
                rate_func=smooth
            )
            texto_estado = nuevo_texto_estado

            self.play(
                flechas[i].animate.set_color(MARRON_OSCURO),
                pipeline[i+1].caja_principal.animate.scale(1.05).set_color(NARANJA_TERRACOTA), 
                pipeline[i+1].adorno.animate.shift(UP*0.1), 
                rate_func=there_and_back, 
                run_time=0.5
            )

        self.next_slide()

        prediccion_txt = Text('      Mancha"', font=FUENTE, font_size=42, color=NARANJA_TERRACOTA)
        prediccion_txt.move_to(pipeline[-1].caja_principal.get_center())
        
        self.play(
            FadeOut(estela),
            ReplacementTransform(punto_flujo, prediccion_txt), 
            FadeOut(panel_estado, shift=DOWN*0.2),
            FadeOut(texto_estado, shift=DOWN*0.2),
            run_time=0.8
        )
        self.next_slide()

        self.play(
            prediccion_txt.animate.move_to(posicion_destino_mancha),
            run_time=1.2,
            path_arc=-PI/4
        )
        
        frase_final = VGroup(frase_base, prediccion_txt)
        self.play(Flash(prediccion_txt, color=NARANJA_TERRACOTA, line_length=0.4, flash_radius=1.5, run_time=1))
        self.play(Circumscribe(frase_final, color=NARANJA_TERRACOTA, time_width=2, stroke_width=4))
        
        self.next_slide()
        self.limpiar_pantalla()

    def slide_embeddings(self):
        """Diapositiva: Slide Embeddings — Por qué vectores, analogía geométrica, lookup."""

        llanuras_fondo = crear_llanuras_manchegas()
        adornos = self._crear_adornos_esquinas(escala=0.55, buff=0.5)
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.4))

        titulo_p1 = Text("Embeddings: ", font=FUENTE, font_size=42, weight=BOLD, color=TINTA_NEGRA)
        titulo_p2 = Text("De Tokens a Significado", font=FUENTE, font_size=42, weight=BOLD, color=NARANJA_TERRACOTA)
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 5, RIGHT * 5, color=MARRON_OSCURO).next_to(titulo_completo, DOWN)
        VGroup(titulo_completo, linea).to_edge(UP, buff=0.5)

        self._animar_entrada_slide(titulo_completo, linea, fondo=llanuras_fondo, adornos=adornos)

        # ══════════════════════════════════════════════════════════════════
        # ACTO 1 — El problema: los IDs no tienen significado
        # ══════════════════════════════════════════════════════════════════
        pregunta = Text(
            "¿Por qué no usar el ID del token directamente?",
            font=FUENTE, font_size=28, color=MARRON_OSCURO, weight=BOLD
        ).next_to(linea, DOWN, buff=0.5)
        self.play(Write(pregunta))
        self.next_slide()

        tokens = ['"rey"', '"reina"', '"hombre"', '"mujer"']
        ids    = ['4726', '9034', '512', '8801']
        colores_tok = [NARANJA_TERRACOTA, VERDE_OLIVA, LADRILLO_VIVO, AZUL_NOCHE]
        
        tok_group = VGroup()
        for palabra, id_str, color in zip(tokens, ids, colores_tok):
            pal = Text(palabra, font=FUENTE, font_size=22, color=color, weight=BOLD)
            flecha = Arrow(LEFT * 0.4, RIGHT * 0.4, color=MARRON_OSCURO,
                           stroke_width=3, max_tip_length_to_length_ratio=0.25)
            id_bg = RoundedRectangle(corner_radius=0.1, width=1.2, height=0.5,
                                     fill_color=MARRON_OSCURO, fill_opacity=1, stroke_width=0)
            id_txt = Text(id_str, font="Monospace", font_size=18,
                          color=PAPEL_CREMA, weight=BOLD).move_to(id_bg)
            fila = VGroup(pal, flecha, VGroup(id_bg, id_txt)).arrange(RIGHT, buff=0.2)
            tok_group.add(fila)

        tok_group.arrange(DOWN, buff=0.3)

        # Caja de problema: las distancias numéricas son arbitrarias
        problema_bg = RoundedRectangle(
            corner_radius=0.15, width=5.2, height=2.6,
            fill_color=SALMON_CLARO, fill_opacity=0.92,
            stroke_color=LADRILLO_VIVO, stroke_width=2.5
        )

        prob_titulo = Text("El problema", font=FUENTE, font_size=20,
                           color=LADRILLO_VIVO, weight=BOLD)
        prob_items = VGroup(
            Text("• |4726 - 9034| = 4308 <- distancia rey-reina?", font=FUENTE, font_size=16, color=TINTA_NEGRA),
            Text("• |512 - 8801| = 8289 <- distancia hombre-mujer?", font=FUENTE, font_size=16, color=TINTA_NEGRA),
            Text("• Los IDs son arbitrarios: no codifican", font=FUENTE, font_size=16, color=TINTA_NEGRA),
            Text("  ninguna relación semántica.", font=FUENTE, font_size=16, color=TINTA_NEGRA),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        prob_contenido = VGroup(prob_titulo, prob_items).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        grupo_problema = VGroup(problema_bg, prob_contenido)
        prob_contenido.move_to(problema_bg)

        # Distribuir lado a lado balanceadamente debajo de la pregunta
        acto1_layout = VGroup(tok_group, grupo_problema).arrange(RIGHT, buff=1.0).next_to(pregunta, DOWN, buff=0.7)

        self.play(LaggedStart(*[FadeIn(f, shift=RIGHT * 0.3) for f in tok_group], lag_ratio=0.15))
        self.play(FadeIn(problema_bg, shift=LEFT * 0.3), Write(prob_contenido), run_time=1.2)
        self.next_slide()

        self.play(FadeOut(tok_group, pregunta, grupo_problema))

        # ══════════════════════════════════════════════════════════════════
        # ACTO 2 — La solución: vectores en espacio de alta dimensión
        # ══════════════════════════════════════════════════════════════════
        solucion_titulo = Text(
            "La solución: representar cada token como un vector",
            font=FUENTE, font_size=26, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.5)
        self.play(Write(solucion_titulo))

        ventajas = [
            ("Cercanía = similitud semántica",   VERDE_OLIVA,        "rey aprox. reina en el espacio"),
            ("Aritmética de conceptos",          NARANJA_TERRACOTA,  "rey − hombre + mujer ≈ reina"),
            ("Parámetros entrenables",           OCRE_CERVANTINO,    "El modelo aprende los pesos"),
            ("Información densa",                LADRILLO_VIVO,      "768 floats vs. un entero"),
        ]

        cards_list = []
        for titulo_v, color, desc in ventajas:
            card_bg = RoundedRectangle(
                corner_radius=0.14, width=5.0, height=0.9,
                fill_color=FONDO_CAJA, fill_opacity=1,
                stroke_color=color, stroke_width=2.2
            )
            dot = Dot(radius=0.09, color=color).move_to(card_bg.get_left() + RIGHT * 0.35)
            t1 = Text(titulo_v, font=FUENTE, font_size=17, color=color, weight=BOLD)
            t2 = Text(desc, font=FUENTE, font_size=14, color=MARRON_OSCURO)
            textos = VGroup(t1, t2).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            textos.next_to(dot, RIGHT, buff=0.2)
            
            # CORRECCIÓN: usar .append() en lugar de .add()
            cards_list.append(VGroup(card_bg, dot, textos)) 

        # Cuadrícula 2x2 para no desbordar por debajo y verse más limpio
        cards = VGroup(*cards_list).arrange_in_grid(rows=2, cols=2, buff=(0.5, 0.4)).next_to(solucion_titulo, DOWN, buff=0.8)

        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.2), run_time=1.4)
        self.next_slide()

        self.play(FadeOut(solucion_titulo, cards))

        # ══════════════════════════════════════════════════════════════════
        # ACTO 3 — Analogía geométrica: aritmética vectorial cervantina
        # ══════════════════════════════════════════════════════════════════
        geo_label = Text(
            "Geometría del significado", font=FUENTE, font_size=24,
            color=MARRON_OSCURO, weight=BOLD, slant=ITALIC
        ).next_to(linea, DOWN, buff=0.4)
        self.play(FadeIn(geo_label, shift=DOWN * 0.15))

        ejes = Axes(
            x_range=[-3.5, 3.5, 1], y_range=[-2.8, 2.8, 1],
            x_length=7.0, y_length=4.5,
            axis_config={"color": MARRON_OSCURO, "stroke_width": 1.5,
                         "include_ticks": False, "stroke_opacity": 0.5}
        ).next_to(geo_label, DOWN, buff=0.4) # Anclado dinámicamente

        etiq_x = Text("género →", font=FUENTE, font_size=16, color=MARRON_OSCURO, slant=ITALIC)
        etiq_x.next_to(ejes.x_axis.get_right(), DOWN, buff=0.15)
        etiq_y = Text("nobleza ↑", font=FUENTE, font_size=16, color=MARRON_OSCURO, slant=ITALIC)
        etiq_y.next_to(ejes.y_axis.get_top(), LEFT, buff=0.15)

        self.play(Create(ejes), Write(etiq_x), Write(etiq_y))

        puntos = {
            "Rey":      (np.array([2.0,  1.8, 0]), NARANJA_TERRACOTA, UR),
            "Reina":    (np.array([-2.0, 1.8, 0]), VERDE_OLIVA,       UL),
            "Hombre":   (np.array([2.0, -1.8, 0]), NARANJA_TERRACOTA, DR),
            "Mujer":    (np.array([-2.0,-1.8, 0]), VERDE_OLIVA,       DL),
        }

        def vec_obj(nombre, color, dir_lbl):
            coord, _, _ = puntos[nombre]
            p = ejes.c2p(*coord)
            flecha = Arrow(ejes.c2p(0,0), p, color=color, buff=0, stroke_width=4,
                           max_tip_length_to_length_ratio=0.12)
            lbl = Text(nombre, font=FUENTE, font_size=20, color=color, weight=BOLD)
            lbl.set_background_stroke(color=PAPEL_CREMA, width=3)
            lbl.next_to(p, dir_lbl, buff=0.15)
            return VGroup(flecha, lbl)

        v_rey    = vec_obj("Rey",    NARANJA_TERRACOTA, UR)
        v_hombre = vec_obj("Hombre", NARANJA_TERRACOTA, DR)
        v_reina  = vec_obj("Reina",  VERDE_OLIVA,       UL)
        v_mujer  = vec_obj("Mujer",  VERDE_OLIVA,       DL)

        self.play(Create(v_rey), Create(v_hombre), run_time=0.9)
        self.play(Create(v_reina), Create(v_mujer), run_time=0.9)

        coord_rey    = ejes.c2p(*puntos["Rey"][0])
        coord_reina  = ejes.c2p(*puntos["Reina"][0])
        coord_hombre = ejes.c2p(*puntos["Hombre"][0])
        coord_mujer  = ejes.c2p(*puntos["Mujer"][0])

        flecha_gen1 = DashedLine(coord_rey, coord_reina, color=LAVANDA, stroke_width=3)
        flecha_gen2 = DashedLine(coord_hombre, coord_mujer, color=LAVANDA, stroke_width=3)
        lbl_gen = Text("mismo Δ género", font=FUENTE, font_size=16, color=LAVANDA, weight=BOLD)
        lbl_gen.set_background_stroke(color=PAPEL_CREMA, width=3)
        lbl_gen.move_to(ejes.c2p(0, 2.2))

        self.play(Create(flecha_gen1), Create(flecha_gen2), Write(lbl_gen))
        self.next_slide()

        formula = MathTex(
            r"\vec{\text{Rey}} - \vec{\text{Hombre}} + \vec{\text{Mujer}}",
            r"\approx \vec{\text{Reina}}",
            font_size=32, color=TINTA_NEGRA
        ).to_edge(DOWN, buff=0.3)
        formula.set_background_stroke(color=PAPEL_CREMA, width=4)

        self.play(Write(formula))
        self.play(
            Indicate(v_reina, color=ORO_VIEJO, scale_factor=1.12),
            Flash(ejes.c2p(*puntos["Reina"][0]), color=ORO_VIEJO, line_length=0.4, num_lines=10),
        )
        self.next_slide()

        self.play(FadeOut(ejes, etiq_x, etiq_y, v_rey, v_hombre, v_reina, v_mujer,
                          flecha_gen1, flecha_gen2, lbl_gen, formula, geo_label))

        # ══════════════════════════════════════════════════════════════════
        # ACTO 4 — Mecanismo real: lookup en la matriz de embeddings
        # ══════════════════════════════════════════════════════════════════
        mec_label = Text(
            "¿Cómo funciona en la práctica?",
            font=FUENTE, font_size=26, color=MARRON_OSCURO, weight=BOLD
        ).next_to(linea, DOWN, buff=0.4)
        self.play(Write(mec_label))

        # Componentes a crear
        input_word = Text('"quijote"', font=FUENTE, font_size=34, color=NARANJA_TERRACOTA, weight=BOLD)
        bg_id = RoundedRectangle(corner_radius=0.18, width=2.4, height=0.72,
                                  fill_color=MARRON_OSCURO, fill_opacity=1, stroke_width=0)
        id_txt = Text("ID: 1605", font="Monospace", font_size=22, color=PAPEL_CREMA, weight=BOLD).move_to(bg_id)
        grupo_id = VGroup(bg_id, id_txt)

        # Matriz
        filas_m, cols_m = 9, 11
        fila_sel_idx = 4
        matriz_v = VGroup()
        for i in range(filas_m):
            fila = VGroup()
            for j in range(cols_m):
                color_fill = NARANJA_TERRACOTA if i == fila_sel_idx else PAPEL_CREMA
                opac = 0.25 if i == fila_sel_idx else 0.5
                cuadro = RoundedRectangle(
                    corner_radius=0.04, width=0.32, height=0.32,
                    fill_color=color_fill, fill_opacity=opac
                ).set_stroke(CAJA_INFERIOR, opacity=0.55, width=1)
                fila.add(cuadro)
            matriz_v.add(fila.arrange(RIGHT, buff=0.07))
        matriz_v.arrange(DOWN, buff=0.07)

        lbl_matriz = Text("Matriz de Embeddings", font=FUENTE, font_size=18, weight=BOLD, color=PAPEL_TAN)
        lbl_dim = Text("(vocab × 768)", font=FUENTE, font_size=15, color=MARRON_OSCURO)
        grupo_matriz = VGroup(lbl_matriz, lbl_dim, matriz_v).arrange(DOWN, buff=0.15)

        # Disposición horizontal dinámica para Palabra/ID y Matriz
        acto4_top_layout = VGroup(input_word, grupo_matriz).arrange(RIGHT, buff=1.8).next_to(mec_label, DOWN, buff=0.5)
        
        # Mover grupo ID exactamente a donde está el input word
        grupo_id.move_to(input_word)

        self.play(FadeIn(input_word, shift=RIGHT * 0.3))
        self.next_slide()
        self.play(ReplacementTransform(input_word, grupo_id))

        flecha_a_matriz = Arrow(grupo_id.get_right(), matriz_v.get_left(), buff=0.2,
                                color=MARRON_OSCURO, stroke_width=3, max_tip_length_to_length_ratio=0.2)
        lbl_lookup = Text("lookup", font=FUENTE, font_size=16, color=MARRON_OSCURO, slant=ITALIC)
        lbl_lookup.next_to(flecha_a_matriz, UP, buff=0.1)

        self.play(GrowArrow(flecha_a_matriz), Write(lbl_lookup))
        self.next_slide()

        self.play(FadeIn(grupo_matriz, shift=LEFT * 0.2))
        self.next_slide()

        fila_highlight = SurroundingRectangle(
            matriz_v[fila_sel_idx], color=NARANJA_TERRACOTA,
            buff=0.05, stroke_width=3, corner_radius=0.06
        )
        self.play(
            Create(fila_highlight),
            matriz_v[fila_sel_idx].animate.set_fill(NARANJA_TERRACOTA, opacity=0.85),
        )
        self.next_slide()

        # Vector resultante
        valores_ejemplo = ["0.12", "-0.4", "0.83", "-0.2", "0.51", "0.07", "-0.9", "0.34", "···", "0.71"]
        vector_visual = VGroup()
        for val in valores_ejemplo:
            es_puntos = val == "···"
            bloque_v = RoundedRectangle(
                corner_radius=0.06, width=0.72, height=0.68,
                fill_color=NARANJA_TERRACOTA if not es_puntos else FONDO_CAJA,
                fill_opacity=0.85 if not es_puntos else 0,
            ).set_stroke(MARRON_OSCURO if not es_puntos else CAJA_INFERIOR, 1.5 if not es_puntos else 0.5)
            txt_v = Text(val, font="Monospace", font_size=14, color=TINTA_NEGRA if not es_puntos else MARRON_OSCURO).move_to(bloque_v)
            vector_visual.add(VGroup(bloque_v, txt_v))

        vector_visual.arrange(RIGHT, buff=0.08)

        lbl_vec = Text("Vector semántico de quijote", font=FUENTE, font_size=20, weight=BOLD, color=TINTA_NEGRA)
        lbl_dim2 = Text("(768 dimensiones, aprox.)", font=FUENTE, font_size=15, color=PAPEL_TAN)
        grupo_vector = VGroup(lbl_vec, lbl_dim2, vector_visual).arrange(DOWN, buff=0.15)
        
        # Centramos el vector justo debajo de la matriz (o centrado global)
        grupo_vector.next_to(grupo_matriz, DOWN, buff=0.6).align_to(mec_label, DOWN).to_edge(DOWN, buff=0.3)

        self.play(
            ReplacementTransform(fila_highlight.copy(), vector_visual),
            Write(lbl_vec), Write(lbl_dim2),
            run_time=1.4
        )
        self.play(
            vector_visual.animate.scale(1.04),
            rate_func=there_and_back, run_time=0.7
        )

        self.next_slide()
        self.limpiar_pantalla()

    def slide_position_embeddings(self):
        # TÍTULO PRINCIPAL (Fijado arriba)
        titulo_p1 = Text("Embeddings de ", font=FUENTE, font_size=42, weight=BOLD, color=TINTA_NEGRA)
        titulo_p2 = Text("Posición", font=FUENTE, font_size=42, weight=BOLD, color=NARANJA_TERRACOTA)
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1).to_edge(UP, buff=0.5)
        linea = Line(LEFT * 5, RIGHT * 5, color=MARRON_OSCURO).next_to(titulo_completo, DOWN)

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo_completo, linea, fondo=llanuras_fondo)

        # --- ESCENA 1: EL PROBLEMA ---
        label_problema = Text("El problema: los Transformers no ven el orden", 
                              font=FUENTE, font_size=26, weight=BOLD, color=NARANJA_TERRACOTA)
        label_problema.move_to(UP * 1.5)

        frase_a_words = ["El", "perro", "muerde", "al", "hombre"]
        frase_b_words = ["El", "hombre", "muerde", "al", "perro"]

        def hacer_fila_tokens(palabras):
            cajas = VGroup()
            for w in palabras:
                rect = RoundedRectangle(corner_radius=0.1, width=1.5, height=0.6,
                                        fill_color=PAPEL_CREMA, fill_opacity=0.9,
                                        stroke_color=MARRON_OSCURO, stroke_width=2)
                txt = Text(w, font=FUENTE, font_size=22, color=TINTA_NEGRA)
                txt.move_to(rect.get_center())
                cajas.add(VGroup(rect, txt))
            cajas.arrange(RIGHT, buff=0.25)
            return cajas

        fila_a = hacer_fila_tokens(frase_a_words)
        fila_b = hacer_fila_tokens(frase_b_words)

        lbl_a = Text("Frase A:", font=FUENTE, font_size=20, weight=BOLD, color=MARRON_OSCURO)
        lbl_b = Text("Frase B:", font=FUENTE, font_size=20, weight=BOLD, color=MARRON_OSCURO)

        grupo_a = VGroup(lbl_a, fila_a).arrange(RIGHT, buff=0.3)
        grupo_b = VGroup(lbl_b, fila_b).arrange(RIGHT, buff=0.3)
        bloque_frases = VGroup(grupo_a, grupo_b).arrange(DOWN, buff=0.8).move_to(DOWN * 0.5)

        self.play(Write(label_problema))
        self.play(FadeIn(grupo_a, shift=RIGHT*0.3))
        self.play(FadeIn(grupo_b, shift=RIGHT*0.3))
        self.next_slide()

        self.play(
            fila_a[1][0].animate.set_fill(ROJO_TOMATE, opacity=0.85),
            fila_a[1][1].animate.set_color(BLANCO),
            fila_b[1][0].animate.set_fill(ROJO_TOMATE, opacity=0.85),
            fila_b[1][1].animate.set_color(BLANCO),
            fila_a[4][0].animate.set_fill(VERDE_OLIVA, opacity=0.85),
            fila_a[4][1].animate.set_color(BLANCO),
            fila_b[4][0].animate.set_fill(VERDE_OLIVA, opacity=0.85),
            fila_b[4][1].animate.set_color(BLANCO),
        )

        nota_orden = Text(
            "Mismas palabras · distinto orden → significado opuesto\n"
            "Sin posición, el modelo las vería IGUAL",
            font=FUENTE, font_size=21, color=NARANJA_TERRACOTA
        ).next_to(bloque_frases, DOWN, buff=1.0)
        
        self.play(Write(nota_orden))
        self.next_slide()

        self.play(
            FadeOut(label_problema), FadeOut(grupo_a), FadeOut(grupo_b), FadeOut(nota_orden)
        )

        # --- ESCENA 2: LA ATENCIÓN ---
        label_atencion = Text("¿Por qué? La atención no depende del orden", 
                              font=FUENTE, font_size=26, weight=BOLD, color=NARANJA_TERRACOTA)
        label_atencion.move_to(UP * 1.5)

        explicacion = Text(
            "En Attention, cada token mira a TODOS los demás a la vez.\n"
            "No hay concepto de 'izquierda' ni 'derecha'.\n"
            "Es como un conjunto desordenado, no una secuencia.",
            font=FUENTE, font_size=22, color=TINTA_NEGRA, line_spacing=1.4
        ).next_to(label_atencion, DOWN, buff=0.5)

        nodo_pos = [LEFT*2.2 + UP*0.6, LEFT*0.8 + DOWN*0.6, RIGHT*0.8 + DOWN*0.6, RIGHT*2.2 + UP*0.6]
        nodo_palabras = ["perro", "muerde", "al", "hombre"]
        nodos = VGroup()
        for pos, palabra in zip(nodo_pos, nodo_palabras):
            circ = Circle(radius=0.4, fill_color=PAPEL_TAN, fill_opacity=0.9,
                          stroke_color=MARRON_OSCURO, stroke_width=2)
            lbl = Text(palabra, font=FUENTE, font_size=18, color=TINTA_NEGRA)
            lbl.move_to(circ.get_center())
            nodos.add(VGroup(circ, lbl).move_to(pos))

        lineas_attn = VGroup()
        for i in range(len(nodos)):
            for j in range(i+1, len(nodos)):
                ln = Line(nodos[i].get_center(), nodos[j].get_center(),
                          stroke_width=1.5, stroke_color=MARRON_OSCURO, stroke_opacity=0.4)
                lineas_attn.add(ln)

        red_atencion = VGroup(lineas_attn, nodos).next_to(explicacion, DOWN, buff=0.5)

        self.play(Write(label_atencion), Write(explicacion))
        self.next_slide()

        self.play(FadeOut(explicacion))
        self.play(red_atencion.animate.move_to(DOWN * 0.5))

        nota_sin_orden = Text("Todos conectados con todos · sin orden implícito",
                              font=FUENTE, font_size=20, color=MARRON_OSCURO).next_to(red_atencion, DOWN, buff=0.7)
        
        self.play(FadeIn(nota_sin_orden, shift=UP*0.2))
        self.next_slide()

        self.play(FadeOut(label_atencion), FadeOut(red_atencion), FadeOut(nota_sin_orden))

        # --- ESCENA 3: LA SOLUCIÓN ---
        label_solucion = Text("La solución: inyectar la posición en cada token", 
                              font=FUENTE, font_size=26, weight=BOLD, color=VERDE_OLIVA)
        label_solucion.move_to(UP * 1.5)

        def crear_vector_mini(valores, color_principal, ancho=0.65, alto=0.5):
            vector = VGroup()
            for val in valores:
                es_puntos = (val == "·")
                rect = RoundedRectangle(corner_radius=0.05, width=ancho, height=alto,
                                        fill_color=color_principal if not es_puntos else FONDO_CAJA,
                                        fill_opacity=0.85 if not es_puntos else 0)
                rect.set_stroke(MARRON_OSCURO if not es_puntos else FONDO_CAJA, 1.5)
                txt = Text(val, font="Monospace", font_size=15, color=TINTA_NEGRA)
                txt.move_to(rect.get_center())
                vector.add(VGroup(rect, txt))
            return vector.arrange(RIGHT, buff=0.08)

        token_word = Text("perro", font=FUENTE, font_size=30, weight=BOLD, color=TINTA_NEGRA)
        pos_tag = Text("(posición 1)", font=FUENTE, font_size=20, color=PAPEL_TAN)
        token_label_group = VGroup(token_word, pos_tag).arrange(DOWN, buff=0.15)

        vec_tok = crear_vector_mini(["0.12", "-0.30", "0.55", "·"], PAPEL_TAN)
        vec_pos_emb = crear_vector_mini(["0.84", "-0.54", "0.14", "·"], NARANJA_TERRACOTA)
        vec_comb = crear_vector_mini(["0.96", "-0.84", "0.69", "·"], CAJA_INFERIOR)

        lbl_tok = Text("Token Embedding\n(¿qué palabra?)", font=FUENTE, font_size=17, weight=BOLD, color=MARRON_OSCURO)
        lbl_pos_emb = Text("Position Embedding\n(¿en qué lugar?)", font=FUENTE, font_size=17, weight=BOLD, color=NARANJA_TERRACOTA)
        lbl_comb = Text("Vector final =\nsignificado + posición", font=FUENTE, font_size=18, weight=BOLD, color=TINTA_NEGRA)
        
        columna_vectores = VGroup(vec_tok, vec_pos_emb).arrange(DOWN, buff=0.8).move_to(DOWN * 0.2)
        mas = Text("+", font=FUENTE, font_size=40, weight=BOLD, color=TINTA_NEGRA).move_to(columna_vectores.get_center())
        
        lbl_tok.next_to(vec_tok, RIGHT, buff=0.5)
        lbl_pos_emb.next_to(vec_pos_emb, RIGHT, buff=0.5)
        lbl_pos_emb.align_to(lbl_tok, LEFT) 

        token_label_group.next_to(columna_vectores, LEFT, buff=1.5)
        
        flecha_tok = Arrow(token_label_group.get_right(), vec_tok.get_left(), color=MARRON_OSCURO, buff=0.2, max_tip_length_to_length_ratio=0.15)
        flecha_pos_arr = Arrow(token_label_group.get_right(), vec_pos_emb.get_left(), color=NARANJA_TERRACOTA, buff=0.2, max_tip_length_to_length_ratio=0.15)

        self.play(Write(label_solucion))
        self.play(FadeIn(token_label_group))
        self.play(GrowArrow(flecha_tok), FadeIn(vec_tok), Write(lbl_tok))
        self.play(GrowArrow(flecha_pos_arr), FadeIn(vec_pos_emb), Write(lbl_pos_emb))
        self.play(Write(mas))
        self.next_slide()

        sep_line = Line(vec_pos_emb.get_left() + LEFT*0.2, vec_pos_emb.get_right() + RIGHT*0.2, color=MARRON_OSCURO, stroke_width=2)
        sep_line.next_to(vec_pos_emb, DOWN, buff=0.3)
        vec_comb.next_to(sep_line, DOWN, buff=0.3)
        lbl_comb.next_to(vec_comb, RIGHT, buff=0.5).align_to(lbl_tok, LEFT)

        self.play(Create(sep_line))
        self.play(
            ReplacementTransform(vec_tok.copy(), vec_comb),
            ReplacementTransform(vec_pos_emb.copy(), vec_comb),
            Write(lbl_comb)
        )
        self.play(Indicate(vec_comb, color=NARANJA_TERRACOTA, scale_factor=1.05))
        self.next_slide()

        self.play(*[FadeOut(m) for m in [
            label_solucion, token_label_group, flecha_tok, flecha_pos_arr,
            vec_tok, lbl_tok, vec_pos_emb, lbl_pos_emb, mas, sep_line, vec_comb, lbl_comb
        ]])

        # --- ESCENA 4: CONTEXT WINDOW ---
        titulo_tabla = Text("Una fila por posición → la tabla es el Context Window",
                            font=FUENTE, font_size=24, weight=BOLD, color=TINTA_NEGRA).move_to(UP * 1.5)

        def crear_fila_pos(num, color, opacidad=0.55):
            etiq = Text(f"Pos {num}:", font=FUENTE, font_size=18, color=TINTA_NEGRA)
            celdas = VGroup(*[
                RoundedRectangle(corner_radius=0.03, width=0.42, height=0.32,
                                 fill_color=color, fill_opacity=opacidad)
                .set_stroke(MARRON_OSCURO, 1)
                for _ in range(8)
            ]).arrange(RIGHT, buff=0.05)
            puntos = Text("···", font_size=14, color=MARRON_OSCURO).next_to(celdas, RIGHT, buff=0.1)
            return VGroup(etiq, celdas, puntos).arrange(RIGHT, buff=0.3)

        f0  = crear_fila_pos(0,    PAPEL_TAN,       0.7)
        f1  = crear_fila_pos(1,    NARANJA_TERRACOTA, 0.5)
        f2  = crear_fila_pos(2,    PAPEL_TAN,       0.5)
        fv  = Text("·  ·  ·", font_size=28, color=MARRON_OSCURO).rotate(0)
        fn  = crear_fila_pos(1023, CAJA_INFERIOR,   0.5)

        tabla = VGroup(f0, f1, f2, fv, fn).arrange(DOWN, buff=0.28)
        llave = Brace(tabla, direction=LEFT, color=MARRON_OSCURO)
        lbl_llave = Text("1 024 posiciones\n(block_size)", font=FUENTE, font_size=18, color=MARRON_OSCURO, line_spacing=1.2)
        lbl_llave.next_to(llave, LEFT, buff=0.2)

        grupo_tabla_completa = VGroup(lbl_llave, llave, tabla)
        
        nota_limite = Text(
            "¡No puede haber una fila 1 024!\n→ El modelo no puede\nprocesar textos más largos.\nEso es el Context Window.",
            font=FUENTE, font_size=21, weight=BOLD, color=NARANJA_TERRACOTA, line_spacing=1.3
        )

        slide4_layout = VGroup(grupo_tabla_completa, nota_limite).arrange(RIGHT, buff=0.8).next_to(titulo_tabla, DOWN, buff=0.6)

        caja_fn = SurroundingRectangle(fn, color=NARANJA_TERRACOTA, stroke_width=4, buff=0.07, corner_radius=0.06)

        self.play(Write(titulo_tabla))
        self.play(FadeIn(tabla, shift=UP*0.3))
        self.play(GrowFromCenter(llave), Write(lbl_llave))
        self.next_slide()

        self.play(Create(caja_fn))
        self.play(Write(nota_limite))
        self.play(Indicate(caja_fn, color=NARANJA_TERRACOTA, scale_factor=1.05))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_layer_normalization(self):
        """Diapositiva: Slide Layer Normalization."""

        def crear_cajita(texto, bg_color, borde_color=MARRON_OSCURO, w=2.6, h=0.7, tam_fuente=20):
            caja = RoundedRectangle(corner_radius=0.1, width=w, height=h, 
                                    fill_color=bg_color, fill_opacity=1, 
                                    stroke_color=borde_color, stroke_width=2)
            lbl = Text(texto, font_size=tam_fuente, color=TINTA_NEGRA).move_to(caja.get_center())
            return VGroup(caja, lbl)

        def crear_vector_visual(numeros, bg_color, borde_color=MARRON_OSCURO):
            bloques = VGroup(*[
                crear_cajita(num, bg_color, borde_color, w=1.6, h=0.7, tam_fuente=18) 
                for num in numeros
            ]).arrange(RIGHT, buff=0.1)
            
            return bloques 

        titulo_p1 = Text("Layer ", font_size=42, weight=BOLD, color=TINTA_NEGRA)
        titulo_p2 = Text("Normalization", font_size=42, weight=BOLD, color=NARANJA_TERRACOTA) 
        titulo_completo = VGroup(titulo_p1, titulo_p2).arrange(RIGHT, buff=0.1)
        linea = Line(LEFT * 4, RIGHT * 4, color=MARRON_OSCURO).next_to(titulo_completo, DOWN)
        grupo_titulo = VGroup(titulo_completo, linea).to_edge(UP)
        
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo_completo, linea, fondo=llanuras_fondo)
        self.next_slide()

        nota_caos = Text("Sin normalización: Los valores explotan o desaparecen", font_size=24, color=NARANJA_TERRACOTA).move_to(UP * 1.5)

        valores_inestables = ["8459.1", "-7302.4", "0.00001", "5120.9", "-9999.9", "..."]
        vec_inestable = crear_vector_visual(valores_inestables, bg_color=SALMON_CLARO, borde_color=NARANJA_TERRACOTA)
        vec_inestable.next_to(nota_caos, DOWN, buff=0.8).set_x(0) 
        
        self.play(Write(nota_caos))
        self.play(FadeIn(vec_inestable, shift=UP))
        self.play(Indicate(vec_inestable, color=NARANJA_TERRACOTA, scale_factor=1.1))
        self.next_slide()

        nota_estable = Text("Con LayerNorm: Se fuerza una Media=0 y Varianza=1", font_size=24, color=MARRON_OSCURO).move_to(UP * 1.5)
        
        valores_estables = ["1.34", "-1.15", "0.00", "0.89", "-1.52", "..."]
        vec_estable = crear_vector_visual(valores_estables, bg_color=CREMA_CALIDA, borde_color=MARRON_OSCURO)
        vec_estable.next_to(nota_estable, DOWN, buff=0.8).set_x(0)
        
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
        paso_2 = Text("2. Antes de MLP", font_size=24, color=MARRON_OSCURO)
        
        textos_izq = VGroup(nota_final, paso_1, paso_2).arrange(DOWN, aligned_edge=LEFT, buff=0.4).to_edge(LEFT, buff=1).shift(UP * 0.5)

        b_in = crear_cajita("Input", CREMA_CALIDA)       
        b_ln1 = crear_cajita("LayerNorm 1", BEIGE_MEDIO)  
        b_attn = crear_cajita("Attention", SALMON_ATENCION, borde_color=LADRILLO_VIVO)
        b_ln2 = crear_cajita("LayerNorm 2", BEIGE_MEDIO) 
        b_mlp = crear_cajita("MLP", ARENA_DORADA)    
        b_out = crear_cajita("Output", CREMA_CALIDA)       

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
            b_ln1[0].animate.set_fill(LADRILLO_VIVO) 
        )
        self.next_slide()

        resalto_2 = SurroundingRectangle(b_ln2, color=NARANJA_TERRACOTA, stroke_width=4, buff=0.05)
        self.play(
            FadeIn(paso_2, shift=RIGHT), 
            Create(resalto_2), 
            b_ln2[0].animate.set_fill(LADRILLO_VIVO) 
        )
        self.next_slide()

        self.limpiar_pantalla()

    def slide_mha_acto1_intuicion(self):

        """Diapositiva: Slide Mha Acto1 Intuicion."""

        titulo, linea = self.crear_titulo(
            "Multi-Head Self-Attention",
            palabra_clave="Attention",
            color_clave=NARANJA_TERRACOTA
        )

        subtitulo = Text(
            "La Intuición (Q, K, V)",
            font=FUENTE, font_size=24, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.5)

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)
        self.play(FadeIn(subtitulo, shift=DOWN))
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

        idx_query = 6
        idx_key_fuerte = 4
        idx_key_debil = 1

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
        """Diapositiva: Slide Mha Acto2 Formula."""

        titulo, linea = self.crear_titulo("Multi-Head Self-Attention", palabra_clave="Attention", color_clave=NARANJA_TERRACOTA)
        subtitulo = Text("La Ecuación de Atención", font=FUENTE, font_size=24, color=MARRON_OSCURO).next_to(linea, DOWN)
        
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)
        self.play(FadeIn(subtitulo, shift=DOWN))
        self.next_slide()
        
        formula = MathTex(
            r"\text{Attention}(Q, K, V) = ",
            r"\text{softmax}",
            r"\left( \frac{",
            r"Q K^T",
            r"}{",
            r"\sqrt{d_k}",
            r"} \right) ",
            r"V",
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

        """Diapositiva: Slide Mha Acto3 Calculo."""

        titulo, linea = self.crear_titulo("Flujo Cálculo Self-Attention", palabra_clave="Flujo", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)
        self.next_slide()

        def crear_caja(texto, bg_color, ancho=1.6, alto=1.2, font_size=24, es_vertical=False):
            w = ancho if not es_vertical else alto
            h = alto if not es_vertical else ancho
            
            caja = RoundedRectangle(
                corner_radius=0.15, width=w, height=h,
                fill_color=bg_color, fill_opacity=1,
                stroke_color=TINTA_NEGRA, stroke_width=2
            )

            if '\\' in texto or '^' in texto:
                lbl = MathTex(texto, color=TINTA_NEGRA, font_size=font_size+8)
            else:
                lbl = Text(texto, font_size=font_size, color=TINTA_NEGRA)
            
            lbl.move_to(caja.get_center())
            return VGroup(caja, lbl)

        col_1_x = -6.0  
        col_2_x = -4.2 
        col_3_x = -2.4  
        col_4_x = -0.4 
        col_5_x = 1.4   
        col_6_x = 3.2   
        col_7_x = 5.2  
        col_8_x = 6.5  

        fila_q_y = 1.1
        fila_k_y = 0.0
        fila_v_y = -1.1

        X = MathTex("X", color=TINTA_NEGRA, font_size=40).move_to([col_1_x, 0, 0])
        nodo_x = Dot(radius=0.06, color=TINTA_NEGRA).next_to(X, RIGHT, buff=0.15)
        linea_x = Line(X.get_right(), nodo_x.get_center(), color=TINTA_NEGRA, stroke_width=3)

        color_w = LAVANDA 
        W_q = crear_caja(r"W^{(q)}", color_w, ancho=1.4, alto=0.8).move_to([col_2_x, fila_q_y, 0])
        W_k = crear_caja(r"W^{(k)}", color_w, ancho=1.4, alto=0.8).move_to([col_2_x, fila_k_y, 0])
        W_v = crear_caja(r"W^{(v)}", color_w, ancho=1.4, alto=0.8).move_to([col_2_x, fila_v_y, 0])

        def ruta_angulada_l2r(start, end_mobj):
            p_mid = [start[0], end_mobj.get_y(), 0]
            l1 = Line(start, p_mid, color=TINTA_NEGRA, stroke_width=3)
            l2 = Arrow(p_mid, end_mobj.get_left(), buff=0.05, color=TINTA_NEGRA, stroke_width=3)
            return VGroup(l1, l2)

        ruta_q = ruta_angulada_l2r(nodo_x.get_center(), W_q)
        ruta_k = Arrow(nodo_x.get_center(), W_k.get_left(), buff=0.05, color=TINTA_NEGRA, stroke_width=3)
        ruta_v = ruta_angulada_l2r(nodo_x.get_center(), W_v)


        Q = MathTex("Q", color=TINTA_NEGRA, font_size=36).move_to([col_3_x, fila_q_y, 0])
        K = MathTex("K", color=TINTA_NEGRA, font_size=36).move_to([col_3_x, fila_k_y, 0])
        V = MathTex("V", color=TINTA_NEGRA, font_size=36).move_to([col_3_x, fila_v_y, 0])

        a_wq = Arrow(W_q.get_right(), Q.get_left(), buff=0.1, color=TINTA_NEGRA, stroke_width=3)
        a_wk = Arrow(W_k.get_right(), K.get_left(), buff=0.1, color=TINTA_NEGRA, stroke_width=3)
        a_wv = Arrow(W_v.get_right(), V.get_left(), buff=0.1, color=TINTA_NEGRA, stroke_width=3)

        y_qk_branch = (fila_q_y + fila_k_y) / 2
        matmul_1 = crear_caja("mat mul", NARANJA_CLARO, ancho=1.2, alto=1.8).move_to([col_4_x, y_qk_branch, 0])
  
        a_q_mm = Arrow(Q.get_right(), [matmul_1.get_left()[0], Q.get_y(), 0], buff=0.05, color=TINTA_NEGRA, stroke_width=3)
        a_k_mm = Arrow(K.get_right(), [matmul_1.get_left()[0], K.get_y(), 0], buff=0.05, color=TINTA_NEGRA, stroke_width=3)

        scale_box = crear_caja("scale", AMARILLO_PALIDO, ancho=1.2, alto=1.8).move_to([col_5_x, y_qk_branch, 0])
        a_mm_sc = Arrow(matmul_1.get_right(), scale_box.get_left(), buff=0.05, color=TINTA_NEGRA, stroke_width=3)

        softmax_box = crear_caja("softmax", MENTA_PALIDA, ancho=1.2, alto=1.8).move_to([col_6_x, y_qk_branch, 0])
        a_sc_sm = Arrow(scale_box.get_right(), softmax_box.get_left(), buff=0.05, color=TINTA_NEGRA, stroke_width=3)

        matmul_2 = crear_caja("mat mul", NARANJA_CLARO, ancho=1.8, alto=2.8).move_to([col_7_x, 0, 0])
        
        a_sm_mm2 = Arrow(softmax_box.get_right(), [matmul_2.get_left()[0], softmax_box.get_y(), 0], buff=0.05, color=TINTA_NEGRA, stroke_width=3)
        a_v_mm2 = Arrow(V.get_right(), [matmul_2.get_left()[0], V.get_y(), 0], buff=0.05, color=TINTA_NEGRA, stroke_width=3)

        Y = MathTex("Y", color=TINTA_NEGRA, font_size=40).move_to([col_8_x, 0, 0])
        a_mm2_y = Arrow(matmul_2.get_right(), Y.get_left(), buff=0.05, color=TINTA_NEGRA, stroke_width=3)

        zona_texto = DOWN * 2.8
        
        panel_fondo = RoundedRectangle(
            corner_radius=0.2, width=12.5, height=0.8, 
            fill_color=PERGAMINO_CLARO, fill_opacity=1, 
            stroke_color=TINTA_NEGRA, stroke_width=2
        ).move_to(zona_texto)

        txt_1 = Text("1. Proyección: La entrada X se divide en matrices Query, Key y Value.", font_size=22, color=TINTA_NEGRA).move_to(panel_fondo.get_center())
        txt_2 = Text("2. Similitud: Se multiplican Q y K para ver qué tokens importan más.", font_size=22, color=TINTA_NEGRA).move_to(panel_fondo.get_center())
        txt_3 = Text("3. Estabilización: Se escala y aplica Softmax para obtener porcentajes (0 a 1).", font_size=22, color=TINTA_NEGRA).move_to(panel_fondo.get_center())
        txt_4 = Text("4. Contexto Final: Se filtran los Values (V) según esos porcentajes.", font_size=22, color=TINTA_NEGRA).move_to(panel_fondo.get_center())

        self.play(FadeIn(panel_fondo, shift=UP), Write(txt_1)) 
        self.play(Write(X), Create(linea_x), FadeIn(nodo_x))
        
        self.play(AnimationGroup(Create(ruta_q), Create(ruta_k), Create(ruta_v), lag_ratio=0.2))
        self.play(AnimationGroup(GrowFromCenter(W_q), GrowFromCenter(W_k), GrowFromCenter(W_v), lag_ratio=0.1))
        
        self.play(
            AnimationGroup(Create(a_wq), Create(a_wk), Create(a_wv), lag_ratio=0.1),
            AnimationGroup(FadeIn(Q, shift=RIGHT), FadeIn(K, shift=RIGHT), FadeIn(V, shift=RIGHT), lag_ratio=0.1)
        )
        self.next_slide()

        self.play(FadeTransform(txt_1, txt_2))
        self.play(Create(a_q_mm), Create(a_k_mm))
        self.play(GrowFromCenter(matmul_1))
        self.next_slide()

        self.play(FadeTransform(txt_2, txt_3))
        self.play(Create(a_mm_sc))
        self.play(GrowFromCenter(scale_box))
        self.play(Create(a_sc_sm))
        self.play(GrowFromCenter(softmax_box))
        self.next_slide()

        self.play(FadeTransform(txt_3, txt_4))
        self.play(Create(a_sm_mm2), Create(a_v_mm2)) 
        self.play(GrowFromCenter(matmul_2))
        self.next_slide()

        self.play(Create(a_mm2_y))
        self.play(FadeIn(Y, shift=RIGHT), Flash(Y, color=NARANJA_TERRACOTA, line_length=0.3))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_mha_acto4_multihead(self):
        """Diapositiva: Slide Mha Acto4 Multihead."""

        titulo, linea = self.crear_titulo("Multi-Head Self-Attention", palabra_clave="Attention", color_clave=NARANJA_TERRACOTA)
        subtitulo = Text("¿Por qué 'Multi-Head'?", font=FUENTE, font_size=24, color=MARRON_OSCURO).next_to(linea, DOWN)
        
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)
        self.play(FadeIn(subtitulo, shift=DOWN))
        self.next_slide()

        vector_completo = Rectangle(width=10, height=0.8, fill_color=FONDO_CAJA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2)
        txt_dim = Text("Vector de Embedding de 768 dimensiones", font=FUENTE, font_size=20, color=TINTA_NEGRA).move_to(vector_completo)
        
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
        """Diapositiva: Slide Causal Masking."""

        titulo, linea = self.crear_titulo("Causal Masking", palabra_clave="Masking", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)
        
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
        
        etiquetas_filas = VGroup()
        for i, t in enumerate(tokens):
            etiqueta = Text(t, font=FUENTE, font_size=14, color=MARRON_OSCURO)
            etiqueta.next_to(mat_scores[i][0], LEFT, buff=0.25)
            etiquetas_filas.add(etiqueta)

        etiquetas_cols = VGroup()
        for j, t in enumerate(tokens):
            etiqueta = Text(t, font=FUENTE, font_size=14, color=MARRON_OSCURO).rotate(PI/4)
            etiqueta.next_to(mat_scores[0][j], UP, buff=0.15)
            etiquetas_cols.add(etiqueta)

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

    def slide_arquitectura_neurona(self):
        """Diapositiva: Slide Arquitectura Neurona."""

        titulo, linea = self.crear_titulo("La Capa MLP: Una Función Matemática", palabra_clave="Función", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        eq_top = MathTex(r"f : \mathbb{R}^{768} \rightarrow \mathbb{R}^{768}", color=TINTA_NEGRA, font_size=42).next_to(linea, DOWN, buff=0.3)
        self.play(Write(eq_top))

        r = 0.15
        c_linea = GRAY_B
        
        capa_in = VGroup(*[Circle(radius=r, fill_color=PAPEL_TAN, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(5)]).arrange(DOWN, buff=0.2)
        
        hid_1 = VGroup(*[Circle(radius=r, fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)
        hid_2 = VGroup(*[Circle(radius=r, fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)
        hid_3 = VGroup(*[Circle(radius=r, fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2) for _ in range(7)]).arrange(DOWN, buff=0.1)

        capas_profundas = VGroup(hid_1, hid_2, hid_3).arrange(RIGHT, buff=0.8)
        capa_out = VGroup(*[Circle(radius=r, fill_color=MARRON_OSCURO, fill_opacity=1, stroke_color=TINTA_NEGRA, stroke_width=2) for _ in range(5)]).arrange(DOWN, buff=0.2)

        red = VGroup(capa_in, capas_profundas, capa_out).arrange(RIGHT, buff=1.8).move_to(DOWN * 0.5)

        conexiones_in_h1 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in capa_in for n2 in hid_1])
        conexiones_h1_h2 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in hid_1 for n2 in hid_2])
        conexiones_h2_h3 = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in hid_2 for n2 in hid_3])
        conexiones_h3_out = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=1, color=c_linea, stroke_opacity=0.3) for n1 in hid_3 for n2 in capa_out])

        brace_in = Brace(capa_in, direction=LEFT, color=MARRON_OSCURO)
        lbl_in = MathTex(r"x \in \mathbb{R}^{768}", color=TINTA_NEGRA, font_size=28).next_to(brace_in, LEFT)
        
        brace_hid = Brace(capas_profundas, direction=UP, color=NARANJA_TERRACOTA)
        lbl_hid = MathTex(r"3072", color=NARANJA_TERRACOTA, font_size=32).next_to(brace_hid, UP)
        
        brace_out = Brace(capa_out, direction=RIGHT, color=MARRON_OSCURO)
        lbl_out = MathTex(r"f(x) \in \mathbb{R}^{768}", color=TINTA_NEGRA, font_size=28).next_to(brace_out, RIGHT)

        flecha_1 = MathTex(r"\xrightarrow{\quad \phi_1 \quad}", color=NARANJA_TERRACOTA).next_to(capas_profundas[0], DOWN, buff=0.5)
        flecha_2 = MathTex(r"\xrightarrow{\quad \phi_2 \quad}", color=NARANJA_TERRACOTA).next_to(capas_profundas[1], DOWN, buff=0.5)
        flecha_3 = MathTex(r"\xrightarrow{\quad \phi_3 \quad}", color=NARANJA_TERRACOTA).next_to(capas_profundas[2], DOWN, buff=0.5)
        eq_composicion = MathTex(r"f(x) = \phi_3(\phi_2(\phi_1(x)))", color=TINTA_NEGRA, font_size=32).next_to(VGroup(flecha_1, flecha_3), DOWN, buff=0.3)


        self.play(FadeIn(capa_in), GrowFromCenter(brace_in), Write(lbl_in))
        self.next_slide() 

        self.play(LaggedStartMap(Create, conexiones_in_h1, lag_ratio=0.01), run_time=0.8)
        self.play(FadeIn(hid_1, shift=LEFT*0.2), Write(flecha_1))
        
        self.play(LaggedStartMap(Create, conexiones_h1_h2, lag_ratio=0.01), run_time=0.6)
        self.play(FadeIn(hid_2, shift=LEFT*0.2), Write(flecha_2))
        
        self.play(LaggedStartMap(Create, conexiones_h2_h3, lag_ratio=0.01), run_time=0.6)
        self.play(FadeIn(hid_3, shift=LEFT*0.2), Write(flecha_3))
        
        self.play(GrowFromCenter(brace_hid), Write(lbl_hid), Write(eq_composicion))
        self.next_slide() 

        self.play(LaggedStartMap(Create, conexiones_h3_out, lag_ratio=0.01), run_time=0.8)
        self.play(FadeIn(capa_out, shift=LEFT*0.2), GrowFromCenter(brace_out), Write(lbl_out))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_zoom_neurona(self):
        """Diapositiva: Slide Zoom Neurona."""

        titulo, linea = self.crear_titulo("La Capa MLP: Dentro de una Neurona", palabra_clave="Neurona", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        r_mini = 0.08
        columna_in = VGroup(*[Circle(radius=r_mini, fill_color=PAPEL_TAN, fill_opacity=1, stroke_color=MARRON_OSCURO) for _ in range(3)]).arrange(DOWN, buff=0.1)
        columna_hid = VGroup(*[Circle(radius=r_mini, fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_color=MARRON_OSCURO) for _ in range(5)]).arrange(DOWN, buff=0.1)
        
        mini_red = VGroup(columna_in, columna_hid).arrange(RIGHT, buff=0.6)

        mini_red.to_edge(LEFT, buff=0.8) 
        
        conexiones_mini = VGroup(*[Line(n1.get_right(), n2.get_left(), stroke_width=0.5, color=GRAY_B) for n1 in columna_in for n2 in columna_hid])
        
        self.play(FadeIn(mini_red), FadeIn(conexiones_mini))

        neurona_objetivo = columna_hid[2]
        resaltador = Circle(radius=r_mini * 2, color=MARRON_OSCURO, stroke_width=3).move_to(neurona_objetivo)
        self.play(Create(resaltador))
        self.next_slide()

        posicion_neurona = RIGHT * 2.2 + UP * 0.4
        neurona_gigante = Circle(radius=1.6, fill_color=NARANJA_TERRACOTA, fill_opacity=0.1, stroke_color=NARANJA_TERRACOTA, stroke_width=4).move_to(posicion_neurona)
        
        sumatoria = MathTex(r"\Sigma", color=TINTA_NEGRA, font_size=55).move_to(neurona_gigante).shift(LEFT * 0.6)
        separador = Line(neurona_gigante.get_top() + DOWN*0.1, neurona_gigante.get_bottom() + UP*0.1, color=NARANJA_TERRACOTA)
        
        ejes_act = Axes(x_range=[-2, 2], y_range=[-0.5, 1.5], x_length=1.0, y_length=1.0, axis_config={"color": MARRON_OSCURO, "include_ticks": False}).move_to(neurona_gigante).shift(RIGHT * 0.6)
        curva_act = ejes_act.plot(lambda x: x if x > 0 else 0, color=MARRON_OSCURO) 
        grupo_activacion = VGroup(ejes_act, curva_act)

        lineas_zoom = VGroup(
            DashedLine(resaltador.get_top(), neurona_gigante.get_top(), color=GRAY_B),
            DashedLine(resaltador.get_bottom(), neurona_gigante.get_bottom(), color=GRAY_B)
        )
        
        self.play(Create(lineas_zoom), FadeIn(neurona_gigante))
        self.play(Write(sumatoria), Create(separador), FadeIn(grupo_activacion))
        self.next_slide()

        self.play(FadeOut(lineas_zoom))

        entradas_text = VGroup(
            MathTex(r"x_1", color=TINTA_NEGRA),
            MathTex(r"x_2", color=TINTA_NEGRA),
            MathTex(r"\vdots", color=TINTA_NEGRA), 
            MathTex(r"x_n", color=TINTA_NEGRA)
        ).arrange(DOWN, buff=0.4)
        
        entradas_text.move_to(LEFT * 2.0 + UP * 0.4)

        flechas_in = VGroup(*[Arrow(e.get_right(), neurona_gigante.get_left(), buff=0.15, color=MARRON_OSCURO, max_tip_length_to_length_ratio=0.08) for e in entradas_text if e.tex_string != r"\vdots"])
        
        pesos_text = VGroup(
            MathTex(r"w_1", color=NARANJA_TERRACOTA).move_to(flechas_in[0].get_center() + UP * 0.35).scale(0.8),
            MathTex(r"w_2", color=NARANJA_TERRACOTA).move_to(flechas_in[1].get_center() + DOWN * 0.2).scale(0.8),
            MathTex(r"w_n", color=NARANJA_TERRACOTA).move_to(flechas_in[2].get_center() + DOWN * 0.35).scale(0.8)
        )

        bias_text = MathTex(r"b", color=MARRON_OSCURO).next_to(neurona_gigante, UP, buff=0.4)
        flecha_bias = Arrow(bias_text.get_bottom(), neurona_gigante.get_top(), buff=0.1, color=MARRON_OSCURO)

        flecha_out = Arrow(neurona_gigante.get_right(), neurona_gigante.get_right() + RIGHT * 1.2, color=MARRON_OSCURO)
        salida_text = MathTex(r"\phi(x)", color=TINTA_NEGRA).next_to(flecha_out, RIGHT)

        self.play(LaggedStart(
            *[AnimationGroup(FadeIn(e, shift=RIGHT), GrowArrow(f)) for e, f in zip([entradas_text[0], entradas_text[1], entradas_text[3]], flechas_in)],
            FadeIn(entradas_text[2]), 
            lag_ratio=0.2
        ))
        self.play(FadeIn(pesos_text, shift=UP))
        self.play(FadeIn(bias_text, shift=DOWN), GrowArrow(flecha_bias))
        self.play(GrowArrow(flecha_out), Write(salida_text))
        self.next_slide()

        eq_final = MathTex(r"\phi(x)", r"=", r"\sigma", r"(", r"\sum x_i \cdot w_i", r"+ b", r")", color=TINTA_NEGRA, font_size=46).move_to(DOWN * 1.8)
        
        eq_final[2].set_color(MARRON_OSCURO) 
        eq_final[4].set_color(NARANJA_TERRACOTA) 
        eq_final[5].set_color(MARRON_OSCURO) 

        nota_w = Text("Suma ponderada\n(Lo que la red aprende)", font=FUENTE, font_size=16, color=NARANJA_TERRACOTA).next_to(eq_final[4], DOWN, buff=0.6)
        flecha_w = Arrow(nota_w.get_top(), eq_final[4].get_bottom(), buff=0.1, color=NARANJA_TERRACOTA)
        
        nota_act = Text("Activación no lineal\n(La flexibilidad)", font=FUENTE, font_size=16, color=MARRON_OSCURO).next_to(eq_final[2], DOWN, buff=0.6).shift(LEFT * 0.5)
        flecha_act = Arrow(nota_act.get_top(), eq_final[2].get_bottom(), buff=0.1, color=MARRON_OSCURO)

        self.play(Write(eq_final))
        self.play(FadeIn(nota_w), GrowArrow(flecha_w))
        self.play(FadeIn(nota_act), GrowArrow(flecha_act))
        self.next_slide() 

        self.limpiar_pantalla()

    def slide_activacion(self):
        """Diapositiva: Slide Activacion."""

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()
        
        titulo, linea = self.crear_titulo(
            "Función de Activación: GELU", 
            palabra_clave="GELU", 
            color_clave=NARANJA_TERRACOTA
        )
        
        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        formula_principal = MathTex(
            r"\text{GELU}(x) = x \cdot \Phi(x)",
            color=TINTA_NEGRA, font_size=42
        )
        formula_aprox = MathTex(
            r"\approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3)\right)\right)",
            color=MARRON_OSCURO, font_size=26
        )
        
        grupo_formulas = VGroup(formula_principal, formula_aprox).arrange(DOWN, buff=0.4)
        
        texto_contexto = Text(
            "Apagado de las neuronas", 
            font=FUENTE, font_size=20, color=NARANJA_TERRACOTA, line_spacing=1.2
        )
        
        grupo_izq = VGroup(grupo_formulas, texto_contexto).arrange(DOWN, buff=0.8)
        grupo_izq.to_edge(LEFT, buff=1.0).shift(UP * 1.0)

        self.play(Write(formula_principal))
        self.play(FadeIn(formula_aprox, shift=UP))
        self.play(FadeIn(texto_contexto, shift=RIGHT))

        ejes = Axes(
            x_range=[-3, 3, 1], y_range=[-1, 3, 1],
            x_length=6, y_length=4.5,
            axis_config={"color": MARRON_OSCURO, "include_numbers": True, "font_size": 16}
        ).to_edge(RIGHT, buff=0.8).shift(DOWN * 0.5)

        curva_relu = ejes.plot(lambda x: np.maximum(0, x), color=GRAY_C, stroke_width=4)
        lbl_relu = Text("ReLU", font=FUENTE, font_size=26, color=GRAY_C).next_to(ejes.c2p(2, 2), UL, buff=0.2)

        curva_gelu = ejes.plot(
            lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
            color=NARANJA_TERRACOTA, stroke_width=5
        )
        lbl_gelu = Text("GELU", font=FUENTE, font_size=26, color=NARANJA_TERRACOTA).next_to(ejes.c2p(2.5, 2.5), DR, buff=0.1)
        
        punto_minimo = ejes.c2p(-0.75, -0.17)
        lbl_suavizado = Text("Apagado suave", font=FUENTE, font_size=16, color=MARRON_QUIJOTE).next_to(punto_minimo, DOWN, buff=0.5).shift(RIGHT * 1)
        flecha_suav = Arrow(lbl_suavizado.get_left(), punto_minimo, buff=0.1, color=MARRON_QUIJOTE, tip_length=0.15)

        self.play(Create(ejes), Write(lbl_relu))
        self.play(Create(curva_relu))
        self.next_slide()

        nodo = Circle(radius=0.7, fill_color=PAPEL_CREMA, fill_opacity=1, stroke_color=GRAY_C, stroke_width=4)
        lbl_nodo = Text("ReLU", font=FUENTE, font_size=20, color=GRAY_C).move_to(nodo)
        
        flecha_in = Arrow(ORIGIN, RIGHT * 1.5, buff=0.1, color=MARRON_OSCURO)
        val_in = MathTex("x = -1", font_size=28, color=TINTA_NEGRA).next_to(flecha_in, UP, buff=0.1)
        grupo_in = VGroup(val_in, flecha_in)
        
        flecha_out = Arrow(ORIGIN, RIGHT * 1.5, buff=0.1, color=GRAY_C)
        val_out = MathTex("0", font_size=32, color=GRAY_C).next_to(flecha_out, UP, buff=0.1)
        grupo_out = VGroup(val_out, flecha_out)

        diagrama_neurona = VGroup(grupo_in, nodo, grupo_out).arrange(RIGHT, buff=0.1)
        diagrama_neurona.to_corner(DL, buff=1.0).shift(UP * 0.5)
        
        lbl_nodo.move_to(nodo)
        
        cruz_muerte = Cross(val_out, stroke_color=RED, stroke_width=5, scale_factor=0.6)

        self.play(FadeIn(nodo), Write(lbl_nodo))
        self.play(GrowArrow(flecha_in), FadeIn(val_in, shift=RIGHT))
        
        val_in_anim = val_in.copy()
        self.play(val_in_anim.animate.move_to(nodo).scale(0.5).set_opacity(0), run_time=0.8)
        
        self.play(GrowArrow(flecha_out), FadeIn(val_out, shift=RIGHT))
        self.play(Create(cruz_muerte))
        self.next_slide()

        lbl_nodo_gelu = Text("GELU", font=FUENTE, font_size=20, color=NARANJA_TERRACOTA).move_to(nodo)
        
        self.play(
            ReplacementTransform(curva_relu, curva_gelu),
            ReplacementTransform(lbl_relu, lbl_gelu),
            nodo.animate.set_stroke(color=NARANJA_TERRACOTA),
            ReplacementTransform(lbl_nodo, lbl_nodo_gelu),
            flecha_out.animate.set_color(NARANJA_TERRACOTA),
            FadeOut(cruz_muerte)
        )
        
        val_out_gelu = MathTex("-0.15", font_size=32, color=NARANJA_TERRACOTA).next_to(flecha_out, UP, buff=0.1)
        chispa = Star(n=5, outer_radius=0.25, inner_radius=0.12, color=MARRON_QUIJOTE, fill_opacity=1).next_to(val_out_gelu, RIGHT, buff=0.2)

        val_in_anim_2 = val_in.copy()
        self.play(val_in_anim_2.animate.move_to(nodo).scale(0.5).set_opacity(0), run_time=0.8)

        self.play(
            ReplacementTransform(val_out, val_out_gelu),
            Create(chispa)
        )
        
        self.play(FadeIn(lbl_suavizado, shift=UP), GrowArrow(flecha_suav))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_capa_transformer(self):
        """Diapositiva: Slide Capa Transformer."""

        escala = 0.65 

        def crear_nodo(texto, ancho=2.5 * escala, alto=0.8 * escala, resaltado=False):
            bg = NARANJA_TERRACOTA if resaltado else PAPEL_CREMA
            borde = MARRON_OSCURO
            txt_color = WHITE if resaltado else TINTA_NEGRA
            
            caja = RoundedRectangle(
                corner_radius=0.15 * escala, width=ancho, height=alto, 
                fill_color=bg, fill_opacity=1, stroke_color=borde, stroke_width=2.5 * escala
            )
            txt = Text(texto, font=FUENTE, font_size=20 * escala, color=txt_color)
            return VGroup(caja, txt)

        titulo, linea = self.crear_titulo("Arquitectura: Transformer Layer", palabra_clave="Transformer", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)
        
        pos_explicacion = RIGHT * 0.5 + UP * 1.5 

        EJE_X_RESIDUAL = LEFT * 3.5
        EJE_X_BLOQUES = LEFT * 0.5

        nodo_input = crear_nodo("Input").move_to(UP * 2.5 * escala + EJE_X_RESIDUAL)
        
        add_1 = VGroup(
            Circle(radius=0.25 * escala, fill_color=PAPEL_CREMA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2 * escala),
            Text("+", font_size=24 * escala, color=TINTA_NEGRA, weight=BOLD)
        ).move_to(UP * 0.2 * escala + EJE_X_RESIDUAL)
        
        add_2 = VGroup(
            Circle(radius=0.25 * escala, fill_color=PAPEL_CREMA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2 * escala),
            Text("+", font_size=24 * escala, color=TINTA_NEGRA, weight=BOLD)
        ).move_to(DOWN * 2.2 * escala + EJE_X_RESIDUAL)
        
        nodo_output = crear_nodo("Output").move_to(DOWN * 3.3 * escala + EJE_X_RESIDUAL)

        nodo_ln1 = crear_nodo("Layer Norm").move_to(UP * 1.4 * escala + EJE_X_BLOQUES)
        nodo_attn = crear_nodo("Self-Attention", resaltado=True).move_to(UP * 0.2 * escala + EJE_X_BLOQUES)
        
        nodo_ln2 = crear_nodo("Layer Norm").move_to(DOWN * 1.0 * escala + EJE_X_BLOQUES)
        nodo_mlp = crear_nodo("MLP", resaltado=True).move_to(DOWN * 2.2 * escala + EJE_X_BLOQUES)

        grosor_linea = 2.5 * escala

        linea_baja_input = Line(nodo_input.get_bottom(), add_1.get_top(), stroke_color=MARRON_OSCURO, stroke_width=grosor_linea)
        punto_split_1 = linea_baja_input.point_from_proportion(0.3)
        flecha_res_1 = Arrow(punto_split_1, add_1.get_top(), buff=0, color=MARRON_OSCURO, stroke_width=grosor_linea, max_tip_length_to_length_ratio=0.15)
        txt_res_1 = Text("Residual Stream", font=FUENTE, font_size=12 * escala, color=GRAY_B).rotate(PI/2).next_to(flecha_res_1, LEFT, buff=0.1)

        esquina_1 = [nodo_ln1.get_center()[0], punto_split_1[1], 0]
        linea_der_1 = Line(punto_split_1, esquina_1, stroke_color=MARRON_OSCURO, stroke_width=grosor_linea)
        flecha_ln1 = Arrow(esquina_1, nodo_ln1.get_top(), buff=0, color=MARRON_OSCURO, stroke_width=grosor_linea)
        flecha_attn = Arrow(nodo_ln1.get_bottom(), nodo_attn.get_top(), buff=0, color=MARRON_OSCURO, stroke_width=grosor_linea)
        flecha_retorno_1 = Arrow(nodo_attn.get_left(), add_1.get_right(), buff=0, color=MARRON_OSCURO, stroke_width=grosor_linea)

        linea_baja_add1 = Line(add_1.get_bottom(), add_2.get_top(), stroke_color=MARRON_OSCURO, stroke_width=grosor_linea)
        punto_split_2 = linea_baja_add1.point_from_proportion(0.3)
        flecha_res_2 = Arrow(punto_split_2, add_2.get_top(), buff=0, color=MARRON_OSCURO, stroke_width=grosor_linea, max_tip_length_to_length_ratio=0.15)
        txt_res_2 = Text("Residual Stream", font=FUENTE, font_size=12 * escala, color=GRAY_B).rotate(PI/2).next_to(flecha_res_2, LEFT, buff=0.1)

        esquina_2 = [nodo_ln2.get_center()[0], punto_split_2[1], 0]
        linea_der_2 = Line(punto_split_2, esquina_2, stroke_color=MARRON_OSCURO, stroke_width=grosor_linea)
        flecha_ln2 = Arrow(esquina_2, nodo_ln2.get_top(), buff=0, color=MARRON_OSCURO, stroke_width=grosor_linea)
        flecha_mlp = Arrow(nodo_ln2.get_bottom(), nodo_mlp.get_top(), buff=0, color=MARRON_OSCURO, stroke_width=grosor_linea)
        flecha_retorno_2 = Arrow(nodo_mlp.get_left(), add_2.get_right(), buff=0, color=MARRON_OSCURO, stroke_width=grosor_linea)

        flecha_out = Arrow(add_2.get_bottom(), nodo_output.get_top(), buff=0, color=MARRON_OSCURO, stroke_width=grosor_linea)

        self.play(FadeIn(nodo_input, shift=DOWN * escala))
        
        self.play(Create(linea_baja_input))
        self.play(Create(linea_der_1), Create(flecha_ln1))
        self.play(FadeIn(nodo_ln1, shift=DOWN * escala))
        self.play(Create(flecha_attn), FadeIn(nodo_attn, shift=DOWN * escala))
        self.play(Create(flecha_retorno_1), FadeIn(add_1, scale=0.5))
        self.play(FadeIn(txt_res_1))
        
        self.play(Create(linea_baja_add1))
        self.play(Create(linea_der_2), Create(flecha_ln2))
        self.play(FadeIn(nodo_ln2, shift=DOWN * escala))
        self.play(Create(flecha_mlp), FadeIn(nodo_mlp, shift=DOWN * escala))
        self.play(Create(flecha_retorno_2), FadeIn(add_2, scale=0.5))
        self.play(FadeIn(txt_res_2))
        
        self.play(Create(flecha_out), FadeIn(nodo_output, shift=UP * escala))
        self.next_slide()
        
        def crear_bloque_texto(titulo, items):
            head = Text(titulo, font=FUENTE, font_size=28, color=NARANJA_TERRACOTA, weight=BOLD)
            bullets = VGroup(*[Text(f"• {item}", font=FUENTE, font_size=22, color=TINTA_NEGRA) for item in items])
            bullets.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(head, DOWN, aligned_edge=LEFT, buff=0.4)
            return VGroup(head, bullets).move_to(pos_explicacion).align_to(pos_explicacion, LEFT)

        grupo_residual = VGroup(linea_baja_input, add_1, linea_baja_add1, add_2, flecha_out)
        caja1 = SurroundingRectangle(grupo_residual, color=NARANJA_TERRACOTA, buff=0.2, stroke_width=3)
        texto1 = crear_bloque_texto("Residual Stream (Autopista):", [
            "Mantiene la información original intacta.",
            "Permite redes muy profundas sin perder gradiente.",
            "Los bloques suman (+) información."
        ])

        self.play(Create(caja1), FadeIn(texto1, shift=LEFT))
        self.next_slide()
        self.play(FadeOut(caja1), FadeOut(texto1))

        caja2 = VGroup(
            SurroundingRectangle(nodo_ln1, color=MARRON_OSCURO, buff=0.1, stroke_width=3),
            SurroundingRectangle(nodo_ln2, color=MARRON_OSCURO, buff=0.1, stroke_width=3)
        )
        texto2 = crear_bloque_texto("Pre-Layer Normalization:", [
            "Normaliza antes de entrar al bloque.",
            "Esencial para estabilidad matemática.",
            "Estándar en modelos como GPT."
        ])

        self.play(Create(caja2), FadeIn(texto2, shift=LEFT))
        self.next_slide()
        self.play(FadeOut(caja2), FadeOut(texto2))

        caja3 = VGroup(
            SurroundingRectangle(nodo_attn, color=NARANJA_TERRACOTA, buff=0.15, stroke_width=3),
            SurroundingRectangle(nodo_mlp, color=NARANJA_TERRACOTA, buff=0.15, stroke_width=3)
        )
        texto3 = crear_bloque_texto("Bloques de Cómputo:", [
            "Attention: Relaciona palabras entre sí.",
            "MLP: Procesa palabras individualmente."
        ])

        self.play(Create(caja3), FadeIn(texto3, shift=LEFT))
        self.next_slide()
        self.play(FadeOut(caja3), FadeOut(texto3))

        self.limpiar_pantalla()

    def slide_entrenamiento(self):

        """Diapositiva: Slide Entrenamiento."""

        titulo, linea = self.crear_titulo(
            "Entrenamiento: Ajustando Perillas", 
            palabra_clave="Perillas", 
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        def crear_caja(texto, ancho=2.5, bg_color=FONDO_CAJA, txt_color=TINTA_NEGRA, 
                       opacidad=1.0, borde_color=MARRON_OSCURO, borde_grosor=1, peso=NORMAL):
            """Crea un VGroup estándar con un rectángulo redondeado y texto."""
            caja = RoundedRectangle(
                corner_radius=0.1, width=ancho, height=0.6, 
                fill_color=bg_color, fill_opacity=opacidad, 
                stroke_color=borde_color, stroke_width=borde_grosor
            )
            txt = Text(texto, font=FUENTE, font_size=16, color=txt_color, weight=peso)
            return VGroup(caja, txt)

        def crear_perilla(angulo):
            """Crea la representación visual de un parámetro ajustable."""
            base = Circle(radius=0.25, fill_color=FONDO_CAJA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2)
            indicador = Line(base.get_center(), base.get_center() + UP*0.25, color=NARANJA_TERRACOTA, stroke_width=4)
            indicador.rotate(angulo, about_point=base.get_center())
            return VGroup(base, indicador)


        estado_ui = crear_caja("Iniciando Motor de Entrenamiento...", ancho=7.5).to_edge(UP, buff=1.2)
        self.play(FadeIn(estado_ui, shift=DOWN))

        EJE_Y = DOWN * 0.2

        self.play(
            estado_ui[1].animate.set_text("FASE 1: Forward Pass (Predicción inicial)"), 
            estado_ui[0].animate.set_stroke(NARANJA_TERRACOTA, width=2)
        )

        lbl_in = Text("Contexto", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        tokens_in = VGroup(*[crear_caja(word, ancho=1.1) for word in ["En un", "lugar", "de la"]]).arrange(DOWN, buff=0.1)
        grupo_entrada = VGroup(lbl_in, tokens_in).arrange(DOWN, buff=0.3).move_to(LEFT * 4.5 + EJE_Y)

        modelo_bg = RoundedRectangle(corner_radius=0.2, width=3.5, height=2.6, fill_color=PAPEL_CREMA, fill_opacity=1, stroke_color=MARRON_OSCURO, stroke_width=2)
        lbl_modelo = Text("Transformer\n(Red Neuronal)", font=FUENTE, font_size=18, color=TINTA_NEGRA, weight=BOLD).next_to(modelo_bg.get_top(), DOWN, buff=0.2)
        
        grupo_perillas = VGroup(
            crear_perilla(PI/4), crear_perilla(-PI/3), crear_perilla(PI)
        ).arrange(RIGHT, buff=0.4).move_to(modelo_bg.get_center() + DOWN*0.1)
        
        lbl_pesos = Text("Parámetros (Pesos)", font=FUENTE, font_size=14, color=TINTA_NEGRA).next_to(grupo_perillas, DOWN, buff=0.25)
        grupo_modelo = VGroup(modelo_bg, lbl_modelo, grupo_perillas, lbl_pesos).move_to(EJE_Y)

        lbl_out = Text("Predicción", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        prob_incorrecta = crear_caja("playa? (85%)", bg_color=PAPEL_TAN, opacidad=0.8, borde_color=NARANJA_TERRACOTA, borde_grosor=2)
        prob_correcta = crear_caja("Mancha (2%)", bg_color=CAJA_INFERIOR, txt_color=MARRON_OSCURO, opacidad=0.5, borde_color=CAJA_INFERIOR)
        grupo_probs = VGroup(prob_incorrecta, prob_correcta).arrange(DOWN, buff=0.1)
        grupo_salida = VGroup(lbl_out, grupo_probs).arrange(DOWN, buff=0.3).move_to(RIGHT * 4.5 + EJE_Y)

        flujo_in = DashedLine(grupo_entrada.get_right(), modelo_bg.get_left(), color=MARRON_OSCURO, stroke_width=3)
        tubo_out = Line(modelo_bg.get_right(), grupo_salida.get_left(), color=MARRON_OSCURO, stroke_width=4)

        self.play(FadeIn(grupo_entrada, shift=RIGHT))
        self.play(Create(flujo_in))
        self.play(FadeIn(grupo_modelo, scale=0.9))
        self.play(GrowFromCenter(tubo_out))
        self.play(FadeIn(grupo_salida, shift=LEFT))
        self.next_slide()

        self.play(
            estado_ui[1].animate.set_text("FASE 2: Función de Pérdida (Loss)"), 
            estado_ui[0].animate.set_stroke(NARANJA_TERRACOTA, width=2)
        )

        lbl_target = Text("Target Real:", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        caja_target = crear_caja("Mancha (100%)")
        grupo_target = VGroup(lbl_target, caja_target).arrange(DOWN, buff=0.1).next_to(grupo_salida, DOWN, buff=0.4)

        nodo_loss = MathTex(r"\mathcal{L}", font_size=40, color=NARANJA_TERRACOTA)
        medidor_bg = RoundedRectangle(width=1.5, height=0.2, corner_radius=0.1, stroke_color=MARRON_OSCURO, fill_color=FONDO_CAJA, fill_opacity=1)
        medidor_fill = RoundedRectangle(width=1.3, height=0.2, corner_radius=0.1, stroke_width=0, fill_color=NARANJA_TERRACOTA, fill_opacity=1).align_to(medidor_bg, LEFT)
        
        caja_error = VGroup(
            Text("Error cuantificado:", font=FUENTE, font_size=14, color=TINTA_NEGRA),
            VGroup(medidor_bg, medidor_fill),
            Text("¡Pérdida muy alta!", font=FUENTE, font_size=14, color=NARANJA_TERRACOTA, weight=BOLD)
        ).arrange(DOWN, buff=0.1)

        panel_loss = VGroup(nodo_loss, caja_error).arrange(RIGHT, buff=0.4).move_to(DOWN * 2.7)

        self.play(FadeIn(grupo_target, shift=UP))
        
        self.play(
            Indicate(prob_incorrecta, color=NARANJA_TERRACOTA, scale_factor=1.1),
            Indicate(caja_target, color=NARANJA_TERRACOTA, scale_factor=1.1)
        )
        
        self.play(FadeIn(panel_loss, shift=UP))
        self.play(Flash(medidor_fill, color=NARANJA_TERRACOTA, line_length=0.2))
        self.next_slide()

        self.play(
            estado_ui[1].animate.set_text("FASE 3: Backpropagation (Ajuste)"), 
            estado_ui[0].animate.set_stroke(NARANJA_TERRACOTA, width=3)
        )

        tubo_back = Arrow(panel_loss.get_top(), modelo_bg.get_bottom(), color=NARANJA_TERRACOTA, stroke_width=4, buff=0.1)
        lbl_grad = MathTex(r"\nabla W", font_size=24, color=NARANJA_TERRACOTA).next_to(tubo_back, RIGHT, buff=0.1)

        self.play(Create(tubo_back), Write(lbl_grad))

        self.play(
            Rotate(grupo_perillas[0][1], angle=-PI/2, about_point=grupo_perillas[0][0].get_center()),
            Rotate(grupo_perillas[1][1], angle=PI/1.5, about_point=grupo_perillas[1][0].get_center()),
            Rotate(grupo_perillas[2][1], angle=-PI/4, about_point=grupo_perillas[2][0].get_center()),
            modelo_bg.animate.set_stroke(NARANJA_TERRACOTA, width=3),
            run_time=2,
            rate_func=there_and_back_with_pause 
        )

        lbl_pesos_nuevos = Text("Parámetros (Ajustados)", font=FUENTE, font_size=14, color=MARRON_OSCURO, weight=BOLD).move_to(lbl_pesos)

        self.play(
            Transform(lbl_pesos, lbl_pesos_nuevos),
            modelo_bg.animate.set_stroke(MARRON_OSCURO, width=2),
            FadeOut(tubo_back, lbl_grad)
        )
        self.next_slide()

        self.play(
            estado_ui[1].animate.set_text("FASE 4: Nuevo Forward Pass (Éxito)"), 
            estado_ui[0].animate.set_stroke(MARRON_OSCURO, width=2)
        )

        flujo_exito = DashedLine(grupo_entrada.get_right(), modelo_bg.get_left(), color=NARANJA_TERRACOTA, stroke_width=4)
        tubo_exito = Line(modelo_bg.get_right(), grupo_salida.get_left(), color=NARANJA_TERRACOTA, stroke_width=5)

        self.play(
            Indicate(modelo_bg, color=PAPEL_TAN),
            Transform(flujo_in, flujo_exito),
            Transform(tubo_out, tubo_exito),
            run_time=1.5
        )

        prob_correcta_nueva = crear_caja("Mancha (98%)", bg_color=PAPEL_TAN, borde_color=MARRON_OSCURO, borde_grosor=2, peso=BOLD).move_to(prob_incorrecta)
        prob_incorrecta_nueva = crear_caja("playa? (1%)", bg_color=CAJA_INFERIOR, txt_color=MARRON_OSCURO, opacidad=0.5, borde_color=CAJA_INFERIOR).move_to(prob_correcta)

        self.play(
            FadeOut(panel_loss, grupo_target),
            Transform(prob_incorrecta, prob_correcta_nueva),
            Transform(prob_correcta, prob_incorrecta_nueva)
        )
        
        self.play(Wiggle(prob_incorrecta, scale_value=1.05))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_descenso_gradiente(self):
        """Diapositiva: Descenso del Gradiente — intuición, mecánica y conexión al entrenamiento."""

        titulo, linea = self.crear_titulo(
            "Descenso del Gradiente: Bajando la Colina",
            palabra_clave="Gradiente:",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        # ── ACTO 1: EL PROBLEMA — ¿QUÉ QUEREMOS MINIMIZAR? ──────────────────────
        # Mostrar primero la idea antes de los ejes
        pregunta = Text("El modelo tiene millones de parámetros.\n¿Cómo sabe hacia dónde ajustarlos?",
                        font=FUENTE, font_size=24, weight=BOLD, color=TINTA_NEGRA,
                        line_spacing=1.4).move_to(UP * 1.5)
        self.play(Write(pregunta))
        self.next_slide()

        analogia = Text(
            "Imagina que estás en una montaña con los ojos vendados.\n"
            "Solo puedes sentir la pendiente bajo tus pies.\n"
            "Estrategia: siempre da un paso cuesta abajo.",
            font=FUENTE, font_size=21, color=MARRON_OSCURO, line_spacing=1.4
        ).next_to(pregunta, DOWN, buff=0.5)
        self.play(FadeIn(analogia, shift=UP * 0.2))
        self.next_slide()

        self.play(FadeOut(pregunta), FadeOut(analogia))

        # ── ACTO 2: LA MECÁNICA — DESCENSO PASO A PASO ───────────────────────────
        def func_costo(x):
            return np.sin(x) + 0.3 * (x ** 2) + 2

        def gradiente_costo(x):
            return np.cos(x) + 0.6 * x

        ejes = Axes(
            x_range=[-3, 5, 1],
            y_range=[0, 10, 2],
            x_length=6.5,
            y_length=4.5,
            axis_config={"color": MARRON_OSCURO, "include_numbers": False}
        ).shift(LEFT * 1.8 + DOWN * 0.3)

        lbl_x = ejes.get_x_axis_label(
            Text("Parámetro  w", font=FUENTE, font_size=15, color=MARRON_OSCURO),
            edge=RIGHT, direction=DOWN
        )
        lbl_y = ejes.get_y_axis_label(
            Text("Loss  L", font=FUENTE, font_size=15, color=MARRON_OSCURO),
            edge=UP, direction=UP
        )

        curva = ejes.plot(func_costo, color=NARANJA_TERRACOTA, stroke_width=4)
        area  = ejes.get_area(curva, opacity=0.08, color=NARANJA_TERRACOTA)

        # Mínimo y stickman meta
        x_minimo  = -0.89
        p_minimo  = ejes.c2p(x_minimo, func_costo(x_minimo))

        def crear_stickman():
            cabeza  = Circle(radius=0.15, color=MARRON_OSCURO,
                             fill_color=PAPEL_CREMA, fill_opacity=1)
            cuerpo  = Line(cabeza.get_bottom(),
                           cabeza.get_bottom() + DOWN * 0.4,
                           color=MARRON_OSCURO, stroke_width=3)
            brazos  = VGroup(
                Line(cuerpo.get_center() + UP * 0.1,
                     cuerpo.get_center() + UP * 0.3 + LEFT * 0.2,
                     color=MARRON_OSCURO, stroke_width=3),
                Line(cuerpo.get_center() + UP * 0.1,
                     cuerpo.get_center() + UP * 0.3 + RIGHT * 0.2,
                     color=MARRON_OSCURO, stroke_width=3)
            )
            piernas = VGroup(
                Line(cuerpo.get_bottom(),
                     cuerpo.get_bottom() + DOWN * 0.3 + LEFT * 0.2,
                     color=MARRON_OSCURO, stroke_width=3),
                Line(cuerpo.get_bottom(),
                     cuerpo.get_bottom() + DOWN * 0.3 + RIGHT * 0.2,
                     color=MARRON_OSCURO, stroke_width=3)
            )
            mastil  = Line(brazos[0].get_end(),
                           brazos[0].get_end() + UP * 0.5,
                           color=TINTA_NEGRA, stroke_width=2)
            bandera = Polygon(
                mastil.get_end(),
                mastil.get_end() + DOWN * 0.15 + LEFT * 0.3,
                mastil.get_end() + DOWN * 0.3,
                fill_color=NARANJA_TERRACOTA, fill_opacity=1, stroke_width=0
            )
            return VGroup(cabeza, cuerpo, brazos, piernas, mastil, bandera)

        stickman = crear_stickman().scale(0.6).next_to(p_minimo, UP, buff=0)
        lbl_meta  = Text("mínimo", font=FUENTE, font_size=13,
                         color=NARANJA_TERRACOTA, weight=BOLD).next_to(stickman, UP, buff=0.08)

        self.play(Create(ejes), Write(lbl_x), Write(lbl_y))
        self.play(Create(curva), FadeIn(area))
        self.play(FadeIn(stickman, shift=UP * 0.3), Write(lbl_meta))

        # Panel lateral — ecuación y leyenda compacta
        lr = 0.8
        eq_update = MathTex(
            r"w \leftarrow w - \underbrace{\eta}_{\text{lr}} \cdot \underbrace{\nabla L}_{\text{pendiente}}",
            font_size=30, color=MARRON_OSCURO
        )
        eq_frame = SurroundingRectangle(eq_update, color=MARRON_OSCURO,
                                        buff=0.25, corner_radius=0.1)
        eq_grupo = VGroup(eq_frame, eq_update)

        lbl_lr   = Text(f"η = {lr}  (learning rate)", font=FUENTE,
                        font_size=16, color=TINTA_NEGRA)
        lbl_grad = Text("∇L = pendiente en w", font=FUENTE,
                        font_size=16, color=NARANJA_TERRACOTA)

        # Indicadores dinámicos
        txt_pos  = Text("w = 4.00", font="Monospace", font_size=16, color=MARRON_OSCURO)
        txt_grad_val = Text("∇L = —", font="Monospace", font_size=16, color=NARANJA_TERRACOTA)
        txt_loss = Text("L = —", font="Monospace", font_size=16, color=ROJO_TOMATE)

        panel = VGroup(eq_grupo, lbl_lr, lbl_grad,
                       txt_pos, txt_grad_val, txt_loss
                       ).arrange(DOWN, buff=0.32, aligned_edge=LEFT
                       ).next_to(ejes, RIGHT, buff=0.55).align_to(ejes, UP).shift(DOWN * 0.3)

        self.play(FadeIn(panel, shift=LEFT * 0.2))
        self.next_slide()

        # Punto inicial y animación de descenso
        x_val = 4.0
        punto  = Dot(ejes.c2p(x_val, func_costo(x_val)),
                     color=TINTA_NEGRA, radius=0.13)
        self.play(FadeIn(punto, scale=0.5))

        rastros = VGroup()

        for i in range(7):
            grad_actual = gradiente_costo(x_val)
            x_next      = x_val - lr * grad_actual
            x_next      = max(-2.8, min(4.8, x_next))  # no salir de los ejes

            p_actual = ejes.c2p(x_val,  func_costo(x_val))
            p_next   = ejes.c2p(x_next, func_costo(x_next))

            # Actualizar panel
            nuevo_pos  = Text(f"w = {x_val:.2f}",
                              font="Monospace", font_size=16, color=MARRON_OSCURO
                              ).move_to(txt_pos, aligned_edge=LEFT)
            nuevo_grad = Text(f"∇L = {grad_actual:+.2f}",
                              font="Monospace", font_size=16,
                              color=VERDE_OLIVA if abs(grad_actual) < 0.5 else NARANJA_TERRACOTA
                              ).move_to(txt_grad_val, aligned_edge=LEFT)
            nuevo_loss = Text(f"L = {func_costo(x_val):.2f}",
                              font="Monospace", font_size=16, color=ROJO_TOMATE
                              ).move_to(txt_loss, aligned_edge=LEFT)

            self.play(
                Transform(txt_pos,      nuevo_pos),
                Transform(txt_grad_val, nuevo_grad),
                Transform(txt_loss,     nuevo_loss),
                run_time=0.35
            )

            # Tangente
            dx      = 0.55
            t_start = ejes.c2p(x_val - dx, func_costo(x_val) - grad_actual * dx)
            t_end   = ejes.c2p(x_val + dx, func_costo(x_val) + grad_actual * dx)
            tangente = Line(t_start, t_end, color=AZUL_NOCHE,
                            stroke_width=2.5).set_opacity(0.55)
            self.play(Create(tangente), run_time=0.25)

            # Flecha de salto
            angulo_salto = -TAU / 6 if grad_actual > 0 else TAU / 6
            flecha_salto = CurvedArrow(p_actual, p_next, angle=angulo_salto,
                                       color=NARANJA_TERRACOTA, stroke_width=2.8)
            self.play(Create(flecha_salto), run_time=0.35)

            # Rastro de posiciones anteriores
            rastro = Dot(p_actual, color=MARRON_OSCURO, radius=0.055).set_opacity(0.35)
            rastros.add(rastro)
            self.add(rastro)

            self.play(
                punto.animate.move_to(p_next),
                FadeOut(tangente),
                FadeOut(flecha_salto),
                run_time=0.5
            )
            x_val = x_next

        # Llegó al mínimo
        self.play(
            Wiggle(stickman, scale_value=1.18, rotation_angle=0.04 * TAU),
            Flash(punto, color=NARANJA_TERRACOTA, line_length=0.45, flash_radius=0.32,
                  num_lines=12),
            run_time=1.0
        )
        self.next_slide()

        # ── ACTO 3: CONEXIÓN AL ENTRENAMIENTO REAL ───────────────────────────────
        self.play(
            FadeOut(ejes), FadeOut(lbl_x), FadeOut(lbl_y),
            FadeOut(curva), FadeOut(area),
            FadeOut(stickman), FadeOut(lbl_meta),
            FadeOut(punto), FadeOut(rastros),
            FadeOut(panel)
        )

        lbl_conexion = Text("En la práctica, esto sucede en cada paso de entrenamiento",
                            font=FUENTE, font_size=24, weight=BOLD,
                            color=TINTA_NEGRA).move_to(UP * 2.5)
        self.play(Write(lbl_conexion))

        pasos = [
            (PAPEL_TAN,        "① Forward pass",
             "El modelo predice la siguiente\npalabra del texto."),
            (SALMON_ATENCION,  "② Calcular Loss",
             "Se mide qué tan equivocada\nfue la predicción."),
            (NARANJA_TERRACOTA,"③ Backprop",
             "Se calcula el gradiente de\ncada parámetro (∇L)."),
            (VERDE_OLIVA,      "④ Actualizar  w ← w − η·∇L",
             "Cada peso da un pequeño\npaso cuesta abajo."),
        ]

        cajas = VGroup()
        for color, tit, desc in pasos:
            rect = RoundedRectangle(corner_radius=0.12, width=2.8, height=1.9,
                                    fill_color=color, fill_opacity=0.25,
                                    stroke_color=color, stroke_width=2.2)
            t_tit  = Text(tit,  font=FUENTE, font_size=16, weight=BOLD,
                          color=TINTA_NEGRA, line_spacing=1.2)
            t_desc = Text(desc, font=FUENTE, font_size=15,
                          color=MARRON_OSCURO, line_spacing=1.2)
            contenido = VGroup(t_tit, t_desc).arrange(DOWN, buff=0.18)
            contenido.scale_to_fit_width(rect.width - 0.35).move_to(rect)
            cajas.add(VGroup(rect, contenido))

        flechas_flujo = VGroup()
        cajas.arrange(RIGHT, buff=0.5).move_to(DOWN * 0.3)
        for i in range(len(cajas) - 1):
            f = Arrow(cajas[i].get_right(), cajas[i + 1].get_left(),
                      buff=0.08, color=MARRON_OSCURO,
                      max_tip_length_to_length_ratio=0.25, stroke_width=3)
            flechas_flujo.add(f)

        # ─── AQUÍ ESTÁ LA SOLUCIÓN ──────────────────────────────────────────────
        animaciones_lagged = []
        for i in range(len(cajas)):
            anims_del_grupo = [FadeIn(cajas[i], shift=UP * 0.2)]
            
            # Solo agregamos Create(flecha) si el índice es válido
            if i < len(flechas_flujo):
                anims_del_grupo.append(Create(flechas_flujo[i]))
                
            animaciones_lagged.append(AnimationGroup(*anims_del_grupo))

        # Evitar fallos si la lista llegara a estar vacía
        if animaciones_lagged:
            self.play(LaggedStart(*animaciones_lagged, lag_ratio=0.25))
        # ────────────────────────────────────────────────────────────────────────

        self.next_slide()

        # Nota final: el ciclo se repite miles de veces
        nota_final = Text(
            "Este ciclo se repite miles de veces hasta que el Loss converge.",
            font=FUENTE, font_size=20, weight=BOLD, color=NARANJA_TERRACOTA
        ).next_to(cajas, DOWN, buff=0.5)
        self.play(Write(nota_final))
        self.play(Indicate(nota_final, color=ORO_VIEJO, scale_factor=1.06))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_backpropagation(self):
        titulo, linea = self.crear_titulo(
            "Backpropagation: Regla de la Cadena",
            palabra_clave="Regla de la Cadena",
            color_clave=NARANJA_TERRACOTA
        )
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        pregunta = Text(
            "El Loss sabe que el modelo se equivocó.\n"
            "Pero… ¿qué parámetro tuvo la culpa?",
            font=FUENTE, font_size=24, weight=BOLD,
            color=TINTA_NEGRA, line_spacing=1.4
        ).move_to(UP * 1.8)
        self.play(Write(pregunta))
        self.next_slide()

        cadena_palabras = ["L", "GELU", "suma", "mul", "w₀"]
        cadena_nodos = VGroup()
        for i, p in enumerate(cadena_palabras):
            rect = RoundedRectangle(corner_radius=0.1, width=1.4, height=0.6,
                                    fill_color=PAPEL_CREMA, fill_opacity=0.9,
                                    stroke_color=MARRON_OSCURO, stroke_width=2)
            lbl = Text(p, font=FUENTE, font_size=20,
                       color=NARANJA_TERRACOTA if i == 0 else TINTA_NEGRA, weight=BOLD)
            lbl.move_to(rect)
            cadena_nodos.add(VGroup(rect, lbl))

        cadena_nodos.arrange(RIGHT, buff=0.55).next_to(pregunta, DOWN, buff=0.55)

        flechas_cadena = VGroup(*[
            Arrow(cadena_nodos[i].get_right(), cadena_nodos[i + 1].get_left(),
                  buff=0.08, color=MARRON_OSCURO,
                  max_tip_length_to_length_ratio=0.3, stroke_width=2.5)
            for i in range(len(cadena_nodos) - 1)
        ])

        lbl_cadena = Text("¿Cuánto contribuyó cada uno al error?",
                          font=FUENTE, font_size=19, color=MARRON_OSCURO
                          ).next_to(cadena_nodos, DOWN, buff=0.35)

        animaciones_lagged = []
        for i in range(len(cadena_nodos)):
            anims = [FadeIn(cadena_nodos[i], shift=RIGHT * 0.15)]
            if i < len(flechas_cadena):
                anims.append(Create(flechas_cadena[i]))
            animaciones_lagged.append(AnimationGroup(*anims))

        if animaciones_lagged:
            self.play(LaggedStart(*animaciones_lagged, lag_ratio=0.2))

        self.play(Write(lbl_cadena))
        self.next_slide()

        self.play(FadeOut(pregunta), FadeOut(cadena_nodos),
                  FadeOut(flechas_cadena), FadeOut(lbl_cadena))

        lbl_herramienta = Text("La herramienta: Regla de la Cadena",
                               font=FUENTE, font_size=26, weight=BOLD,
                               color=TINTA_NEGRA).move_to(UP * 2.6)
        self.play(Write(lbl_herramienta))

        analogia_lbl = Text(
            "Si f depende de g, y g depende de x,\n"
            "el efecto total de x sobre f se obtiene multiplicando las pendientes intermedias:",
            font=FUENTE, font_size=20, color=MARRON_OSCURO, line_spacing=1.4
        ).next_to(lbl_herramienta, DOWN, buff=0.4)
        self.play(FadeIn(analogia_lbl, shift=UP * 0.2))
        self.next_slide()

        eq_cadena = MathTex(
            r"\frac{\partial L}{\partial w} = "
            r"\frac{\partial L}{\partial \hat{y}} \cdot "
            r"\frac{\partial \hat{y}}{\partial \text{sum}} \cdot "
            r"\frac{\partial \text{sum}}{\partial \text{mul}} \cdot "
            r"\frac{\partial \text{mul}}{\partial w}",
            font_size=34, color=TINTA_NEGRA
        ).next_to(analogia_lbl, DOWN, buff=0.5)

        self.play(Write(eq_cadena), run_time=2.0)
        self.next_slide()

        caja_eq = SurroundingRectangle(eq_cadena, color=NARANJA_TERRACOTA,
                                       buff=0.2, corner_radius=0.1, stroke_width=2.5)
        nota_local = Text(
            "Cada nodo solo necesita conocer su gradiente local.\n"
            "Backprop los encadena de derecha a izquierda.",
            font=FUENTE, font_size=19, color=NARANJA_TERRACOTA, line_spacing=1.3
        ).next_to(eq_cadena, DOWN, buff=0.4)
        self.play(Create(caja_eq))
        self.play(Write(nota_local))
        self.next_slide()

        self.play(FadeOut(lbl_herramienta), FadeOut(analogia_lbl),
                  FadeOut(eq_cadena), FadeOut(caja_eq), FadeOut(nota_local))

        lbl_grafo = Text("Grafo computacional: forward → backward",
                         font=FUENTE, font_size=24, weight=BOLD,
                         color=TINTA_NEGRA).move_to(UP * 2.7)
        self.play(Write(lbl_grafo))

        EJE_Y = UP * 1.0

        def crear_nodo_op(texto, pos, es_texto=False):
            circ = Circle(radius=0.48, fill_color=FONDO_CAJA, fill_opacity=1,
                          stroke_color=MARRON_OSCURO, stroke_width=3)
            circ.move_to(pos)
            if es_texto:
                etq = Text(texto, font=FUENTE, font_size=15,
                           color=MARRON_OSCURO, weight=BOLD).move_to(circ)
            else:
                etq = MathTex(texto, font_size=38,
                              color=MARRON_OSCURO).move_to(circ)
            return VGroup(circ, etq), circ

        grp_mul, nd_mul = crear_nodo_op(r"\times", LEFT * 3.5 + EJE_Y)
        grp_sum, nd_sum = crear_nodo_op(r"+", LEFT * 0.0 + EJE_Y)
        grp_gelu, nd_gelu = crear_nodo_op("GELU", RIGHT * 3.5 + EJE_Y, es_texto=True)

        tx0 = MathTex("x_0", font_size=30, color=TINTA_NEGRA
                      ).move_to(nd_mul.get_left() + LEFT * 1.4 + UP * 0.55)
        tw0 = MathTex("w_0", font_size=30, color=AZUL_NOCHE
                      ).move_to(nd_mul.get_left() + LEFT * 1.4 + DOWN * 0.55)
        tb = MathTex("b", font_size=30, color=AZUL_NOCHE
                     ).move_to(nd_sum.get_bottom() + DOWN * 1.1)
        ty = MathTex(r"\hat{y}", font_size=34, color=NARANJA_TERRACOTA
                     ).move_to(nd_gelu.get_right() + RIGHT * 1.4)

        def arista(a, b, color=MARRON_OSCURO, sw=2):
            return Line(a, b, color=color, stroke_width=sw, z_index=-1).add_tip(
                tip_length=0.18, tip_width=0.18)

        f_x0 = arista(tx0.get_right(), nd_mul.get_left() + UP * 0.2)
        f_w0 = arista(tw0.get_right(), nd_mul.get_left() + DOWN * 0.2, color=AZUL_NOCHE)
        f_b = arista(tb.get_top(), nd_sum.get_bottom(), color=AZUL_NOCHE)
        f_ms = arista(nd_mul.get_right(), nd_sum.get_left())
        f_sg = arista(nd_sum.get_right(), nd_gelu.get_left())
        f_out = arista(nd_gelu.get_right(), ty.get_left(), color=NARANJA_TERRACOTA)

        lbl_ms = Text("mul", font=FUENTE, font_size=14,
                      color=MARRON_OSCURO).next_to(f_ms, UP, buff=0.08)
        lbl_sg = Text("sum", font=FUENTE, font_size=14,
                      color=MARRON_OSCURO).next_to(f_sg, UP, buff=0.08)

        grafo_fwd = VGroup(grp_mul, grp_sum, grp_gelu,
                           tx0, tw0, tb, ty,
                           f_x0, f_w0, f_b, f_ms, f_sg, f_out,
                           lbl_ms, lbl_sg)

        lbl_fwd = Text("① Forward pass", font=FUENTE, font_size=18,
                       weight=BOLD, color=VERDE_OLIVA).next_to(lbl_grafo, DOWN, buff=0.15)
        self.play(Write(lbl_fwd))
        self.play(LaggedStart(
            AnimationGroup(FadeIn(tx0), FadeIn(tw0)),
            AnimationGroup(Create(f_x0), Create(f_w0)),
            DrawBorderThenFill(grp_mul),
            AnimationGroup(Create(f_ms), FadeIn(lbl_ms), FadeIn(tb), Create(f_b)),
            DrawBorderThenFill(grp_sum),
            AnimationGroup(Create(f_sg), FadeIn(lbl_sg)),
            DrawBorderThenFill(grp_gelu),
            AnimationGroup(Create(f_out), Write(ty)),
            lag_ratio=0.25, run_time=2.5
        ))
        self.next_slide()

        lbl_bwd = Text("② Backward pass  (gradientes de derecha a izquierda)",
                       font=FUENTE, font_size=18, weight=BOLD,
                       color=NARANJA_TERRACOTA).move_to(lbl_fwd)
        self.play(ReplacementTransform(lbl_fwd, lbl_bwd))

        rutas_back = [
            (nd_gelu.get_left(), nd_sum.get_right(), r"\partial\hat{y}/\partial\text{sum}"),
            (nd_sum.get_left(), nd_mul.get_right(), r"\partial\text{sum}/\partial\text{mul}"),
            (nd_mul.get_left() + DOWN * 0.2, tw0.get_right(), r"\partial\text{mul}/\partial w_0"),
            (nd_sum.get_bottom(), tb.get_top(), r"\partial\text{sum}/\partial b"),
        ]

        for p_start, p_end, grad_tex in rutas_back:
            flash_line = Line(p_start, p_end, color=NARANJA_TERRACOTA, stroke_width=5)
            grad_lbl = MathTex(grad_tex, font_size=20, color=NARANJA_TERRACOTA
                               ).move_to(flash_line.get_center() + UP * 0.35)
            self.play(ShowPassingFlash(flash_line, time_width=0.55), run_time=0.7)
            self.play(FadeIn(grad_lbl, shift=UP * 0.1), run_time=0.35)
            self.play(FadeOut(grad_lbl), run_time=0.25)

        caja_w0 = SurroundingRectangle(tw0, color=NARANJA_TERRACOTA,
                                       buff=0.1, corner_radius=0.08, stroke_width=2.5)
        caja_b = SurroundingRectangle(tb, color=NARANJA_TERRACOTA,
                                      buff=0.1, corner_radius=0.08, stroke_width=2.5)
        lbl_upd_w = MathTex(r"-\eta\,\Delta w_0", font_size=20,
                            color=NARANJA_TERRACOTA).next_to(caja_w0, LEFT, buff=0.15)
        lbl_upd_b = MathTex(r"-\eta\,\Delta b", font_size=20,
                            color=NARANJA_TERRACOTA).next_to(caja_b, LEFT, buff=0.15)

        self.play(Create(caja_w0), Create(caja_b))
        self.play(FadeIn(lbl_upd_w, shift=RIGHT * 0.15),
                  FadeIn(lbl_upd_b, shift=RIGHT * 0.15))
        self.play(
            Indicate(lbl_upd_w, color=ORO_VIEJO, scale_factor=1.2),
            Indicate(lbl_upd_b, color=ORO_VIEJO, scale_factor=1.2),
        )
        self.next_slide()

        self.play(FadeOut(grafo_fwd), FadeOut(lbl_bwd),
                  FadeOut(caja_w0), FadeOut(caja_b),
                  FadeOut(lbl_upd_w), FadeOut(lbl_upd_b), FadeOut(lbl_grafo))

        lbl_red = Text("En la red completa: millones de parámetros, mismo principio",
                       font=FUENTE, font_size=22, weight=BOLD,
                       color=TINTA_NEGRA).move_to(UP * 2.7)
        self.play(Write(lbl_red))

        capas_config = [3, 5, 4, 2]
        nodos_red = VGroup()
        for i, n in enumerate(capas_config):
            capa = VGroup(*[
                Circle(radius=0.22, fill_color=FONDO_CAJA, fill_opacity=1,
                       stroke_color=MARRON_OSCURO, stroke_width=2.5)
                for _ in range(n)
            ]).arrange(DOWN, buff=0.45)
            capa.move_to(RIGHT * (i * 2.6 - 3.9) + DOWN * 0.35)
            nodos_red.add(capa)

        conexiones_fwd_red = VGroup()
        conexiones_back_por_capa = []
        for i in range(len(capas_config) - 1):
            grupo_back = VGroup()
            for n1 in nodos_red[i]:
                for n2 in nodos_red[i + 1]:
                    ln_fwd = Line(n1.get_right(), n2.get_left(),
                                  stroke_width=1.5, color=MARRON_OSCURO,
                                  z_index=-1).set_opacity(0.25)
                    conexiones_fwd_red.add(ln_fwd)
                    ln_back = Line(n2.get_left(), n1.get_right(),
                                   stroke_width=4, color=NARANJA_TERRACOTA)
                    grupo_back.add(ln_back)
            conexiones_back_por_capa.append(grupo_back)

        nombres_capas = ["Entrada", "Capa 1", "Capa 2", "Salida"]
        lbls_capas = VGroup(*[
            Text(nombres_capas[i], font=FUENTE, font_size=14, color=MARRON_OSCURO
                 ).next_to(nodos_red[i], DOWN, buff=0.3)
            for i in range(len(capas_config))
        ])

        self.play(LaggedStart(
            *[GrowFromCenter(n) for capa in nodos_red for n in capa],
            lag_ratio=0.06
        ), run_time=1.2)
        self.play(Create(conexiones_fwd_red), FadeIn(lbls_capas), run_time=1.0)

        txt_loss_red = MathTex(r"L", font_size=30,
                               color=ROJO_TOMATE).next_to(nodos_red[-1], RIGHT, buff=0.5)
        self.play(Write(txt_loss_red),
                  Indicate(nodos_red[-1], color=ROJO_TOMATE, scale_factor=1.12))
        self.next_slide()

        lbl_bwd_red = Text("Gradientes fluyendo hacia atrás →",
                           font=FUENTE, font_size=19, weight=BOLD,
                           color=NARANJA_TERRACOTA).next_to(lbl_red, DOWN, buff=0.18)
        self.play(Write(lbl_bwd_red))

        for i in reversed(range(len(capas_config) - 1)):
            destellos = [ShowPassingFlash(l.copy(), time_width=0.45)
                         for l in conexiones_back_por_capa[i]]
            self.play(
                AnimationGroup(*destellos),
                Indicate(nodos_red[i], color=NARANJA_TERRACOTA, scale_factor=1.1),
                run_time=1.0
            )

        conclusion = Text(
            "GPT-2 tiene 124 millones de parámetros.\nBackprop calcula el gradiente de todos en un solo paso.",
            font=FUENTE, font_size=20, weight=BOLD,
            color=NARANJA_TERRACOTA, line_spacing=1.3
        ).next_to(nodos_red, DOWN, buff=0.55).set_x(0)
        self.play(Write(conclusion))
        self.play(Indicate(conclusion, color=ORO_VIEJO, scale_factor=1.05))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_adam(self):
        """Diapositiva: Slide Adam."""

        titulo_intro, linea_intro = self.crear_titulo(
            "Optimizador Adam", 
            palabra_clave="Adam", 
            color_clave=NARANJA_TERRACOTA
        )
        self.play(Write(titulo_intro), GrowFromCenter(linea_intro))

        formula_adam = MathTex(
            r"w_{t + 1} \leftarrow w_{t} - ", 
            r"\frac{\eta}{\sqrt{v_{t}} + \epsilon} ", 
            r"\cdot m_{t}", 
            color=TINTA_NEGRA, font_size=60
        ).shift(UP * 2)

        self.play(Write(formula_adam))
        self.next_slide()

        ejes_m = Axes(x_range=[0, 4], y_range=[0, 2], x_length=4, y_length=2).shift(LEFT*3.2 + DOWN*1.5)
        
        func_m = lambda x: 0.5 * np.cos(3*x) - 0.2 * x + 1.5
        terreno_m = ejes_m.plot(func_m, color=MARRON_OSCURO)
        
        lbl_m = Text("m: Momentum", font_size=24, color=NARANJA_TERRACOTA).next_to(ejes_m, UP)

        bola_gd = Dot(color=MARRON_OSCURO).move_to(ejes_m.c2p(0.5, func_m(0.5)))
        bola_adam = Dot(color=NARANJA_TERRACOTA).move_to(ejes_m.c2p(0.5, func_m(0.5)))

        self.play(
            formula_adam[1].animate.set_opacity(0.3),
            formula_adam[2].animate.set_color(NARANJA_TERRACOTA).scale(1.2),
            Create(ejes_m), Create(terreno_m), Write(lbl_m),
            FadeIn(bola_gd), FadeIn(bola_adam)
        )

        tracker_gd = ValueTracker(0.5)
        tracker_adam = ValueTracker(0.5)

        bola_gd.add_updater(lambda d: d.move_to(ejes_m.c2p(tracker_gd.get_value(), func_m(tracker_gd.get_value()))))
        bola_adam.add_updater(lambda d: d.move_to(ejes_m.c2p(tracker_adam.get_value(), func_m(tracker_adam.get_value()))))

        self.play(
            tracker_gd.animate.set_value(1.2), 
            tracker_adam.animate.set_value(3.2), 
            run_time=2.5, rate_func=smooth
        )
        
        bola_gd.clear_updaters()
        bola_adam.clear_updaters()
        self.next_slide()

        ejes_v = Axes(x_range=[0, 4], y_range=[0, 4], x_length=4, y_length=2).shift(RIGHT*3.2 + DOWN*1.5)
        func_v = lambda x: 0.8 * (x - 2.5)**2 + 0.5
        terreno_v = ejes_v.plot(func_v, color=MARRON_OSCURO)
        
        lbl_v = Text("v: Varianza", font_size=24, color=MARRON_OSCURO).next_to(ejes_v, UP)

        self.play(
            formula_adam[2].animate.set_color(TINTA_NEGRA).set_opacity(0.3).scale(1/1.2),
            formula_adam[1].animate.set_opacity(1).set_color(NARANJA_TERRACOTA).scale(1.2),
            Create(ejes_v), Create(terreno_v), Write(lbl_v)
        )

        x_inicio = 0.8
        punto_base = Dot(ejes_v.c2p(x_inicio, func_v(x_inicio)), color=TINTA_NEGRA)
        self.play(FadeIn(punto_base))

        x_sobreimpulso = 3.8
        flecha_mala = Arrow(
            start=ejes_v.c2p(x_inicio, func_v(x_inicio)),
            end=ejes_v.c2p(x_sobreimpulso, func_v(x_sobreimpulso)),
            color=MARRON_OSCURO, buff=0.1
        )
        lbl_mala = Text("Gradiente puro: Paso gigante", font_size=16, color=MARRON_OSCURO).next_to(flecha_mala, UP)
        
        self.play(GrowArrow(flecha_mala), Write(lbl_mala))
        self.wait(0.5)
        
        x_ideal = 2.0
        flecha_buena = Arrow(
            start=ejes_v.c2p(x_inicio, func_v(x_inicio)),
            end=ejes_v.c2p(x_ideal, func_v(x_ideal)),
            color=NARANJA_TERRACOTA, buff=0.1
        )
        lbl_buena = Text("ADAM adapta el paso: Frena en bajadas fuertes", font_size=16, color=NARANJA_TERRACOTA).next_to(flecha_buena, DOWN)

        self.play(
            ReplacementTransform(flecha_mala, flecha_buena),
            ReplacementTransform(lbl_mala, lbl_buena)
        )
        self.wait(0.5)

        tracker_v = ValueTracker(x_inicio)
        punto_v = Dot(color=NARANJA_TERRACOTA)
        punto_v.add_updater(lambda d: d.move_to(ejes_v.c2p(tracker_v.get_value(), func_v(tracker_v.get_value()))))
        
        self.add(punto_v)
        self.remove(punto_base)
        
        self.play(
            tracker_v.animate.set_value(x_ideal), 
            run_time=2, 
            rate_func=smooth
        )
        punto_v.clear_updaters()
        self.next_slide()

        self.play(FadeOut(VGroup(
            ejes_m, terreno_m, lbl_m, bola_gd, bola_adam, 
            ejes_v, terreno_v, lbl_v, flecha_buena, lbl_buena, punto_v, 
            formula_adam, titulo_intro, linea_intro
        )))

        titulo, linea = self.crear_titulo("ADAM vs GD: El Descenso Final", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        def func_costo(x):
            return np.cos(3*x)*0.5 + 0.2 * (x**2) + 2

        ejes_gd = Axes(x_range=[-3, 5, 1], y_range=[0, 8, 2], x_length=5.5, y_length=4, 
                       axis_config={"color": MARRON_OSCURO}).shift(LEFT * 3.4 + DOWN * 0.5)
        ejes_adam = Axes(x_range=[-3, 5, 1], y_range=[0, 8, 2], x_length=5.5, y_length=4, 
                         axis_config={"color": NARANJA_TERRACOTA}).shift(RIGHT * 3.4 + DOWN * 0.5)

        lbl_titulo_gd = Text("Gradient Descent Normal", font_size=24, color=MARRON_OSCURO).next_to(ejes_gd, UP)
        lbl_titulo_adam = Text("Optimizador ADAM", font_size=24, color=NARANJA_TERRACOTA).next_to(ejes_adam, UP)

        self.play(
            Create(VGroup(ejes_gd, ejes_gd.plot(func_costo, color=MARRON_OSCURO))),
            Create(VGroup(ejes_adam, ejes_adam.plot(func_costo, color=NARANJA_TERRACOTA))),
            Write(lbl_titulo_gd),
            Write(lbl_titulo_adam)
        )

        pasos_gd, pasos_adam = [4.0], [4.0]
        x_g, x_a = 4.0, 4.0
        m, v = 0, 0
        
        for t in range(1, 60):
            g_a = -np.sin(3*x_a)*1.5 + 0.4*x_a 
            m = 0.9 * m + 0.1 * g_a
            v = 0.999 * v + 0.001 * (g_a**2)
            m_hat = m / (1 - 0.9**t)
            v_hat = v / (1 - 0.999**t)
            
            if abs(x_a) < 0.05 and t > 25:
                x_a = 0.0
            else:
                x_a -= 0.35 * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            pasos_adam.append(x_a)
            g_g = -np.sin(3*x_g)*1.5 + 0.4*x_g
            x_g -= 0.12 * g_g
            pasos_gd.append(x_g)

        trail_gd_points = [ejes_gd.c2p(x, func_costo(x)) for x in pasos_gd]
        trail_adam_points = [ejes_adam.c2p(x, func_costo(x)) for x in pasos_adam]
        
        linea_rastro_gd = VMobject(color=MARRON_OSCURO).set_points_smoothly(trail_gd_points)
        linea_rastro_adam = VMobject(color=NARANJA_TERRACOTA).set_points_smoothly(trail_adam_points)
        
        punto_gd = Dot(trail_gd_points[0], color=MARRON_OSCURO)
        punto_adam = Dot(trail_adam_points[0], color=NARANJA_TERRACOTA)
        
        self.play(FadeIn(punto_gd), FadeIn(punto_adam))

        self.play(
            MoveAlongPath(punto_gd, linea_rastro_gd),
            Create(linea_rastro_gd),
            MoveAlongPath(punto_adam, linea_rastro_adam),
            Create(linea_rastro_adam),
            run_time=4, rate_func=smooth
        )

        lbl_resultado_gd = Text("Mínimo Local", font_size=20, color=MARRON_OSCURO).next_to(punto_gd, DOWN)
        lbl_resultado_adam = Text("Mínimo Global", font_size=20, color=NARANJA_TERRACOTA).next_to(punto_adam, DOWN)

        self.play(
            Write(lbl_resultado_gd),
            Write(lbl_resultado_adam),
            Flash(punto_adam, color=NARANJA_TERRACOTA)
        )

        self.next_slide()
        
        panel_textos = VGroup(
            Text("¿Por qué ganó ADAM?", font_size=26, color=TINTA_NEGRA, weight=BOLD),
            Text("• El Momentum saltó los baches locales.", font_size=20, color=MARRON_OSCURO),
            Text("• La Varianza controló los pasos bruscos.", font_size=20, color=MARRON_OSCURO)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        fondo_panel = Rectangle(
            width=panel_textos.width + 1, height=panel_textos.height + 0.7, 
            color=PAPEL_CREMA, fill_color=FONDO_CAJA, fill_opacity=1
        )
        panel_final = VGroup(fondo_panel, panel_textos).to_edge(DOWN, buff=0.2)

        self.play(FadeIn(panel_final, shift=UP))
        self.next_slide()
        self.limpiar_pantalla()

    def slide_residual_connections(self):
        """Diapositiva: Slide Residual Connections."""

        titulo, linea = self.crear_titulo("Conexiones Residuales: La Autopista", palabra_clave="Conexiones Residuales:", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        lbl_forward = Text("1. Forward Pass (Bifurcación)", font=FUENTE, font_size=18, color=TINTA_NEGRA, weight=BOLD)
        
        eq_forward = MathTex(r"y = x + \text{Capa}(x)", font_size=36, color=MARRON_OSCURO)
        desc_forward = Text("La entrada original 'x' se suma a la salida de la capa.", font=FUENTE, font_size=14, color=TINTA_NEGRA).set_opacity(0.8)
        
        grupo_forward = VGroup(lbl_forward, eq_forward, desc_forward).arrange(DOWN, buff=0.25)
        caja_forward = SurroundingRectangle(grupo_forward, color=MARRON_OSCURO, fill_color=FONDO_CAJA, fill_opacity=1, corner_radius=0.2, buff=0.25)
        
        bloque1_completo = VGroup(caja_forward, grupo_forward).next_to(linea, DOWN, buff=0.3)

        self.play(FadeIn(bloque1_completo, shift=DOWN))
        self.next_slide()

        lbl_backward = Text("2. Backward Pass (Acumulación)", font=FUENTE, font_size=18, color=TINTA_NEGRA, weight=BOLD)
        
        eq_back_izq = MathTex(r"\nabla x =", font_size=36, color=TINTA_NEGRA)
        eq_back_skip = MathTex(r"\nabla y", font_size=36, color=NARANJA_TERRACOTA)
        eq_back_plus = MathTex(r"+", font_size=36, color=TINTA_NEGRA)
        eq_back_capa = MathTex(r"\nabla x_{\text{capa}}", font_size=36, color=MARRON_OSCURO)
        
        grupo_eq_back = VGroup(eq_back_izq, eq_back_skip, eq_back_plus, eq_back_capa).arrange(RIGHT, buff=0.2)
        
        txt_skip = Text("Pasa intacto (¡La Autopista!)", font=FUENTE, font_size=14, color=NARANJA_TERRACOTA, weight=BOLD).next_to(eq_back_skip, DOWN, buff=0.6)
        txt_capa = Text("Gradiente transformado", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(eq_back_capa, DOWN, buff=1.0) 
        flecha_skip = Arrow(txt_skip.get_top(), eq_back_skip.get_bottom(), buff=0.1, color=NARANJA_TERRACOTA, stroke_width=3, max_tip_length_to_length_ratio=0.15)
        flecha_capa = Arrow(txt_capa.get_top(), eq_back_capa.get_bottom(), buff=0.1, color=MARRON_OSCURO, stroke_width=3, max_tip_length_to_length_ratio=0.15)

        grupo_textos_back = VGroup(lbl_backward, grupo_eq_back, txt_skip, txt_capa, flecha_skip, flecha_capa)
        
        grupo_eq_back.next_to(lbl_backward, DOWN, buff=0.4)
        txt_skip.next_to(eq_back_skip, DOWN, buff=0.7)
        txt_capa.next_to(eq_back_capa, DOWN, buff=1.1)
        flecha_skip.put_start_and_end_on(txt_skip.get_top() + UP*0.1, eq_back_skip.get_bottom() + DOWN*0.1)
        flecha_capa.put_start_and_end_on(txt_capa.get_top() + UP*0.1, eq_back_capa.get_bottom() + DOWN*0.1)

        caja_backward = SurroundingRectangle(grupo_textos_back, color=PAPEL_TAN, fill_color=PAPEL_CREMA, fill_opacity=1, corner_radius=0.2, buff=0.25)
        bloque2_completo = VGroup(caja_backward, grupo_textos_back).next_to(bloque1_completo, DOWN, buff=0.4)

        self.play(Create(caja_backward), FadeIn(lbl_backward, shift=UP))
        self.play(Write(eq_back_izq), Write(eq_back_plus))
        self.next_slide()

        self.play(FadeIn(eq_back_skip, shift=DOWN))
        self.play(GrowArrow(flecha_skip), FadeIn(txt_skip))
        self.next_slide()

        self.play(FadeIn(eq_back_capa, shift=DOWN))
        self.play(GrowArrow(flecha_capa), FadeIn(txt_capa))
        self.next_slide()

        lbl_impacto = Text("Sin esta suma directa, el gradiente desaparecería en redes profundas.", font=FUENTE, font_size=16, color=TINTA_NEGRA, weight=BOLD)
        caja_impacto = SurroundingRectangle(lbl_impacto, color=NARANJA_TERRACOTA, fill_color=CAJA_INFERIOR, fill_opacity=0.5, corner_radius=0.1, buff=0.2)
        bloque3_completo = VGroup(caja_impacto, lbl_impacto).next_to(bloque2_completo, DOWN, buff=0.5)

        self.play(FadeIn(bloque3_completo, shift=UP))
        self.play(Indicate(bloque3_completo, color=NARANJA_TERRACOTA))
        
        self.next_slide()
        self.limpiar_pantalla()
    
    def slide_training_techniques(self):
        """Diapositiva: Slide Training Techniques."""

        titulo, linea = self.crear_titulo("Estabilizadores y Optimización", palabra_clave="Estabilizadores", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        lbl_clip = Text("1. Gradient Clipping (Evitar Explosiones)", font=FUENTE, font_size=22, color=TINTA_NEGRA, weight=BOLD)
        
        eq_clip_izq = MathTex(r"g \leftarrow g \cdot", font_size=42, color=TINTA_NEGRA)
        eq_clip_der = MathTex(r"\min\left(1, \frac{\text{max\_norm}}{||g||_2}\right)", font_size=42, color=NARANJA_TERRACOTA)
        grupo_eq_clip = VGroup(eq_clip_izq, eq_clip_der).arrange(RIGHT, buff=0.2)
        
        nota_clip = Text("Si el gradiente es muy grande, reducimos su magnitud preservando su dirección.", font=FUENTE, font_size=14, color=MARRON_OSCURO)
        
        grupo_clip_total = VGroup(lbl_clip, grupo_eq_clip, nota_clip).arrange(DOWN, buff=0.6)
        caja_clip = SurroundingRectangle(grupo_clip_total, color=PAPEL_TAN, fill_color=FONDO_CAJA, fill_opacity=1, corner_radius=0.2, buff=0.4)
        bloque_clip = VGroup(caja_clip, grupo_clip_total).next_to(linea, DOWN, buff=1)

        self.play(FadeIn(caja_clip), FadeIn(lbl_clip, shift=DOWN))
        self.play(Write(eq_clip_izq), Write(eq_clip_der))
        self.play(FadeIn(nota_clip, shift=UP))
        self.next_slide()
        
        self.play(FadeOut(bloque_clip, shift=UP))

        lbl_drop = Text("2. Dropout (Prevenir Memorización)", font=FUENTE, font_size=22, color=TINTA_NEGRA, weight=BOLD)
        
        eq_drop = MathTex(r"y = x \odot M", font_size=42, color=MARRON_OSCURO)
        eq_bernoulli = MathTex(r"M \sim \text{Bernoulli}(1 - p)", font_size=32, color=NARANJA_TERRACOTA)
        grupo_eq_drop = VGroup(eq_drop, eq_bernoulli).arrange(DOWN, buff=0.3)
        
        nota_drop = Text("Apaga aleatoriamente el 10% de las neuronas (p=0.1) para forzar redundancia.", font=FUENTE, font_size=14, color=MARRON_OSCURO)

        grupo_drop_total = VGroup(lbl_drop, grupo_eq_drop, nota_drop).arrange(DOWN, buff=0.6)
        caja_drop = SurroundingRectangle(grupo_drop_total, color=PAPEL_TAN, fill_color=FONDO_CAJA, fill_opacity=1, corner_radius=0.2, buff=0.4)
        bloque_drop = VGroup(caja_drop, grupo_drop_total).next_to(linea, DOWN, buff=1)

        self.play(FadeIn(caja_drop), FadeIn(lbl_drop, shift=DOWN))
        self.play(Write(eq_drop))
        self.play(FadeIn(eq_bernoulli, shift=UP))
        self.play(FadeIn(nota_drop, shift=UP))
        self.next_slide()
        
        self.play(FadeOut(bloque_drop, shift=UP))

        lbl_adamw = Text("3. AdamW (Decoupled Weight Decay)", font=FUENTE, font_size=22, color=TINTA_NEGRA, weight=BOLD)
        
        txt_wd = Text("Paso 1: Decaimiento de Pesos", font=FUENTE, font_size=12, color=NARANJA_TERRACOTA)
        eq_wd = MathTex(r"\theta_{t} = \theta_{t-1}(1 - \eta \lambda)", font_size=34, color=NARANJA_TERRACOTA)
        grupo_wd = VGroup(txt_wd, eq_wd).arrange(DOWN, buff=0.1)

        txt_mv = Text("Paso 2: Promedios Móviles del Gradiente", font=FUENTE, font_size=12, color=MARRON_OSCURO)
        eq_mv = MathTex(r"m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad | \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2", font_size=28, color=MARRON_OSCURO)
        grupo_mv = VGroup(txt_mv, eq_mv).arrange(DOWN, buff=0.1)

        txt_update = Text("Paso 3: Actualización Final", font=FUENTE, font_size=12, color=TINTA_NEGRA)
        eq_update = MathTex(r"\theta_{t} \leftarrow \theta_{t} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}", font_size=38, color=TINTA_NEGRA)
        grupo_update = VGroup(txt_update, eq_update).arrange(DOWN, buff=0.1)

        grupo_eqs_adam = VGroup(grupo_wd, grupo_mv, grupo_update).arrange(DOWN, buff=0.4)
        
        nota_adamw = Text("A diferencia de Adam normal, la regularización NO se mezcla con el momentum.", font=FUENTE, font_size=14, color=NARANJA_TERRACOTA, weight=BOLD)

        grupo_adamw_total = VGroup(lbl_adamw, grupo_eqs_adam, nota_adamw).arrange(DOWN, buff=0.5)
        caja_adamw = SurroundingRectangle(grupo_adamw_total, color=MARRON_OSCURO, fill_color=PAPEL_CREMA, fill_opacity=1, corner_radius=0.2, buff=0.4)
        bloque_adamw = VGroup(caja_adamw, grupo_adamw_total).next_to(linea, DOWN, buff=0.3)

        self.play(FadeIn(caja_adamw), FadeIn(lbl_adamw, shift=DOWN))
        
        self.play(FadeIn(txt_wd), Write(eq_wd))
        self.next_slide()
        
        self.play(FadeIn(txt_mv), Write(eq_mv))
        self.next_slide()
        
        self.play(FadeIn(txt_update), Write(eq_update))
        self.play(FadeIn(nota_adamw, shift=UP))
        
        self.play(Indicate(grupo_wd, color=NARANJA_TERRACOTA))

        self.next_slide()
        self.limpiar_pantalla()

    def slide_training_metrics(self):
        """Diapositiva: Training Metrics — Loss y PPL explicados con intuición y evolución visual."""

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Métricas de Entrenamiento: Loss y Perplejidad",
            palabra_clave="Métricas",
            color_clave=NARANJA_TERRACOTA
        )
        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        # ── ACTO 1: QUÉ MIDE EL LOSS ─────────────────────────────────────────────
        # Intuición: el modelo predice la siguiente palabra → Loss mide qué tan equivocado está
        pregunta = Text("¿Cómo sabe el modelo si está aprendiendo?",
                        font=FUENTE, font_size=26, weight=BOLD, color=TINTA_NEGRA).move_to(UP * 1.6)
        self.play(Write(pregunta))
        self.next_slide()

        # Mostrar una predicción concreta con probabilidades
        contexto_txt = Text('"Esta que llaman por ahí Fortuna es una mujer..."',
                            font=FUENTE, font_size=22, color=MARRON_OSCURO).move_to(UP * 0.6)
        pregunta_sig = Text("¿Cuál es la siguiente palabra?",
                            font=FUENTE, font_size=20, color=PAPEL_TAN).next_to(contexto_txt, DOWN, buff=0.2)
        self.play(FadeIn(contexto_txt), Write(pregunta_sig))
        self.next_slide()

        # Distribución de probabilidades del modelo (mal vs bien entrenado)
        palabras_pred = ["borracha", "bella", "rica", "alta", "noble"]
        probs_mal  = [0.05, 0.10, 0.08, 0.12, 0.05]   # disperso → caos
        probs_bien = [0.72, 0.10, 0.07, 0.06, 0.05]   # concentrado → aprendido

        bar_w, bar_max_h = 0.55, 1.8
        base_y = DOWN * 1.3  # Bajamos la base de las barras para que no choquen con el texto

        def hacer_barras(probs, color_correcto, color_resto):
            grupo = VGroup()
            for i, (p, pal) in enumerate(zip(probs, palabras_pred)):
                h = p * bar_max_h / max(probs_bien)
                rect = Rectangle(width=bar_w, height=h,
                                 fill_color=color_correcto if i == 0 else color_resto,
                                 fill_opacity=0.85, stroke_width=1.2,
                                 stroke_color=MARRON_OSCURO)
                lbl_p = Text(f"{int(p*100)}%", font="Monospace", font_size=15,
                             color=TINTA_NEGRA).next_to(rect, UP, buff=0.08)
                lbl_w = Text(pal, font=FUENTE, font_size=15,
                             color=TINTA_NEGRA).next_to(rect, DOWN, buff=0.1)
                barra = VGroup(rect, lbl_p, lbl_w)
                barra.move_to(RIGHT * (i - 2) * 1.1 + base_y)
                # alinear base
                barra.shift(DOWN * (bar_max_h - h) / 2)
                grupo.add(barra)
            return grupo

        barras_mal  = hacer_barras(probs_mal,  ROJO_TOMATE,     PAPEL_TAN)
        barras_bien = hacer_barras(probs_bien, VERDE_OLIVA, PAPEL_TAN)

        lbl_mal  = Text("Modelo sin entrenar — Loss ≈ 8",
                        font=FUENTE, font_size=19, weight=BOLD, color=ROJO_TOMATE).next_to(barras_mal, DOWN, buff=0.5)
        lbl_bien = Text("Modelo entrenado — Loss ≈ 2.8",
                        font=FUENTE, font_size=19, weight=BOLD, color=VERDE_OLIVA).next_to(barras_bien, DOWN, buff=0.5)

        self.play(FadeOut(pregunta_sig))
        self.play(LaggedStart(*[FadeIn(b, shift=UP*0.2) for b in barras_mal], lag_ratio=0.1))
        self.play(Write(lbl_mal))
        self.next_slide()

        self.play(
            ReplacementTransform(barras_mal, barras_bien),
            ReplacementTransform(lbl_mal, lbl_bien)
        )
        self.next_slide()

        self.play(FadeOut(pregunta), FadeOut(contexto_txt), FadeOut(barras_bien), FadeOut(lbl_bien))

        # ── ACTO 2: LAS DOS MÉTRICAS LADO A LADO ─────────────────────────────────
        lbl_metricas = Text("Dos caras de la misma moneda",
                            font=FUENTE, font_size=26, weight=BOLD, color=TINTA_NEGRA).move_to(UP * 1.6)

        # Loss — lado izquierdo
        caja_loss = RoundedRectangle(corner_radius=0.15, width=4.8, height=3.2,
                                     fill_color=FONDO_CAJA, fill_opacity=1,
                                     stroke_color=MARRON_OSCURO, stroke_width=2).move_to(LEFT * 2.8 + DOWN * 0.4)
        tit_loss = Text("Cross-Entropy Loss", font=FUENTE, font_size=20,
                        weight=BOLD, color=MARRON_OSCURO).move_to(caja_loss.get_top() + DOWN * 0.35)
        eq_loss = MathTex(r"L = -\frac{1}{N}\sum_{i=1}^{N} \ln P(x_i)",
                          color=TINTA_NEGRA).scale(0.85).next_to(tit_loss, DOWN, buff=0.3)
        desc_loss = Text(
            "Probabilidad de asignar la palabra correcta",
            font=FUENTE, font_size=17, color=MARRON_OSCURO, line_spacing=1.3
        ).next_to(eq_loss, DOWN, buff=0.3)
        escala_loss = Text("0 = perfecto  ·  8+ = caos", font=FUENTE, font_size=16,
                           color=NARANJA_TERRACOTA, weight=BOLD).next_to(desc_loss, DOWN, buff=0.25)
        grupo_loss = VGroup(caja_loss, tit_loss, eq_loss, desc_loss, escala_loss)

        # PPL — lado derecho
        caja_ppl = RoundedRectangle(corner_radius=0.15, width=4.8, height=3.2,
                                    fill_color=FONDO_CAJA, fill_opacity=1,
                                    stroke_color=NARANJA_TERRACOTA, stroke_width=2).move_to(RIGHT * 2.8 + DOWN * 0.4)
        tit_ppl = Text("Perplejidad (PPL)", font=FUENTE, font_size=20,
                       weight=BOLD, color=NARANJA_TERRACOTA).move_to(caja_ppl.get_top() + DOWN * 0.35)
        eq_ppl = MathTex(r"PPL = e^{\,L}",
                         color=NARANJA_TERRACOTA).scale(1.1).next_to(tit_ppl, DOWN, buff=0.3)
        desc_ppl = Text(
            "¿Entre cuántas palabras\nduda el modelo?\n(branching factor)",
            font=FUENTE, font_size=17, color=MARRON_OSCURO, line_spacing=1.3
        ).next_to(eq_ppl, DOWN, buff=0.3)
        escala_ppl = Text("1 = perfecto  ·  3000+ = caos", font=FUENTE, font_size=16,
                          color=NARANJA_TERRACOTA, weight=BOLD).next_to(desc_ppl, DOWN, buff=0.25)
        grupo_ppl = VGroup(caja_ppl, tit_ppl, eq_ppl, desc_ppl, escala_ppl)

        self.play(Write(lbl_metricas))
        self.play(FadeIn(grupo_loss, shift=RIGHT * 0.3))
        self.next_slide()
        self.play(FadeIn(grupo_ppl, shift=LEFT * 0.3))
        self.next_slide()

        # Resaltar la relación entre ambas
        flecha_rel = CurvedArrow(caja_loss.get_right(), caja_ppl.get_left(),
                                 color=ORO_VIEJO, angle=-PI/4, stroke_width=3)
        lbl_rel = Text("e^L", font="Monospace", font_size=20,
                       color=ORO_VIEJO, weight=BOLD).move_to(flecha_rel.get_center() + UP * 0.3)
        self.play(Create(flecha_rel), Write(lbl_rel))
        self.next_slide()

        self.play(FadeOut(lbl_metricas), FadeOut(grupo_loss), FadeOut(grupo_ppl),
                  FadeOut(flecha_rel), FadeOut(lbl_rel))

        # ── ACTO 3: CURVA DE LOSS EN EL TIEMPO ───────────────────────────────────
        lbl_curva = Text("Así cae el Loss durante el entrenamiento",
                         font=FUENTE, font_size=24, weight=BOLD, color=TINTA_NEGRA).move_to(UP * 1.8)

        ax = Axes(
            x_range=[0, 16000, 4000],
            y_range=[0, 9, 3],
            x_length=7.5,
            y_length=3.8,
            axis_config={"include_tip": False, "color": MARRON_OSCURO},
        ).move_to(DOWN * 0.4)  # Centrado un poco más abajo para no chocar con el título

        x_label = Text("Pasos de entrenamiento", font=FUENTE, font_size=15,
                       color=MARRON_OSCURO).next_to(ax, DOWN, buff=0.2)
        y_label = Text("Loss", font=FUENTE, font_size=15,
                       color=MARRON_OSCURO).next_to(ax, LEFT, buff=0.2)

        # Curva de decaimiento realista: caída rápida al inicio, luego suaviza
        def curva_loss(t):
            return 2.85 + 5.27 * np.exp(-t / 4200)

        curva = ax.plot(curva_loss, x_range=[0, 16000], color=NARANJA_TERRACOTA, stroke_width=4)

        # Puntos clave con sus valores
        puntos_datos = [(0, 8.12, "3 360"), (4000, 4.45, "85"), (16000, 2.85, "17")]
        dots = VGroup()
        anotaciones = VGroup()
        for paso, loss_v, ppl_v in puntos_datos:
            y_v = curva_loss(paso)
            d = Dot(ax.c2p(paso, y_v), radius=0.1,
                    color=VERDE_OLIVA if paso == 16000 else MARRON_OSCURO)
            nota = Text(f"Loss {loss_v}\nPPL {ppl_v}",
                        font="Monospace", font_size=14,
                        color=VERDE_OLIVA if paso == 16000 else MARRON_OSCURO)
            nota.next_to(d, UR if paso < 10000 else UL, buff=0.15)
            dots.add(d)
            anotaciones.add(nota)

        self.play(Write(lbl_curva))
        self.play(Create(ax), Write(x_label), Write(y_label))
        self.play(Create(curva), run_time=2.5)
        self.play(LaggedStart(
            *[AnimationGroup(FadeIn(dots[i]), Write(anotaciones[i])) for i in range(3)],
            lag_ratio=0.4
        ))
        self.next_slide()

        self.play(FadeOut(lbl_curva), FadeOut(ax), FadeOut(x_label), FadeOut(y_label),
                  FadeOut(curva), FadeOut(dots), FadeOut(anotaciones))

        # ── ACTO 4: LA EVOLUCIÓN EN EL TEXTO GENERADO ────────────────────────────
        lbl_evol = Text("Lo que ve el Loss, tú lo puedes leer",
                        font=FUENTE, font_size=24, weight=BOLD, color=TINTA_NEGRA).move_to(UP * 1.5)
        self.play(Write(lbl_evol))

        pasos_info = [
            ("0",      "8.12", "3 360", ROJO_TOMATE,
             '"Esta q7e llama8 p0r a#í F0rtun4\nes una muj8r..."'),
            ("4 000",  "4.45", "85",    PAPEL_TAN,
             '"Esta que llaman por ahí Fortuna\nes una mujer..."'),
            ("16 000", "2.85", "17",    VERDE_OLIVA,
             '"Esta que llaman por ahí Fortuna\nes una mujer borracha y antojadiza."'),
        ]

        def hacer_burbuja(paso, loss_v, ppl_v, color, texto):
            rect = RoundedRectangle(corner_radius=0.18, height=2.0, width=9.5,
                                    fill_color=PAPEL_CREMA, fill_opacity=1,
                                    stroke_color=color, stroke_width=2.5)
            icono_circ = Circle(radius=0.28, fill_color=color,
                                fill_opacity=1, stroke_width=0)
            icono_lbl = Text("M", font=FUENTE, font_size=21,
                             color=BLANCO, weight=BOLD).move_to(icono_circ)
            icono = VGroup(icono_circ, icono_lbl).next_to(rect, LEFT, buff=0.25).shift(UP * 0.4)

            contenido = Paragraph(texto, font=FUENTE, font_size=21,
                                  color=TINTA_NEGRA, line_spacing=1.3, alignment="left")
            contenido.scale_to_fit_width(rect.width - 1.0).move_to(rect)

            barra_info = Text(
                f"Paso {paso}   Loss {loss_v}   PPL {ppl_v}",
                font="Monospace", font_size=15, color=color
            ).next_to(rect, DOWN, buff=0.12, aligned_edge=RIGHT)

            return VGroup(rect, icono, contenido, barra_info).center().move_to(DOWN * 0.5)

        burbujas = [hacer_burbuja(*d) for d in pasos_info]

        actual = burbujas[0]
        self.play(FadeIn(actual, scale=0.92))
        self.next_slide()

        for sig in burbujas[1:]:
            self.play(FadeTransform(actual, sig), run_time=1.4)
            actual = sig
            self.next_slide()

        self.limpiar_pantalla()

    def slide_model_in_action(self):
        """Diapositiva: Slide Model In Action."""

        titulo, linea = self.crear_titulo("Molinete AI: Demostración en Vivo", palabra_clave="Vivo", color_clave=NARANJA_TERRACOTA)
        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, fondo=llanuras_fondo)

        terminal_window = RoundedRectangle(corner_radius=0.2, width=11, height=6, color=MARRON_OSCURO, fill_color=FONDO_CAJA, fill_opacity=0.95)
        terminal_header = Rectangle(width=11, height=0.5, color=MARRON_OSCURO, fill_color=MARRON_OSCURO, fill_opacity=1).align_to(terminal_window, UP)
        
        dot_1 = Circle(radius=0.08, color=ROJO_MAC, fill_color=ROJO_MAC, fill_opacity=1).move_to(terminal_header.get_left() + RIGHT*0.4)
        dot_2 = Circle(radius=0.08, color=AMARILLO_MAC, fill_color=AMARILLO_MAC, fill_opacity=1).next_to(dot_1, RIGHT, buff=0.15)
        dot_3 = Circle(radius=0.08, color=VERDE_MAC, fill_color=VERDE_MAC, fill_opacity=1).next_to(dot_2, RIGHT, buff=0.15)
        

        terminal_title = Text("demostración en vivo", font_size=14, color=FONDO_CAJA, weight=BOLD).move_to(terminal_header) 
        
        terminal_ui = VGroup(terminal_window, terminal_header, dot_1, dot_2, dot_3, terminal_title)
        terminal_ui.next_to(linea, DOWN, buff=0.4)

        self.play(FadeIn(terminal_ui, shift=UP))

        start_pos = terminal_header.get_bottom() + DOWN*0.4 + LEFT*5.2

        comando = Text("$ cargo run --release", font_size=16, font="Monospace", color=TINTA_NEGRA).move_to(start_pos, aligned_edge=LEFT)
        
        barra_vacia = Text("Cargando el modelo: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%", font_size=16, font="Monospace", color=TINTA_NEGRA).next_to(comando, DOWN, buff=0.3, aligned_edge=LEFT)

        barra_llena = Text("Cargando el modelo: [████████████████████████████████████████] 100%", font_size=16, font="Monospace", color=NARANJA_TERRACOTA).next_to(comando, DOWN, buff=0.3, aligned_edge=LEFT)

        msg_listo = Text("¡Modelo cargado! Ya puedes chatear.", font_size=16, font="Monospace", color=TINTA_NEGRA).next_to(barra_llena, DOWN, buff=0.3, aligned_edge=LEFT)
        
        prompt_user = Text("Tú: En un lugar de la Mancha", font_size=16, font="Monospace", color=NARANJA_TERRACOTA).next_to(msg_listo, DOWN, buff=0.5, aligned_edge=LEFT)
        prompt_model = Text("Molinete: , de cuyo nombre no quiero acordarme, no ha mucho...", font_size=16, font="Monospace", color=TINTA_NEGRA).next_to(prompt_user, DOWN, buff=0.3, aligned_edge=LEFT)

        self.play(Write(comando), run_time=1)
        self.wait(0.3)
        
        self.play(Write(barra_vacia), run_time=0.5)
        self.wait(0.2)
        self.play(Transform(barra_vacia, barra_llena), run_time=2) 
        self.wait(0.3)
        
        self.play(FadeIn(msg_listo, shift=UP*0.1))
        self.wait(0.5)
        
        self.play(AddTextLetterByLetter(prompt_user), run_time=1.5)
        self.wait(0.5)
        
        self.play(AddTextLetterByLetter(prompt_model), run_time=3.5)
        
        self.wait(1)
        self.next_slide()

        transicion_lbl = Text("Cambiando al Demo", font_size=24, color=NARANJA_TERRACOTA, weight=BOLD)
        caja_transicion = SurroundingRectangle(transicion_lbl, color=MARRON_OSCURO, fill_color=FONDO_CAJA, fill_opacity=1, buff=0.4)
        grupo_transicion = VGroup(caja_transicion, transicion_lbl).move_to(terminal_window.get_center())

        contenido_terminal = VGroup(comando, barra_vacia, msg_listo, prompt_user, prompt_model)
        
        self.play(contenido_terminal.animate.set_opacity(0.1), FadeIn(grupo_transicion, scale=0.8))
        
        self.next_slide()
        self.limpiar_pantalla()

    def slide_final(self):
        """Diapositiva Final — sin sol, layout mejorado y elementos inferiores más abajo."""

        # ==========================================
        # 1. FUNCIONES AUXILIARES DE CONSTRUCCIÓN
        # ==========================================
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
            # Cuerpo
            base = Polygon([-0.85, -1.5, 0], [0.85, -1.5, 0], [0.52, 1.0, 0], [-0.52, 1.0, 0],
                           color=LADRILLO, fill_opacity=1, stroke_color=MADERA_OSCURA, stroke_width=2)
            puerta = RoundedRectangle(corner_radius=0.18, width=0.55, height=0.75,
                                      color=MADERA_OSCURA, fill_color=MADERA_CLARA, fill_opacity=1, stroke_width=2).move_to(base.get_bottom() + UP*0.38)
            ventana = Circle(radius=0.16, color=MADERA_OSCURA, fill_color=AZUL_NOCHE,
                             fill_opacity=0.8, stroke_width=2).move_to(base.get_center() + UP*0.32)
            techo = Polygon([-0.6, 1.0, 0], [0, 1.85, 0], [0.6, 1.0, 0],
                            color=TEJA, fill_opacity=1, stroke_color=MADERA_OSCURA, stroke_width=2)
            cuerpo = VGroup(base, puerta, ventana, techo)

            # Aspas
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

            # Bajamos el molino a DOWN * 1.8 (arrastrará a los personajes con él)
            molino = VGroup(cuerpo, aspas, eje).scale(0.7).to_edge(LEFT, buff=0.8).shift(DOWN * 1.8)
            return cuerpo, aspas, eje, aspas.get_center()

        def construir_qr():
            qr_real = ImageMobject(r"assets\qr_github_molineteai.png").scale(0.85)
            fondo_qr = RoundedRectangle(corner_radius=0.25, width=qr_real.width + 0.5, height=qr_real.height + 0.5,
                                        color=NARANJA_TERRACOTA, stroke_width=4, fill_color=PAPEL_CREMA, fill_opacity=1)
            estr_l = estrella_deco(fondo_qr.get_corner(UL) + RIGHT*0.28 + DOWN*0.28)
            estr_r = estrella_deco(fondo_qr.get_corner(UR) + LEFT*0.28  + DOWN*0.28)
            
            # También bajé el QR a 1.2 para equilibrar visualmente
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


        # ==========================================
        # 2. INSTANCIACIÓN DE OBJETOS
        # ==========================================
        marco_ext, marco_int, estrellas_esq = construir_marco_y_estrellas()
        gracias, linea_deco, sub, estrellas_tit = construir_textos()
        cuerpo_molino, aspas, eje, centro_giro_aspas = construir_molino()
        fondo_qr, qr_img, estr_qr_l, estr_qr_r, lbl_qr, url_lbl = construir_qr()
        creditos = construir_creditos()
        
        # Personajes anclados. Al bajar el molino, ellos bajan.
        quijote = crear_rust_quijote().scale(0.85).next_to(cuerpo_molino, RIGHT, buff=0.6, aligned_edge=DOWN)
        sancho = crear_rust_sancho().scale(0.85).next_to(quijote, RIGHT, buff=0.4, aligned_edge=DOWN)


        # ==========================================
        # 3. LÍNEA DE TIEMPO (ANIMACIONES)
        # ==========================================
        
        self.add(crear_llanuras_manchegas())

        # Marco
        self.play(Create(marco_ext), Create(marco_int), run_time=1.0)
        self.play(LaggedStart(*[GrowFromCenter(e) for e in estrellas_esq], lag_ratio=0.2), run_time=0.8)

        # Textos de agradecimiento
        self.play(Write(gracias), run_time=1.0)
        self.play(Create(linea_deco), FadeIn(sub, shift=UP*0.2))
        self.play(LaggedStart(*[GrowFromCenter(s) for s in estrellas_tit], lag_ratio=0.1), run_time=0.9)

        # Créditos 
        self.play(LaggedStart(*[FadeIn(c, shift=UP*0.15) for c in creditos], lag_ratio=0.25), run_time=1.0)

        # Aparición simultánea: Molino + Quijote + Sancho
        self.play(
            FadeIn(cuerpo_molino, shift=DOWN*0.5),
            GrowFromCenter(aspas),
            FadeIn(eje),
            FadeIn(quijote, shift=DOWN*0.2),
            FadeIn(sancho, shift=DOWN*0.2),
            run_time=1.2
        )

        # Aparición QR
        self.play(
            DrawBorderThenFill(fondo_qr), FadeIn(qr_img),
            GrowFromCenter(estr_qr_l), GrowFromCenter(estr_qr_r),
            Write(lbl_qr), FadeIn(url_lbl, shift=UP*0.15),
            run_time=1.4
        )

        # Bucle final
        self.play(Rotate(aspas, angle=2*PI*4, about_point=centro_giro_aspas, run_time=10, rate_func=linear))

    def slide_por_que_rust(self):
        """Diapositiva: Slide Por Que Rust."""

        titulo, linea = self.crear_titulo("¿Por qué Rust?", palabra_clave="Rust", color_clave=NARANJA_TERRACOTA)
        adornos = self._crear_adornos_esquinas()

        logo_py = ImageMobject("assets/logo_python.png").set_height(2)
        py_txt1 = Text("Fácil prototipado", font=FUENTE, font_size=24, color=MARRON_OSCURO)
        py_txt2 = Text("✗ Pausas por GC", font=FUENTE, font_size=24, color=ROJO_CONTRA)
        py_txt3 = Text("✗ Lento en inferencia", font=FUENTE, font_size=24, color=ROJO_CONTRA)
        py_grupo = VGroup(py_txt1, py_txt2, py_txt3).arrange(DOWN, buff=0.3)
        bloque_py = Group(logo_py, py_grupo).arrange(DOWN, buff=0.5)

        logo_cpp = ImageMobject("assets/logo_cpp.png").set_height(2)
        cpp_txt1 = Text("Máximo rendimiento", font=FUENTE, font_size=24, color=MARRON_OSCURO)
        cpp_txt2 = Text("✗ Ya muy visto en AI", font=FUENTE, font_size=24, color=ROJO_CONTRA)
        cpp_txt3 = Text("✗ Riesgo en memoria", font=FUENTE, font_size=24, color=ROJO_CONTRA)
        cpp_grupo = VGroup(cpp_txt1, cpp_txt2, cpp_txt3).arrange(DOWN, buff=0.3)
        bloque_cpp = Group(logo_cpp, cpp_grupo).arrange(DOWN, buff=0.5)

        logo_rust = ImageMobject("assets/logo_rust.png").set_height(2.5)
        rust_txt1 = Text("✓ Rendimiento C++", font=FUENTE, font_size=26, color=TINTA_NEGRA)
        rust_txt2 = Text("✓ Memoria segura", font=FUENTE, font_size=26, color=TINTA_NEGRA)
        rust_txt3 = Text("✓ No tan visto en AI", font=FUENTE, font_size=26, color=TINTA_NEGRA)
        rust_grupo = VGroup(rust_txt1, rust_txt2, rust_txt3).arrange(DOWN, buff=0.3)
        bloque_rust = Group(logo_rust, rust_grupo).arrange(DOWN, buff=0.5)

        contenido_principal = Group(bloque_py, bloque_cpp, bloque_rust).arrange(RIGHT, buff=1.0)
        contenido_principal.next_to(linea, DOWN, buff=0.5)

        if contenido_principal.width > 12.0:
            contenido_principal.width = 12.0
            contenido_principal.next_to(linea, DOWN, buff=0.5)

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=llanuras_fondo)
        self.next_slide()

        self.play(FadeIn(logo_py, shift=DOWN * 0.5))
        self.play(FadeIn(py_txt1, shift=UP * 0.2))
        self.next_slide()
        self.play(FadeIn(py_txt2), FadeIn(py_txt3))
        self.play(Wiggle(py_txt2), Wiggle(py_txt3))
        self.next_slide()
        
        self.play(logo_py.animate.set_opacity(0.2), py_grupo.animate.set_opacity(0.2))

        self.play(FadeIn(logo_cpp, shift=DOWN * 0.5))
        self.play(FadeIn(cpp_txt1, shift=UP * 0.2))
        self.next_slide()
        self.play(FadeIn(cpp_txt2), FadeIn(cpp_txt3))
        self.play(Wiggle(cpp_txt2), Wiggle(cpp_txt3))
        self.next_slide()

        self.play(logo_cpp.animate.set_opacity(0.2), cpp_grupo.animate.set_opacity(0.2))

        logo_rust.generate_target()
        logo_rust.rotate(-math.pi / 2).scale(0.5)
        self.play(
            FadeIn(logo_rust),
            MoveToTarget(logo_rust),
            run_time=1.2
        )
        self.next_slide()

        self.play(
            LaggedStart(
                FadeIn(rust_txt1, shift=RIGHT * 0.3),
                FadeIn(rust_txt2, shift=RIGHT * 0.3),
                FadeIn(rust_txt3, shift=RIGHT * 0.3),
                lag_ratio=0.3
            ),
            run_time=1.5
        )
        self.next_slide()

        caja_rust = RoundedRectangle(
            corner_radius=0.2,
            width=bloque_rust.width + 0.8,
            height=bloque_rust.height + 0.8,
            stroke_color=NARANJA_TERRACOTA,
            stroke_width=4,
            fill_color=FONDO_CAJA,
            fill_opacity=0.3
        ).move_to(bloque_rust)

        self.play(Create(caja_rust))
        self.play(
            rust_txt1.animate.set_color(NARANJA_TERRACOTA),
            rust_txt2.animate.set_color(NARANJA_TERRACOTA),
            rust_txt3.animate.set_color(NARANJA_TERRACOTA),
            Indicate(logo_rust, color=NARANJA_TERRACOTA, scale_factor=1.1)
        )
        self.next_slide()

        self.limpiar_pantalla()

    def diapo_codigo(self, codigo_fuente: str, titulo_archivo: str = "codigo.rs"):
        """Muestra un bloque de código Rust en un editor estilizado tipo macOS."""
        from pygments.style import Style
        from pygments.token import Keyword, Name, String, Number, Operator, Punctuation, Token, Comment

        class EstiloCervantino(Style):
            background_color = FONDO_CAJA
            styles = {
                Token: TINTA_NEGRA,             
                Keyword: f'bold {NARANJA_TERRACOTA}',
                Keyword.Type: f'bold {MARRON_OSCURO}', 
                String: MARRON_OSCURO,            
                Number: PAPEL_TAN,            
                Name.Function: NARANJA_TERRACOTA,     
                Operator: TINTA_NEGRA,          
                Punctuation: TINTA_NEGRA,   
                Comment: f'italic {CAJA_INFERIOR}',  
            }

        bloque_codigo = Code(
            code_string=codigo_fuente,
            language="rust",
            formatter_style=EstiloCervantino, 
            background="rectangle"
        ).scale(0.8)

        max_alto = 6.0 
        max_ancho = 12.0
        
        if bloque_codigo.height > max_alto:
            bloque_codigo.scale_to_fit_height(max_alto)
        if bloque_codigo.width > max_ancho:
            bloque_codigo.scale_to_fit_width(max_ancho)

        if len(bloque_codigo) > 0:
            bloque_codigo[0].set_opacity(0)
            
        if len(bloque_codigo) > 1:
            bloque_codigo[1].set_color(PAPEL_TAN)

        alto_header = 0.6
        padding_x = 1.2
        padding_y = 1.0

        ancho_caja = max(bloque_codigo.width + padding_x, 6.0)
        alto_caja = bloque_codigo.height + padding_y + alto_header

        sombra = RoundedRectangle(
            corner_radius=0.1,
            width=ancho_caja, height=alto_caja,
            color=BLACK,
            fill_color=BLACK,
            fill_opacity=0.25,
            stroke_width=0
        ).shift(DOWN * 0.15 + RIGHT * 0.15)

        editor_bg = RoundedRectangle(
            corner_radius=0.1, 
            width=ancho_caja, height=alto_caja,
            color=MARRON_OSCURO,
            stroke_width=3,
            fill_color=FONDO_CAJA,
            fill_opacity=1,
        )

        editor_header = Rectangle(
            width=ancho_caja, height=alto_header,
            color=MARRON_OSCURO,
            stroke_width=3,
            fill_color=MARRON_OSCURO,
            fill_opacity=1,
        ).align_to(editor_bg, UP)

        dot_1 = Circle(radius=0.08, color=TINTA_NEGRA, fill_color=CAJA_INFERIOR, fill_opacity=1, stroke_width=1.5)
        dot_2 = Circle(radius=0.08, color=TINTA_NEGRA, fill_color=CAJA_INFERIOR, fill_opacity=1, stroke_width=1.5)
        dot_3 = Circle(radius=0.08, color=TINTA_NEGRA, fill_color=CAJA_INFERIOR, fill_opacity=1, stroke_width=1.5)
        
        botones = VGroup(dot_1, dot_2, dot_3).arrange(RIGHT, buff=0.2)
        botones.move_to(editor_header.get_left() + RIGHT * 0.5)

        file_title = Text(
            titulo_archivo, font="Times New Roman", font_size=20, color=PAPEL_CREMA, weight=BOLD
        ).move_to(editor_header)

        editor_ui = VGroup(sombra, editor_bg, editor_header, botones, file_title)
        editor_ui.move_to(ORIGIN)
        
        area_util = editor_bg.get_center() + DOWN * (alto_header / 2)
        bloque_codigo.move_to(area_util)

        self.play(
            FadeIn(sombra, shift=UP * 0.1),
            DrawBorderThenFill(editor_bg),
            run_time=0.8
        )

        self.play(
            FadeIn(editor_header, shift=DOWN * 0.2),
            FadeIn(botones, shift=RIGHT * 0.1),
            Write(file_title),
            run_time=0.6
        )
        self.next_slide()

        animaciones_lineas = []
        for numero, linea in zip(bloque_codigo[1], bloque_codigo[2]):
            animaciones_lineas.append(FadeIn(VGroup(numero, linea), shift=UP * 0.15, scale=0.95))

        tiempo_animacion_codigo = max(1.5, len(animaciones_lineas) * 0.08)
        self.play(LaggedStart(*animaciones_lineas, lag_ratio=0.1), run_time=tiempo_animacion_codigo)
        
        self.next_slide()

        self.play(FadeOut(VGroup(editor_ui, bloque_codigo), shift=DOWN * 0.3), run_time=0.8)

    def slide_matmul(self):
        """Diapositiva: Slide Matmul."""

        titulo, linea = self.crear_titulo(
            "Multiplicación: El 'Dot Product'", 
            palabra_clave="'Dot Product'", 
            color_clave=NARANJA_TERRACOTA
        )

        camino_mancha = FunctionGraph(lambda x: 0.5 * math.sin(x) - 0.5, color=MARRON_OSCURO).set_opacity(0.15)
        camino_punteado = DashedVMobject(camino_mancha, num_dashes=45, dashed_ratio=0.5)
        
        lanza_fondo = Line(LEFT * 7 + DOWN * 2, RIGHT * 7 + UP * 2, color=NARANJA_TERRACOTA, stroke_width=2).set_opacity(0.15)
        
        decoracion_quijote = VGroup(camino_punteado, lanza_fondo).set_z_index(-2)

        adornos = self._crear_adornos_esquinas()
        adornos[1].add_updater(lambda m, dt: m.rotate(dt * 0.15))

        llanuras_fondo = crear_llanuras_manchegas()
        self._animar_entrada_slide(titulo, linea, adornos=adornos, fondo=VGroup(llanuras_fondo, decoracion_quijote))

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
        ).to_edge(DOWN, buff=0.6)
        
        self.play(Write(def_matmul))
        self.next_slide()
        
        self.limpiar_pantalla()

    def slide_temperature(self):
        """Diapositiva: Slide Temperature."""

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "La Temperatura: Controlando la Locura", 
            palabra_clave="Temperatura", 
            color_clave=NARANJA_TERRACOTA
        )
        
        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        label_eq = Text("Softmax con Temperatura (T)", font=FUENTE, font_size=18, color=MARRON_OSCURO)
        
        eq_temp = MathTex(
            r"P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}", 
            color=TINTA_NEGRA
        ).scale(1.3)
        
        eq_temp.set_color_by_tex("T", NARANJA_TERRACOTA)
        
        math_group = VGroup(label_eq, eq_temp).arrange(DOWN, buff=0.3).move_to(ORIGIN).shift(UP * 0.5)

        self.play(FadeIn(label_eq, shift=DOWN*0.2), Write(eq_temp), run_time=1.5)
        self.play(eq_temp.animate.scale(1.1).set_glow(0.3), rate_func=there_and_back, run_time=1)
        self.next_slide()

        explicacion = Tex(
            r"$T \to 0$: Determinista, conservador (Como Sancho)\\$T > 1$: Aleatorio, creativo, ``alucinaciones'' (Como el Quijote)",
            font_size=28, color=MARRON_OSCURO, tex_environment="flushleft"
        ).next_to(math_group, DOWN, buff=0.8)

        self.play(FadeIn(explicacion, shift=UP))
        self.next_slide()

        self.play(FadeOut(math_group, explicacion))

        rect_prompt = RoundedRectangle(corner_radius=0.15, height=1.2, width=8)
        rect_prompt.set_fill(color=MARRON_OSCURO, opacity=0.1).set_stroke(color=MARRON_OSCURO, width=1.5)
        
        user_label = Text("Usuario", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(rect_prompt, UP, aligned_edge=LEFT).shift(DOWN*0.1 + RIGHT*0.2)
        
        texto_prompt = Text(
            "Prompt: \"En un lugar de la Mancha...\"", 
            font=FUENTE, font_size=22, color=TINTA_NEGRA, weight=BOLD
        ).move_to(rect_prompt)

        grupo_prompt = VGroup(rect_prompt, user_label, texto_prompt).to_edge(UP, buff=1.5)
        
        self.play(FadeIn(grupo_prompt, shift=DOWN))

        def crear_respuesta(temp_val, intento, texto, color_perfil, titulo_perfil):
            sombra = RoundedRectangle(corner_radius=0.15, height=2.2, width=8)
            sombra.set_fill(MARRON_OSCURO, opacity=0.1).set_stroke(width=0)
            sombra.shift(RIGHT * 0.08 + DOWN * 0.08)
            
            rect = RoundedRectangle(corner_radius=0.15, height=2.2, width=8)
            rect.set_fill(color=PAPEL_CREMA, opacity=1).set_stroke(color=MARRON_OSCURO, width=1.5)
            
            icon = Circle(radius=0.25, color=color_perfil, fill_opacity=1)
            label_icon = Text(titulo_perfil[5], font=FUENTE, font_size=20, color=WHITE, weight=BOLD).move_to(icon)
            user_icon = VGroup(icon, label_icon).next_to(rect, LEFT, buff=0.3).shift(UP * 0.5)
            
            username = Text(f"{titulo_perfil} (T={temp_val})", font=FUENTE, font_size=14, color=MARRON_OSCURO).next_to(rect, UP, aligned_edge=LEFT).shift(UP*0.1)

            contenido = Paragraph(
                texto, font=FUENTE, font_size=22, color=TINTA_NEGRA, 
                line_spacing=1.3, alignment="left"
            ).scale_to_fit_width(rect.width - 0.8).move_to(rect)

            bubble_group = VGroup(sombra, rect, user_icon, username, contenido)
            
            info = Text(f"Generación - Intento #{intento}", 
                        font="Monospace", font_size=16, color=color_perfil).next_to(rect, DOWN, buff=0.15, aligned_edge=RIGHT)
            
            return VGroup(bubble_group, info).next_to(grupo_prompt, DOWN, buff=1.0)

        r_sancho_1 = crear_respuesta("0.1", 1, "\"...de cuyo nombre no quiero acordarme.\"", MARRON_OSCURO, "Sancho")
        r_sancho_2 = crear_respuesta("0.1", 2, "\"...de cuyo nombre no quiero acordarme.\"", MARRON_OSCURO, "Sancho")
        
        r_quijote_1 = crear_respuesta("1.5", 1, "\"...donde los dragones mecánicos beben aceite de oliva.\"", NARANJA_TERRACOTA, "El Quijote")
        r_quijote_2 = crear_respuesta("1.5", 2, "\"...los molinos me hablan en código binario al amanecer.\"", NARANJA_TERRACOTA, "El Quijote")

        actual = r_sancho_1
        self.play(FadeIn(actual, shift=UP))
        self.next_slide()

        self.play(FadeTransform(actual, r_sancho_2), run_time=1)
        actual = r_sancho_2
        self.next_slide()

        self.play(FadeTransform(actual, r_quijote_1), run_time=1.5)
        actual = r_quijote_1
        self.next_slide()

        self.play(FadeTransform(actual, r_quijote_2), run_time=1)
        self.next_slide()

        self.limpiar_pantalla()

    def slide_residual(self):
        """Diapositiva: Slide Residual."""

        escala = 0.85 

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Conexiones Residuales: El Atajo de Sancho", 
            palabra_clave="Residuales", 
            color_clave=NARANJA_TERRACOTA
        )
        
        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        pos_x = LEFT * 4
        pos_f = LEFT * 0.5
        pos_add = RIGHT * 2.5
        pos_y = RIGHT * 4.5

        label_x = MathTex("x", font_size=48 * escala, color=TINTA_NEGRA).move_to(pos_x)
        
        nodo_f = RoundedRectangle(corner_radius=0.2, width=1.5 * escala, height=2.2 * escala, fill_color=MARRON_QUIJOTE, fill_opacity=1, stroke_color=WHITE, stroke_width=3)
        label_f = MathTex("f", font_size=48 * escala, color=WHITE).move_to(nodo_f)
        grupo_f = VGroup(nodo_f, label_f).move_to(pos_f)
        
        nodo_add = Circle(radius=0.4 * escala, fill_color=MARRON_QUIJOTE, fill_opacity=1, stroke_color=WHITE, stroke_width=3)
        label_add = MathTex("+", font_size=40 * escala, color=WHITE).move_to(nodo_add)
        grupo_add = VGroup(nodo_add, label_add).move_to(pos_add)
        
        label_y = MathTex("y", font_size=48 * escala, color=TINTA_NEGRA).move_to(pos_y)
        
        eq_final = MathTex("y = x + f(x)", font_size=42 * escala, color=TINTA_NEGRA).next_to(grupo_f, DOWN, buff=1.2)

        arrow_x_f = Arrow(label_x.get_right(), grupo_f.get_left(), buff=0.2, color=MARRON_OSCURO, stroke_width=3)
        arrow_f_add = Arrow(grupo_f.get_right(), grupo_add.get_left(), buff=0.2, color=MARRON_OSCURO, stroke_width=3)
        arrow_add_y = Arrow(grupo_add.get_right(), label_y.get_left(), buff=0.2, color=MARRON_OSCURO, stroke_width=3)

        punto_inicio_skip = pos_x + RIGHT * 0.8
        p1 = punto_inicio_skip
        p2 = p1 + UP * 2.2
        p3 = np.array([grupo_add.get_center()[0], p2[1], 0])
        p4 = grupo_add.get_top() + UP * 0.1
        
        line_up = Line(p1, p2, color=MARRON_OSCURO, stroke_width=3)
        line_across = Line(p2, p3, color=MARRON_OSCURO, stroke_width=3)
        arrow_down = Arrow(p3, p4, buff=0, color=MARRON_OSCURO, stroke_width=3)
        skip_connection = VGroup(line_up, line_across, arrow_down)

        self.play(FadeIn(label_x, shift=RIGHT))
        self.play(GrowArrow(arrow_x_f), FadeIn(grupo_f, shift=RIGHT))
        self.play(GrowArrow(arrow_f_add), FadeIn(grupo_add, scale=0.5))
        self.play(Create(skip_connection))
        self.play(GrowArrow(arrow_add_y), FadeIn(label_y, shift=RIGHT))
        self.play(Write(eq_final))
        self.next_slide()

        pos_texto = DOWN * 3
        
        txt_desc_1 = Text("La señal original viaja por la red...", font=FUENTE, font_size=24 * escala, color=TINTA_NEGRA).move_to(pos_texto)
        txt_desc_2 = Text("El Quijote f(x) transforma el mundo, pero la señal se desvanece.", font=FUENTE, font_size=24 * escala, color=MARRON_QUIJOTE).move_to(pos_texto)
        txt_desc_3 = Text("Sancho lleva una copia intacta de la realidad por un 'atajo'.", font=FUENTE, font_size=24 * escala, color=NARANJA_TERRACOTA).move_to(pos_texto)
        txt_desc_4 = Text("La realidad de Sancho y la visión del Quijote se suman.", font=FUENTE, font_size=24 * escala, color=MARRON_OSCURO).move_to(pos_texto)

        def crear_imagen_pixelada(resolucion="alta"):
            cuadros = []
            filas, cols = (6, 6) if resolucion == "alta" else (3, 3)
            lado = 0.15 if resolucion == "alta" else 0.3
            colores = [NARANJA_TERRACOTA, MARRON_OSCURO, MARRON_QUIJOTE, OCRE_CERVANTINO]
            
            for i in range(filas):
                for j in range(cols):
                    color = colores[(i*j) % len(colores)]
                    cuadro = Square(side_length=lado, fill_color=color, fill_opacity=1, stroke_width=0.5, stroke_color=WHITE)
                    cuadros.append(cuadro)
                    
            img = VGroup(*cuadros).arrange_in_grid(rows=filas, cols=cols, buff=0)
            if resolucion == "baja":
                img.set_opacity(0.6) 
            return img

        img_alta_x = crear_imagen_pixelada("alta").move_to(pos_x).shift(UP*1.2)
        img_baja_f = crear_imagen_pixelada("baja").move_to(pos_f).shift(UP*1.2)
        img_alta_copia = crear_imagen_pixelada("alta").move_to(pos_x).shift(UP*1.2)
        img_final_y = crear_imagen_pixelada("alta").move_to(pos_y).shift(UP*1.2)

        txt_input = Text("Imagen Real", font=FUENTE, font_size=18, color=MARRON_OSCURO).next_to(img_alta_x, UP)
        self.play(FadeIn(img_alta_x, shift=DOWN), FadeIn(txt_input), FadeIn(txt_desc_1))
        self.next_slide()

        self.play(ReplacementTransform(txt_desc_1, txt_desc_2))
        txt_f = Text("Visión Alterada f(x)", font=FUENTE, font_size=18, color=MARRON_QUIJOTE).next_to(img_baja_f, UP)
        self.play(
            ReplacementTransform(img_alta_x.copy(), img_baja_f),
            FadeIn(txt_f)
        )
        self.next_slide()

        self.play(ReplacementTransform(txt_desc_2, txt_desc_3))
        txt_skip = Text("La Copia de Sancho", font=FUENTE, font_size=18, color=NARANJA_TERRACOTA).next_to(line_across, UP)
        self.play(FadeIn(txt_skip))
        self.play(
            img_alta_copia.animate.move_to(p2).shift(UP*0.5),
            run_time=0.8
        )
        self.play(
            img_alta_copia.animate.move_to(p3).shift(UP*0.5),
            run_time=1.5
        )
        self.play(
            img_alta_copia.animate.move_to(pos_add).shift(UP*1.2),
            run_time=0.8
        )
        self.next_slide()

        self.play(ReplacementTransform(txt_desc_3, txt_desc_4))
        txt_output = Text("Realidad Recuperada", font=FUENTE, font_size=18, color=MARRON_OSCURO).next_to(img_final_y, UP)
        
        self.play(
            FadeOut(img_baja_f, shift=RIGHT),
            img_alta_copia.animate.move_to(pos_y).shift(UP*1.2),
            FadeIn(txt_output)
        )
        
        caja_eq = SurroundingRectangle(eq_final, color=NARANJA_TERRACOTA, buff=0.2, stroke_width=2)
        self.play(Create(caja_eq))
        self.next_slide()

        self.limpiar_pantalla()

    def slide_dropout(self):
        """Diapositiva: Slide Dropout."""

        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        herradura_fondo = crear_herradura().scale(0.6).to_corner(DL).shift(UP*0.3 + RIGHT*0.3)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Dropout: El Arte de Olvidar", 
            palabra_clave="Dropout", 
            color_clave=NARANJA_TERRACOTA
        )
        
        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo, herradura_fondo))

        label_proposito = Text(
            "Es una técnica de regularización.", 
            font=FUENTE, font_size=24, color=TINTA_NEGRA, weight=BOLD
        )

        label_eq = Text(
            "Se basa en una Distribución de Bernoulli:", 
            font=FUENTE, font_size=20, color=MARRON_OSCURO
        )
        
        eq_dropout = MathTex(
            r"r_i \sim \text{Bernoulli}(p)", 
            r"\quad \Rightarrow \quad",
            r"y_i = \frac{r_i \cdot x_i}{p}",
            color=TINTA_NEGRA
        ).scale(1.2)
        
        eq_dropout.set_color_by_tex("p", NARANJA_TERRACOTA)
        eq_dropout.set_color_by_tex("r_i", MARRON_OSCURO)

        explicacion = Tex(
            r"Al escalar por $p$, mantenemos la esperanza matemática: $\mathbb{E}[y_i] = x_i$",
            font_size=24, color=MARRON_OSCURO
        )
        
        grupo_intro = VGroup(label_proposito, label_eq, eq_dropout, explicacion).arrange(DOWN, buff=0.4).move_to(ORIGIN)

        self.play(FadeIn(grupo_intro, shift=DOWN*0.2), run_time=1.5)
        self.next_slide()

        self.play(FadeOut(grupo_intro))

        tamaños_capas = [4, 5, 5, 5, 4]
        colores_capas = [BLUE_D, NARANJA_TERRACOTA, NARANJA_TERRACOTA, NARANJA_TERRACOTA, GREEN_D]
        nombres_capas = ["Input", "Dense 1", "Dense 2", "Dense 3", "Output"]
        
        nodos = VGroup()
        etiquetas = VGroup()
        
        for size, color, nombre in zip(tamaños_capas, colores_capas, nombres_capas):
            capa = VGroup(*[Dot(radius=0.15, color=color) for _ in range(size)]).arrange(DOWN, buff=0.4)
            nodos.add(capa)
            
            etiqueta = Text(nombre, font=FUENTE, font_size=16, color=MARRON_OSCURO, weight=BOLD)
            etiqueta.next_to(capa, UP, buff=0.3)
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
                    grupo_conexiones.add(Line(n1.get_center(), n2.get_center(), stroke_width=1.5, color=MARRON_OSCURO, stroke_opacity=0.3))
            conexiones.add(grupo_conexiones)

        red_grupo = VGroup(conexiones, nodos, etiquetas)

        self.play(FadeIn(red_grupo))
        self.next_slide()

        def aplicar_dropout_capa(indice_capa, indices_apagar, texto_explicativo):
            animaciones = []
            capa = nodos[indice_capa]
            
            for idx in indices_apagar:
                animaciones.append(capa[idx].animate.set_color(GRAY).set_opacity(0.2))
                
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
            self.next_slide()
            self.play(FadeOut(texto))

        aplicar_dropout_capa(1, [1, 4], "Dense 1: Desactivamos el 40% de las neuronas al azar.")
        aplicar_dropout_capa(2, [0, 2, 3], "Dense 2: Otras neuronas deben aprender a compensar la pérdida.")
        aplicar_dropout_capa(3, [1, 4], "Dense 3: Ninguna neurona se vuelve indispensable.")

        texto_metafora = Text(
            "\"No confíes la batalla a un solo caballero,\nhaz que toda la orden luche unida.\"",
            font=FUENTE, font_size=24, color=MARRON_OSCURO, slant=ITALIC
        ).to_edge(DOWN, buff=0.5)

        self.play(Write(texto_metafora))
        self.next_slide()

    def slide_cache_blocking(self):
        """Diapositiva: Cache Blocking (Tiling) — sin referencias a código, animaciones mejoradas."""
        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Cache Blocking: Alforjas bien Rellenas", 
            palabra_clave="Cache", 
            color_clave=NARANJA_TERRACOTA
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo), adornos=adornos)

        ram_box = RoundedRectangle(corner_radius=0.15, height=4.2, width=2.8,
                                   stroke_color=MARRON_OSCURO, stroke_width=3,
                                   fill_color=PAPEL_CREMA, fill_opacity=0.85)
        ram_box.move_to(LEFT * 4.2 + DOWN * 0.4)
        label_ram = Text("RAM", font=FUENTE, font_size=26, color=TINTA_NEGRA,
                         weight=BOLD).move_to(ram_box.get_top() + DOWN * 0.38)
        sub_ram = Text("Memoria principal", font=FUENTE, font_size=14,
                       color=MARRON_OSCURO).next_to(label_ram, DOWN, buff=0.05)

        matriz_ram = VGroup(*[
            Square(side_length=0.35,
                   stroke_color=MARRON_OSCURO, stroke_opacity=0.55,
                   fill_color=BEIGE_MEDIO, fill_opacity=0.65)
            for _ in range(36)
        ]).arrange_in_grid(6, 6, buff=0.03).next_to(sub_ram, DOWN, buff=0.18)

        cpu_box = RoundedRectangle(corner_radius=0.18, height=3.8, width=3.8,
                                   stroke_color=MARRON_OSCURO, stroke_width=3,
                                   fill_color=PAPEL_CREMA, fill_opacity=0.25)
        cpu_box.move_to(RIGHT * 3.2 + DOWN * 0.4)
        label_cpu = Text("CPU", font=FUENTE, font_size=26, color=TINTA_NEGRA,
                         weight=BOLD).move_to(cpu_box.get_top() + DOWN * 0.38)

        cache_box = RoundedRectangle(corner_radius=0.12, height=1.5, width=1.5,
                                     stroke_color=NARANJA_TERRACOTA, stroke_width=3,
                                     fill_color=NARANJA_TERRACOTA, fill_opacity=0.18)
        cache_box.next_to(label_cpu, DOWN, buff=0.45)
        label_cache = Text("Caché L1", font=FUENTE, font_size=15,
                           color=TINTA_NEGRA, weight=BOLD,
                           line_spacing=0.9).move_to(cache_box.get_center())

        flecha_bus = Arrow(
            ram_box.get_right(), cpu_box.get_left(),
            color=MARRON_OSCURO, stroke_width=3, buff=0.12
        )
        label_bus = Text("Bus de datos", font=FUENTE, font_size=14,
                         color=MARRON_OSCURO).next_to(flecha_bus, UP, buff=0.1)

        self.play(DrawBorderThenFill(ram_box), Write(label_ram), FadeIn(sub_ram))
        self.play(Create(matriz_ram, lag_ratio=0.04), run_time=1.2)
        self.play(DrawBorderThenFill(cpu_box), Write(label_cpu))
        self.play(GrowArrow(flecha_bus), FadeIn(label_bus))
        self.play(DrawBorderThenFill(cache_box), Write(label_cache))
        self.next_slide()

        texto_prob = Text(
            "Multiplicar matrices exige leer filas y columnas enteras a la vez",
            font=FUENTE, font_size=20, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=0.22)
        self.play(Write(texto_prob))

        fila_datos = matriz_ram[0:6].copy()
        col_datos  = VGroup(*[matriz_ram[i * 6] for i in range(6)]).copy()
        self.play(
            fila_datos.animate.set_fill(ROJO_TOMATE, opacity=0.85),
            col_datos.animate.set_fill(ROJO_TOMATE, opacity=0.85),
        )
        self.next_slide()

        grupo_prob = VGroup(fila_datos, col_datos)
        self.play(grupo_prob.animate.move_to(cache_box.get_center()).scale(0.75), run_time=1.2)

        texto_error = Text(
            "¡Demasiado grande! La caché se desborda — hay que releer desde RAM",
            font=FUENTE, font_size=19, color=ROJO_TOMATE, weight=BOLD
        ).next_to(linea, DOWN, buff=0.22)
        self.play(FadeTransform(texto_prob, texto_error))
        self.play(
            Wiggle(cache_box, scale_value=1.12, rotation_angle=0.04),
            Flash(cache_box, color=ROJO_TOMATE, line_length=0.38, num_lines=10)
        )
        self.next_slide()
        self.play(FadeOut(grupo_prob))

        texto_sol = Text(
            "Solución: dividir la matriz en mosaicos que sí caben en caché",
            font=FUENTE, font_size=20, color=VERDE_OLIVA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.22)
        self.play(FadeTransform(texto_error, texto_sol))

        bloques_indices = [[0,1,6,7], [2,3,8,9], [4,5,10,11]]
        colores_bloques = [VERDE_OLIVA, ORO_VIEJO, NARANJA_TERRACOTA]

        for bloque_idx, (indices, color) in enumerate(zip(bloques_indices, colores_bloques)):
            bloque = VGroup(*[matriz_ram[i] for i in indices]).copy()
            self.play(bloque.animate.set_fill(color, opacity=0.85))
            self.play(bloque.animate.move_to(cache_box.get_center()).scale(0.85), run_time=0.7)
            self.play(
                Flash(cache_box, color=color, line_length=0.28, num_lines=8),
                Indicate(bloque, color=color, scale_factor=1.08),
                run_time=0.6
            )
            self.play(FadeOut(bloque, shift=RIGHT * 0.5), run_time=0.4)

        self.next_slide()

        self.play(FadeOut(texto_sol))
        concl = Text(
            "Menos viajes a RAM → hasta 10× más velocidad en multiplicación de matrices",
            font=FUENTE, font_size=20, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.25)
        self.play(
            Write(concl),
            Indicate(cache_box, color=ORO_VIEJO, scale_factor=1.12),
            run_time=1.2
        )
        self.play(Flash(cache_box.get_center(), color=ORO_VIEJO, line_length=0.5, num_lines=12))

        self.next_slide()
        self.limpiar_pantalla()

    def slide_parallel_rayon(self):
        """Diapositiva: Paralelización — sin referencias a código, animaciones mejoradas."""
        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Paralelización: La Fuerza de los Escuderos", 
            palabra_clave="Paralelización", 
            color_clave=VERDE_OLIVA
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo), adornos=adornos)

        matriz = VGroup(*[
            Square(side_length=0.6,
                   stroke_color=MARRON_OSCURO, stroke_width=1.5,
                   fill_color=PAPEL_CREMA, fill_opacity=0.6)
            for _ in range(36)
        ]).arrange_in_grid(6, 6, buff=0.03).move_to(DOWN * 0.3)

        self.play(FadeIn(matriz, lag_ratio=0.03), run_time=1.0)
        self.next_slide()

        texto_seq = Text(
            "Un solo núcleo calcula cada celda del resultado en serie, una tras otra",
            font=FUENTE, font_size=20, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=0.22)
        self.play(Write(texto_seq))

        trabajador = Circle(radius=0.22, color=MARRON_QUIJOTE, fill_opacity=0.9,
                            stroke_width=2).next_to(texto_seq, RIGHT, buff=0.3)
        self.play(FadeIn(trabajador))

        animaciones_lentas = [
            cuadro.animate.set_fill(MARRON_QUIJOTE, opacity=0.75)
            for cuadro in matriz[0:6]
        ]
        self.play(LaggedStart(*animaciones_lentas, lag_ratio=0.9), run_time=3.0)

        label_lento = Text("Lento: O(m·n·k) en serie", font=FUENTE, font_size=18,
                           color=ROJO_CONTRA, weight=BOLD).next_to(matriz, UP, buff=0.28)
        self.play(FadeIn(label_lento, shift=DOWN*0.2))
        self.next_slide()

        self.play(
            FadeOut(texto_seq), FadeOut(trabajador), FadeOut(label_lento),
            *[cuadro.animate.set_fill(PAPEL_CREMA, opacity=0.6) for cuadro in matriz[0:6]],
            run_time=0.6
        )

        texto_div = Text(
            "La biblioteca divide el buffer de resultado en franjas de filas independientes",
            font=FUENTE, font_size=20, color=VERDE_OLIVA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.22)
        self.play(Write(texto_div))

        chunk1 = matriz[0:12]
        chunk2 = matriz[12:24]
        chunk3 = matriz[24:36]

        self.play(
            chunk1.animate.shift(UP * 0.28),
            chunk3.animate.shift(DOWN * 0.28),
            run_time=0.9
        )
        self.next_slide()

        colores_hilos = [ROJO_TOMATE, VERDE_OLIVA, ORO_VIEJO]
        nombres_hilos = ["Hilo 1", "Hilo 2", "Hilo 3"]
        chunks        = [chunk1, chunk2, chunk3]

        label_t = []
        for i, (chunk, color, nombre) in enumerate(zip(chunks, colores_hilos, nombres_hilos)):
            lbl = Text(nombre, font=FUENTE, font_size=18, color=color, weight=BOLD)
            lbl.next_to(chunk, LEFT, buff=0.35)
            label_t.append(lbl)

        self.play(
            LaggedStart(*[FadeIn(lbl, shift=RIGHT*0.3) for lbl in label_t], lag_ratio=0.2),
            run_time=0.8
        )

        texto_par = Text(
            "Cada hilo escribe en su franja exclusiva — sin esperas ni bloqueos",
            font=FUENTE, font_size=20, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.22)
        self.play(FadeTransform(texto_div, texto_par))

        self.play(
            chunk1.animate.set_fill(ROJO_TOMATE, opacity=0.82),
            chunk2.animate.set_fill(VERDE_OLIVA, opacity=0.82),
            chunk3.animate.set_fill(ORO_VIEJO,   opacity=0.82),
            run_time=1.0
        )
        self.play(
            *[Flash(chunk.get_center(), color=color, line_length=0.38, num_lines=8)
              for chunk, color in zip(chunks, colores_hilos)],
            run_time=0.8
        )
        self.next_slide()

        self.play(
            *[FadeOut(lbl) for lbl in label_t],
            FadeOut(texto_par),
            chunk1.animate.shift(DOWN * 0.28),
            chunk3.animate.shift(UP  * 0.28),
            run_time=0.8
        )

        concl = Text(
            "Resultado: el tiempo se divide entre el número de núcleos disponibles",
            font=FUENTE, font_size=21, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.3)
        self.play(Write(concl), Indicate(matriz, color=ORO_VIEJO, scale_factor=1.04))
        self.play(Flash(matriz.get_center(), color=ORO_VIEJO, line_length=0.6, num_lines=14))

        self.next_slide()
        self.limpiar_pantalla()

    def slide_batched_matmul(self):
        """Diapositiva: Batched MatMul — sin referencias a código, animaciones mejoradas."""
        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Batched MatMul: Sabios Independientes", 
            palabra_clave="Batched", 
            color_clave=OCRE_CERVANTINO
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo), adornos=adornos)

        def crear_mini_matriz(color_borde):
            return VGroup(*[
                Square(side_length=0.46,
                       stroke_color=color_borde, stroke_width=2,
                       fill_color=PAPEL_CREMA, fill_opacity=0.35)
                for _ in range(16)
            ]).arrange_in_grid(4, 4, buff=0.03)

        colores = [ROJO_TOMATE, VERDE_OLIVA, ORO_VIEJO]
        nombres = ["Cabeza 1", "Cabeza 2", "Cabeza 3"]

        matrices = VGroup(
            crear_mini_matriz(ROJO_TOMATE),
            crear_mini_matriz(VERDE_OLIVA),
            crear_mini_matriz(ORO_VIEJO)
        ).arrange(RIGHT, buff=1.8).move_to(DOWN * 0.3)

        labels = VGroup(*[
            Text(nombres[i], font=FUENTE, font_size=17, color=colores[i], weight=BOLD)
            .next_to(matrices[i], UP, buff=0.18)
            for i in range(3)
        ])

        sublabels = VGroup(*[
            Text("Atención\nindependiente", font=FUENTE, font_size=13,
                 color=MARRON_OSCURO, line_spacing=0.85)
            .next_to(matrices[i], DOWN, buff=0.18)
            for i in range(3)
        ])

        self.play(
            LaggedStart(
                *[FadeIn(matrices[i], shift=UP*0.2) for i in range(3)],
                lag_ratio=0.2
            ),
            run_time=1.0
        )
        self.play(
            LaggedStart(*[Write(labels[i]) for i in range(3)], lag_ratio=0.2),
            LaggedStart(*[FadeIn(sublabels[i]) for i in range(3)], lag_ratio=0.2),
            run_time=0.8
        )
        self.next_slide()

        texto_seq = Text(
            "Sin batching: cada cabeza de atención espera a que la anterior termine",
            font=FUENTE, font_size=20, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=0.22)
        self.play(Write(texto_seq))

        for i, color in enumerate(colores):
            self.play(
                matrices[i].animate.set_fill(color, opacity=0.72),
                run_time=0.75
            )

        label_lento = Text("3 cabezas × T tiempo = 3T en total", font=FUENTE,
                           font_size=18, color=ROJO_CONTRA, weight=BOLD)
        label_lento.next_to(matrices, UP, buff=0.65)
        self.play(FadeIn(label_lento, shift=DOWN*0.2))
        self.next_slide()

        self.play(
            FadeOut(texto_seq), FadeOut(label_lento),
            *[matrices[i].animate.set_fill(PAPEL_CREMA, opacity=0.35) for i in range(3)],
        )

        texto_batch = Text(
            "Con batching: las cabezas se agrupan en un solo tensor y se despachan juntas",
            font=FUENTE, font_size=20, color=OCRE_CERVANTINO, weight=BOLD
        ).next_to(linea, DOWN, buff=0.22)
        self.play(Write(texto_batch))

        batch_box = SurroundingRectangle(
            matrices, color=OCRE_CERVANTINO, buff=0.38,
            stroke_width=4, corner_radius=0.15
        )
        label_tensor = Text("Tensor  [lote × cabezas × secuencia × secuencia]",
                            font=FUENTE, font_size=16, color=OCRE_CERVANTINO, weight=BOLD)
        label_tensor.next_to(batch_box, DOWN, buff=0.22)

        self.play(Create(batch_box), FadeIn(label_tensor, shift=UP*0.15))
        self.next_slide()

        texto_par = Text(
            "Cada cabeza se asigna a un hilo distinto — todas calculan al mismo tiempo",
            font=FUENTE, font_size=20, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.22)
        self.play(FadeTransform(texto_batch, texto_par))

        self.play(
            matrices[0].animate.set_fill(ROJO_TOMATE, opacity=0.85),
            matrices[1].animate.set_fill(VERDE_OLIVA, opacity=0.85),
            matrices[2].animate.set_fill(ORO_VIEJO,   opacity=0.85),
            run_time=0.9
        )
        self.play(
            Flash(matrices[0].get_center(), color=ROJO_TOMATE, line_length=0.35, num_lines=8),
            Flash(matrices[1].get_center(), color=VERDE_OLIVA, line_length=0.35, num_lines=8),
            Flash(matrices[2].get_center(), color=ORO_VIEJO,   line_length=0.35, num_lines=8),
            run_time=0.8
        )
        self.play(Flash(batch_box, color=OCRE_CERVANTINO, line_length=0.5, num_lines=14))
        self.play(Indicate(batch_box, color=OCRE_CERVANTINO, scale_factor=1.04))

        label_rapido = Text("3 cabezas × T tiempo = ~T en total  (3× más rápido)",
                            font=FUENTE, font_size=18, color=VERDE_OLIVA, weight=BOLD)
        label_rapido.next_to(matrices, UP, buff=0.65)
        self.play(FadeIn(label_rapido, shift=DOWN*0.2))

        self.next_slide()
        self.limpiar_pantalla()

    def slide_intro_matmul_optimizacion(self):
        """Diapositiva introductoria: Por qué optimizar MatMul en un Transformer."""
        sol_fondo      = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "MatMul: El Corazón del Transformer",
            palabra_clave="MatMul",
            color_clave=NARANJA_TERRACOTA
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo), adornos=adornos)

        pregunta = Text(
            "¿Por qué optimizar con la multiplicación de matrices?",
            font=FUENTE, font_size=24, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=0.35)
        self.play(Write(pregunta))
        self.next_slide()

        radio = 1.6
        datos_pastel = [
            (0.833, NARANJA_TERRACOTA, "MatMul\n83.3 %"),
            (0.064, VERDE_OLIVA,       "Elementwise\n6.4 %"),
            (0.058, OCRE_CERVANTINO,   "Norm.\n5.8 %"),
            (0.045, BEIGE_MEDIO,       "Comm.\n4.5 %"),
        ]

        sectores       = VGroup()
        labels_grafica = VGroup()
        lineas_grafica = VGroup()

        angulo_actual = PI / 2

        for i, (porc, color, nombre) in enumerate(datos_pastel):
            angulo_sector = porc * TAU

            sector = Sector(
                radius=radio, 
                angle=-angulo_sector,
                start_angle=angulo_actual,
                color=color, fill_opacity=0.92,
                stroke_color=BLANCO, stroke_width=1.5
            )
            sectores.add(sector)

            angulo_medio = angulo_actual - (angulo_sector / 2)
            direccion = np.array([np.cos(angulo_medio), np.sin(angulo_medio), 0])

            fs        = 20 if i == 0 else 14
            fw        = BOLD if i == 0 else NORMAL
            lbl_color = BLANCO if i == 0 else color
            lbl = Text(nombre, font=FUENTE, font_size=fs, color=lbl_color,
                       weight=fw, line_spacing=0.85)

            if i == 0:
                lbl.move_to(direccion * (radio * 0.55))
                labels_grafica.add(lbl)
                lineas_grafica.add(VMobject())
            else:
                punto_borde    = direccion * radio
                punto_exterior = direccion * (radio + 0.3)
                linea = Line(punto_borde, punto_exterior,
                             color=color, stroke_width=1.5)
                direccion_texto = RIGHT if direccion[0] >= 0 else LEFT
                lbl.next_to(punto_exterior, direccion_texto, buff=0.15)
                lineas_grafica.add(linea)
                labels_grafica.add(lbl)

            angulo_actual -= angulo_sector

        borde_grafica = Circle(radius=radio, color=MARRON_OSCURO, stroke_width=2)

        grafica_completa = VGroup(sectores, borde_grafica, lineas_grafica, labels_grafica)

        razones_data = [
            (NARANJA_TERRACOTA, "83.3 % de los FLOPs",
             "Atención, proyecciones Q/K/V y el bloque MLP\nson todas multiplicaciones de matrices."),
            (VERDE_OLIVA,       "El cuello de botella real",
             "Cada token generado dispara decenas de MatMuls.\nOptimizarlas acelera toda la inferencia."),
            (ORO_VIEJO,         "Técnicas apilables",
             "SIMD, cache blocking, Rayon y batching se\ncombinan para multiplicar la ganancia total."),
        ]

        razones_grupo = VGroup()
        for color, titulo_r, cuerpo_r in razones_data:
            icono  = Star(n=5, outer_radius=0.18, inner_radius=0.08,
                          color=color, fill_opacity=1, stroke_width=0)
            tit    = Text(titulo_r, font=FUENTE, font_size=19, color=color, weight=BOLD)
            cue    = Text(cuerpo_r, font=FUENTE, font_size=14, color=TINTA_NEGRA, line_spacing=0.9)
            textos = VGroup(tit, cue).arrange(DOWN, buff=0.05, aligned_edge=LEFT)
            fila   = VGroup(icono, textos).arrange(RIGHT, buff=0.25, aligned_edge=UP)
            razones_grupo.add(fila)

        razones_grupo.arrange(DOWN, buff=0.45, aligned_edge=LEFT)

        contenido_principal = VGroup(grafica_completa, razones_grupo)
        contenido_principal.arrange(RIGHT, buff=1.0)
        contenido_principal.next_to(pregunta, DOWN, buff=0.55)

        self.play(
            LaggedStart(*[GrowFromCenter(s) for s in sectores], lag_ratio=0.15),
            run_time=1.2
        )
        self.play(Create(borde_grafica), run_time=0.4)

        self.play(FadeIn(labels_grafica[0]))
        animaciones_etiquetas = [
            AnimationGroup(Create(lineas_grafica[j]), FadeIn(labels_grafica[j]))
            for j in range(1, len(datos_pastel))
        ]
        self.play(LaggedStart(*animaciones_etiquetas, lag_ratio=0.3), run_time=1.2)
        self.next_slide()

        for fila in razones_grupo:
            self.play(
                GrowFromCenter(fila[0]),
                FadeIn(fila[1], shift=RIGHT * 0.2),
                run_time=0.7
            )
        self.next_slide()

        
        concl = Text(
            "Las 4 técnicas que veremos atacan directamente este 83.3 %",
            font=FUENTE, font_size=23, color=NARANJA_TERRACOTA, weight=BOLD
        )
        concl.next_to(contenido_principal, DOWN, buff=0.8)
        
        self.play(Write(concl))

        punto_flash = sectores[0].get_center() + (
            np.array([np.cos(PI / 2 - 0.833 * TAU / 2),
                      np.sin(PI / 2 - 0.833 * TAU / 2), 0]) * radio * 0.5
        )
        self.play(
            Indicate(sectores[0], color=ORO_VIEJO, scale_factor=1.06),
            Flash(punto_flash, color=ORO_VIEJO, line_length=0.5, num_lines=12),
            run_time=1.0
        )

        self.next_slide()
        self.limpiar_pantalla()

    def slide_simd(self):
        """Diapositiva: SIMD Vectorization — sin referencias a código, animaciones mejoradas."""
        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "SIMD: Una Lanza, Cuatro Gigantes", 
            palabra_clave="SIMD", 
            color_clave=ROJO_TOMATE
        )
        adornos = self._crear_adornos_esquinas()
        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo), adornos=adornos)

        array_size = 12
        valores = [1.2, 0.8, 3.1, 2.4, 0.5, 1.9, 2.7, 0.3, 1.6, 3.0, 0.9, 2.1]
        fila = VGroup(*[
            Square(side_length=0.72, stroke_color=MARRON_OSCURO, stroke_width=2,
                   fill_color=PAPEL_CREMA, fill_opacity=0.5)
            for _ in range(array_size)
        ]).arrange(RIGHT, buff=0.05).move_to(DOWN * 0.5)

        etiquetas_val = VGroup(*[
            Text(f"{valores[i]}", font=FUENTE, font_size=14, color=MARRON_OSCURO).move_to(fila[i])
            for i in range(array_size)
        ])
        indices = VGroup(*[
            Text(str(i), font=FUENTE, font_size=13, color=MARRON_OSCURO).next_to(fila[i], DOWN, buff=0.08)
            for i in range(array_size)
        ])

        self.play(
            LaggedStart(*[DrawBorderThenFill(fila[i]) for i in range(array_size)], lag_ratio=0.07),
            run_time=1.2
        )
        self.play(FadeIn(etiquetas_val, lag_ratio=0.05), FadeIn(indices), run_time=0.8)
        self.next_slide()

        texto_escalar = Text(
            "Modo escalar: la CPU procesa un número por ciclo de reloj",
            font=FUENTE, font_size=21, color=TINTA_NEGRA
        ).next_to(linea, DOWN, buff=0.25)
        self.play(Write(texto_escalar))

        reloj = Circle(radius=0.22, color=MARRON_OSCURO, stroke_width=2, fill_color=PAPEL_CREMA, fill_opacity=0.9)
        manecilla = Line(ORIGIN, UP*0.16, color=MARRON_OSCURO, stroke_width=2)
        reloj_grupo = VGroup(reloj, manecilla).next_to(texto_escalar, RIGHT, buff=0.2)
        self.play(FadeIn(reloj_grupo))

        tick_colors = [MARRON_QUIJOTE, MARRON_QUIJOTE, MARRON_QUIJOTE]
        for i in range(3):
            self.play(
                fila[i].animate.set_fill(tick_colors[i], opacity=0.85),
                etiquetas_val[i].animate.set_color(BLANCO),
                Rotate(manecilla, angle=PI/2, about_point=reloj.get_center()),
                run_time=0.55
            )
        
        label_ciclos = Text("3 ciclos → 3 elementos", font=FUENTE, font_size=18, color=ROJO_CONTRA, weight=BOLD)
        label_ciclos.next_to(fila, UP, buff=0.3)
        self.play(FadeIn(label_ciclos, shift=DOWN*0.2))
        self.next_slide()

        self.play(
            FadeOut(texto_escalar), FadeOut(reloj_grupo), FadeOut(label_ciclos),
            *[fila[i].animate.set_fill(PAPEL_CREMA, opacity=0.5) for i in range(3)],
            *[etiquetas_val[i].animate.set_color(MARRON_OSCURO) for i in range(3)],
        )

        texto_simd = Text(
            "Modo vectorial: la CPU calcula 4 números en un solo ciclo",
            font=FUENTE, font_size=21, color=ROJO_TOMATE, weight=BOLD
        ).next_to(linea, DOWN, buff=0.25)
        self.play(Write(texto_simd))

        caja_simd = SurroundingRectangle(
            fila[0:4], color=ROJO_TOMATE, stroke_width=4, buff=0.06,
            corner_radius=0.08
        )
        label_reg = Text("Registro vectorial  ×4", font=FUENTE, font_size=16,
                         color=ROJO_TOMATE, weight=BOLD).next_to(caja_simd, UP, buff=0.18)
        grupo_simd = VGroup(caja_simd, label_reg)

        self.play(Create(caja_simd), FadeIn(label_reg, shift=DOWN*0.15))

        self.play(
            LaggedStart(
                *[fila[i].animate.set_fill(ROJO_TOMATE, opacity=0.85) for i in range(4)],
                lag_ratio=0
            ),
            Flash(caja_simd, color=ROJO_TOMATE, line_length=0.35, num_lines=12),
            run_time=0.7
        )
        label_1ciclo = Text("1 ciclo → 4 elementos", font=FUENTE, font_size=18,
                            color=VERDE_OLIVA, weight=BOLD).next_to(fila, UP, buff=0.3)
        self.play(FadeIn(label_1ciclo, shift=DOWN*0.2))
        self.next_slide()

        texto_avance = Text(
            "El registro avanza de 4 en 4 — cubriendo todo el array en pocos ciclos",
            font=FUENTE, font_size=20, color=MARRON_OSCURO
        ).next_to(linea, DOWN, buff=0.25)
        self.play(FadeTransform(texto_simd, texto_avance), FadeOut(label_1ciclo))

        saltos = [fila[4:8], fila[8:12]]
        colors_avance = [ORO_VIEJO, NARANJA_TERRACOTA]

        for idx, (elementos, color) in enumerate(zip(saltos, colors_avance)):
            nuevo_centro = elementos.get_center()
            offset_y = grupo_simd.get_center()[1] - caja_simd.get_center()[1]
            self.play(
                grupo_simd.animate.move_to(nuevo_centro + UP * offset_y),
                run_time=0.5, rate_func=smooth
            )
            self.play(
                LaggedStart(*[elementos[i].animate.set_fill(color, opacity=0.85)
                              for i in range(len(elementos))], lag_ratio=0),
                Flash(caja_simd, color=color, line_length=0.3, num_lines=10),
                run_time=0.6
            )

        self.next_slide()

        self.play(FadeOut(grupo_simd), FadeOut(texto_avance))
        conclusion = Text(
            "Resultado: hasta 8× más operaciones por ciclo de CPU",
            font=FUENTE, font_size=23, color=NARANJA_TERRACOTA, weight=BOLD
        ).next_to(linea, DOWN, buff=0.3)
        self.play(
            Indicate(fila, color=ORO_VIEJO, scale_factor=1.04),
            Write(conclusion),
            run_time=1.2
        )
        self.play(Flash(fila.get_center(), color=ORO_VIEJO, line_length=0.5, num_lines=14))

        self.next_slide()
        self.limpiar_pantalla()

    def slide_roadmap(self):
        """Diapositiva: Hoja de Ruta de la Presentación"""
        sol_fondo = crear_sol_cervantino().scale(0.8).to_corner(UR).shift(DOWN*0.2 + LEFT*0.2)
        llanuras_fondo = crear_llanuras_manchegas()

        titulo, linea = self.crear_titulo(
            "Hoja de Ruta: El Camino a Seguir", 
            palabra_clave="Ruta", 
            color_clave=ORO_VIEJO
        )

        self._animar_entrada_slide(titulo, linea, fondo=VGroup(llanuras_fondo, sol_fondo))

        pasos = [
            ("Operaciones\nTensoriales", ROJO_TOMATE),
            ("Arquitectura\ndel Modelo", VERDE_OLIVA),
            ("Fase de\nEntrenamiento", ORO_VIEJO),
            ("Muestra\nPráctica", NARANJA_TERRACOTA)
        ]

        nodos = VGroup()
        textos = VGroup()
        iconos = VGroup()
        flechas = VGroup()

        for i, (texto, color) in enumerate(pasos):
            anillo = Circle(radius=0.5, stroke_color=color, stroke_width=5, fill_color=PAPEL_CREMA, fill_opacity=0.9)
            numero = Text(str(i + 1), font=FUENTE, font_size=36, color=color, weight=BOLD)
            numero.move_to(anillo.get_center())
            grupo_nodo = VGroup(anillo, numero)
            
            label = Text(texto, font=FUENTE, font_size=18, color=TINTA_NEGRA, weight=BOLD, line_spacing=1)
            
            if i == 0:
                icono = VGroup(*[
                    Square(side_length=0.15, fill_color=color, fill_opacity=0.8, stroke_width=1, stroke_color=color) 
                    for _ in range(4)
                ]).arrange_in_grid(2, 2, buff=0.05)
            elif i == 1:
                n1 = Dot(radius=0.06, color=color).move_to(LEFT*0.15 + UP*0.1)
                n2 = Dot(radius=0.06, color=color).move_to(LEFT*0.15 + DOWN*0.1)
                n3 = Dot(radius=0.06, color=color).move_to(RIGHT*0.15)
                l1 = Line(n1.get_center(), n3.get_center(), stroke_width=2, color=color)
                l2 = Line(n2.get_center(), n3.get_center(), stroke_width=2, color=color)
                icono = VGroup(l1, l2, n1, n2, n3)
            elif i == 2:
                ejes = VGroup(
                    Line(DOWN*0.15 + LEFT*0.15, DOWN*0.15 + RIGHT*0.2, stroke_width=2, color=color),
                    Line(DOWN*0.15 + LEFT*0.15, UP*0.2 + LEFT*0.15, stroke_width=2, color=color)
                )
                curva = Line(UP*0.1 + LEFT*0.05, DOWN*0.05 + RIGHT*0.15, stroke_width=3, color=color).add_tip(tip_length=0.1, tip_width=0.1)
                icono = VGroup(ejes, curva)
            else:
                icono = Triangle(fill_color=color, fill_opacity=0.8, stroke_width=1, stroke_color=color).scale(0.2).rotate(-90 * DEGREES)
                
            nodos.add(grupo_nodo)
            textos.add(label)
            iconos.add(icono)

        nodos.arrange(RIGHT, buff=1.5).move_to(DOWN * 0.2)
        
        for i in range(len(pasos)):
            textos[i].next_to(nodos[i], DOWN, buff=0.4)
            iconos[i].next_to(nodos[i], UP, buff=0.3)

        for i in range(len(pasos) - 1):
            flecha = Line(
                nodos[i].get_right() + RIGHT*0.1, 
                nodos[i+1].get_left() + LEFT*0.1, 
                color=MARRON_OSCURO, stroke_width=4
            ).add_tip(tip_length=0.2, tip_width=0.2)
            flechas.add(flecha)

        for i in range(len(pasos)):
            self.play(
                DrawBorderThenFill(nodos[i][0]), 
                Write(nodos[i][1]), 
                run_time=0.6
            )
            
            if i == 0:
                self.play(Write(textos[i]), DrawBorderThenFill(iconos[i], lag_ratio=0.2), run_time=0.8)
            elif i == 1:
                self.play(Write(textos[i]), Create(iconos[i], lag_ratio=0.3), run_time=0.8)
            elif i == 2:
                self.play(Write(textos[i]), Create(iconos[i][0]), run_time=0.4)
                self.play(Create(iconos[i][1]), run_time=0.4)
            else:
                self.play(Write(textos[i]), DrawBorderThenFill(iconos[i]), run_time=0.5)
                self.play(Flash(iconos[i], color=NARANJA_TERRACOTA, line_length=0.2), run_time=0.4)
            
            if i < len(pasos) - 1:
                self.play(Create(flechas[i]), run_time=0.6)

        self.next_slide()

        self.play(
            *[Indicate(nodos[i][0], color=MARRON_QUIJOTE, scale_factor=1.1) for i in range(len(pasos))],
            run_time=1.2
        )
        
        self.next_slide()
        self.limpiar_pantalla()

    def slide_tokenizacion(self):
        """Diapositiva: Slide Tokenizacion."""

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