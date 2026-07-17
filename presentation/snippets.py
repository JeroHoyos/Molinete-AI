
RUST_SNIPPETS = {
    # src/tensor.rs (struct Tensor)
    "tensor.rs": """/// Un pergamino multidimensional para los calculos del Transformer
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Todos los valores en una hilera plana de memoria
    pub datos: Vec<f32>,

    /// Las dimensiones del tensor, ej. [lote, secuencia, embedding]
    pub forma: Vec<usize>,

    /// Cuantos pasos dar en memoria para avanzar en cada dimension
    pub saltos: Vec<usize>,
}""",

    # src/tensor.rs (matmul, camino secuencial)
    "matmul_base.rs": """/// Multiplicacion clasica de matrices: tres bucles anidados
let mut resultado = vec![0.0; m * n];

for i in 0..m {
    for j in 0..n {
        /// Producto punto: fila i de A con columna j de B
        let mut suma = 0.0;
        for l in 0..k {
            suma += self.datos[i * k + l] * other.datos[l * n + j];
        }
        resultado[i * n + j] = suma;
    }
}

Tensor::new(resultado, vec![m, n])""",

    # src/tensor.rs (matmul_interno_simd)
    "simd_vectorization.rs": """#[inline(always)]
fn matmul_interno_simd(val_a: f32, b: &[f32], resultado: &mut [f32]) {
    /// Bucle sencillo que LLVM convierte en instrucciones SIMD (AVX2/NEON):
    /// cuatro gigantes atacados al mismo tiempo con una sola lanza
    for (r, &val_b) in resultado.iter_mut().zip(b.iter()) {
        *r += val_a * val_b;
    }
}""",

    # src/tensor.rs (matmul_paralelo_bloques, bucles con bloqueo de cache)
    "cache_blocking.rs": """/// Escuadrones de 8x8: caben completos en la cache L1
const TAM_BLOQUE: usize = 8;

for j_inicio in (0..n).step_by(TAM_BLOQUE) {
    let j_fin = (j_inicio + TAM_BLOQUE).min(n);

    for k_inicio in (0..k).step_by(TAM_BLOQUE) {
        let k_fin = (k_inicio + TAM_BLOQUE).min(k);

        /// El bucle interno recorre memoria contigua: aqui actua la lanza SIMD
        for i in i_inicio..i_fin {
            let fila = (i - i_inicio) * n;
            for k_idx in k_inicio..k_fin {
                Self::matmul_interno_simd(
                    self.datos[i * k + k_idx],
                    &other.datos[k_idx * n + j_inicio..k_idx * n + j_fin],
                    &mut bloque_resultado[fila + j_inicio..fila + j_fin],
                );
            }
        }
    }
}""",

    # src/tensor.rs (matmul_paralelo_bloques, reparto entre hilos)
    "parallel.rs": """/// Rayon reparte las filas del resultado entre todos los nucleos:
/// cada hilo recibe TAM_BLOQUE filas y combate por su cuenta
resultado
    .par_chunks_mut(TAM_BLOQUE * n)
    .enumerate()
    .for_each(|(i_bloque, bloque_resultado)| {
        let i_inicio = i_bloque * TAM_BLOQUE;
        let i_fin = (i_inicio + TAM_BLOQUE).min(m);

        /// ... cada bloque calcula sus filas sin tocar las de otros
    });""",

    # src/tokenizador.rs (struct TokenizadorBPE)
    "BDPtokenizer.rs": """#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenizadorBPE {
    /// Asocia cada token con su ID. Comienza con 256 tokens base,
    /// uno por cada byte posible: "<00>", "<01>", ..., "<ff>"
    vocabulario: HashMap<String, usize>,

    /// Reglas de fusion aprendidas durante el entrenamiento;
    /// se aplican en orden para codificar texto nuevo
    fusiones: Vec<(String, String)>,
}""",

    # src/tokenizador.rs (conteo de pares en paralelo)
    "pair_counts.rs": """let conteo_pares: HashMap<(String, String), usize> = tokens
    /// Rayon reparte fragmentos del corpus entre los hilos
    .par_chunks(tam_fragmento)
    .enumerate()
    /// fold: cada hilo cuenta pares en su propio HashMap (sin locks)
    .fold(HashMap::new, |mut conteos, (idx, fragmento)| {
        /// windows(2) recorre cada par adyacente: [A,B], [B,C], ...
        for ventana in fragmento.windows(2) {
            let par = (ventana[0].clone(), ventana[1].clone());
            *conteos.entry(par).or_insert(0) += 1;
        }
        /// (mas el par que cruza la frontera entre dos fragmentos)
        conteos
    })
    /// reduce: consolida los conteos locales en un total global
    .reduce(HashMap::new, |mut a, b| {
        for (par, n) in b { *a.entry(par).or_insert(0) += n; }
        a
    });""",

    # src/gpt2_entrenable.rs (forward: embeddings de token y posicion)
    "embedding.rs": """pub fn forward(&self, ids_entrada: &[usize]) -> (Tensor, CacheGPT2) {
    let long_sec = ids_entrada.len();
    let n_embd = self.config.n_embd;

    /// Cada token recibe su embedding mas el embedding de su posicion
    let mut embebidos = Vec::new();
    for (pos, &id_token) in ids_entrada.iter().enumerate() {
        let inicio_token = id_token * n_embd;
        let inicio_pos = pos * n_embd;

        for i in 0..n_embd {
            embebidos.push(
                self.embedding_tokens.datos[inicio_token + i]
                    + self.embedding_posiciones.datos[inicio_pos + i],
            );
        }
    }
    let mut x = Tensor::new(embebidos, vec![long_sec, n_embd]);

    /// ... y el pergamino sigue su camino por los bloques transformer
}""",

    # src/layers/layer_norm.rs (forward)
    "normalization.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheNormCapa) {
    /// 1. Estadisticas de cada vector de la secuencia
    let media    = x.mean(-1, true);
    let varianza = x.var(-1, true);
    let desv_est = varianza.add_scalar(self.eps).sqrt(); /// eps evita dividir por 0

    /// 2. Normalizacion: media 0, varianza 1
    let x_centrado = x.sub(&media);
    let x_norm     = x_centrado.div(&desv_est);

    /// 3. Escala y desplazamiento aprendibles: y = gamma * x_norm + beta
    let y = x_norm.mul(&self.gamma).add(&self.beta);

    let cache = CacheNormCapa { x: x.clone(), x_norm, media, desv_est };
    (y, cache)
}""",

    # src/layers/attention.rs (forward)
    "attention.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheAtencion) {
    /// 1. Cada token proyecta su Query, Key y Value
    let (q, cache_q) = self.proy_q.forward(x);
    let (k, cache_k) = self.proy_k.forward(x);
    let (v, cache_v) = self.proy_v.forward(x);

    /// 2. Puntuaciones: Q @ K_T / sqrt(d_k), la escala doma los productos
    let escala = (self.n_embd as f32).sqrt();
    let puntuaciones = q.matmul(&k.transpose(-2, -1)).mul_scalar(1.0 / escala);

    /// 3. Mascara causal: el pacto del caballero, nadie mira al futuro
    let enmascaradas = puntuaciones.masked_fill(&tensor_mascara, -1e9);

    /// 4. Softmax: las puntuaciones se vuelven porcentajes de atencion
    let pesos_atencion = enmascaradas.softmax(-1);
    /// 5. Cada posicion recibe una mezcla ponderada de los Values
    let salida_atencion = pesos_atencion.matmul(&v);

    /// 6. Proyeccion final y dropout residual
    let (y_proy, cache_salida) = self.proy_salida.forward(&salida_atencion);
    let (y, cache_dropout) = self.dropout_resid.forward(&y_proy);
    (y, cache)
}""",

    # src/layers/activation.rs (gelu_forward)
    "gelu.rs": """pub fn gelu_forward(x: &Tensor) -> Tensor {
    /// GELU(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let resultado = x.datos
        .par_iter()
        .map(|&valor| {
            let interno = (2.0 / std::f32::consts::PI).sqrt()
                * (valor + 0.044715 * valor.powi(3));
            0.5 * valor * (1.0 + interno.tanh())
        })
        .collect();

    Tensor::new(resultado, x.forma.clone())
}""",

    # src/tensor.rs (softmax 2D por fila)
    "softmax.rs": """/// Cada fila se procesa en paralelo: un caballero por fila
let resultado: Vec<f32> = (0..filas)
    .into_par_iter()
    .flat_map_iter(|i| {
        let fila = &self.datos[i * columnas..(i + 1) * columnas];

        /// Restamos el maximo para que exp() no desborde
        let maximo = fila.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let valores_exp: Vec<f32> =
            fila.iter().map(|&x| (x - maximo).exp()).collect();

        /// Normalizamos: cada fila suma exactamente 1.0
        let suma: f32 = valores_exp.iter().sum();
        valores_exp.into_iter().map(move |val| val / suma)
    })
    .collect();""",

    # src/layers/mlp.rs (forward)
    "mlp_forward.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheMLP) {
    /// 1. Expansion: abrimos el espacio de representacion (n_embd -> 4*n_embd)
    let (h, cache_fc1) = self.fc1.forward(x);

    /// 2. Activacion GELU: la no-linealidad suave del alquimista
    let h_activada = gelu_forward(&h);

    /// 3. Compresion: volvemos a la dimension original (4*n_embd -> n_embd)
    let (y_proy, cache_fc2) = self.fc2.forward(&h_activada);

    /// 4. Dropout residual: el ejercicio de la humildad
    let (y, cache_dropout_resid) = self.dropout_resid.forward(&y_proy);

    let cache = CacheMLP {
        cache_fc1, h, h_activada, cache_fc2, cache_dropout_resid,
    };
    (y, cache)
}""",

    # src/gpt2_entrenable.rs (calcular_perdida)
    "compute_loss.rs": """pub fn calcular_perdida(&self, logits: &Tensor, objetivos: &[usize]) -> f32 {
    let tamano_vocab = self.config.vocab_size;
    let mut perdida_total = 0.0;

    for (i, &objetivo) in objetivos.iter().enumerate() {
        let inicio = i * tamano_vocab;
        let slice_logits = &logits.datos[inicio..inicio + tamano_vocab];

        /// Estabilidad numerica: restar el maximo antes de exponenciar
        let logit_maximo = slice_logits.iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let suma_exp: f32 = slice_logits.iter()
            .map(|&x| (x - logit_maximo).exp()).sum();

        /// Entropia cruzada: -log P(token correcto)
        let log_prob = (slice_logits[objetivo] - logit_maximo) - suma_exp.ln();
        perdida_total -= log_prob;
    }

    perdida_total / objetivos.len() as f32
}""",

    # src/layers/linear.rs (backward)
    "linear_backward.rs": """pub fn backward(&self, grad_salida: &Tensor, cache: &CacheLineal)
    -> GradientesLineales
{
    /// grad_W = x_T @ grad_y: cada peso responde por su influencia
    let grad_peso = cache.x.transpose(-2, -1).matmul(grad_salida);

    /// grad_b = suma de grad_y: el sesgo acumula el error de todas las posiciones
    let datos_grad_sesgo: Vec<f32> = (0..self.sesgo.datos.len())
        .map(|i| {
            let mut suma = 0.0;
            for fila in 0..grad_salida.forma[0] {
                suma += grad_salida.datos[fila * grad_salida.forma[1] + i];
            }
            suma
        })
        .collect();
    let grad_sesgo = Tensor::new(datos_grad_sesgo, self.sesgo.forma.clone());

    /// grad_x = grad_y @ W_T: el error se propaga a la capa anterior
    let grad_x = grad_salida.matmul(&self.peso.transpose(-2, -1));

    GradientesLineales { peso: grad_peso, sesgo: grad_sesgo, x: grad_x }
}""",

    # src/layers/block.rs (backward)
    "block_backward.rs": """pub fn backward(&self, grad_salida: &Tensor, cache: &CacheBloque)
    -> GradientesBloque {
    /// Segunda residual: el gradiente se bifurca en dos caminos
    let mut grad_x_despues_atencion = grad_salida.clone(); /// camino directo

    /// Camino del MLP: dropout -> fc2 -> GELU -> fc1
    let grads_mlp = self.mlp.backward(grad_salida, &cache.cache_mlp);
    let grads_ln2 = self.ln2.backward(&grads_mlp.x, &cache.cache_ln2);
    /// Acumulamos: atajo + camino MLP
    for i in 0..grad_x_despues_atencion.datos.len() {
        grad_x_despues_atencion.datos[i] += grads_ln2.x.datos[i];
    }

    /// Primera residual: otra bifurcacion (atajo + camino Atencion)
    let grads_atencion = self.atencion
        .backward(&grad_x_despues_atencion, &cache.cache_atencion);
    let grads_ln1 = self.ln1.backward(&grads_atencion.x, &cache.cache_ln1);
    let mut grad_x = grad_x_despues_atencion;
    for i in 0..grad_x.datos.len() {
        grad_x.datos[i] += grads_ln1.x.datos[i];
    }

    /// Se devuelven los gradientes de ln1, atencion, ln2, mlp y x
}""",

    # src/optimizador.rs (actualizar_adamw)
    "adamw_update.rs": """/// Correccion de sesgo: m y v arrancan en 0 y necesitan calibrarse
let bias_correction1 = 1.0 - optimizer.beta1.powf(step);
let bias_correction2 = 1.0 - optimizer.beta2.powf(step);

param.datos.par_iter_mut()
    .zip(grad.datos.par_iter())
    .zip(m.datos.par_iter_mut().zip(v.datos.par_iter_mut()))
    .for_each(|((param_val, &grad_val), (m_val, v_val))| {
        /// Weight decay desacoplado: solo en matrices de pesos 2D
        if aplicar_decay {
            *param_val *= 1.0 - lr * weight_decay;
        }
        /// Primer momento: la inercia del gradiente (momentum)
        *m_val = beta1 * *m_val + (1.0 - beta1) * grad_val;

        /// Segundo momento: la varianza del gradiente
        *v_val = beta2 * *v_val + (1.0 - beta2) * grad_val * grad_val;

        /// Paso adaptativo: cada parametro avanza a su propio ritmo
        let m_hat = *m_val / bias_correction1;
        let v_hat = *v_val / bias_correction2;
        *param_val -= lr * m_hat / (v_hat.sqrt() + epsilon);
    });""",

    # src/layers/dropout.rs (forward)
    "dropout.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheDropout) {
    if !self.entrenando || self.tasa == 0.0 {
        /// En inferencia todas las neuronas contribuyen sin cambios
        return (x.clone(), CacheDropout { mascara: None, escala: 1.0 });
    }

    /// Inverted dropout: las sobrevivientes se escalan por 1/(1-tasa)
    /// para mantener la misma esperanza matematica
    let escala = 1.0 / (1.0 - self.tasa);
    let mut mascara = Vec::with_capacity(x.datos.len());
    let mut salida = Tensor::ceros(x.forma.clone());

    for i in 0..x.datos.len() {
        let mantener = rand::random::<f32>() > self.tasa; /// sobrevive?
        mascara.push(mantener);
        if mantener {
            salida.datos[i] = x.datos[i] * escala;
        }
        /// La neurona silenciada aporta cero: descansa en su lecho
    }

    (salida, CacheDropout { mascara: Some(mascara), escala })
}""",

    # src/python_bindings.rs (PyTokenizadorBPE + modulo)
    "python_bindings.rs": """use pyo3::prelude::*;

/// Envoltura PyO3: expone el tokenizador Rust como clase Python
#[pyclass(name = "TokenizadorBPE")]
pub struct PyTokenizadorBPE {
    pub inner: TokenizadorBPE,
}

#[pymethods]
impl PyTokenizadorBPE {
    #[new]
    pub fn new(tam_vocabulario: usize) -> Self {
        Self { inner: TokenizadorBPE::new(tam_vocabulario) }
    }

    pub fn codificar(&self, texto: &str) -> Vec<usize> {
        self.inner.codificar(texto)
    }
}

/// Registra las clases en el modulo Python `molineteai`
pub fn molineteai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConfig>()?;
    m.add_class::<PyTokenizadorBPE>()?;
    m.add_class::<PyGPT2Entrenable>()?;
    Ok(())
}""",

    # src/gpt2_entrenable.rs (generar)
    "temperature.rs": """pub fn generar(&self, prompt: &[usize], max_tokens: usize,
               temperatura: f32) -> Vec<usize> {
    let mut tokens = prompt.to_vec();

    for _ in 0..max_tokens {
        let (logits, _) = self.forward(&tokens);
        /// Logits de la ultima posicion: el siguiente token en disputa
        let inicio = (tokens.len() - 1) * self.config.vocab_size;
        let ultimos = &logits.datos[inicio..inicio + self.config.vocab_size];

        /// La temperatura templa la audacia: <1 conservador, >1 aventurero
        let escalados: Vec<f32> =
            ultimos.iter().map(|&x| x / temperatura).collect();

        /// Softmax con estabilidad numerica
        let maximo = escalados.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = escalados.iter()
            .map(|&x| (x - maximo).exp()).collect();
        let suma: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&x| x / suma).collect();
        tokens.push(muestrear_de_probs(&probs));

        if tokens.len() >= self.config.block_size { break; }
    }
    tokens
}""",
}
