from colores import *

RUST_SNIPPETS = {
    "tensor.rs": """pub struct Tensor {
    pub datos: Vec<f32>,
    pub forma: Vec<usize>,
    pub saltos: Vec<usize>,
}""",

    "matmul_base.rs": """let mut resultado = vec![0.0; m * n];
for i in 0..m {
    for j in 0..n {
        let mut suma = 0.0;
        for l in 0..k {
            suma += self.datos[i * k + l] * other.datos[l * n + j];
        }
        resultado[i * n + j] = suma;
    }
}
Tensor::new(resultado, vec![m, n])""",

    "cache_blocking.rs": """const TAM_BLOQUE: usize = 8;

resultado
    .par_chunks_mut(TAM_BLOQUE * n)
    .enumerate()
    .for_each(|(i_bloque, bloque_resultado)| {
        let i_inicio = i_bloque * TAM_BLOQUE;
        for j_inicio in (0..n).step_by(TAM_BLOQUE) {
            for k_inicio in (0..k).step_by(TAM_BLOQUE) {
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

    "parallel.rs": """resultado
    .par_chunks_mut(TAM_BLOQUE * n)
    .enumerate()
    .for_each(|(i_bloque, bloque_resultado)| {
        let i_inicio = i_bloque * TAM_BLOQUE;
        let i_fin = (i_inicio + TAM_BLOQUE).min(m);
    });""",

    "batched_matmul.rs": """resultado
    .par_chunks_mut(sec1 * sec2)
    .enumerate()
    .for_each(|(idx_lc, bloque)| {
        let b = idx_lc / num_cabezas;
        let h = idx_lc % num_cabezas;
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

    "simd_vectorization.rs": """fn matmul_interno_simd(val_a: f32, b: &[f32], resultado: &mut [f32]) {
    for (r, &val_b) in resultado.iter_mut().zip(b.iter()) {
        *r += val_a * val_b;
    }
}""",

    "BDPtokenizer.rs": """#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenizadorBPE {
    vocabulario: HashMap<String, usize>,
    fusiones: Vec<(String, String)>,
}""",

    "pair_counts.rs": """let conteo_pares: HashMap<(String, String), usize> = tokens
    .par_chunks(tam_fragmento)
    .enumerate()
    .fold(HashMap::new, |mut conteos_locales, (idx_fragmento, fragmento)| {
        for ventana in fragmento.windows(2) {
            let par = (ventana[0].clone(), ventana[1].clone());
            *conteos_locales.entry(par).or_insert(0) += 1;
        }
        if let (Some(ultimo), Some(siguiente)) = (
            fragmento.last(),
            tokens.get(idx_fragmento * tam_fragmento + fragmento.len()),
        ) {
            *conteos_locales.entry((ultimo.clone(), siguiente.clone())).or_insert(0) += 1;
        }
        conteos_locales
    })
    .reduce(HashMap::new, |mut a, b| {
        for (par, conteo) in b { *a.entry(par).or_insert(0) += conteo; }
        a
    });""",

    "embedding.rs": """pub fn forward(&self, ids_tokens: &[Vec<usize>]) -> Tensor {
    let mut x = self.embedding_tokens.forward(ids_tokens);

    let posiciones: Vec<Vec<usize>> = vec![(0..long_sec).collect()];
    let emb_pos = self.embedding_posicion.forward(&posiciones);
    for l in 0..tamano_lote {
        for s in 0..long_sec {
            for e in 0..self.configuracion.n_embd {
                x.datos[(l * long_sec + s) * self.configuracion.n_embd + e]
                    += emb_pos.datos[s * self.configuracion.n_embd + e];
            }
        }
    }
    for bloque in &self.bloques { x = bloque.forward(&x); }
    x = self.ln_final.forward(&x);
    self.cabeza_lm.forward(&x)
}""",

    "normalization.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheNormCapa) {
    let media    = x.mean(-1, true);
    let varianza = x.var(-1, true);
    let desv_est = varianza.add_scalar(self.eps).sqrt();

    let x_centrado = x.sub(&media);
    let x_norm     = x_centrado.div(&desv_est);

    let y = x_norm.mul(&self.gamma).add(&self.beta);

    let cache = CacheNormCapa { x: x.clone(), x_norm, media, desv_est };
    (y, cache)
}""",

    "attention.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheAtencion) {
    let (q, cache_q) = self.proy_q.forward(x);
    let (k, cache_k) = self.proy_k.forward(x);
    let (v, cache_v) = self.proy_v.forward(x);

    let escala = (self.n_embd as f32).sqrt();
    let puntuaciones = q.matmul(&k.transpose(-2, -1)).mul_scalar(1.0 / escala);

    let puntuaciones_enmascaradas = puntuaciones.masked_fill(&tensor_mascara, -1e9);

    let pesos_atencion = puntuaciones_enmascaradas.softmax(-1);
    let salida_atencion = pesos_atencion.matmul(&v);

    let (y_proy, cache_salida) = self.proy_salida.forward(&salida_atencion);
    let (y, cache_dropout_resid) = self.dropout_resid.forward(&y_proy);
    (y, /* cache ... */)
}""",

    "gelu.rs": """pub fn gelu_forward(x: &Tensor) -> Tensor {
    let resultado = x
        .datos
        .par_iter()
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

    "softmax.rs": """let resultado: Vec<f32> = (0..filas)
    .into_par_iter()
    .flat_map_iter(|i| {
        let fila = &self.datos[i * columnas..(i + 1) * columnas];

        let maximo = fila.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let valores_exp: Vec<f32> = fila.iter().map(|&x| (x - maximo).exp()).collect();

        let suma: f32 = valores_exp.iter().sum();
        valores_exp.into_iter().map(move |val| val / suma)
    })
    .collect();""",

    "mlp_forward.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheMLP) {
    let (h, cache_fc1) = self.fc1.forward(x);

    let h_activada = gelu_forward(&h);

    let (y_proy, cache_fc2) = self.fc2.forward(&h_activada);

    let (y, cache_dropout_resid) = self.dropout_resid.forward(&y_proy);

    let cache = CacheMLP { cache_fc1, h, h_activada, cache_fc2, cache_dropout_resid };
    (y, cache)
}""",

    "compute_loss.rs": """pub fn calcular_perdida(&self, logits: &Tensor, objetivos: &[usize]) -> f32 {
    let mut perdida_total = 0.0;
    for (i, &objetivo) in objetivos.iter().enumerate() {
        let inicio = i * self.config.vocab_size;
        let slice_logits = &logits.datos[inicio..inicio + self.config.vocab_size];

        let logit_max = slice_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let suma_exp: f32 = slice_logits.iter().map(|&x| (x - logit_max).exp()).sum();

        let log_prob = (slice_logits[objetivo] - logit_max) - suma_exp.ln();
        perdida_total -= log_prob;
    }
    perdida_total / objetivos.len() as f32
}""",

    "linear_backward.rs": """pub fn backward(&self, grad_salida: &Tensor, cache: &CacheLineal) -> GradientesLineales {
    let grad_peso = cache.x.transpose(-2, -1).matmul(grad_salida);

    let datos_grad_sesgo: Vec<f32> = (0..self.sesgo.datos.len())
        .map(|i| (0..grad_salida.forma[0])
            .map(|fila| grad_salida.datos[fila * grad_salida.forma[1] + i])
            .sum())
        .collect();
    let grad_sesgo = Tensor::new(datos_grad_sesgo, self.sesgo.forma.clone());

    let grad_x = grad_salida.matmul(&self.peso.transpose(-2, -1));

    GradientesLineales { peso: grad_peso, sesgo: grad_sesgo, x: grad_x }
}""",

    "block_backward.rs": """pub fn backward(&self, grad_salida: &Tensor, cache: &CacheBloque) -> GradientesBloque {
    let mut grad_x_despues_atencion = grad_salida.clone();

    let grads_mlp = self.mlp.backward(grad_salida, &cache.cache_mlp);
    let grads_ln2 = self.ln2.backward(&grads_mlp.x, &cache.cache_ln2);
    for i in 0..grad_x_despues_atencion.datos.len() {
        grad_x_despues_atencion.datos[i] += grads_ln2.x.datos[i];
    }

    let grads_atencion = self.atencion.backward(&grad_x_despues_atencion, &cache.cache_atencion);
    let grads_ln1 = self.ln1.backward(&grads_atencion.x, &cache.cache_ln1);
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
    let correcc1 = 1.0 - optimizer.beta1.powf(optimizer.step as f32);
    let correcc2 = 1.0 - optimizer.beta2.powf(optimizer.step as f32);

    param.datos.par_iter_mut()
        .zip(grad.datos.par_iter())
        .zip(m.datos.par_iter_mut().zip(v.datos.par_iter_mut()))
        .for_each(|((param_val, &grad_val), (m_val, v_val))| {
            if aplicar_decay { *param_val *= 1.0 - lr * weight_decay; }

            *m_val = optimizer.beta1 * *m_val + (1.0 - optimizer.beta1) * grad_val;

            *v_val = optimizer.beta2 * *v_val + (1.0 - optimizer.beta2) * grad_val * grad_val;

            let m_hat = *m_val / correcc1;
            let v_hat = *v_val / correcc2;
            *param_val -= lr * m_hat / (v_hat.sqrt() + optimizer.epsilon);
        });
}""",

    "dropout.rs": """pub fn forward(&self, x: &Tensor) -> (Tensor, CacheDropout) {
    if !self.entrenando || self.tasa == 0.0 {
        return (x.clone(), CacheDropout { mascara: None, escala: 1.0 });
    }

    let escala = 1.0 / (1.0 - self.tasa);
    let mut mascara = Vec::with_capacity(x.datos.len());
    let mut salida  = Tensor::ceros(x.forma.clone());

    for i in 0..x.datos.len() {
        let mantener = rand::random::<f32>() > self.tasa;
        mascara.push(mantener);
        if mantener {
            salida.datos[i] = x.datos[i] * escala;
        }
    }
    let cache = CacheDropout { mascara: Some(mascara), escala };
    (salida, cache)
}""",

    "python_bindings.rs": """use pyo3::prelude::*;

#[pyclass]
pub struct TokenizadorBPE { inner: crate::TokenizadorBPE }

#[pymethods]
impl TokenizadorBPE {
    #[new]
    pub fn new(tam_vocab: usize) -> Self {
        Self { inner: crate::TokenizadorBPE::nuevo(tam_vocab) }
    }
    pub fn entrenar(&mut self, texto: &str, tam_vocab: usize) {
        self.inner.entrenar(texto, tam_vocab);
    }
    pub fn codificar(&self, texto: &str) -> Vec<usize> {
        self.inner.codificar(texto)
    }
    pub fn decodificar(&self, ids: Vec<usize>) -> String {
        self.inner.decodificar(&ids)
    }
}

#[pymodule]
fn molineteai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TokenizadorBPE>()?;
    m.add_class::<GPT2Entrenable>()?;
    m.add_class::<Config>()?;
    Ok(())
}""",

    "temperature.rs": """pub fn generar(&self, prompt: &[usize], max_tokens: usize, temperatura: f32) -> Vec<usize> {
    let mut tokens = prompt.to_vec();

    for _ in 0..max_tokens {
        let (logits, _) = self.forward(&tokens);

        let inicio_ultima = (tokens.len() - 1) * self.config.vocab_size;
        let ultimos_logits = &logits.datos[inicio_ultima..inicio_ultima + self.config.vocab_size];

        let logits_escalados: Vec<f32> = ultimos_logits.iter().map(|&x| x / temperatura).collect();

        let logit_max = logits_escalados.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let valores_exp: Vec<f32> = logits_escalados.iter().map(|&x| (x - logit_max).exp()).collect();
        let suma: f32 = valores_exp.iter().sum();
        let probs: Vec<f32> = valores_exp.iter().map(|&x| x / suma).collect();

        tokens.push(muestrear_de_probs(&probs));
        if tokens.len() >= self.config.block_size { break; }
    }
    tokens
}"""
}
