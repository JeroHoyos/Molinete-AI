//! # El Olvido Virtuoso — Dropout
//!
//! Como el caballero que de vez en cuando olvida alguna hazaña para no volverse
//! arrogante, el Dropout obliga al modelo a no depender de ninguna neurona en
//! particular. En cada paso de entrenamiento, algunas neuronas se "duermen"
//! al azar, forzando al resto a aprender representaciones más robustas.
//!
//! Durante la inferencia (cuando Molinete ya habla con el mundo), todas las
//! neuronas permanecen despiertas — el olvido fue solo una práctica de humildad.

use crate::tensor::Tensor;

/// Capa de dropout entrenable — el ejercicio de la humildad neuronal
pub struct DropoutEntrenable {
    pub tasa: f32,       // Probabilidad de silenciar cada neurona (0.0 = sin silencio)
    pub entrenando: bool, // true durante el entrenamiento, false en inferencia
}

impl DropoutEntrenable {
    /// Forja una nueva capa de dropout con la tasa de silencio dada
    pub fn new(tasa: f32) -> Self {
        assert!((0.0..=1.0).contains(&tasa), "La tasa de dropout debe estar entre 0.0 y 1.0");
        Self { tasa, entrenando: true }
    }

    /// Paso hacia adelante — el sorteo del olvido
    ///
    /// Durante el entrenamiento, cada neurona se silencia con probabilidad `tasa`.
    /// Las sobrevivientes se escalan por 1/(1-tasa) para mantener la misma
    /// esperanza matemática — el llamado "inverted dropout".
    pub fn forward(&self, x: &Tensor) -> (Tensor, CacheDropout) {
        if !self.entrenando || self.tasa == 0.0 {
            // En inferencia, todas las neuronas contribuyen sin modificación
            return (x.clone(), CacheDropout { mascara: None, escala: 1.0 });
        }

        if self.tasa >= 1.0 {
            // Silencio total — caso extremo, raramente usado
            return (Tensor::ceros(x.forma.clone()),
                    CacheDropout { mascara: Some(vec![false; x.datos.len()]), escala: 1.0 });
        }

        // El escudo de compensación: las neuronas que sobreviven reciben más responsabilidad
        let escala = 1.0 / (1.0 - self.tasa);
        let mut mascara = Vec::with_capacity(x.datos.len());
        let mut salida = Tensor::ceros(x.forma.clone());

        for i in 0..x.datos.len() {
            let mantener = rand::random::<f32>() > self.tasa; // ¿sobrevive este escudero?
            mascara.push(mantener);
            if mantener {
                salida.datos[i] = x.datos[i] * escala; // el sobreviviente carga más peso
            }
            // La neurona silenciada aporta cero — descansa en su lecho
        }

        (salida, CacheDropout { mascara: Some(mascara), escala })
    }

    /// Paso hacia atrás — el error solo fluye por las neuronas que sobrevivieron
    ///
    /// Usamos la misma máscara del forward: las neuronas que fueron silenciadas
    /// no reciben gradiente (su contribución al error fue nula).
    pub fn backward(&self, grad_salida: &Tensor, cache: &CacheDropout) -> Tensor {
        if let Some(mascara) = &cache.mascara {
            let mut grad_entrada = Tensor::ceros(grad_salida.forma.clone());
            for (i, &mantener) in mascara.iter().enumerate() {
                if mantener {
                    grad_entrada.datos[i] = grad_salida.datos[i] * cache.escala;
                }
                // La neurona silenciada no propaga gradiente — su lección es la ausencia
            }
            grad_entrada
        } else {
            // Sin dropout activo, el gradiente fluye libremente
            grad_salida.clone()
        }
    }
}

/// Tesoro del dropout — guarda la máscara para el paso hacia atrás
pub struct CacheDropout {
    /// La máscara del sorteo: true = neurona viva, false = neurona silenciada
    pub mascara: Option<Vec<bool>>,
    /// El factor de compensación aplicado a las sobrevivientes
    pub escala: f32,
}
