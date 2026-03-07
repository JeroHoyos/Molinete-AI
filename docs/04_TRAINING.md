# Training Infrastructure: Teaching the Model

Este documento acompaña el artículo  
"Building an LLM From Scratch in Rust, Part 4: Training Infrastructure".

La fase de entrenamiento convierte **pesos aleatorios en un modelo funcional**.  
Aquí se implementan:

- Backpropagation a través de todas las capas
- Optimizador **AdamW**
- **Gradient clipping**
- **Data loading**
- Métricas y logging de entrenamiento

## Archivos relevantes

- `src/gpt2_trainable.rs` — Modelo entrenable con forward/backward
- `src/train.rs` — Carga de datos y utilidades de entrenamiento
- `src/training_logger.rs` — Registro de métricas
- `examples/04_training_infrastructure.rs` — Demostración del sistema de entrenamiento

## Ejecutar el ejemplo

```bash
cargo run --release --example 04_training_infrastructure
```

Este ejemplo demuestra:

- entrenamiento de tokenizer
- creación del data loader
- cálculo de pérdida
- simulación de progreso de entrenamiento

---

# Pérdida base (Baseline Loss)

Con pesos aleatorios, el modelo asigna probabilidad uniforme a todos los tokens.

La pérdida esperada es:

```
loss = log(vocab_size)
```

Ejemplo:

```
vocab_size = 512
log(512) ≈ 6.24
```

Si la pérdida inicial difiere mucho de este valor, probablemente hay un error en la implementación.

---

# Progresión típica del entrenamiento

Un entrenamiento saludable muestra:

- **Loss disminuyendo**
- **Validation loss cercana a training loss**
- **Perplexity descendiendo**

Valores esperados:

| Métrica | Inicio | Buen rango |
|-------|------|-------------|
| Loss | ~6.2 | 2 – 4 |
| Perplexity | ~512 | 20 – 100 |

Señales de problemas:

- Loss constante → learning rate muy bajo
- Loss NaN → learning rate muy alto
- Val loss subiendo → overfitting

---

# Data Loader

El loader tokeniza el texto una vez y genera batches durante el entrenamiento.

```rust
let mut loader = TextDataLoader::new(&text, &tokenizer, seq_len, batch_size);

while let Some((inputs, targets)) = loader.next_batch() {
    // inputs: [batch, seq_len]
    // targets: [batch, seq_len]
}
```

### División de datos

```
90% entrenamiento
10% validación
```

### Generación de secuencias

Se usa **ventana deslizante**:

```
input  = tokens[t : t+seq_len]
target = tokens[t+1 : t+seq_len+1]
```

---

# Cálculo de la pérdida

Se usa **cross entropy** con estabilidad numérica.

```rust
let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
```

Restar el máximo evita overflow en `exp()`.

---

# Gradient Clipping

Previene gradientes explosivos.

```rust
if norm > max_norm {
    scale = max_norm / norm
}
```

Valor típico:

```
max_norm = 1.0
```

---

# Optimizador AdamW

AdamW usa **momentos de primer y segundo orden**.

```
m = β1 * m + (1 − β1) * grad
v = β2 * v + (1 − β2) * grad²
```

Actualización:

```
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

Parámetros típicos:

```
β1 = 0.9
β2 = 0.95
eps = 1e-8
weight_decay = 0.1
```

AdamW aplica **weight decay separado del gradiente**, lo que mejora la regularización.

---

# Caching para Backpropagation

Durante el forward se guardan activaciones necesarias para calcular gradientes.

```rust
pub struct BlockCache {
    ln1_cache
    attn_cache
    ln2_cache
    mlp_cache
}
```

Sin cache habría que recalcular el forward durante backprop.

---

# Métricas de entrenamiento

Se registran:

| Métrica | Descripción |
|-------|-------------|
| train_loss | pérdida en datos de entrenamiento |
| val_loss | pérdida en validación |
| perplexity | exp(loss) |
| learning_rate | tasa de aprendizaje |
| grad_norm | magnitud del gradiente |

Formato CSV:

```
step,learning_rate,train_loss,val_loss,perplexity
```

---

# Checkpoints

Dos tipos:

### Inferencia

```
model.bin
```

Contiene:

- pesos del modelo
- configuración

### Entrenamiento

```
checkpoint.bin
```

Contiene además:

- estado del optimizador
- contador de pasos

Esto permite **reanudar el entrenamiento sin perder momentum**.

---

# Problemas comunes

## Loss = NaN

Causas comunes:

- learning rate demasiado alto
- gradientes explosivos
- overflow en softmax

Solución:

```
reducir LR
activar gradient clipping
```

---

## Loss no disminuye

Posibles causas:

- learning rate muy bajo
- gradientes cero
- modelo demasiado pequeño

Verificar:

```rust
println!("{}", compute_grad_norm(&grads));
```

---

## Entrenamiento lento

Siempre usar:

```bash
cargo run --release
```

Modo debug puede ser **10-100× más lento**.

---

## Falta de memoria

El uso de memoria crece con:

```
batch_size
seq_len²
n_embd
n_layers
```

Soluciones:

- reducir batch size
- reducir sequence length
- usar modelo más pequeño

---

# Hiperparámetros recomendados

| Modelo | Learning Rate | Batch | Seq Len |
|------|---------------|------|---------|
| <1M | 3e-4 | 16 | 64 |
| 1-10M | 3e-4 | 8-16 | 128 |
| 10-50M | 1e-4 | 4-8 | 128-256 |

---

# Cuándo detener el entrenamiento

Detener cuando:

```
validation loss deja de mejorar
```

Usar siempre el **checkpoint con mejor validation loss**, no el último.

---

# Próximos pasos

Una vez validada la infraestructura:

1. verificar gradientes
2. entrenar un modelo pequeño
3. observar métricas
4. generar texto durante entrenamiento

---

# Lecturas recomendadas

- AdamW — Loshchilov & Hutter (2019)
- Adam Optimizer — Kingma & Ba (2014)
- Backpropagation — Andrej Karpathy
- Layer Normalization — Ba et al. (2016)