# Operaciones con Tensores: la base de las redes neuronales

Este documento acompaña el artículo  
["Building an LLM From Scratch in Rust, Part 2: Tensor Operations"](https://www.tag1.com/how-to/part2-tensor-operations-building-an-llm-from-scratch/).

Las redes neuronales manipulan datos como **tensores**.  
Casi todas las operaciones en un modelo de lenguaje se reducen a operaciones tensoriales:

- La **atención** usa multiplicación de matrices.
- La **normalización de capas** usa estadísticas sobre tensores.
- Los **embeddings** convierten tokens en vectores (tensores).

Esta implementación construye la biblioteca de tensores que utilizará todo el modelo.

## Archivos relevantes

- `src/tensor.rs` — Implementación principal del tensor  
- `examples/02_tensor_operations.rs` — Ejemplo de uso de las operaciones

## Ejecutar el ejemplo

```bash
cargo run --release --example 02_tensor_operations
```

El ejemplo demuestra:

- creación de tensores
- multiplicación de matrices
- operaciones elemento a elemento
- broadcasting
- softmax estable
- reshape y transpose
- operaciones estadísticas
- máscaras causales

## Crear tensores

Tres formas básicas:

```rust
let tensor = Tensor::new(vec![1.0,2.0,3.0,4.0], vec![2,2]);
let zeros = Tensor::zeros(vec![3,4]);
let range = Tensor::arange(0,10);
```

El constructor verifica que los datos coincidan con la forma del tensor.

## Multiplicación de matrices

Es la operación más importante en redes neuronales.

Se usa en:

- capas lineales
- atención (`Q @ K^T`)
- capas feedforward

La implementación usa dos estrategias:

**Secuencial** para matrices pequeñas.

```rust
for i in 0..m {
    for j in 0..n {
        for k in 0..p {
            result[i*n+j] += a[i*p+k] * b[k*n+j];
        }
    }
}
```

**Paralela** para matrices grandes usando Rayon y bloques en caché.

Esto divide el trabajo entre varios núcleos de CPU y mejora el rendimiento.

## Operaciones elemento a elemento

Se aplican posición por posición.

Ejemplo:

```rust
let result = a.add(&b);
let scaled = tensor.mul_scalar(2.0);
```

Rayon permite paralelizar estas operaciones en tensores grandes.

## Softmax y estabilidad numérica

Softmax convierte puntuaciones en probabilidades.

Implementación básica:

```
softmax(x) = exp(x) / sum(exp(x))
```

El problema es que `exp(300)` puede desbordar.

La solución es restar el máximo:

```rust
let max = row.max();
exp(x - max)
```

Esto mantiene los valores en un rango seguro sin cambiar las probabilidades relativas.

## Broadcasting

Permite operar tensores con formas diferentes.

Ejemplo:

```
[2x3] + [3]
```

El vector se aplica a cada fila automáticamente.

Esto se usa frecuentemente para añadir **bias** en redes neuronales.

## Reshape y Transpose

**Reshape**

Reinterpreta los datos con otra forma sin moverlos.

```rust
tensor.reshape(&[3,2]);
```

**Transpose**

Intercambia dimensiones.

```rust
tensor.transpose(0,1);
```

Se usa frecuentemente en cálculos de atención.

## Operaciones estadísticas

Necesarias para **Layer Normalization**.

```rust
let mean = x.mean(-1, true);
let var = x.var(-1, true);
```

Luego se normaliza:

```
(x - mean) / sqrt(var + eps)
```

y se aplican parámetros aprendibles `gamma` y `beta`.

## Masked Fill

En modelos autoregresivos el modelo **no puede ver tokens futuros**.

Esto se logra con una máscara causal:

```
[0 1 1]
[0 0 1]
[0 0 0]
```

Las posiciones futuras se reemplazan por `-inf`, lo que hace que su probabilidad sea cero tras aplicar softmax.

## Estructura principal

```rust
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}
```

Los datos se almacenan como un vector plano en memoria (**row-major**).

`shape` define las dimensiones y `strides` permite indexar correctamente.

## Optimizaciones

La implementación incluye:

- paralelización con **Rayon**
- multiplicación de matrices por **bloques en caché**
- operaciones elemento a elemento paralelas
- softmax numéricamente estable

Esto permite ejecutar operaciones en milisegundos incluso en CPU.

## Limitaciones

A diferencia de bibliotecas como PyTorch, esta implementación **prioriza claridad sobre rendimiento**.

No incluye:

- soporte para GPU
- diferenciación automática
- gestión avanzada de memoria
- SIMD manual
- cientos de operaciones adicionales

El objetivo es **entender cómo funcionan los transformers desde cero**.

## Problemas comunes

**Error de dimensiones**

La multiplicación `A @ B` requiere que:

```
columnas(A) = filas(B)
```

**Error en reshape**

El número total de elementos debe mantenerse constante.

**Broadcasting no soportado**

Solo se implementan los patrones usados por transformers.

## Siguiente paso

Con tokenización y operaciones tensoriales completas, el siguiente paso es construir la **arquitectura del Transformer**:

- embeddings
- multi-head self-attention
- feedforward networks
- layer normalization
- conexiones residuales

Estas piezas permitirán construir un modelo que procese tokens y produzca logits para el siguiente token.

## Lecturas recomendadas

- Cache-Oblivious Algorithms (optimización de matrices)
- Floating Point Arithmetic — Goldberg
- NumPy Broadcasting
- Documentación de Rayon