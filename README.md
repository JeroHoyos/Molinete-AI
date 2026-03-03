<div align="center">

# Molinete AI  
## Construyendo un Transformer con Rust

</div>

**Autor:**  
Jerónimo Hoyos Botero  

**Repositorio principal:**  
https://github.com/JeroHoyos/Molinete-AI  

**Repositorio original:**  
https://github.com/tag1consulting/feste  

---

## ¿Qué es Molinete AI?

Molinete AI es un fork de **Feste**, una implementación desde cero de un modelo Transformer tipo GPT-2 en Rust desarrollada por Tag1 Consulting como acompañamiento a la serie *Building an LLM From Scratch in Rust*.

Mientras que Feste entrena el modelo con las obras completas de Shakespeare, **Molinete AI propone entrenarlo con la obra de Miguel de Cervantes**, estableciendo una contraposición lingüística y cultural:

- Feste → Shakespeare (inglés isabelino)
- Molinete AI → Cervantes (español del Siglo de Oro)

El objetivo no es solo replicar el experimento original, sino reinterpretarlo en español y convertirlo en una guía técnica rigurosa.

---

## ¿Por qué “Molinete”?

El nombre hace referencia al universo cervantino.  
Si Feste toma su identidad del bufón ingenioso en *Twelfth Night*, Molinete AI toma relación a los fieros oponentes que tuvo Hildago contra los molinos de viento.

---

## Qué es este proyecto

Un modelo Transformer completamente entrenable, implementado desde cero en Rust, sin frameworks de deep learning.

Incluye:

- Tokenización BPE.
- Implementación manual de tensores.
- Multi-Head Self-Attention.
- Máscara causal.
- Feed Forward Networks.
- Normalización y conexiones residuales.
- Infraestructura de entrenamiento.
- Generación autoregresiva de texto.

El propósito es comprender cómo funcionan los modelos de lenguaje implementando cada componente explícitamente.

---

## Diferencias frente al repositorio original

Este fork agrega:

- Scripts experimentales para observar ejemplos y el comportamiento interno de cada componente del modelo (tokenización, atención, capas feed forward, normalización y generación).
- Una presentación desarrollada en Manim que explica visualmente la arquitectura del Transformer y el flujo de información entre capas.
- Documentación completa en español, con explicaciones adicionales orientadas a comprender mejor el código y su estructura interna.

El enfoque es pedagógico y analítico, priorizando la comprensión detallada del funcionamiento del modelo.

---

## Serie de Documentación

Siguiendo la estructura conceptual del repositorio original, el proyecto cubre:

1. Tokenización
2. Operaciones tensoriales
3. Arquitectura del modelo
4. Infraestructura de entrenamiento
5. Experimentos de entrenamiento y generación

---

