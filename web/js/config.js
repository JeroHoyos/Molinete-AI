'use strict';

// ─────────────────────────────────────────────────
// Metadata de módulos + estado global
// ─────────────────────────────────────────────────
const MODS = {
  '1':  { name:'Tokenizadores BPE',             sub:'¿Cómo convierte GPT-2 el texto en números?',      desc:'Entrena tokenizadores BPE con vocabularios de 256 a 1536 tokens. Observa tasas de compresión, estadísticas del vocabulario y ejemplos de codificación/decodificación.',                            cat:'learn' },
  '2':  { name:'Operaciones Tensoriales',        sub:'Las matemáticas detrás de la atención',           desc:'Matmul, softmax, broadcasting y máscaras causales implementadas desde cero en Rust con SIMD y Rayon.',                                                                                               cat:'learn' },
  '3':  { name:'Arquitectura GPT-2',             sub:'De 50K a 163M parámetros',                        desc:'Construye y analiza cuatro tamaños de GPT-2. Tabla de parámetros, estructura de capas y un forward pass con tokens de Cervantes.',                                                                 cat:'learn' },
  '4':  { name:'Infraestructura de Entrenamiento',sub:'Prepara el corpus para entrenar',                desc:'Tokeniza el corpus, divide en 90% train / 10% validación y estima los recursos necesarios para cada tamaño.',                                                                                       cat:'learn' },
  '5':  { name:'GPT-2 Diminuto',                 sub:'~50K parámetros · 2–5 minutos',                   desc:'El punto de partida ideal. LR=3e-3, 3000 pasos, ventana de 64 tokens. Al terminar genera texto con prompts del Quijote.',                                                                          cat:'train' },
  '6':  { name:'GPT-2 Pequeño',                  sub:'~200K parámetros · 15–20 minutos',                desc:'Más capacidad y vocabulario BPE de 1024 tokens. El texto generado comienza a mostrar patrones del español.',                                                                                        cat:'train' },
  '7':  { name:'GPT-2 Mediano',                  sub:'~4M parámetros · 1–3 horas',                      desc:'Entrenamiento serio sobre el corpus completo. LR=3e-4 con warmup coseno y early stopping.',                                                                                                         cat:'train' },
  '8':  { name:'GPT-2 Small completo',           sub:'~163M parámetros · toda la noche',                desc:'La arquitectura original de OpenAI (2019) desde cero en Rust. 200K pasos, LR=1e-4. Requiere confirmación.',                                                                                        cat:'train' },
  '10': { name:'Chat con modelo',                sub:'Generación interactiva · Interactivo',            desc:'Carga un checkpoint y genera texto en español estilo Cervantes. Escribe prompts y recibe respuestas del modelo que entrenaste.',                                                                      cat:'chat'  },
  '11': { name:'Descargar Corpus de Cervantes',  sub:'Obras completas · Project Gutenberg · ~7 MB',     desc:'Descarga Don Quijote, Novelas Ejemplares, La Galatea y más. Se concatenan en cervantes.txt listo para entrenar.',                                                                                   cat:'learn' },
};

const S = {
  ws:           null,
  connected:    false,
  running:      false,
  currentId:    null,
  cat:          null,
  outputBuf:    '',
  totalSteps:   0,
  lossHistory:  [],
  trainStarted: false,
};
