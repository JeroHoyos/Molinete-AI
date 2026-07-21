'use strict';

// ─────────────────────────────────────────────────
// Metadata de módulos + estado global
// ─────────────────────────────────────────────────
const MODS = {
  '1':  { name:'Tokenizadores BPE',              sub:'¿Cómo convierte GPT-2 el texto en números?',  cat:'learn' },
  '2':  { name:'Operaciones Tensoriales',        sub:'Las matemáticas detrás de la atención',       cat:'learn' },
  '3':  { name:'Arquitectura GPT-2',             sub:'De 50K a 163M parámetros',                    cat:'learn' },
  '4':  { name:'Infraestructura de Entrenamiento', sub:'Prepara el corpus para entrenar',           cat:'learn' },
  '5':  { name:'GPT-2 50K',                      sub:'2–5 minutos de entrenamiento',                cat:'train' },
  '6':  { name:'GPT-2 200K',                     sub:'15–20 minutos de entrenamiento',              cat:'train' },
  '7':  { name:'GPT-2 4M',                       sub:'1–3 horas de entrenamiento',                  cat:'train' },
  '8':  { name:'GPT-2 163M',                     sub:'toda la noche entrenando',                    cat:'train' },
  '10': { name:'Conversar y comparar',           sub:'Chatea con un modelo o compara varios a la vez', cat:'chat' },
  '11': { name:'Descargar Corpus de Cervantes',  sub:'Obras completas · Project Gutenberg · ~7 MB', cat:'learn' },
  '12': { name:'Comparar modelos',               sub:'El mismo prompt en hasta 4 paneles',          cat:'comp'  },
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
  elapsedSec:   0,
  bestVal:      null,   // { val, step } — mejor pérdida de validación vista
  chartFit:     false,  // escala X: false = run completo, true = ajustada a los datos
  chatTemp:     0.8,
  chatMax:      100,
  compMax:      80,
};
