'use strict';

// ─────────────────────────────────────────────────
// Utilidades: texto, formato, estado, scroll
// ─────────────────────────────────────────────────
function stripAnsi(s) {
  return s.replace(/\x1b\[[0-9;?]*[a-zA-Z]/g, '');
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Duración legible a partir de segundos: "42 s", "3 min 05 s", "1 h 12 min"
function fmtDur(s) {
  if (!isFinite(s) || s < 0) return '—';
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h} h ${m} min`;
  if (m) return `${m} min ${String(sec).padStart(2, '0')} s`;
  return `${sec} s`;
}

// Velocidad de entrenamiento legible
function fmtRate(pasosPorSeg) {
  if (!isFinite(pasosPorSeg) || pasosPorSeg <= 0) return '—';
  if (pasosPorSeg >= 10) return `${pasosPorSeg.toFixed(0)} pasos/s`;
  if (pasosPorSeg >= 1)  return `${pasosPorSeg.toFixed(1)} pasos/s`;
  return `${(pasosPorSeg * 60).toFixed(1)} pasos/min`;
}

const STATUS_COLOR = { ok: '#55813f', err: '#b3402e', run: '#bf5a2a' };
function setStatus(text, type = 'ok') {
  $statusDot.style.background = STATUS_COLOR[type] || '#96896d';
  $statusText.textContent = text;
}

function scrollChat() {
  $chatMessages.scrollTop = $chatMessages.scrollHeight;
}
