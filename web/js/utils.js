'use strict';

// ─────────────────────────────────────────────────
// Utilidades: texto, estado, scroll
// ─────────────────────────────────────────────────
function stripAnsi(s) {
  return s.replace(/\x1b\[[0-9;?]*[a-zA-Z]/g, '');
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

const STATUS_COLOR = { ok: '#3fb950', err: '#f85149', run: '#d29922' };
function setStatus(text, type = 'ok') {
  $statusDot.style.background = STATUS_COLOR[type] || '#374151';
  $statusText.textContent = text;
}

function scrollChat() {
  $chatMessages.scrollTop = $chatMessages.scrollHeight;
}
