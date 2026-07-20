'use strict';

// ─────────────────────────────────────────────────
// Modal de arquitectura — diagrama SVG animado con el
// lenguaje visual de la presentación Manim (colores.py):
// tarjetas crema con sombra, atención terracota, MLP oliva,
// residuales que saltan por arriba y llave "× N capas".
// ─────────────────────────────────────────────────
const $archModal   = $('arch-modal');
const $archName    = $('arch-name');
const $archSub     = $('arch-sub');
const $archDiagram = $('arch-diagram');

// Paleta de la presentación (presentation/colores.py)
const AD = {
  marron:    '#3D3834',  // MARRON_OSCURO — trazos y flechas
  terracota: '#A36536',  // NARANJA_TERRACOTA — atención y residuales
  oliva:     '#6B8E23',  // VERDE_OLIVA — MLP
  cajaInf:   '#E0C2A8',  // CAJA_INFERIOR — entrada / salida
  fondoCaja: '#FCF3E4',  // FONDO_CAJA — círculos "+" y normas
  celeste:   '#C9E4F5',  // CELESTE_PALIDO — embedding de tokens
  lavanda:   '#C4C4FF',  // LAVANDA — embedding posicional
  amarillo:  '#FFFFCC',  // AMARILLO_PALIDO — softmax
  tinta:     '#1A1A1A',  // TINTA_NEGRA
  blanco:    '#FFFFFF',
  pulso:     '#BF5A2A',
};

function _tarjeta(cx, cy, w, h, fill, color, titulo, dim, delay) {
  const x = cx - w / 2, y = cy - h / 2;
  const ty = dim ? cy - 6 : cy + 6;
  const dimTxt = dim
    ? `<text class="ad2-dim" x="${cx}" y="${cy + 18}" text-anchor="middle" fill="${color === AD.blanco ? 'rgba(255,255,255,.85)' : '#5a4a33'}">${dim}</text>`
    : '';
  return `
    <g class="ad2-card ad2-in" style="animation-delay:${delay}s">
      <rect x="${x + 4}" y="${y + 5}" width="${w}" height="${h}" rx="12" fill="${AD.marron}" opacity="0.13"/>
      <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${fill}" stroke="${AD.marron}" stroke-width="2.5"/>
      <text class="ad2-title" x="${cx}" y="${ty}" text-anchor="middle" fill="${color}">${titulo}</text>
      ${dimTxt}
    </g>`;
}

function _suma(cx, cy, delay) {
  return `
    <g class="ad2-pop" style="animation-delay:${delay}s">
      <circle cx="${cx}" cy="${cy}" r="16" fill="${AD.fondoCaja}" stroke="${AD.terracota}" stroke-width="3"/>
      <text class="ad2-plus" x="${cx}" y="${cy + 6}" text-anchor="middle" fill="${AD.terracota}">+</text>
    </g>`;
}

function _flecha(x1, x2, y, delay, color = AD.marron) {
  const dir = x2 > x1 ? 1 : -1;
  const fin = x2 - dir * 9;
  return `
    <line class="ad2-draw" style="animation-delay:${delay}s" x1="${x1}" y1="${y}" x2="${fin}" y2="${y}" stroke="${color}" stroke-width="3" pathLength="1"/>
    <polygon class="ad2-pop" style="animation-delay:${delay + 0.18}s" fill="${color}"
      points="${x2},${y} ${x2 - dir * 11},${y - 5.5} ${x2 - dir * 11},${y + 5.5}"/>`;
}

function _flechaAbajo(x, y1, y2, delay, color = AD.marron) {
  return `
    <line class="ad2-draw" style="animation-delay:${delay}s" x1="${x}" y1="${y1}" x2="${x}" y2="${y2 - 9}" stroke="${color}" stroke-width="3" pathLength="1"/>
    <polygon class="ad2-pop" style="animation-delay:${delay + 0.18}s" fill="${color}"
      points="${x},${y2} ${x - 5.5},${y2 - 11} ${x + 5.5},${y2 - 11}"/>`;
}

function _codo(puntos, delay, color = AD.marron) {
  // Conector en ángulo recto que termina con flecha hacia abajo
  const d = 'M ' + puntos.map(p => p.join(' ')).join(' L ');
  const [fx, fy] = puntos[puntos.length - 1];
  return `
    <path class="ad2-draw" style="animation-delay:${delay}s" d="${d}" fill="none" stroke="${color}" stroke-width="3" pathLength="1"/>
    <polygon class="ad2-pop" style="animation-delay:${delay + 0.3}s" fill="${color}"
      points="${fx},${fy + 9} ${fx - 5.5},${fy - 2} ${fx + 5.5},${fy - 2}"/>`;
}

function construirDiagrama(d) {
  const fmt = x => x.toLocaleString('es');
  const oculto = d.embd * 4;
  const dimCabeza = Math.round(d.embd / d.cabezas);
  const reducirMovimiento = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ── Fila 1: entrada y embeddings ──────────────────────────────
  const Y1 = 64;
  const fila1 =
    _tarjeta(110, Y1, 160, 58, AD.cajaInf, AD.tinta, 'Entrada', `${fmt(d.ctx)} tokens`, 0) +
    _flecha(190, 240, Y1, 0.15) +
    _tarjeta(335, Y1, 190, 58, AD.celeste, AD.tinta, 'Embedding tokens', `${fmt(d.vocab)} × ${fmt(d.embd)}`, 0.3) +
    _flecha(430, 452, Y1, 0.45) +
    _suma(468, Y1, 0.55) +
    _flecha(584, 484, Y1, 0.45) +
    _tarjeta(679, Y1, 190, 58, AD.lavanda, AD.tinta, 'Embedding posicional', `${fmt(d.ctx)} × ${fmt(d.embd)}`, 0.3) +
    _codo([[468, 82], [468, 130], [90, 130], [90, 231]], 0.7);

  // ── Fila 2: la capa Transformer (como en la presentación) ─────
  const Y2 = 280, YS = 208;
  const fila2 =
    _tarjeta(90, Y2, 120, 80, AD.cajaInf, AD.tinta, 'Input', `${fmt(d.ctx)} × ${fmt(d.embd)}`, 1.0) +
    _flecha(150, 195, Y2, 1.15) +
    _tarjeta(290, Y2, 190, 80, AD.terracota, AD.blanco, 'Self-Attention', `${d.cabezas} ${d.cabezas === 1 ? 'cabeza' : 'cabezas'} · ${fmt(dimCabeza)} dims`, 1.25) +
    _flecha(385, 430, Y2, 1.4) +
    _suma(446, Y2, 1.5) +
    // residual 1: rodea la atención por arriba
    `<path class="ad2-draw" style="animation-delay:1.55s" d="M 172 ${Y2} L 172 ${YS} L 446 ${YS} L 446 ${Y2 - 28}" fill="none" stroke="${AD.terracota}" stroke-width="3" pathLength="1"/>` +
    `<polygon class="ad2-pop" style="animation-delay:1.85s" fill="${AD.terracota}" points="446,${Y2 - 19} 440.5,${Y2 - 30} 451.5,${Y2 - 30}"/>` +
    _flecha(462, 507, Y2, 1.7) +
    _tarjeta(572, Y2, 130, 80, AD.oliva, AD.blanco, 'MLP', `${fmt(d.embd)} → ${fmt(oculto)} → ${fmt(d.embd)}`, 1.8) +
    _flecha(637, 682, Y2, 1.95) +
    _suma(698, Y2, 2.05) +
    // residual 2: rodea el MLP por arriba
    `<path class="ad2-draw" style="animation-delay:2.1s" d="M 484 ${Y2} L 484 ${YS} L 698 ${YS} L 698 ${Y2 - 28}" fill="none" stroke="${AD.terracota}" stroke-width="3" pathLength="1"/>` +
    `<polygon class="ad2-pop" style="animation-delay:2.4s" fill="${AD.terracota}" points="698,${Y2 - 19} 692.5,${Y2 - 30} 703.5,${Y2 - 30}"/>` +
    _flecha(714, 759, Y2, 2.25) +
    _tarjeta(824, Y2, 130, 80, AD.cajaInf, AD.tinta, 'Output', `${fmt(d.ctx)} × ${fmt(d.embd)}`, 2.35) +
    // llave inferior "× N capas"
    `<path class="ad2-draw" style="animation-delay:2.55s" d="M 195 344 L 195 356 L 714 356 L 714 344" fill="none" stroke="${AD.marron}" stroke-width="2.5" pathLength="1"/>` +
    `<text class="ad2-capas ad2-in" style="animation-delay:2.7s" x="454" y="382" text-anchor="middle" fill="${AD.marron}">× ${d.capas} capas</text>`;

  // ── Fila 3: cabeza del modelo ─────────────────────────────────
  const Y3 = 480;
  const fila3 =
    _codo([[824, 320], [824, 410], [110, 410], [110, 442]], 2.8) +
    _tarjeta(110, Y3, 160, 58, AD.fondoCaja, AD.tinta, 'LayerNorm final', '', 3.0) +
    _flecha(190, 240, Y3, 3.1) +
    _tarjeta(360, Y3, 240, 58, AD.cajaInf, AD.tinta, 'Proyección al vocabulario', `${fmt(d.embd)} → ${fmt(d.vocab)} logits`, 3.2) +
    _flecha(480, 525, Y3, 3.3) +
    _tarjeta(660, Y3, 270, 58, AD.amarillo, AD.tinta, 'Softmax', 'probabilidad del siguiente token', 3.4);

  // ── Pulsos: un token recorriendo el flujo en bucle ────────────
  const pulsos = reducirMovimiento ? '' : `
    <circle class="ad2-pulse-dot" r="5.5" fill="${AD.pulso}" opacity="0">
      <animate attributeName="opacity" values="0;1" dur="0.01s" begin="3.9s" fill="freeze"/>
      <animateMotion dur="5s" begin="3.9s" repeatCount="indefinite" path="M 90 ${Y2} L 824 ${Y2}"/>
    </circle>
    <circle class="ad2-pulse-dot" r="4.5" fill="${AD.terracota}" opacity="0">
      <animate attributeName="opacity" values="0;1" dur="0.01s" begin="4.6s" fill="freeze"/>
      <animateMotion dur="5s" begin="4.6s" repeatCount="indefinite" path="M 172 ${Y2} L 172 ${YS} L 446 ${YS} L 446 ${Y2}"/>
    </circle>`;

  return `
    <svg class="ad2" viewBox="0 0 960 520" role="img" aria-label="Diagrama de la arquitectura GPT-2">
      <g transform="translate(0,-6)">
        ${pulsos}
        ${fila1}
        ${fila2}
        ${fila3}
      </g>
    </svg>`;
}

function abrirArch(card) {
  if (!card) return;
  const d = {
    embd:    parseInt(card.dataset.embd, 10),
    capas:   parseInt(card.dataset.capas, 10),
    cabezas: parseInt(card.dataset.cabezas, 10),
    ctx:     parseInt(card.dataset.ctx, 10),
    vocab:   parseInt(card.dataset.vocab, 10),
  };
  $archName.textContent = card.dataset.name || '';
  $archSub.textContent  = `${card.dataset.params} parámetros · embd ${d.embd} · ${d.capas} capas · contexto ${d.ctx}`;
  $archDiagram.innerHTML = construirDiagrama(d);
  $archModal.classList.remove('hidden');
  $('arch-close').focus();
}

function cerrarArch() {
  $archModal.classList.add('hidden');
  $archDiagram.innerHTML = '';
}

document.querySelectorAll('.model-info-btn').forEach(btn => {
  btn.addEventListener('click', e => {
    e.stopPropagation();
    abrirArch(btn.closest('.model-card'));
  });
});
$('arch-close').addEventListener('click', cerrarArch);
$('arch-backdrop').addEventListener('click', cerrarArch);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !$archModal.classList.contains('hidden')) cerrarArch();
});
