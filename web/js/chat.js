'use strict';

// ─────────────────────────────────────────────────
// Vista de chat — burbujas, sugerencias y texto de preparación
// ─────────────────────────────────────────────────
const SUGERENCIAS = [
  'En un lugar de la Mancha',
  'Dulcinea del Toboso',
  'Sancho Panza respondió',
  'El ingenioso hidalgo',
];

function renderChatSuggestions() {
  $chatSuggest.innerHTML = '';
  for (const texto of SUGERENCIAS) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'suggest-chip';
    chip.textContent = texto;
    chip.addEventListener('click', () => {
      if (!S.running) return;
      $userInput.value = texto;
      submitChat();
    });
    $chatSuggest.appendChild(chip);
  }
  $chatSuggest.classList.remove('hidden');
}

// ─────────────────────────────────────────────────
// Constructores compartidos (chat y comparador)
// ─────────────────────────────────────────────────

// Texto corrido con el prompt resaltado
function construirTexto(prefijo, text) {
  const cuerpo = document.createElement('div');
  cuerpo.className = 'bubble-text';
  if (prefijo) {
    const p = document.createElement('span');
    p.className = 'bubble-prompt';
    p.textContent = prefijo;
    cuerpo.appendChild(p);
    cuerpo.appendChild(document.createTextNode(text));
  } else {
    cuerpo.textContent = text;
  }
  return cuerpo;
}

// Vista de tokens: flujo coloreado con separador prompt → generado,
// clic en cada token para ver su ID, y leyenda con los recuentos.
function construirVistaTokens(tokens, tokensPrefijo, ids, idsPrefijo) {
  const nP = (tokensPrefijo || []).length;
  const tv = document.createElement('div');
  tv.className = 'bubble-tokens';

  const flujo = document.createElement('div');
  flujo.className = 'bubble-tok-flow';
  let idx = 0;
  const pintar = (lista, listaIds, esPrompt) => {
    lista.forEach((t, i) => {
      const s = document.createElement('span');
      s.className = `tok-i c${idx % 5}` + (esPrompt ? ' tok-prompt' : '');
      // Espacios y saltos visibles para que se vea qué abarca cada token
      s.textContent = t.replace(/\n/g, '↵').replace(/ /g, '·');
      s.dataset.pos = idx + 1;
      s.dataset.tok = t;
      s.dataset.prompt = esPrompt ? '1' : '';
      if (listaIds && listaIds[i] != null) s.dataset.id = listaIds[i];
      s.addEventListener('click', e => { e.stopPropagation(); seleccionarToken(s); });
      flujo.appendChild(s);
      idx++;
    });
  };
  pintar(tokensPrefijo || [], idsPrefijo || [], true);
  if (nP && tokens.length) {
    const sep = document.createElement('span');
    sep.className = 'tok-sep';
    sep.textContent = '→';
    sep.title = 'Fin del prompt, empieza la generación';
    flujo.appendChild(sep);
  }
  pintar(tokens, ids || [], false);
  tv.appendChild(flujo);

  const cnt = document.createElement('div');
  cnt.className = 'bubble-tok-count';
  if (nP) {
    cnt.innerHTML =
      `<span class="tok-leg"><span class="tok-leg-key prompt"></span>${nP} del prompt</span>` +
      `<span class="tok-leg"><span class="tok-leg-key gen"></span>${tokens.length} generados</span>` +
      `<span class="tok-leg-total">${nP + tokens.length} tokens</span>`;
  } else {
    cnt.innerHTML = `<span class="tok-leg-total">${tokens.length} tokens</span>`;
  }
  tv.appendChild(cnt);
  return tv;
}

function addChatBubble(role, text, prefijo, tokens, tokensPrefijo, ids, idsPrefijo) {
  if ($chatWelcome.style.display !== 'none') $chatWelcome.style.display = 'none';
  const row = document.createElement('div');
  row.className = `bubble-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `bubble-avatar ${role === 'model' ? 'avatar-model' : 'avatar-user'}`;
  avatar.innerHTML = role === 'model'
    ? `<svg width="14" height="16" viewBox="0 0 28 32" fill="currentColor"><path d="M10 32V18L12 14H16L18 18V32H10Z" opacity="0.85"/><path d="M11 14 Q14 9 17 14 Z"/><circle cx="14" cy="13" r="2.4"/><path d="M15.5 11.5 L23 4 Q25 3 25 5 L18 13 Z"/><path d="M15.5 14.5 L23 22 Q25 24 23 25 L16 16 Z"/><path d="M12.5 14.5 L5 22 Q3 24 3 22 L10 14 Z"/><path d="M12.5 11.5 L5 4 Q3 3 5 1 L13.5 10 Z"/></svg>`
    : 'Tú';

  const bubble = document.createElement('div');
  bubble.className = `bubble bubble-${role} tok-dual`;
  bubble.appendChild(construirTexto(prefijo, text));
  if (tokens && tokens.length) {
    bubble.appendChild(construirVistaTokens(tokens, tokensPrefijo, ids, idsPrefijo));
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  $chatBubbleList.appendChild(row);
  scrollChat();
}

// ─────────────────────────────────────────────────
// Etiqueta emergente al hacer clic en un token
// ─────────────────────────────────────────────────
let _tokPop = null;

function cerrarTokenPop() {
  document.querySelector('.tok-i.sel')?.classList.remove('sel');
  if (_tokPop) { _tokPop.remove(); _tokPop = null; }
}

function seleccionarToken(span) {
  const yaSel = span.classList.contains('sel');
  cerrarTokenPop();
  if (yaSel) return;                 // segundo clic sobre el mismo → cerrar
  span.classList.add('sel');

  const id     = span.dataset.id;
  const pos    = span.dataset.pos;
  const esP    = span.dataset.prompt === '1';
  const literal = JSON.stringify(span.dataset.tok);

  const pop = document.createElement('div');
  pop.className = 'tok-pop';
  pop.innerHTML =
    (id != null && id !== ''
      ? `<div class="tok-pop-id">ID <b>${escHtml(id)}</b></div>`
      : '') +
    `<div class="tok-pop-meta">Posición ${escHtml(pos)} · ${esP ? 'prompt' : 'generado'}</div>` +
    `<div class="tok-pop-lit">${escHtml(literal)}</div>`;
  document.body.appendChild(pop);
  _tokPop = pop;

  // Posicionar centrado sobre el token
  const r = span.getBoundingClientRect();
  const pr = pop.getBoundingClientRect();
  let left = r.left + r.width / 2 - pr.width / 2;
  left = Math.max(8, Math.min(left, window.innerWidth - pr.width - 8));
  let top = r.top - pr.height - 8;
  pop.classList.toggle('abajo', top < 8);
  if (top < 8) top = r.bottom + 8;
  pop.style.left = left + 'px';
  pop.style.top  = top + 'px';
}

// Cerrar al hacer clic fuera, al hacer scroll o con Escape
document.addEventListener('click', e => {
  if (_tokPop && !e.target.closest('.tok-i')) cerrarTokenPop();
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') cerrarTokenPop(); });
document.addEventListener('scroll', cerrarTokenPop, true);

function addBubbleInfo(text) {
  const div = document.createElement('div');
  div.className = 'bubble-info fade-up';
  div.textContent = text;
  $chatBubbleList.appendChild(div);
  scrollChat();
}

function appendChatSetup(text) {
  // La carga del modelo se comunica con eventos estructurados; del texto
  // plano del proceso solo interesan advertencias y errores.
  const stripped = stripAnsi(text)
    .split('\n')
    .map(l => l.slice(l.lastIndexOf('\r') + 1))
    .filter(l => /⚠|error/i.test(l))
    .join('\n')
    .trim();
  if (!stripped) return;
  const pre = document.createElement('pre');
  pre.className = 'chat-setup-pre';
  pre.textContent = stripped;
  $chatBubbleList.appendChild(pre);
  scrollChat();
}
