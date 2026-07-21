'use strict';

// ─────────────────────────────────────────────────
// Comparador de modelos — selector múltiple, paneles
// lado a lado y ronda de generación con el mismo prompt.
// ─────────────────────────────────────────────────
const $compView       = $('comp-view');
const $compWelcome    = $('comp-welcome');
const $compWelcomeText= $('comp-welcome-text');
const $compLoadingDots= $('comp-loading');
const $compPromptLine = $('comp-prompt-line');
const $compGrid       = $('comp-grid');
const $compInputBar   = $('comp-input-bar');
const $compInput      = $('comp-input');
const $compSend       = $('comp-send');
const $compCount      = $('comp-count');
const $compTokensBtn  = $('comp-tokens-btn');

const COMP_MAX = 4;
let _compSel = [];   // índices (1-based) seleccionados en el picker

function resetCompView() {
  $compWelcome.style.display = 'flex';
  $compWelcomeText.textContent = 'Buscando checkpoints entrenados…';
  $compLoadingDots.style.display = 'flex';
  document.getElementById('comp-picker-wrap')?.remove();
  document.querySelectorAll('.comp-wrap > .chat-setup-pre, .comp-wrap > .bubble-info').forEach(n => n.remove());
  $compPromptLine.classList.add('hidden');
  $compPromptLine.textContent = '';
  $compGrid.innerHTML = '';
  $compInputBar.classList.add('hidden');
  $compView.classList.remove('ver-tokens');
  $compTokensBtn.classList.remove('active');
  $compTokensBtn.setAttribute('aria-pressed', 'false');
  $compInput.disabled = true;
  $compSend.disabled = true;
  S.compMax = 80;
  $('comp-max-value').textContent = String(S.compMax);
  _compSel = [];
}

// ── Selector múltiple de checkpoints ──────────────────────────────
function renderCompPicker(modelos) {
  $compLoadingDots.style.display = 'none';
  document.getElementById('comp-picker-wrap')?.remove();
  if (!modelos.length) {
    $compWelcomeText.textContent = 'No hay modelos entrenados todavía. Entrena al menos uno para comparar.';
    return;
  }
  $compWelcomeText.textContent = '';
  _compSel = [];

  const wrap = document.createElement('div');
  wrap.id = 'comp-picker-wrap';
  wrap.style.cssText = 'display:flex;flex-direction:column;align-items:center;width:100%;';
  wrap.innerHTML = `
    <div class="ck-picker-heading">Elige hasta ${COMP_MAX} modelos</div>
    <div class="ck-picker-sub">Todos recibirán el mismo prompt, lado a lado</div>`;

  const grid = document.createElement('div');
  grid.className = 'ck-model-grid';

  const go = document.createElement('button');
  go.type = 'button';
  go.className = 'btn btn-primary comp-go';
  go.textContent = 'Comparar';
  go.disabled = true;

  const actualizarGo = () => {
    go.disabled = _compSel.length === 0;
    go.textContent = _compSel.length
      ? `Comparar ${_compSel.length} ${_compSel.length === 1 ? 'modelo' : 'modelos'}`
      : 'Comparar';
  };

  modelos.forEach(m => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'ck-model-card' + (m.tiene_mejor ? ' ck-has-mejor' : '');
    const fecha = m.fecha
      ? `<div class="ck-stat-row"><span class="ck-stat-k">Entrenado</span><span class="ck-stat-v">${escHtml(m.fecha)}</span></div>`
      : '';
    const perp = m.mejor_perp != null
      ? `<div class="ck-stat-row"><span class="ck-stat-k">Perplejidad</span><span class="ck-stat-v">${Number(m.mejor_perp).toFixed(2)}</span></div>`
      : '';
    card.innerHTML = `
      <span class="comp-check">✓</span>
      <div class="ck-model-idx">Modelo ${m.idx}</div>
      <div class="ck-model-display">${escHtml(m.display)}</div>
      <div class="ck-model-folder">${escHtml(m.nombre)}</div>
      <div class="ck-stats-block">
        <div class="ck-stat-row"><span class="ck-stat-k">Pasos</span><span class="ck-stat-v">${(m.pasos||0).toLocaleString()}</span></div>
        ${perp}${fecha}
      </div>`;
    card.addEventListener('click', () => {
      const pos = _compSel.indexOf(m.idx);
      if (pos >= 0) {
        _compSel.splice(pos, 1);
        card.classList.remove('comp-on');
      } else {
        if (_compSel.length >= COMP_MAX) return;
        _compSel.push(m.idx);
        card.classList.add('comp-on');
      }
      actualizarGo();
    });
    grid.appendChild(card);
  });

  go.addEventListener('click', () => {
    if (!_compSel.length) return;
    document.getElementById('comp-picker-wrap')?.remove();
    $compWelcomeText.textContent = 'Cargando modelos…';
    $compLoadingDots.style.display = 'flex';
    wsSend({ action: 'input', value: _compSel.join(',') });
  });

  wrap.appendChild(grid);
  wrap.appendChild(go);
  $compWelcome.appendChild(wrap);
}

// ── Paneles ───────────────────────────────────────────────────────
function compPanel(slot) {
  return $compGrid.querySelector(`.comp-panel[data-slot="${slot}"]`);
}

const _ESTADO_HTML = txt =>
  `<div class="comp-estado"><span class="comp-estado-txt">${txt}</span></div>`;

function compLoadingEv(ev) {
  $compWelcome.style.display = 'none';
  let p = compPanel(ev.slot);
  if (!p) {
    p = document.createElement('div');
    p.className = 'comp-panel';
    p.dataset.slot = ev.slot;
    p.innerHTML = `
      <div class="comp-panel-head">
        <div class="comp-head-row">
          <div class="comp-panel-title"></div>
          <span class="comp-temp-ctl stepper hidden" title="Temperatura de este panel">
            <button type="button" class="ct-minus" aria-label="Bajar temperatura">−</button>
            <output>0.8</output>
            <button type="button" class="ct-plus" aria-label="Subir temperatura">+</button>
          </span>
        </div>
        <div class="comp-panel-sub"></div>
        <div class="comp-panel-chips hidden"></div>
      </div>
      <div class="comp-panel-body tok-dual">${_ESTADO_HTML('Cargando…')}</div>`;

    // Temperatura por panel: el backend acepta "temp <panel> <valor>" (1-based)
    const ctl = p.querySelector('.comp-temp-ctl');
    const out = ctl.querySelector('output');
    let timer = 0;
    const ajustar = delta => {
      const t = Math.min(2.0, Math.max(0.1, Math.round((parseFloat(out.textContent) + delta) * 10) / 10));
      out.textContent = t.toFixed(1);
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (S.running && S.cat === 'comp') {
          wsSend({ action: 'input', value: `temp ${Number(ev.slot) + 1} ${out.textContent}` });
        }
      }, 450);
    };
    ctl.querySelector('.ct-minus').addEventListener('click', () => ajustar(-0.1));
    ctl.querySelector('.ct-plus').addEventListener('click',  () => ajustar(+0.1));

    $compGrid.appendChild(p);
  }
  p.querySelector('.comp-panel-title').textContent = ev.display || ev.nombre || 'Modelo';
  p.querySelector('.comp-panel-sub').textContent = ev.nombre || '';
}

function compLoadedEv(ev) {
  const p = compPanel(ev.slot);
  if (!p) return;
  p.querySelector('.comp-panel-sub').textContent = ev.nombre || '';

  const body = p.querySelector('.comp-panel-body');
  body.innerHTML = '';
  if (ev.ok) {
    // Chips con los datos del checkpoint + stepper de temperatura visible
    const chips = p.querySelector('.comp-panel-chips');
    chips.innerHTML = '';
    const chip = (v, k) => {
      const c = document.createElement('span');
      c.className = 'comp-chip';
      c.innerHTML = `<b>${v}</b> ${k}`;
      chips.appendChild(c);
    };
    if (ev.pasos)              chip(Number(ev.pasos).toLocaleString(), 'pasos');
    if (ev.mejor_perp != null) chip(Number(ev.mejor_perp).toFixed(1), 'perp');
    if (ev.vocab)              chip(Number(ev.vocab).toLocaleString(), 'vocab');
    chips.classList.toggle('hidden', !chips.children.length);

    const ctl = p.querySelector('.comp-temp-ctl');
    if (ev.temp != null) ctl.querySelector('output').textContent = Number(ev.temp).toFixed(1);
    ctl.classList.remove('hidden');

    body.innerHTML = '<div class="comp-vacio">Esperando prompt…</div>';
  } else {
    p.classList.add('err');
    const e = document.createElement('div');
    e.className = 'comp-vacio err';
    e.textContent = 'No se pudo cargar: ' + (ev.error || 'error desconocido');
    body.appendChild(e);
  }
}

function compReadyEv(ev) {
  $compWelcome.style.display = 'none';
  $compLoadingDots.style.display = 'none';
  $compInputBar.classList.remove('hidden');
  $compCount.textContent = `${ev.n} ${ev.n === 1 ? 'panel listo' : 'paneles listos'}`;
  if (ev.max_tok != null) {
    S.compMax = Number(ev.max_tok);
    $('comp-max-value').textContent = String(S.compMax);
  }
  $compInput.disabled = !S.running;
  $compSend.disabled = !S.running;
  if (S.running) $compInput.focus();
}

function compPromptEv(ev) {
  $compPromptLine.classList.remove('hidden');
  $compPromptLine.textContent = `❝ ${ev.text} ❞`;
  $compGrid.querySelectorAll('.comp-panel:not(.err) .comp-panel-body').forEach(b => {
    b.innerHTML = _ESTADO_HTML('En cola…');
  });
}

function compGenEv(ev) {
  const t = compPanel(ev.slot)?.querySelector('.comp-estado-txt');
  if (t) t.textContent = 'Generando…';
}

function compResultEv(ev) {
  const p = compPanel(ev.slot);
  if (!p) return;
  const body = p.querySelector('.comp-panel-body');
  body.innerHTML = '';
  body.appendChild(construirTexto(ev.prompt, ev.text));
  if (ev.tokens && ev.tokens.length) {
    body.appendChild(construirVistaTokens(ev.tokens, ev.tokens_prompt || null, ev.ids || null, ev.ids_prompt || null));
  }
  // Pie con los datos de la generación
  const datos = [];
  if (ev.tokens)       datos.push(`${ev.tokens.length} tokens`);
  if (ev.segs != null) datos.push(`${ev.segs} s`);
  if (ev.temp != null) datos.push(`temp ${Number(ev.temp).toFixed(1)}`);
  if (datos.length) {
    const foot = document.createElement('div');
    foot.className = 'comp-foot';
    foot.textContent = datos.join(' · ');
    body.appendChild(foot);
  }
  body.classList.add('fade-up');
}

// ── Mensajes sueltos en la vista del comparador ───────────────────
function appendCompInfo(text) {
  const div = document.createElement('div');
  div.className = 'bubble-info fade-up';
  div.textContent = text;
  $compGrid.before(div);
}

function appendCompSetup(text) {
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
  $compGrid.before(pre);
}

// ── Envío del prompt ──────────────────────────────────────────────
function submitComp() {
  const val = $compInput.value.trim();
  if (!val || !S.running) return;
  wsSend({ action: 'input', value: val });
  $compInput.value = '';
  $compInput.focus();
}

$compSend.addEventListener('click', submitComp);
$compInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitComp(); }
});
$compTokensBtn.addEventListener('click', () => {
  const activo = $compView.classList.toggle('ver-tokens');
  $compTokensBtn.classList.toggle('active', activo);
  $compTokensBtn.setAttribute('aria-pressed', String(activo));
});

// ── Máximo de tokens (global para todos los paneles) ──────────────
let _compMaxTimer = 0;
function ajustarCompMax(delta) {
  S.compMax = Math.min(400, Math.max(20, S.compMax + delta));
  $('comp-max-value').textContent = String(S.compMax);
  clearTimeout(_compMaxTimer);
  _compMaxTimer = setTimeout(() => {
    if (S.running && S.cat === 'comp') wsSend({ action: 'input', value: `max ${S.compMax}` });
  }, 450);
}
$('comp-max-minus').addEventListener('click', () => ajustarCompMax(-20));
$('comp-max-plus').addEventListener('click',  () => ajustarCompMax(+20));
