'use strict';

// ─────────────────────────────────────────────────
// Vista de entrenamiento — parseo del progreso (Rust),
// curva de pérdida interactiva y tarjetas de muestra.
// Formato de línea (Rust):
// "Paso  123 | Tiempo:   45.6s (+1.2s) | TA: 0.000300 | Entren: 2.3456 | Val: 2.4567 | Perplejidad: 11.67"
// ─────────────────────────────────────────────────
const STEP_RE  = /Paso\s+(\d+)/;
const TIME_RE  = /Tiempo:\s*([\d.]+)s/;
const LOSS_RE  = /Entren[.:]?\s*([\d.]+)/;
const VLOS_RE  = /Val[.:]?\s*([\d.]+)/;
const LR_RE    = /TA:\s*([\d.e+\-]+)/;
const PERP_RE  = /Perplejidad:\s*([\d.]+)/;

function parseTrainProgress(text) {
  for (const line of text.split('\n')) {
    const stepM = STEP_RE.exec(line);
    if (!stepM) {
      // Salida previa al bucle → registro crudo colapsable de la tarjeta
      if (!S.trainStarted && line.trim()) {
        const t = stripAnsi(line).trim();
        if (t) {
          $prepCard.classList.remove('hidden');
          $vocabLog.textContent += t + '\n';
          $vocabLog.scrollTop = $vocabLog.scrollHeight;
        }
      }
      continue;
    }
    const step  = parseInt(stepM[1], 10);
    const total = S.totalSteps || 1;
    const pct   = Math.min(100, (step / total) * 100);

    $progressFill.style.width = pct.toFixed(1) + '%';
    $trainStepLabel.textContent = `Paso ${step.toLocaleString()} / ${total.toLocaleString()}`;
    $trainPctLabel.textContent  = pct.toFixed(0) + '%';

    // Tiempos: transcurrido, velocidad y restante
    const timeM = TIME_RE.exec(line);
    if (timeM) {
      S.elapsedSec = parseFloat(timeM[1]);
      $ttElapsed.textContent = fmtDur(S.elapsedSec);
      if (step > 0 && S.elapsedSec > 0) {
        const rate = step / S.elapsedSec;
        $ttSpeed.textContent = fmtRate(rate);
        if (S.totalSteps) $ttEta.textContent = fmtDur((S.totalSteps - step) / rate);
      }
    }

    const entry = { step, train: NaN, val: NaN };

    const lossM = LOSS_RE.exec(line);
    if (lossM) { entry.train = parseFloat(lossM[1]); $metricTrain.textContent = entry.train.toFixed(4); }

    const vlosM = VLOS_RE.exec(line);
    if (vlosM) {
      entry.val = parseFloat(vlosM[1]);
      $metricVal.textContent = entry.val.toFixed(4);
      if (!S.bestVal || entry.val < S.bestVal.val) {
        S.bestVal = { val: entry.val, step };
        $metricValBest.textContent = `mejor: ${entry.val.toFixed(4)} · paso ${step.toLocaleString()}`;
      }
    }

    const lrM = LR_RE.exec(line);
    if (lrM) $metricLr.textContent = parseFloat(lrM[1]).toExponential(2);

    const perpM = PERP_RE.exec(line);
    if (perpM) $metricPerp.textContent = parseFloat(perpM[1]).toFixed(2);

    if (!isNaN(entry.train) || !isNaN(entry.val)) {
      S.lossHistory.push(entry);
      if (S.lossHistory.length > 800) S.lossHistory.splice(0, S.lossHistory.length - 800);
      scheduleChartDraw();
    }
  }
}

// ─────────────────────────────────────────────────
// Curva de pérdida
// ─────────────────────────────────────────────────
let _chartRaf = 0;
let _chart    = null;   // geometría del último render (para el tooltip)
let _kbIdx    = -1;     // índice seleccionado con teclado

function scheduleChartDraw() {
  if (_chartRaf) return;
  _chartRaf = requestAnimationFrame(() => { _chartRaf = 0; drawLossChart(); });
}

// Ticks "bonitos": valores redondos que cubren [min, max]
function niceTicks(min, max, n) {
  const span = (max - min) || 1;
  const paso0 = span / Math.max(1, n);
  const mag   = Math.pow(10, Math.floor(Math.log10(paso0)));
  const norm  = paso0 / mag;
  const paso  = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const ticks = [];
  for (let v = Math.ceil(min / paso) * paso; v <= max + 1e-9; v += paso) ticks.push(v);
  const dec = Math.max(0, -Math.floor(Math.log10(paso)));
  return { ticks, dec };
}

function drawLossChart() {
  if (S.lossHistory.length < 2) return;
  const W = $lossChartBody.clientWidth  || 700;
  const H = Math.max(180, $lossChartBody.clientHeight || 0) || 260;
  const pad = { t: 14, r: 64, b: 26, l: 48 };
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;

  const maxStepDatos = Math.max(...S.lossHistory.map(p => p.step));
  const maxStep = S.chartFit ? maxStepDatos : (S.totalSteps || maxStepDatos);
  const allLoss = [];
  for (const p of S.lossHistory) {
    if (!isNaN(p.train)) allLoss.push(p.train);
    if (!isNaN(p.val))   allLoss.push(p.val);
  }
  if (!allLoss.length) return;

  let minL = Math.min(...allLoss), maxL = Math.max(...allLoss);
  const margen = (maxL - minL) * 0.08 || 0.1;
  minL -= margen; maxL += margen;

  const X = s => pad.l + (s / maxStep) * cw;
  const Y = l => pad.t + (1 - (l - minL) / (maxL - minL)) * ch;
  _chart = { pad, cw, ch, maxStep, minL, maxL, X, Y };

  // Rejilla y ticks del eje Y
  const { ticks: yTicks, dec } = niceTicks(minL, maxL, 4);
  const grid = yTicks.map(v => {
    const y = Y(v).toFixed(1);
    return `<line class="lc-grid" x1="${pad.l}" y1="${y}" x2="${pad.l + cw}" y2="${y}"/>` +
           `<text class="lc-tick" x="${pad.l - 7}" y="${(+y + 3.5).toFixed(1)}" text-anchor="end">${v.toFixed(dec)}</text>`;
  }).join('');

  // Ticks del eje X (0 · 25 · 50 · 75 · 100 % de los pasos)
  const xTicks = [0, .25, .5, .75, 1].map(f => {
    const s = Math.round(maxStep * f);
    const anchor = f === 0 ? 'start' : f === 1 ? 'end' : 'middle';
    return `<text class="lc-tick" x="${X(s).toFixed(1)}" y="${pad.t + ch + 17}" text-anchor="${anchor}">${s.toLocaleString()}</text>`;
  }).join('');

  // Polilíneas de las series
  const pts = key => S.lossHistory
    .filter(p => !isNaN(p[key]))
    .map(p => `${X(p.step).toFixed(1)},${Y(p[key]).toFixed(1)}`)
    .join(' ');
  const trainPts = pts('train');
  const valPts   = pts('val');

  // Último punto de cada serie: marcador + etiqueta directa con el valor
  const finales = [];
  for (const key of ['train', 'val']) {
    const seq = S.lossHistory.filter(p => !isNaN(p[key]));
    if (seq.length) {
      const u = seq[seq.length - 1];
      finales.push({ key, x: X(u.step), y: Y(u[key]), v: u[key] });
    }
  }
  // Evitar colisión entre las dos etiquetas finales
  if (finales.length === 2 && Math.abs(finales[0].y - finales[1].y) < 13) {
    const mid = (finales[0].y + finales[1].y) / 2;
    const [arriba, abajo] = finales[0].y <= finales[1].y ? [finales[0], finales[1]] : [finales[1], finales[0]];
    arriba.ly = mid - 7; abajo.ly = mid + 7;
  }
  const endMarks = finales.map(f =>
    `<circle class="lc-dot-${f.key}" cx="${f.x.toFixed(1)}" cy="${f.y.toFixed(1)}" r="4.5"/>` +
    `<text class="lc-endlabel" x="${(f.x + 9).toFixed(1)}" y="${((f.ly ?? f.y) + 3.5).toFixed(1)}">${f.v.toFixed(3)}</text>`
  ).join('');

  // Marcador de la mejor pérdida de validación
  const best = S.bestVal
    ? `<circle class="lc-best" cx="${X(S.bestVal.step).toFixed(1)}" cy="${Y(S.bestVal.val).toFixed(1)}" r="5.5"/>`
    : '';

  $lossChart.setAttribute('viewBox', `0 0 ${W} ${H}`);
  $lossChart.setAttribute('preserveAspectRatio', 'none');
  $lossChart.innerHTML = `
    ${grid}
    <line class="lc-axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t + ch}"/>
    <line class="lc-axis" x1="${pad.l}" y1="${pad.t + ch}" x2="${pad.l + cw}" y2="${pad.t + ch}"/>
    ${xTicks}
    ${trainPts ? `<polyline class="lc-line-train" points="${trainPts}"/>` : ''}
    ${valPts   ? `<polyline class="lc-line-val"   points="${valPts}"/>`   : ''}
    ${best}
    ${endMarks}
    <line id="lc-cross" class="lc-cross" y1="${pad.t}" y2="${pad.t + ch}" style="display:none"/>
  `;
  $lossChartEmpty.style.display = 'none';
  $lossChart.style.display = 'block';
  bindChartInteraction();
}

// ─────────────────────────────────────────────────
// Interacción: crosshair + tooltip (puntero y teclado)
// ─────────────────────────────────────────────────
function nearestEntry(step) {
  let mejor = null, dist = Infinity;
  for (const p of S.lossHistory) {
    const d = Math.abs(p.step - step);
    if (d < dist) { dist = d; mejor = p; }
  }
  return mejor;
}

function showChartTip(entry) {
  if (!entry || !_chart) return;
  const cross = $('lc-cross');
  const x = _chart.X(entry.step);
  if (cross) {
    cross.setAttribute('x1', x.toFixed(1));
    cross.setAttribute('x2', x.toFixed(1));
    cross.style.display = 'block';
  }
  $lcTipStep.textContent  = `Paso ${entry.step.toLocaleString()}`;
  $lcTipTrain.textContent = isNaN(entry.train) ? '—' : entry.train.toFixed(4);
  $lcTipVal.textContent   = isNaN(entry.val)   ? '—' : entry.val.toFixed(4);
  $lcTip.style.display = 'block';
  // Posicionar: a la derecha del crosshair, o a la izquierda cerca del borde
  const bw = $lossChartBody.clientWidth;
  const px = x / ((_chart.pad.l + _chart.cw + _chart.pad.r) || 1) * bw;
  const tw = $lcTip.offsetWidth || 160;
  $lcTip.style.left = (px + 14 + tw > bw ? Math.max(4, px - tw - 14) : px + 14) + 'px';
  $lcTip.style.top  = '10px';
}

function hideChartTip() {
  $lcTip.style.display = 'none';
  const cross = $('lc-cross');
  if (cross) cross.style.display = 'none';
  _kbIdx = -1;
}

function bindChartInteraction() {
  if (bindChartInteraction._bound) return;
  bindChartInteraction._bound = true;

  window.addEventListener('resize', () => { if (S.lossHistory.length >= 2) scheduleChartDraw(); });

  $lossChartBody.addEventListener('pointermove', e => {
    if (!_chart || S.lossHistory.length < 2) return;
    const rect = $lossChartBody.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;            // fracción del ancho visible
    const sx = fx * (_chart.pad.l + _chart.cw + _chart.pad.r);  // coordenada en el viewBox
    const step = Math.round(((sx - _chart.pad.l) / _chart.cw) * _chart.maxStep);
    showChartTip(nearestEntry(step));
  });
  $lossChartBody.addEventListener('pointerleave', hideChartTip);

  $lossChartBody.addEventListener('keydown', e => {
    if (!S.lossHistory.length) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      if (_kbIdx < 0) _kbIdx = S.lossHistory.length - 1;
      else _kbIdx += e.key === 'ArrowRight' ? 1 : -1;
      _kbIdx = Math.max(0, Math.min(S.lossHistory.length - 1, _kbIdx));
      showChartTip(S.lossHistory[_kbIdx]);
    } else if (e.key === 'Escape') {
      hideChartTip();
    }
  });
  $lossChartBody.addEventListener('blur', hideChartTip);
}

// ─────────────────────────────────────────────────
// Tarjetas de muestra (panel derecho)
// ─────────────────────────────────────────────────
function addSampleCard(prompt, text, temp) {
  const card = document.createElement('div');
  card.className = 'sample-card';
  card.innerHTML = `
    <div class="sample-prompt">
      ${escHtml(prompt)}
      <span class="sample-temp">temperatura = ${escHtml(temp)}</span>
    </div>
    <div class="sample-text">${escHtml(text)}</div>
  `;
  $samplesList.prepend(card);
  if ($trainContent) $trainContent.scrollTop = 0;
}
