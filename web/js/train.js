'use strict';

// ─────────────────────────────────────────────────
// Vista de entrenamiento — parseo del progreso (Rust),
// curva de pérdida y tarjetas de muestra.
// Formato esperado: "Paso X/Y | pérd: Z | pérd_val: W | LR: V"
// ─────────────────────────────────────────────────
const STEP_RE  = /Paso\s+(\d+)/;
const LOSS_RE  = /Entren[.:]?\s*([\d.]+)/;
const VLOS_RE  = /Val[.:]?\s*([\d.]+)/;
const LR_RE    = /TA:\s*([\d.e+\-]+)/;
const PERP_RE  = /Perplejidad:\s*([\d.]+)/;

function parseTrainProgress(text) {
  for (const line of text.split('\n')) {
    const stepM = STEP_RE.exec(line);
    if (!stepM) {
      if (!S.trainStarted && line.trim()) {
        const t = stripAnsi(line).trim();
        if (t) {
          $vocabLog.style.display = 'block';
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

    const entry = { step, train: NaN, val: NaN };

    const lossM = LOSS_RE.exec(line);
    if (lossM) { entry.train = parseFloat(lossM[1]); $metricTrain.textContent = entry.train.toFixed(4); }

    const vlosM = VLOS_RE.exec(line);
    if (vlosM) { entry.val = parseFloat(vlosM[1]); $metricVal.textContent = entry.val.toFixed(4); }

    const lrM = LR_RE.exec(line);
    if (lrM) $metricLr.textContent = parseFloat(lrM[1]).toExponential(2);

    const perpM = PERP_RE.exec(line);
    if (perpM) $metricPerp.textContent = parseFloat(perpM[1]).toFixed(2);

    if (!isNaN(entry.train) || !isNaN(entry.val)) {
      S.lossHistory.push(entry);
      if (S.lossHistory.length > 800) S.lossHistory.splice(0, S.lossHistory.length - 800);
      drawLossChart();
    }
  }
}

function drawLossChart() {
  if (S.lossHistory.length < 2) return;
  const body = $lossChart.parentElement;
  const W = (body?.clientWidth  || 700);
  const H = Math.max(160, body?.clientHeight || 0) || Math.min(400, 200 + S.lossHistory.length * 0.3);
  const pad = { t: 14, r: 20, b: 28, l: 52 };
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;

  const maxStep = S.totalSteps || Math.max(...S.lossHistory.map(p => p.step));
  const trains  = S.lossHistory.map(p => p.train).filter(v => !isNaN(v));
  const vals    = S.lossHistory.map(p => p.val).filter(v => !isNaN(v));
  const allLoss = [...trains, ...vals];
  if (!allLoss.length) return;

  let minL = Math.min(...allLoss); let maxL = Math.max(...allLoss);
  const pad_ = (maxL - minL) * 0.08 || 0.1;
  minL -= pad_; maxL += pad_;

  const X = s => pad.l + (s / maxStep) * cw;
  const Y = l => pad.t + (1 - (l - minL) / (maxL - minL)) * ch;

  const pts = arr => arr.filter(p => !isNaN(p.v)).map(p => `${X(p.s).toFixed(1)},${Y(p.v).toFixed(1)}`).join(' ');
  const trainPts = pts(S.lossHistory.map(p => ({ s: p.step, v: p.train })));
  const valPts   = pts(S.lossHistory.map(p => ({ s: p.step, v: p.val })));

  // Y grid lines (3 levels)
  const gridLines = [0, 0.5, 1].map(t => {
    const lv = minL + t * (maxL - minL);
    const y = Y(lv);
    return `<line x1="${pad.l}" y1="${y.toFixed(1)}" x2="${pad.l + cw}" y2="${y.toFixed(1)}" stroke="#d9c8aa" stroke-width="1" stroke-dasharray="4,3"/>
            <text x="${(pad.l - 6).toFixed(0)}" y="${(y + 4).toFixed(0)}" fill="#a36536" font-size="10" text-anchor="end">${lv.toFixed(2)}</text>`;
  }).join('');

  // X axis labels
  const x0 = pad.l; const x1 = pad.l + cw;
  const xLabels = `<text x="${x0}" y="${pad.t + ch + 18}" fill="#a36536" font-size="10" text-anchor="middle">0</text>
                   <text x="${x1}" y="${pad.t + ch + 18}" fill="#a36536" font-size="10" text-anchor="end">${maxStep.toLocaleString()}</text>`;

  $lossChart.setAttribute('viewBox', `0 0 ${W} ${H}`);
  $lossChart.setAttribute('preserveAspectRatio', 'none');
  $lossChart.innerHTML = `
    ${gridLines}
    <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t + ch}" stroke="#c9a06a" stroke-width="1"/>
    <line x1="${pad.l}" y1="${pad.t + ch}" x2="${pad.l + cw}" y2="${pad.t + ch}" stroke="#c9a06a" stroke-width="1"/>
    ${trainPts ? `<polyline points="${trainPts}" fill="none" stroke="#a36536" stroke-width="2.5" stroke-linejoin="round"/>` : ''}
    ${valPts   ? `<polyline points="${valPts}"   fill="none" stroke="#6b8e23" stroke-width="2.5" stroke-linejoin="round"/>` : ''}
    ${xLabels}
  `;
  $lossChartEmpty.style.display = 'none';
  $lossChart.style.display = 'block';
  // Redraw on resize
  if (!drawLossChart._bound) {
    drawLossChart._bound = true;
    window.addEventListener('resize', () => { if (S.lossHistory.length >= 2) drawLossChart(); });
  }
}

function addSampleCard(prompt, text, temp) {
  const card = document.createElement('div');
  card.className = 'sample-card';
  card.innerHTML = `
    <div class="sample-prompt">
      ${escHtml(prompt)}
      <span class="sample-temp">temp=${temp}</span>
    </div>
    <div class="sample-text">${escHtml(text)}</div>
  `;
  $samplesList.prepend(card);
  if ($trainContent) $trainContent.scrollTop = 0;
}
