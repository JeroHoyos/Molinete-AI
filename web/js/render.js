'use strict';

// ─────────────────────────────────────────────────
// Parseo de salida — separa eventos __MOL__ del texto normal
// y los despacha a la vista correspondiente.
// ─────────────────────────────────────────────────
const MOL_PREFIX = '__MOL__';
let _lineBuffer = '';

function parseAndRender(chunk) {
  _lineBuffer += chunk;
  const newlineIdx = _lineBuffer.lastIndexOf('\n');
  if (newlineIdx === -1) return;

  const complete = _lineBuffer.slice(0, newlineIdx + 1);
  _lineBuffer = _lineBuffer.slice(newlineIdx + 1);

  const lines = complete.split('\n');
  const plainLines = [];

  for (const line of lines) {
    if (line.startsWith(MOL_PREFIX)) {
      try {
        const event = JSON.parse(line.slice(MOL_PREFIX.length));
        handleMolEvent(event);
      } catch (_) {}
    } else {
      plainLines.push(line);
    }
  }

  if (plainLines.length === 0) return;
  const plain = plainLines.join('\n');

  if (S.cat === 'train') {
    parseTrainProgress(plain);
  } else if (S.cat === 'learn') {
    appendLearnOutput(plain);
  } else if (S.cat === 'chat') {
    appendChatSetup(plain);
  }
}

// ─────────────────────────────────────────────────
// Eventos estructurados __MOL__
// ─────────────────────────────────────────────────
function handleMolEvent(ev) {
  switch (ev.type) {

    // ── Chat events ──
    case 'chat_checkpoints': {
      $chatLoading.style.display = 'none';
      document.getElementById('ck-picker-wrap')?.remove();
      const ms = ev.modelos || [];
      if (ms.length === 0) {
        const el = document.createElement('p');
        el.className = 'ck-no-models';
        el.textContent = 'No hay modelos entrenados. Usa las opciones 5–9 para entrenar uno.';
        $chatWelcome.appendChild(el);
      } else {
        $chatWelcomeText.textContent = '';
        const wrap = document.createElement('div');
        wrap.id = 'ck-picker-wrap';
        wrap.style.cssText = 'display:flex;flex-direction:column;align-items:center;width:100%;';
        wrap.innerHTML = `
          <div class="ck-picker-heading">Elige tu Modelo</div>
          <div class="ck-picker-sub">Selecciona el checkpoint con el que quieres conversar</div>
        `;
        const grid = document.createElement('div');
        grid.className = 'ck-model-grid';
        grid.id = 'ck-model-grid';
        ms.forEach(m => {
          const card = document.createElement('button');
          card.className = 'ck-model-card' + (m.tiene_mejor ? ' ck-has-mejor' : '');
          const perp = m.mejor_perp != null
            ? `<div class="ck-stat-row"><span class="ck-stat-k">Perplejidad</span><span class="ck-stat-v">${Number(m.mejor_perp).toFixed(2)}</span></div>`
            : '';
          const val = m.mejor_val != null
            ? `<div class="ck-stat-row"><span class="ck-stat-k">Mejor val.</span><span class="ck-stat-v">${Number(m.mejor_val).toFixed(4)}</span></div>`
            : '';
          const sizeMb = m.size_mb != null
            ? `<div class="ck-stat-row"><span class="ck-stat-k">Peso</span><span class="ck-stat-v">${m.size_mb} MB</span></div>`
            : '';
          const tagMejor  = m.tiene_mejor  ? '<span class="ck-tag ck-tag-mejor">✓ mejor checkpoint</span>' : '';
          const tagUltimo = m.tiene_ultimo ? '<span class="ck-tag ck-tag-ultimo">↓ último guardado</span>'  : '';
          card.innerHTML = `
            <div class="ck-model-idx">Modelo ${m.idx}</div>
            <div class="ck-model-display">${escHtml(m.display)}</div>
            <div class="ck-model-folder">${escHtml(m.nombre)}</div>
            <div class="ck-stats-block">
              <div class="ck-stat-row"><span class="ck-stat-k">Pasos</span><span class="ck-stat-v">${(m.pasos||0).toLocaleString()}</span></div>
              ${val}${perp}${sizeMb}
            </div>
            <div class="ck-tags">${tagMejor}${tagUltimo}</div>
            <div class="ck-select-btn">Conversar →</div>`;
          card.addEventListener('click', () => {
            document.getElementById('ck-picker-wrap')?.remove();
            $chatWelcomeText.textContent = `Cargando ${m.display}…`;
            $chatLoading.style.display = 'flex';
            wsSend({ action: 'input', value: String(m.idx) });
          });
          grid.appendChild(card);
        });
        wrap.appendChild(grid);
        $chatWelcome.appendChild(wrap);
      }
      break;
    }

    case 'chat_ready':
      document.getElementById('ck-model-grid')?.remove();
      $chatLoading.style.display = 'none';
      $chatWelcome.style.display = 'none';
      if (ev.checkpoint) {
        addBubbleInfo(`Modelo cargado · temp=${ev.temperatura} · max=${ev.max_tok} tokens`);
      }
      $inputBar.classList.remove('hidden');
      $userInput.placeholder = 'Escribe un prompt al estilo Cervantes… (Enter para enviar)';
      $userInput.disabled = !S.running;
      $sendBtn.disabled   = !S.running;
      break;

    case 'chat_user':
      $chatThinking.classList.remove('visible');
      addChatBubble('user', ev.text);
      $chatThinking.classList.add('visible');
      scrollChat();
      break;

    case 'chat_model':
      $chatThinking.classList.remove('visible');
      addChatBubble('model', ev.text);
      scrollChat();
      break;

    case 'chat_info':
      addBubbleInfo(ev.text);
      break;

    // ── Training events ──
    case 'train_meta':
      $tmModel.textContent  = ev.model_name || '—';
      $tmParams.textContent = ev.params_m !== undefined ? `${ev.params_m}M` : (ev.params ? `${(ev.params/1e6).toFixed(2)}M` : '—');
      $tmSteps.textContent  = ev.total_steps ? ev.total_steps.toLocaleString() : '—';
      $tmLr.textContent     = ev.lr || '—';
      $tmVocab.textContent  = ev.vocab || '—';
      S.totalSteps = ev.total_steps || 0;
      $trainStepLabel.textContent = 'Preparando entrenamiento…';
      break;

    case 'train_start':
      $trainStepLabel.textContent = 'Entrenando…';
      S.trainStarted = true;
      if ($trainRightDots) $trainRightDots.classList.remove('hidden');
      break;

    case 'train_done':
      $trainStepLabel.textContent = 'Entrenamiento completado';
      $trainPctLabel.textContent = '100%';
      $progressFill.style.width = '100%';
      if ($trainRightDots) $trainRightDots.classList.add('hidden');
      break;

    case 'sample':
      $('samples-empty').style.display = 'none';
      addSampleCard(ev.prompt, ev.text, ev.temp);
      break;

    case 'checkpoint_saved':
      $checkpointBadge.classList.remove('hidden');
      $checkpointPath.textContent = ev.path || '';
      $trainContent.scrollTop = $trainContent.scrollHeight;
      break;

    // ── Architecture table ──
    case 'arch_table':
      if (ev.rows && ev.rows.length) {
        $archTableWrap.classList.remove('hidden');
        $archTableBody.innerHTML = '';
        for (const r of ev.rows) {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td class="highlight">${r.nombre}</td>
            <td class="num">${r.vocab.toLocaleString()}</td>
            <td class="num">${r.embd}</td>
            <td class="num">${r.cabezas}</td>
            <td class="num">${r.capas}</td>
            <td class="num">${r.params.toLocaleString()}</td>
            <td class="num">${r.mem_mb}</td>
          `;
          $archTableBody.appendChild(tr);
        }
      }
      break;
  }
}
