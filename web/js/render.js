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
  } else if (S.cat === 'comp') {
    appendCompSetup(plain);
  }
}

// ─────────────────────────────────────────────────
// Eventos estructurados __MOL__
// ─────────────────────────────────────────────────
function handleMolEvent(ev) {
  switch (ev.type) {

    // ── Chat ──
    case 'chat_checkpoints': {
      $chatLoading.style.display = 'none';
      document.getElementById('ck-picker-wrap')?.remove();
      const ms = ev.modelos || [];
      if (ms.length === 0) {
        $chatWelcomeText.textContent = '';
        const el = document.createElement('p');
        el.className = 'ck-no-models';
        el.textContent = 'No hay modelos entrenados todavía. Entrena uno desde la portada y vuelve aquí.';
        $chatWelcome.appendChild(el);
      } else {
        $chatWelcomeText.textContent = '';
        const wrap = document.createElement('div');
        wrap.id = 'ck-picker-wrap';
        wrap.style.cssText = 'display:flex;flex-direction:column;align-items:center;width:100%;';
        wrap.innerHTML = `
          <div class="ck-picker-heading">Elige tu modelo</div>
          <div class="ck-picker-sub">Selecciona el checkpoint con el que quieres conversar</div>
        `;
        const grid = document.createElement('div');
        grid.className = 'ck-model-grid';
        grid.id = 'ck-model-grid';
        ms.forEach(m => {
          const card = document.createElement('div');
          card.setAttribute('role', 'button');
          card.tabIndex = 0;
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
          const fecha = m.fecha
            ? `<div class="ck-stat-row"><span class="ck-stat-k">Entrenado</span><span class="ck-stat-v">${escHtml(m.fecha)}</span></div>`
            : '';
          const tagMejor  = m.tiene_mejor  ? '<span class="ck-tag ck-tag-mejor">✓ mejor checkpoint</span>' : '';
          const tagUltimo = m.tiene_ultimo ? '<span class="ck-tag ck-tag-ultimo">último guardado</span>'  : '';
          card.innerHTML = `
            <div class="ck-model-idx">Modelo ${m.idx}</div>
            <div class="ck-model-display">${escHtml(m.display)}</div>
            <div class="ck-model-folder">${escHtml(m.nombre)}</div>
            <div class="ck-stats-block">
              <div class="ck-stat-row"><span class="ck-stat-k">Pasos</span><span class="ck-stat-v">${(m.pasos||0).toLocaleString()}</span></div>
              ${val}${perp}${sizeMb}${fecha}
            </div>
            <div class="ck-tags">${tagMejor}${tagUltimo}</div>
            <div class="ck-select-btn">Conversar →</div>`;

          const seleccionar = () => {
            document.getElementById('ck-picker-wrap')?.remove();
            $chatWelcomeText.textContent = `Cargando ${m.display}…`;
            $chatLoading.style.display = 'flex';
            wsSend({ action: 'input', value: String(m.idx) });
          };
          card.addEventListener('click', seleccionar);
          card.addEventListener('keydown', e => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seleccionar(); }
          });

          // Borrar el modelo del disco (con confirmación en el propio botón)
          const del = document.createElement('button');
          del.type = 'button';
          del.className = 'ck-del-btn';
          del.title = 'Borrar este modelo del disco';
          del.textContent = '×';
          del.addEventListener('click', e => {
            e.stopPropagation();
            if (!del.classList.contains('confirmar')) {
              del.classList.add('confirmar');
              del.textContent = '¿Borrar?';
              setTimeout(() => {
                del.classList.remove('confirmar');
                del.textContent = '×';
              }, 3000);
              return;
            }
            wsSend({ action: 'input', value: `borrar ${m.idx}` });
          });
          card.appendChild(del);
          grid.appendChild(card);
        });
        wrap.appendChild(grid);
        $chatWelcome.appendChild(wrap);
      }
      break;
    }

    case 'chat_ready': {
      document.getElementById('ck-model-grid')?.remove();
      $chatLoading.style.display = 'none';
      $chatWelcome.style.display = 'none';

      // Chip con el modelo cargado + tarjeta de bienvenida con sus datos
      const partes = String(ev.checkpoint || '').split(/[\\/]/);
      const carpeta = partes.length >= 2 ? partes[partes.length - 2] : (ev.checkpoint || '');
      $chatModelName.textContent = ev.nombre || carpeta || 'modelo';

      const stats = [];
      if (ev.pasos)              stats.push([Number(ev.pasos).toLocaleString(), 'pasos']);
      if (ev.mejor_val != null)  stats.push([Number(ev.mejor_val).toFixed(4), 'mejor pérdida']);
      if (ev.mejor_perp != null) stats.push([Number(ev.mejor_perp).toFixed(1), 'perplejidad']);
      if (ev.vocab)              stats.push([Number(ev.vocab).toLocaleString(), 'vocab']);
      const card = document.createElement('div');
      card.className = 'chat-ready-card fade-up';
      card.innerHTML = `
        <div class="crc-icon">
          <svg width="20" height="23" viewBox="0 0 28 32" fill="white"><path d="M10 32V18L12 14H16L18 18V32H10Z" opacity="0.85"/><path d="M11 14 Q14 9 17 14 Z"/><circle cx="14" cy="13" r="2.4"/><path d="M15.5 11.5 L23 4 Q25 3 25 5 L18 13 Z"/><path d="M15.5 14.5 L23 22 Q25 24 23 25 L16 16 Z"/><path d="M12.5 14.5 L5 22 Q3 24 3 22 L10 14 Z"/><path d="M12.5 11.5 L5 4 Q3 3 5 1 L13.5 10 Z"/></svg>
        </div>
        <div class="crc-title">${escHtml(ev.display || carpeta || 'Modelo')}</div>
        <div class="crc-folder">${escHtml(ev.nombre || carpeta)}</div>
        ${stats.length ? `<div class="crc-stats">${stats.map(([v, k]) => `<span class="crc-stat"><b>${v}</b><span>${k}</span></span>`).join('')}</div>` : ''}
        <div class="crc-hint">Listo para conversar. Escribe un prompt o elige una sugerencia.</div>`;
      $chatBubbleList.appendChild(card);
      scrollChat();
      if (ev.temperatura != null) { S.chatTemp = Number(ev.temperatura); $tempValue.textContent = S.chatTemp.toFixed(1); }
      if (ev.max_tok     != null) { S.chatMax  = Number(ev.max_tok);     $maxValue.textContent  = String(S.chatMax); }

      $inputBar.classList.remove('hidden');
      $chatToolbar.classList.remove('hidden');
      renderChatSuggestions();
      $userInput.placeholder = 'Escribe un prompt al estilo Cervantes… (Enter para enviar)';
      $userInput.disabled = !S.running;
      $sendBtn.disabled   = !S.running;
      if (S.running) $userInput.focus();
      break;
    }

    case 'chat_user':
      $chatThinking.classList.remove('visible');
      $chatSuggest.classList.add('hidden');
      addChatBubble('user', ev.text, null, ev.tokens || null, null, ev.ids || null);
      $chatThinking.classList.add('visible');
      scrollChat();
      break;

    case 'chat_model':
      $chatThinking.classList.remove('visible');
      addChatBubble('model', ev.text, ev.prompt, ev.tokens || null, ev.tokens_prompt || null, ev.ids || null, ev.ids_prompt || null);
      scrollChat();
      break;

    case 'chat_info':
      if (S.cat === 'comp') appendCompInfo(ev.text);
      else addBubbleInfo(ev.text);
      break;

    // ── Comparador de modelos ──
    case 'comp_checkpoints': renderCompPicker(ev.modelos || []); break;
    case 'comp_loading':     compLoadingEv(ev); break;
    case 'comp_loaded':      compLoadedEv(ev); break;
    case 'comp_ready':       compReadyEv(ev); break;
    case 'comp_prompt':      compPromptEv(ev); break;
    case 'comp_gen':         compGenEv(ev); break;
    case 'comp_result':      compResultEv(ev); break;
    case 'comp_round_done':  break;

    // ── Preparación del entrenamiento ──
    case 'prep': {
      $prepCard.classList.remove('hidden');
      let row = $prepList.querySelector(`.prep-row[data-paso="${CSS.escape(ev.paso || '')}"]`);
      if (!row) {
        row = document.createElement('div');
        row.className = 'prep-row';
        row.dataset.paso = ev.paso || '';
        row.innerHTML = '<span class="prep-icon"></span><span class="prep-label"></span><span class="prep-value"></span>';
        $prepList.appendChild(row);
      }
      const enCurso = ev.estado === 'run';
      row.classList.toggle('run', enCurso);
      row.querySelector('.prep-icon').textContent  = enCurso ? '' : '✓';
      row.querySelector('.prep-label').textContent = ev.label || '';
      row.querySelector('.prep-value').textContent = ev.value || '';
      break;
    }

    case 'tok_examples': {
      const ejemplos = ev.ejemplos || [];
      if (!ejemplos.length) break;
      $prepCard.classList.remove('hidden');
      $tokExamples.classList.remove('hidden');
      $tokExamplesList.innerHTML = '';
      for (const ej of ejemplos) {
        const frase = document.createElement('div');
        frase.className = 'tok-frase';
        frase.textContent = `"${ej.frase}"`;
        const n = document.createElement('b');
        n.textContent = `${(ej.tokens || []).length} tokens`;
        frase.appendChild(n);
        const chips = document.createElement('div');
        chips.className = 'tok-chips';
        for (const t of ej.tokens || []) {
          const chip = document.createElement('span');
          chip.className = 'tok-chip';
          chip.textContent = t;
          chips.appendChild(chip);
        }
        $tokExamplesList.appendChild(frase);
        $tokExamplesList.appendChild(chips);
      }
      break;
    }

    // ── Entrenamiento ──
    case 'train_meta':
      $tmModel.textContent  = ev.model_name || '—';
      $tmParams.textContent = ev.params_m !== undefined ? `${ev.params_m}M` : (ev.params ? `${(ev.params/1e6).toFixed(2)}M` : '—');
      $tmSteps.textContent  = ev.total_steps ? ev.total_steps.toLocaleString() : '—';
      $tmLr.textContent     = ev.lr || '—';
      $tmVocab.textContent  = ev.vocab ? ev.vocab.toLocaleString() : '—';
      $tmCorpus.textContent = ev.corpus_mb != null ? `${ev.corpus_mb} MB` : '—';
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
      $ttEta.textContent = '0 s';
      if ($trainRightDots) $trainRightDots.classList.add('hidden');
      break;

    case 'sample':
      $('samples-empty').style.display = 'none';
      addSampleCard(ev.prompt, ev.text, ev.temp);
      break;

    case 'checkpoint_saved':
      $checkpointBadge.classList.remove('hidden');
      $checkpointPath.textContent = ev.path || '';
      break;

    // ── Descarga del corpus ──
    case 'descarga_obras': {
      $dlCard.classList.remove('hidden');
      $dlDest.textContent = ev.destino || 'cervantes.txt';
      $dlList.innerHTML = '';
      (ev.obras || []).forEach((titulo, i) => {
        const row = document.createElement('div');
        row.className = 'dl-row';
        row.dataset.idx = i;
        row.innerHTML = '<span class="dl-row-icon"></span><span class="dl-row-title"></span><span class="dl-row-size"></span>';
        row.querySelector('.dl-row-title').textContent = titulo;
        $dlList.appendChild(row);
      });
      $dlProgress.classList.remove('hidden');
      break;
    }

    case 'descarga_confirmar':
      $dlCard.classList.remove('hidden');
      $dlConfirm.classList.remove('hidden');
      $dlConfirmText.textContent = `'${ev.ruta}' ya existe (${ev.mb} MB). ¿Quieres descargarlo de nuevo y sobrescribirlo?`;
      break;

    case 'descarga_obra': {
      const row = $dlList.querySelector(`.dl-row[data-idx="${ev.idx}"]`);
      if (!row) break;
      row.className = 'dl-row' + (ev.estado ? ` ${ev.estado}` : '');
      row.querySelector('.dl-row-icon').textContent = ev.estado === 'ok' ? '✓' : ev.estado === 'err' ? '✕' : '';
      if (ev.mb != null) row.querySelector('.dl-row-size').textContent = `${ev.mb} MB`;
      const total  = $dlList.children.length || 1;
      const hechas = $dlList.querySelectorAll('.dl-row.ok, .dl-row.err').length;
      $dlProgressFill.style.width = ((hechas / total) * 100).toFixed(0) + '%';
      break;
    }

    case 'descarga_fin':
      $dlProgressFill.style.width = '100%';
      $dlDone.classList.remove('hidden');
      $dlDoneText.textContent = `Corpus listo: ${ev.mb} MB en ${ev.ruta}. Ya puedes entrenar modelos.`;
      $dlNote.dataset.final = '1';
      break;

    case 'descarga_cancelada':
      $dlConfirm.classList.add('hidden');
      $dlProgress.classList.add('hidden');
      $dlNote.textContent = 'Operación cancelada. Se conserva el corpus existente.';
      $dlNote.className = 'err';
      $dlNote.dataset.final = '1';
      break;

    // ── Tabla de arquitecturas ──
    case 'arch_table':
      if (ev.rows && ev.rows.length) {
        $archTableWrap.classList.remove('hidden');
        $archTableBody.innerHTML = '';
        for (const r of ev.rows) {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td class="highlight">${escHtml(r.nombre)}</td>
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
