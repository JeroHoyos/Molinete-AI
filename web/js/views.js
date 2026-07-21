'use strict';

// ─────────────────────────────────────────────────
// Cambio de vista: home / módulo, y sincronización de UI
// ─────────────────────────────────────────────────
function showHome() {
  $homeView.classList.remove('hidden');
  $moduleView.classList.add('hidden');
  $hdrHome.classList.remove('hidden');
  $hdrModule.classList.add('hidden');
  document.querySelectorAll('.opt-btn, .model-card').forEach(b => b.classList.remove('opt-btn-active', 'model-card-active'));
  S.currentId = null; S.cat = null;
}

function showModule(id) {
  const mod = MODS[String(id)] || {};
  S.cat = mod.cat || 'learn';
  S.currentId = String(id);

  $homeView.classList.add('hidden');
  $moduleView.classList.remove('hidden');
  $hdrHome.classList.add('hidden');
  $hdrModule.classList.remove('hidden');

  $hdrTaskName.textContent = mod.name || `Módulo ${id}`;
  $hdrTaskSub.textContent  = mod.sub  || '';

  // Reset de vistas
  $chatView.classList.add('hidden');
  $trainView.classList.add('hidden');
  $learnView.classList.add('hidden');
  $('comp-view').classList.add('hidden');
  $inputBar.classList.add('hidden');

  // Reset de estado compartido
  S.outputBuf   = '';
  S.totalSteps  = 0;
  S.lossHistory = [];
  S.bestVal     = null;
  S.elapsedSec  = 0;

  if (S.cat === 'chat') {
    $chatView.classList.remove('hidden');
    // Reset del chat
    $chatBubbleList.innerHTML = '';
    $chatWelcome.style.display = 'flex';
    $chatWelcome.classList.remove('picker-mode');
    $chatWelcomeText.textContent = 'Buscando checkpoints entrenados…';
    $chatLoading.style.display = 'flex';
    $chatThinking.classList.remove('visible');
    $chatSuggest.classList.add('hidden');
    $chatSuggest.innerHTML = '';
    $chatToolbar.classList.add('hidden');
    $chatView.classList.remove('ver-tokens');
    $('chat-tokens-btn').classList.remove('active');
    $('chat-tokens-btn').setAttribute('aria-pressed', 'false');
    S.chatTemp = 0.8; S.chatMax = 100;
    $tempValue.textContent = S.chatTemp.toFixed(1);
    $maxValue.textContent  = String(S.chatMax);
    $userInput.disabled = true;
    $sendBtn.disabled = true;
  } else if (S.cat === 'comp') {
    $('comp-view').classList.remove('hidden');
    resetCompView();
  } else if (S.cat === 'train') {
    $trainView.classList.remove('hidden');
    // Reset del dashboard de entrenamiento
    $tmModel.textContent  = '—';
    $tmParams.textContent = '—';
    $tmSteps.textContent  = '—';
    $tmLr.textContent     = '—';
    $tmVocab.textContent  = '—';
    $tmCorpus.textContent = '—';
    $ttElapsed.textContent = '—';
    $ttSpeed.textContent   = '—';
    $ttEta.textContent     = '—';
    $progressFill.style.width = '0%';
    $trainStepLabel.textContent = 'Esperando inicio…';
    $trainPctLabel.textContent = '0%';
    $metricTrain.textContent = '—';
    $metricVal.textContent = '—';
    $metricValBest.textContent = 'mejor: —';
    $metricLr.textContent = '—';
    $metricPerp.textContent = '—';
    $samplesList.innerHTML = '';
    $('samples-empty').style.display = 'block';
    $prepCard.classList.add('hidden');
    $prepList.innerHTML = '';
    $tokExamples.classList.add('hidden');
    $tokExamplesList.innerHTML = '';
    $prepRaw.removeAttribute('open');
    $vocabLog.textContent = '';
    $checkpointBadge.classList.add('hidden');
    $lossChart.style.display = 'none';
    $lossChart.innerHTML = '';
    $lossChartEmpty.style.display = '';
    $lcTip.style.display = 'none';
    S.chartFit = false;
    document.querySelectorAll('.lc-scale-btn').forEach(b => b.classList.toggle('active', b.dataset.fit === '0'));
    document.querySelectorAll('.train-note').forEach(n => n.remove());
    S.trainStarted = false;
  } else {
    $learnView.classList.remove('hidden');
    $learnView.classList.toggle('dl-mode', String(id) === '11');
    $outputPre.textContent = '';
    $outputPre.classList.add('cursor');
    $terminalNote.className = '';
    $archTableWrap.classList.add('hidden');
    $archTableBody.innerHTML = '';
    // Reset de la tarjeta de descarga
    $dlCard.classList.add('hidden');
    $dlList.innerHTML = '';
    $dlConfirm.classList.add('hidden');
    $dlProgress.classList.add('hidden');
    $dlProgressFill.style.width = '0%';
    $dlDone.classList.add('hidden');
    $dlNote.textContent = '';
    $dlNote.className = '';
    delete $dlNote.dataset.final;
    $dlLog.textContent = '';
    $dlRaw.removeAttribute('open');
  }

  // Estados activos en la home
  document.querySelectorAll('.opt-btn, .model-card').forEach(b => b.classList.remove('opt-btn-active', 'model-card-active'));
  document.querySelector(`.opt-btn[data-id="${id}"]`)?.classList.add('opt-btn-active');
  document.querySelector(`.model-card[data-id="${id}"]`)?.classList.add('model-card-active');
}

function runExample(id) {
  if (!S.connected || S.running) return;
  showModule(id);
  wsSend({ action: 'run', id });
}

// El módulo unificado (10) arranca en la vista de chat; si la selección
// del picker es múltiple, el backend pasa a modo comparación y aquí
// cambiamos de vista sin reiniciar el proceso.
function switchToCompView() {
  if (S.cat === 'comp') return;
  S.cat = 'comp';
  $chatView.classList.add('hidden');
  $inputBar.classList.add('hidden');
  $('comp-view').classList.remove('hidden');
  resetCompView();
  $('comp-welcome-text').textContent = 'Cargando modelos…';
  $('comp-loading').style.display = 'flex';
  $hdrTaskName.textContent = 'Comparar modelos';
  $hdrTaskSub.textContent  = 'El mismo prompt en varios paneles, cada uno con su temperatura';
  syncUI();
}

// ─────────────────────────────────────────────────
// Sincronización de UI según estado
// ─────────────────────────────────────────────────
function syncUI() {
  $stopBtn.classList.toggle('hidden', !S.running);

  document.querySelectorAll('.opt-btn').forEach(el => { el.disabled = S.running; });
  document.querySelectorAll('.model-card').forEach(el => el.classList.toggle('is-disabled', S.running));

  // Entrada del chat
  if (S.cat === 'chat') {
    $userInput.disabled = !(S.connected && S.running);
    $sendBtn.disabled   = !(S.connected && S.running);
  }
  // Entrada del comparador
  if (S.cat === 'comp') {
    $('comp-input').disabled = !(S.connected && S.running);
    $('comp-send').disabled  = !(S.connected && S.running);
  }
}
