'use strict';

// ─────────────────────────────────────────────────
// Cambio de vista: home / módulo, y sincronización de UI
// ─────────────────────────────────────────────────
function showHome() {
  $homeView.classList.remove('hidden');
  $moduleView.classList.add('hidden');
  $hdrHome.classList.remove('hidden');
  $hdrModule.classList.add('hidden');
  document.querySelectorAll('.card, .opt-btn, .model-card').forEach(b => b.classList.remove('card-active', 'opt-btn-active', 'model-card-active'));
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
  $modHeaderName.textContent = mod.name || `Módulo ${id}`;
  $modHeaderDesc.textContent = mod.desc || '';

  // Reset views
  $chatView.classList.add('hidden');
  $trainView.classList.add('hidden');
  $learnView.classList.add('hidden');
  $inputBar.classList.add('hidden');

  // Reset state
  S.outputBuf  = '';
  S.totalSteps = 0;
  S.lossHistory = [];

  if (S.cat === 'chat') {
    $chatView.classList.remove('hidden');
    // Reset chat
    $chatBubbleList.innerHTML = '';
    $chatWelcome.style.display = 'flex';
    $chatWelcomeText.textContent = 'Cargando modelo… selecciona un checkpoint cuando aparezca el prompt.';
    $chatLoading.style.display = 'flex';
    $chatThinking.classList.remove('visible');
    $inputBar.classList.add('hidden');
    $userInput.disabled = true;
    $sendBtn.disabled = true;
  } else if (S.cat === 'train') {
    $trainView.classList.remove('hidden');
    // Reset training dashboard
    $tmModel.textContent = '—';
    $tmParams.textContent = '—';
    $tmSteps.textContent = '—';
    $tmLr.textContent = '—';
    $tmVocab.textContent = '—';
    $progressFill.style.width = '0%';
    $trainStepLabel.textContent = 'Esperando inicio…';
    $trainPctLabel.textContent = '0%';
    $metricTrain.textContent = '—';
    $metricVal.textContent = '—';
    $metricLr.textContent = '—';
    $metricPerp.textContent = '—';
    $samplesList.innerHTML = '';
    $('samples-empty').style.display = 'block';
    $vocabLog.textContent = '';
    $vocabLog.style.display = 'none';
    $checkpointBadge.classList.add('hidden');
    $lossChart.style.display = 'none';
    $lossChart.innerHTML = '';
    $lossChartEmpty.style.display = '';
    S.trainStarted = false;
  } else {
    $learnView.classList.remove('hidden');
    $outputPre.textContent = '';
    $outputPre.classList.add('cursor');
    $terminalNote.className = '';
    $archTableWrap.classList.add('hidden');
    $archTableBody.innerHTML = '';
  }

  // Active states
  document.querySelectorAll('.card, .opt-btn, .model-card').forEach(b => b.classList.remove('card-active', 'opt-btn-active', 'model-card-active'));
  document.querySelector(`.opt-btn[data-id="${id}"]`)?.classList.add('opt-btn-active');
  document.querySelector(`.model-card[data-id="${id}"]`)?.classList.add('model-card-active');
}

function runExample(id) {
  if (!S.connected || S.running) return;
  showModule(id);
  wsSend({ action: 'run', id });
}

// ─────────────────────────────────────────────────
// Sincronización de UI según estado
// ─────────────────────────────────────────────────
function syncUI() {
  $stopBtn.classList.toggle('hidden', !S.running);

  document.querySelectorAll('.card, .opt-btn, .model-card').forEach(el => { el.disabled = S.running; });

  // Chat input
  if (S.cat === 'chat') {
    $userInput.disabled = !(S.connected && S.running);
    $sendBtn.disabled   = !(S.connected && S.running);
  }
}
