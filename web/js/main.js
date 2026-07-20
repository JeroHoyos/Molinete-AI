'use strict';

// ─────────────────────────────────────────────────
// Listeners de eventos + arranque
// ─────────────────────────────────────────────────
document.querySelectorAll('.opt-btn, .model-card').forEach(btn => {
  btn.addEventListener('click', () => runExample(btn.dataset.id));
});
// Las tarjetas de modelo son <div role="button">: activar con Enter o espacio
document.querySelectorAll('.model-card').forEach(card => {
  card.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); runExample(card.dataset.id); }
  });
});

$backBtn.addEventListener('click', () => {
  if (S.running) wsSend({ action: 'stop' });
  showHome();
});
$stopBtn.addEventListener('click', () => wsSend({ action: 'stop' }));
$clearBtn.addEventListener('click', () => {
  S.outputBuf = '';
  $outputPre.textContent = '';
  $terminalNote.className = '';
});

// ── Chat: envío ──
function submitChat() {
  const val = $userInput.value.trim();
  if (!val || !S.running) return;
  wsSend({ action: 'input', value: val });
  $userInput.value = ''; $userInput.style.height = 'auto'; $userInput.focus();
}
$sendBtn.addEventListener('click', submitChat);
$userInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitChat(); } });
$userInput.addEventListener('input', () => {
  $userInput.style.height = 'auto';
  $userInput.style.height = Math.min($userInput.scrollHeight, 130) + 'px';
});

// ── Chat: controles de generación ──
// Los cambios se envían como comandos ("temp X" / "max X") que el
// backend ya entiende; se agrupan para no inundar el chat de avisos.
let _tempTimer = 0, _maxTimer = 0;

function _sendChatCmd(cmd, timerRef) {
  clearTimeout(timerRef.id);
  timerRef.id = setTimeout(() => {
    if (S.running && S.cat === 'chat') wsSend({ action: 'input', value: cmd });
  }, 450);
}
const _tempRef = { id: 0 }, _maxRef = { id: 0 };

function ajustarTemp(delta) {
  S.chatTemp = Math.min(2.0, Math.max(0.1, Math.round((S.chatTemp + delta) * 10) / 10));
  $tempValue.textContent = S.chatTemp.toFixed(1);
  _sendChatCmd(`temp ${S.chatTemp.toFixed(1)}`, _tempRef);
}
function ajustarMax(delta) {
  S.chatMax = Math.min(400, Math.max(20, S.chatMax + delta));
  $maxValue.textContent = String(S.chatMax);
  _sendChatCmd(`max ${S.chatMax}`, _maxRef);
}

// Alternar la vista de tokens en las burbujas
let _avisoTokens = false;
$('chat-tokens-btn').addEventListener('click', () => {
  const activo = $chatView.classList.toggle('ver-tokens');
  $('chat-tokens-btn').classList.toggle('active', activo);
  $('chat-tokens-btn').setAttribute('aria-pressed', String(activo));
  // Sesiones abiertas con el backend anterior no traen tokens: avisar una vez
  if (activo && !_avisoTokens
      && $chatBubbleList.querySelector('.bubble')
      && !$chatBubbleList.querySelector('.bubble-tokens')) {
    _avisoTokens = true;
    addBubbleInfo('Los tokens se muestran en los mensajes nuevos. Si no aparecen, vuelve al inicio y entra al chat de nuevo.');
  }
});

$('temp-minus').addEventListener('click', () => ajustarTemp(-0.1));
$('temp-plus').addEventListener('click',  () => ajustarTemp(+0.1));
$('max-minus').addEventListener('click',  () => ajustarMax(-20));
$('max-plus').addEventListener('click',   () => ajustarMax(+20));

// ── Gráfica de pérdida: escala del eje de pasos ──
document.querySelectorAll('.lc-scale-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    S.chartFit = btn.dataset.fit === '1';
    document.querySelectorAll('.lc-scale-btn').forEach(b => b.classList.toggle('active', b === btn));
    if (S.lossHistory.length >= 2) scheduleChartDraw();
  });
});

// ── Descarga del corpus: confirmación de sobrescritura ──
$('dl-overwrite').addEventListener('click', () => {
  $dlConfirm.classList.add('hidden');
  wsSend({ action: 'input', value: 's' });
});
$('dl-cancel').addEventListener('click', () => {
  $dlConfirm.classList.add('hidden');
  wsSend({ action: 'input', value: 'n' });
});

// ── Molinillo: huevo de pascua ──
let spinning = false;
$('windmill-logo').addEventListener('click', () => {
  if (spinning) return; spinning = true;
  const el = $('windmill-logo');
  let deg = 0;
  const iv = setInterval(() => {
    deg += 18; el.style.transform = `rotate(${deg}deg) scale(1.05)`;
    if (deg >= 720) { clearInterval(iv); spinning = false; el.style.transform = ''; }
  }, 16);
});

// ── QR del repositorio ──
$('qr-btn').addEventListener('click',      () => $('qr-modal').classList.remove('hidden'));
$('qr-close').addEventListener('click',    () => $('qr-modal').classList.add('hidden'));
$('qr-backdrop').addEventListener('click', () => $('qr-modal').classList.add('hidden'));
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') $('qr-modal').classList.add('hidden');
});

// ─────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────
connect();
syncUI();
