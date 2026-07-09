'use strict';

// ─────────────────────────────────────────────────
// Listeners de eventos + arranque
// ─────────────────────────────────────────────────
document.querySelectorAll('.card, .opt-btn, .model-card').forEach(btn => {
  btn.addEventListener('click', () => runExample(btn.dataset.id));
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

// Chat submit
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

// Windmill Easter egg
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

// ─────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────
connect();
syncUI();
