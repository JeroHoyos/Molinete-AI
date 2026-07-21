'use strict';

// ─────────────────────────────────────────────────
// WebSocket: conexión y mensajes del servidor
// ─────────────────────────────────────────────────
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  S.ws = ws;

  ws.onopen = () => { S.connected = true; setStatus('Conectado', 'ok'); syncUI(); };
  ws.onclose = () => {
    S.connected = false; S.running = false;
    setStatus('Desconectado — reconectando…', 'err'); syncUI();
    setTimeout(connect, 2500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = e => { try { handleMsg(JSON.parse(e.data)); } catch (_) {} };
}

function wsSend(obj) {
  if (S.ws?.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(obj));
}

function handleMsg(msg) {
  switch (msg.type) {
    case 'started':
      S.running = true;
      _lineBuffer = '';
      setStatus('Ejecutando…', 'run');
      syncUI();
      break;

    case 'output':
      parseAndRender(msg.text);
      break;

    case 'done':
      S.running = false;
      if (S.cat === 'learn') {
        $outputPre.classList.remove('cursor');
        showNote(msg.code === 0 ? '✓ Proceso completado' : `Proceso terminó con código ${msg.code}`, msg.code === 0);
      } else if (S.cat === 'train') {
        if (msg.code !== 0) {
          const note = document.createElement('div');
          note.className = 'train-note';
          note.textContent = `El proceso terminó con código ${msg.code}`;
          $('train-dashboard').after(note);
        }
      }
      setStatus('Listo', 'ok');
      syncUI();
      break;

    case 'stopped':
      S.running = false;
      if (S.cat === 'learn') showNote('Detenido por el usuario', false);
      setStatus('Listo', 'ok');
      syncUI();
      break;

    case 'error':
      S.running = false;
      if (S.cat === 'learn') showNote(msg.message || 'Error', false);
      setStatus('Error', 'err');
      syncUI();
      break;
  }
}
