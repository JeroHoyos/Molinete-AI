'use strict';

// ─────────────────────────────────────────────────
// Vista aprender/datos — salida cruda de terminal
// ─────────────────────────────────────────────────
function appendLearnOutput(chunk) {
  let i = 0;
  while (i < chunk.length) {
    if (chunk[i] === '\r' && i + 1 < chunk.length && chunk[i+1] === '\n') {
      S.outputBuf += '\n'; i += 2;
    } else if (chunk[i] === '\r') {
      const nl = S.outputBuf.lastIndexOf('\n');
      S.outputBuf = S.outputBuf.slice(0, nl + 1); i++;
    } else {
      let j = i;
      while (j < chunk.length && chunk[j] !== '\r') j++;
      S.outputBuf += chunk.slice(i, j); i = j;
    }
  }
  const MAX = 60_000;
  if (S.outputBuf.length > MAX) {
    const cut = S.outputBuf.indexOf('\n', S.outputBuf.length - MAX * 0.8);
    S.outputBuf = '… [salida truncada] …\n' + S.outputBuf.slice(cut + 1);
  }
  $outputPre.textContent = stripAnsi(S.outputBuf);
  const lc = $('learn-content');
  if (lc) lc.scrollTop = lc.scrollHeight;
}

function showNote(text, ok) {
  $terminalNote.textContent = text;
  $terminalNote.className = ok ? 'ok' : 'err';
}
