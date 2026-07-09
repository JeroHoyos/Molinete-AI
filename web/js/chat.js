'use strict';

// ─────────────────────────────────────────────────
// Vista de chat — burbujas y texto de preparación
// ─────────────────────────────────────────────────
function addChatBubble(role, text) {
  if ($chatWelcome.style.display !== 'none') $chatWelcome.style.display = 'none';
  const row = document.createElement('div');
  row.className = `bubble-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `bubble-avatar ${role === 'model' ? 'avatar-model' : 'avatar-user'}`;
  avatar.innerHTML = role === 'model'
    ? `<svg width="14" height="16" viewBox="0 0 28 32" fill="currentColor"><path d="M10 32V18L12 14H16L18 18V32H10Z" opacity="0.85"/><path d="M11 14 Q14 9 17 14 Z"/><circle cx="14" cy="13" r="2.4"/><path d="M15.5 11.5 L23 4 Q25 3 25 5 L18 13 Z"/><path d="M15.5 14.5 L23 22 Q25 24 23 25 L16 16 Z"/><path d="M12.5 14.5 L5 22 Q3 24 3 22 L10 14 Z"/><path d="M12.5 11.5 L5 4 Q3 3 5 1 L13.5 10 Z"/></svg>`
    : 'Tú';

  const bubble = document.createElement('div');
  bubble.className = `bubble bubble-${role}`;
  bubble.textContent = text;

  row.appendChild(avatar);
  row.appendChild(bubble);
  $chatBubbleList.appendChild(row);
  scrollChat();
}

function addBubbleInfo(text) {
  const div = document.createElement('div');
  div.className = 'bubble-info fade-up';
  div.textContent = text;
  $chatBubbleList.appendChild(div);
  scrollChat();
}

function appendChatSetup(text) {
  const stripped = stripAnsi(text).trim();
  if (!stripped) return;
  // Ignorar la línea del prompt de selección vacío que queda en stdin
  if (stripped === '' || stripped === '>') return;
  const pre = document.createElement('pre');
  pre.style.cssText = 'font-family:"JetBrains Mono",monospace;font-size:12px;line-height:1.6;color:var(--text-2);background:var(--bg-3);border:1px solid var(--border);border-radius:8px;padding:10px 14px;margin:4px 0;white-space:pre-wrap;word-break:break-word;';
  pre.textContent = stripped;
  // No ocultar $chatWelcome aquí — las tarjetas del picker viven dentro.
  // Solo se oculta cuando llega chat_ready.
  $chatBubbleList.appendChild(pre);
  scrollChat();
}
