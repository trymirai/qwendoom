// qwendoom page: draws the frames the server streams over /ws, shows the CPU's counters on the
// monitor's LCD and sends the keyboard (and the on-screen keys) back as key events.
// Protocol: ARCHITECTURE.md, "The browser UI and server".
'use strict';

(() => {
  const W = 320, H = 200;
  const FRAME_BYTES = 1 + 768 + W * H;
  const ACTIVE = new Set(['starting', 'running', 'paused']);

  const $ = (id) => document.getElementById(id);
  const monitor = $('monitor'), screenEl = $('screen'), canvas = $('canvas'), msgEl = $('screen-msg');
  const startBtn = $('start'), startLabel = $('start-label');
  const pauseBtn = $('pause'), pauseLabel = $('pause-label'), led = $('led');
  const lcd = { tokens: $('tokens'), tps: $('tps'), frames: $('frames') };

  // ---------------------------------------------------------------- screen

  const ctx = canvas.getContext('2d', { alpha: false });
  const image = ctx.createImageData(W, H);
  const out = new Uint32Array(image.data.buffer);
  const lut = new Uint32Array(256);
  const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);

  let pending = null;  // newest undrawn frame; older ones are dropped
  let drawScheduled = false;

  function drawFrame(buf) {
    const pal = new Uint8Array(buf, 1, 768);
    const idx = new Uint8Array(buf, 769, W * H);
    for (let i = 0; i < 256; i++) {
      const r = pal[3 * i], g = pal[3 * i + 1], b = pal[3 * i + 2];
      lut[i] = littleEndian ? (0xff000000 | (b << 16) | (g << 8) | r) >>> 0
                            : ((r << 24) | (g << 16) | (b << 8) | 0xff) >>> 0;
    }
    for (let i = 0; i < W * H; i++) out[i] = lut[idx[i]];
    ctx.putImageData(image, 0, 0);
    screenEl.classList.add('live');
  }

  function queueFrame(buf) {
    pending = buf;
    if (drawScheduled) return;
    drawScheduled = true;
    requestAnimationFrame(() => {
      drawScheduled = false;
      if (pending && state !== 'idle') drawFrame(pending);
      pending = null;
    });
  }

  // ---------------------------------------------------------------- LCD

  // 1234567 -> "1 234 567" (thin spaces)
  const group = (n) => String(Math.trunc(Math.abs(n))).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  const known = (v) => typeof v === 'number' && Number.isFinite(v);
  const fmtCount = (v) => known(v) ? (v < 0 ? '-' : '') + group(v) : '—';
  const fmtRate = (v) => !known(v) ? '—' : Math.abs(v) < 100 ? v.toFixed(1) : (v < 0 ? '-' : '') + group(Math.round(v));

  function setText(el, text) { if (el.textContent !== text) el.textContent = text; }

  // A monospace font gives the thin space a full cell, so each separator gets its own span in the
  // UI font (where it is thin); the text stays "1 234 567" for copying and screen readers.
  function setNum(el, text) {
    if (el.textContent === text) return;
    el.replaceChildren(...text.split(' ').flatMap((part, i) => {
      if (!i) return [part];
      const sep = document.createElement('span');
      sep.className = 'sep';
      sep.textContent = ' ';
      return [sep, part];
    }));
  }

  let lastTokens = null, tokenActivity = 0;

  function showStats(m) {
    setNum(lcd.tokens, fmtCount(m.tokens));
    setNum(lcd.tps, fmtRate(m.tps));
    setNum(lcd.frames, fmtCount(m.frames));
    if (known(m.tokens) && m.tokens !== lastTokens) {
      if (lastTokens !== null && m.tokens > lastTokens) tokenActivity = performance.now();
      lastTokens = m.tokens;
    }
    if (typeof m.state === 'string') setState(m.state);
  }

  // ---------------------------------------------------------------- state

  let ws = null, connected = false, everConnected = false, retries = 0;
  let state = 'idle', controller = false, hasController = false, serverMessage = null;

  function setState(s) {
    if (s === state) return;
    state = s;
    render();
  }

  function render() {
    monitor.dataset.state = connected ? state : 'offline';
    const active = ACTIVE.has(state);
    // Start / Stop: begins a game, stops the one this tab drives, or takes control of one another
    // tab is driving
    const stoppable = connected && active && controller;
    startBtn.disabled = !connected;
    startBtn.classList.toggle('stop', stoppable);
    setText(startLabel, stoppable ? 'Stop' : 'Start');
    startBtn.title = connected && active && !controller ? 'Take control of the running game' : '';
    if (state === 'idle') {
      pending = null;  // a frame queued before the stop must not repaint the dark screen
      ctx.fillRect(0, 0, W, H);  // stopped: back to a dark screen
      screenEl.classList.remove('live');
    }
    pauseBtn.disabled = !connected || !controller || !(state === 'running' || state === 'paused');
    pauseBtn.classList.toggle('resume', state === 'paused');
    setText(pauseLabel, state === 'paused' ? 'Resume' : 'Pause');
    let msg = '';
    if (!connected) msg = everConnected ? 'reconnecting…' : 'connecting…';
    else if (state === 'starting') msg = serverMessage || 'loading…';
    else if (state === 'paused') msg = 'paused';
    else if (state === 'error') msg = serverMessage || 'error';
    else if (state === 'ended') msg = serverMessage || 'the game has ended';
    else if (active && !controller && hasController) msg = 'watching: another tab is playing';
    else if (serverMessage && state !== 'idle') msg = serverMessage;
    setText(msgEl, msg);
  }

  // power LED: glows while running and flickers while tokens are flowing
  setInterval(() => {
    const busy = state === 'running' && performance.now() - tokenActivity < 700;
    led.style.setProperty('--glow', busy ? (0.55 + 0.45 * Math.random()).toFixed(2) : '1');
  }, 70);

  // ---------------------------------------------------------------- socket

  function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }

  function connect() {
    const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
    const sock = ws = new WebSocket(url);
    sock.binaryType = 'arraybuffer';
    sock.onopen = () => {
      connected = everConnected = true;
      retries = 0;
      // the server may have restarted: drop the old picture (it sends its current frame, if any)
      pending = null;
      lastTokens = null;
      ctx.fillRect(0, 0, W, H);
      screenEl.classList.remove('live');
      render();
    };
    sock.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        const view = new Uint8Array(ev.data, 0, 1);
        if (ev.data.byteLength === FRAME_BYTES && view[0] === 1) queueFrame(ev.data);
        return;
      }
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === 'stats') showStats(m);
      else if (m.type === 'status') {
        if (m.error) console.warn('qwendoom server:', m.error);
        if ('controller' in m) controller = !!m.controller;
        if ('has_controller' in m) hasController = !!m.has_controller;
        if ('message' in m) serverMessage = m.message || null;
        if (typeof m.state === 'string') state = m.state;
        render();
      }
    };
    sock.onclose = () => {
      if (ws !== sock) return;
      ws = null;
      connected = false;
      controller = false;
      releaseAll(false);
      render();
      const delay = Math.min(5000, 300 * 2 ** retries++) * (0.8 + 0.4 * Math.random());
      setTimeout(connect, delay);
    };
  }

  // ---------------------------------------------------------------- keys

  const CODES = {
    KeyQ: 'left', ArrowLeft: 'left', KeyW: 'forward', ArrowUp: 'forward',
    KeyE: 'right', ArrowRight: 'right', KeyA: 'strafeleft', KeyS: 'back', ArrowDown: 'back',
    KeyD: 'straferight', ControlLeft: 'fire', ControlRight: 'fire', Space: 'use',
    KeyF: 'fire',  // macOS keeps Ctrl+arrows (Spaces, Mission Control): fire while turning
    ShiftLeft: 'run', ShiftRight: 'run', Escape: 'menu', Enter: 'enter', NumpadEnter: 'enter',
    Digit1: 'weapon1', Digit2: 'weapon2', Digit3: 'weapon3',
    Numpad1: 'weapon1', Numpad2: 'weapon2', Numpad3: 'weapon3',
  };
  const tiles = new Map();  // action -> [tile elements]
  for (const el of document.querySelectorAll('.key[data-action]')) {
    const a = el.dataset.action;
    tiles.set(a, [...(tiles.get(a) || []), el]);
  }

  // An action is down while anything holds it (W and ↑ together, a finger on its tile), so the
  // server sees one press and one release.
  const holders = new Map();  // action -> Set of holder ids
  const chars = new Map();    // KeyboardEvent.code -> char sent for it

  function press(action, holder) {
    let set = holders.get(action);
    if (!set) holders.set(action, set = new Set());
    if (set.has(holder)) return;
    set.add(holder);
    if (set.size === 1) {
      send({ type: 'key', action, down: true });
      for (const t of tiles.get(action) || []) t.classList.add('held');
    }
  }

  function release(action, holder) {
    const set = holders.get(action);
    if (!set || !set.delete(holder) || set.size) return;
    send({ type: 'key', action, down: false });
    for (const t of tiles.get(action) || []) t.classList.remove('held');
  }

  function releaseAll(notify = true) {
    for (const [action, set] of holders) {
      if (!set.size) continue;
      set.clear();
      if (notify) send({ type: 'key', action, down: false });
      for (const t of tiles.get(action) || []) t.classList.remove('held');
    }
    for (const [, ch] of chars) if (notify) send({ type: 'key', char: ch, down: false });
    chars.clear();
  }

  // The char a typed key sends: printable ASCII only, since DOOM takes a key as one byte and 'τ'
  // or 'я' would land on an unrelated key (F10, Enter, ...). A letter or digit from another
  // layout falls back to its physical key, so y/n prompts answer on any keyboard.
  function charOf(e) {
    if (e.key >= '!' && e.key <= '~') return e.key.toLowerCase();
    const m = /^Key([A-Z])$|^Digit([0-9])$/.exec(e.code);
    return m ? (m[1] || m[2]).toLowerCase() : null;
  }

  // The game has the keyboard when the screen is focused, or when nothing else is and this tab is
  // playing. Links and buttons keep their usual keys.
  function gameHasFocus() {
    const a = document.activeElement;
    if (a === screenEl) return true;
    if (a && a !== document.body && a !== document.documentElement) return false;
    return connected && controller && ACTIVE.has(state);
  }

  window.addEventListener('keydown', (e) => {
    if (e.metaKey || e.altKey || !gameHasFocus()) return;
    const action = CODES[e.code];
    if (action) {
      e.preventDefault();
      if (!e.repeat) press(action, 'k:' + e.code);
      return;
    }
    // other printable keys go through as characters (y/n prompts and the like)
    if (e.key.length === 1 && !e.ctrlKey && /\S/.test(e.key)) {
      e.preventDefault();
      const ch = charOf(e);
      if (ch === null || e.repeat || chars.has(e.code)) return;
      chars.set(e.code, ch);
      send({ type: 'key', char: ch, down: true });
    }
  });

  window.addEventListener('keyup', (e) => {
    const action = CODES[e.code];
    if (action) {
      if (holders.get(action)?.has('k:' + e.code)) { e.preventDefault(); release(action, 'k:' + e.code); }
      return;
    }
    const ch = chars.get(e.code);
    if (ch !== undefined) {
      e.preventDefault();
      chars.delete(e.code);
      send({ type: 'key', char: ch, down: false });
    }
  });

  // never leave a key stuck down when the tab loses the keyboard
  window.addEventListener('blur', () => releaseAll());
  document.addEventListener('visibilitychange', () => { if (document.hidden) releaseAll(); });

  const focusScreen = () => screenEl.focus({ preventScroll: true });

  for (const [action, els] of tiles) {
    for (const el of els) {
      el.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        e.preventDefault();  // keep the keyboard focus where it is
        try { el.setPointerCapture(e.pointerId); } catch { /* synthetic or already gone */ }
        press(action, 'p:' + e.pointerId);
        focusScreen();
      });
      const up = (e) => release(action, 'p:' + e.pointerId);
      el.addEventListener('pointerup', up);
      el.addEventListener('pointercancel', up);
      el.addEventListener('lostpointercapture', up);
      el.addEventListener('contextmenu', (e) => e.preventDefault());
      el.addEventListener('click', (e) => e.preventDefault());
    }
  }

  // ---------------------------------------------------------------- buttons

  startBtn.addEventListener('click', () => {
    const stopping = ACTIVE.has(state) && controller;
    send({ type: stopping ? 'stop' : 'start' });
    if (!stopping) focusScreen();
  });
  pauseBtn.addEventListener('click', () => {
    send({ type: state === 'paused' ? 'resume' : 'pause' });
    focusScreen();
  });
  screenEl.addEventListener('pointerdown', () => focusScreen());

  render();
  connect();
})();
