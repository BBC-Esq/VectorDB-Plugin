"use strict";

(function () {
  const { ICONS, esc, fmt, $ } = VDB;

  const SEND = '<svg viewBox="0 0 16 16"><path d="M2.4 2.6 14 8 2.4 13.4l1.9-5.4L2.4 2.6Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M4.3 8h5.2" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
  const EJECT = '<svg viewBox="0 0 16 16"><path d="M8 3 13 9H3L8 3Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M3 12.4h10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
  const COPY = '<svg viewBox="0 0 16 16"><rect x="5.2" y="5.2" width="8.3" height="8.3" rx="1.3" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M10.8 5.2V3.8c0-.7-.6-1.3-1.3-1.3H3.8c-.7 0-1.3.6-1.3 1.3v5.7c0 .7.6 1.3 1.3 1.3h1.4" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>';
  const KINDS = { pdf: ["pdf"], word: ["docx", "doc"], text: ["txt", "md", "rtf"], web: ["html", "htm"], email: ["eml", "msg"],
    sheet: ["csv", "xls", "xlsx", "xlsm"], image: ["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"],
    audio: ["mp3", "wav", "m4a", "flac", "ogg", "opus", "aac", "wma", "aiff", "aif", "webm", "mp4", "mkv", "mov", "avi"] };
  const HUES = { pdf: 4, word: 214, text: 190, web: 268, email: 36, sheet: 140, image: 320, audio: 168, other: 220 };
  const PHASE_TEXT = { loading: "Loading", searching: "Searching", answering: "Answering" };

  let state = null;
  let headSignature = "";
  let controlsSignature = "";
  let notesSignature = "";
  let actionsSignature = "";
  const turnSignatures = new Map();
  const expanded = new Set();
  let lastTranscript = 0;
  let lastFocus = 0;
  let askError = "";
  let copied = { id: null, timer: 0 };

  function extOf(name) {
    const dot = String(name).lastIndexOf(".");
    return dot > 0 ? String(name).slice(dot + 1).toLowerCase() : "";
  }

  function kindOf(name) {
    const ext = extOf(name);
    return Object.keys(KINDS).find((k) => KINDS[k].includes(ext)) || "other";
  }

  function badge(name) {
    const ext = extOf(name);
    return `<span class="file-kind" style="--h:${HUES[kindOf(name)]}">${esc(ext ? ext.toUpperCase() : "FILE")}</span>`;
  }

  function score(low, high) {
    if (low == null) return "";
    return low === high || high == null ? low.toFixed(3) : `${low.toFixed(3)}–${high.toFixed(3)}`;
  }

  function busy() {
    return Boolean(state && state.busy);
  }

  function currentDatabase() {
    return state.databases.find((d) => d.name === state.database) || null;
  }

  function currentModel() {
    return state.models.find((m) => m.name === state.local_model) || null;
  }

  function blocker() {
    if (busy()) return "busy";
    if (!state.databases.length) return "Create a database on the Create Database tab first.";
    const db = currentDatabase();
    if (!db) return "Choose a database to query.";
    if (db.model_missing) return "The embedding model this database uses is not downloaded.";
    if (!state.chunks_only && state.readiness) return state.readiness.message;
    if (!$("#question").value.trim()) return "Type a question.";
    return null;
  }

  function sourcesHTML(turn) {
    if (!turn.citations.length) {
      return turn.ended && turn.kind === "answer" && turn.answer.trim() && !turn.error ? '<div class="sources-none">No sources were returned.</div>' : "";
    }
    return '<div class="sources">'
      + `<div class="sources-title">${ICONS.file}<span>Sources</span><span class="n">${fmt.int(turn.citations.length)}</span></div>`
      + turn.citations.map((c) => '<div class="source">'
        + badge(c.name)
        + `<button type="button" class="source-name" data-open="${esc(c.path)}" data-tip-text="${esc(c.path)}">${esc(c.name)}</button>`
        + (c.pages ? `<span class="source-pages">p. ${esc(c.pages)}</span>` : "")
        + `<span class="source-score" data-tip-text="Similarity to the question">${esc(score(c.low, c.high))}</span>`
        + `<button type="button" class="file-action" data-reveal="${esc(c.path)}" data-tip-text="Show in folder">${ICONS.folder}</button>`
        + "</div>").join("")
      + "</div>";
  }

  function tokensHTML(tokens) {
    if (!tokens) return "";
    const total = Math.max(1, tokens.available);
    const parts = [["instructions", tokens.instruction, "a"], ["question", tokens.question, "b"], ["sources", tokens.contexts, "c"], ["answer", tokens.response, "d"]];
    const bar = parts.map(([label, value, cls]) => `<i class="seg ${cls}" style="width:${Math.min(100, (value / total) * 100).toFixed(2)}%" data-tip-text="${esc(label)}: ${fmt.int(value)} tokens"></i>`).join("");
    const left = tokens.remaining;
    return '<div class="tokens">'
      + `<div class="token-bar">${bar}</div>`
      + `<div class="token-text">${fmt.int(tokens.available)}-token context: `
      + parts.map(([label, value, cls]) => `<span class="key ${cls}"></span>${esc(label)} ${fmt.int(value)}`).join(" · ")
      + ` · <b class="${left < 0 ? "over" : ""}">${fmt.int(left)} left</b></div></div>`;
  }

  function chunksHTML(turn) {
    if (!turn.chunks) return "";
    if (!turn.chunks.length) {
      return `<div class="status warn">${ICONS.warn}<div class="status-body">No chunks passed the similarity threshold of ${esc(turn.similarity ?? "—")}. `
        + 'Lower the Similarity setting on the <button type="button" class="link" data-tab="Settings">Settings tab</button> and ask again.</div></div>';
    }
    return '<div class="chunks">' + turn.chunks.map((c, i) => {
      const key = `${turn.id}:${i}`;
      const long = c.text.length > 900 || c.text.split("\n").length > 12;
      const open = expanded.has(key);
      return `<div class="chunk${long && !open ? " clipped" : ""}">`
        + '<div class="chunk-head">'
        + `<span class="chunk-n">${i + 1}</span>${badge(c.name)}`
        + `<button type="button" class="source-name" data-open="${esc(c.path)}" data-tip-text="${esc(c.path)}">${esc(c.name)}</button>`
        + (c.page ? `<span class="source-pages">p. ${esc(c.page)}</span>` : "")
        + (c.score != null ? `<span class="source-score" data-tip-text="Similarity to the question">${c.score.toFixed(3)}</span>` : "")
        + `<button type="button" class="file-action" data-reveal="${esc(c.path)}" data-tip-text="Show in folder">${ICONS.folder}</button>`
        + "</div>"
        + `<div class="chunk-text">${esc(c.text)}</div>`
        + (long ? `<button type="button" class="link chunk-more" data-more="${key}">${open ? "Show less" : "Show more"}</button>` : "")
        + "</div>";
    }).join("") + "</div>";
  }

  function errorHTML(turn) {
    if (!turn.error) return "";
    const hint = /similarity threshold/i.test(turn.error)
      ? ' <button type="button" class="link" data-tab="Settings">Change the Similarity setting</button>'
      : /api key/i.test(turn.error) ? ' <button type="button" class="link" data-act="backend_settings">Open Chat Backend Settings</button>' : "";
    return `<div class="status error turn-error">${ICONS.error}<div class="status-body"><div class="error-text">${esc(turn.error)}</div>${hint}</div></div>`;
  }

  function phaseHTML(turn) {
    if (turn.ended || turn.phase === "answering") return "";
    let text;
    if (turn.phase === "loading") text = `Loading ${esc(turn.model)}. The first question after choosing a model takes longer.`;
    else if (turn.kind === "chunks") text = `Searching <b>${esc(turn.database)}</b>…`;
    else text = `Searching <b>${esc(turn.database)}</b> and waiting for ${esc(turn.model || turn.backend)}…`;
    return `<div class="status phase"><span class="spinner"></span><div class="status-body">${text}</div></div>`;
  }

  function footHTML(turn) {
    if (!turn.ended) return "";
    const hasText = turn.kind === "chunks" ? Boolean(turn.chunks && turn.chunks.length) : Boolean(turn.answer.trim());
    if (!hasText) return "";
    const buttons = [];
    const isCopied = copied.id === turn.id;
    buttons.push(`<button type="button" class="btn ghost small" data-act="copy" data-turn="${turn.id}">${isCopied ? ICONS.check : COPY}${isCopied ? "Copied" : "Copy"}</button>`);
    if (turn.kind === "answer") {
      if (state.speaking === turn.id) {
        buttons.push(`<button type="button" class="btn danger small" data-act="stop_speaking">${ICONS.stop}Stop reading</button>`);
      } else {
        const tip = state.speaking != null ? "Another answer is being read aloud." : `Read the answer aloud with ${state.tts || "the selected voice"}`;
        buttons.push(`<button type="button" class="btn ghost small" data-act="speak" data-turn="${turn.id}" data-tip-text="${esc(tip)}"${state.speaking != null ? " disabled" : ""}>${ICONS.speaker}Speak</button>`);
      }
    }
    return `<div class="turn-foot">${tokensHTML(turn.tokens)}<div class="turn-actions">${buttons.join("")}</div></div>`;
  }

  function turnHTML(turn) {
    const who = turn.kind === "chunks" ? "Chunks only" : (turn.model || turn.backend);
    const time = turn.ended
      ? `<span class="elapsed">${VDB.elapsed(turn.ended - turn.started)}</span>`
      : `<span class="elapsed" data-since="${turn.started}">${VDB.elapsed(Date.now() / 1000 - turn.started)}</span>`;
    const answer = turn.kind === "answer" && turn.answer
      ? `<div class="md${turn.ended ? "" : " streaming"}">${QueryMarkdown.render(turn.answer)}</div>`
      : "";
    return '<div class="turn-head">'
      + `<div class="turn-question">${esc(turn.question)}</div>`
      + `<div class="turn-meta"><span>${esc(turn.database)}</span><span class="dot">·</span><span>${esc(who)}</span><span class="dot">·</span>${time}</div>`
      + "</div>"
      + phaseHTML(turn)
      + answer
      + chunksHTML(turn)
      + errorHTML(turn)
      + sourcesHTML(turn)
      + footHTML(turn);
  }

  function atBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }

  function renderTurns() {
    const convo = $("#convo");
    const stick = atBottom(convo);
    const ids = new Set(state.turns.map((t) => t.id));
    for (const el of [...convo.querySelectorAll(".turn")]) {
      if (!ids.has(Number(el.dataset.turn))) {
        el.remove();
        turnSignatures.delete(Number(el.dataset.turn));
      }
    }
    let added = false;
    for (const turn of state.turns) {
      const signature = JSON.stringify([turn, state.speaking, state.tts, copied.id === turn.id, [...expanded].filter((k) => k.startsWith(`${turn.id}:`))]);
      let el = convo.querySelector(`.turn[data-turn="${turn.id}"]`);
      if (!el) {
        el = document.createElement("article");
        el.className = "turn";
        el.dataset.turn = String(turn.id);
        convo.appendChild(el);
        added = true;
      }
      if (turnSignatures.get(turn.id) !== signature) {
        turnSignatures.set(turn.id, signature);
        el.className = `turn ${turn.kind}${turn.ended ? "" : " live"}${turn.error ? " failed" : ""}`;
        el.innerHTML = turnHTML(turn);
      }
    }
    const empty = $("#convo-empty");
    if (!state.turns.length) {
      if (!empty) convo.insertAdjacentHTML("beforeend", emptyHTML());
    } else if (empty) {
      empty.remove();
    }
    if (added || stick) convo.scrollTop = convo.scrollHeight;
  }

  function emptyHTML() {
    return '<div class="convo-empty" id="convo-empty">'
      + `<div class="empty-icon">${ICONS.database}</div>`
      + '<div class="empty-title">Ask a question about one of your databases</div>'
      + "<div>The answer streams in here with links to the sources it came from. Turn on <b>Chunks only</b> to see the matching passages without asking a chat model.</div>"
      + "</div>";
  }

  function renderHead() {
    const signature = JSON.stringify([state.turns.length, busy(), state.speaking]);
    if (signature === headSignature) return;
    headSignature = signature;
    const count = state.turns.length;
    const speaking = state.speaking != null
      ? `<span class="speaking-pill"><span class="pulse"></span>Reading aloud<button type="button" class="link" data-act="stop_speaking">Stop</button></span>`
      : "";
    $("#convo-head").innerHTML = '<div class="convo-title-row">'
      + `<h2 class="section-title">${ICONS.list}<span>Answers</span>${count ? `<span class="title-count">${fmt.int(count)}</span>` : ""}</h2>`
      + speaking
      + (count ? `<button type="button" class="btn ghost small" data-act="clear"${busy() ? " disabled" : ""}>${ICONS.close}Clear</button>` : "")
      + "</div>";
  }

  function modelOption(m) {
    const notes = [];
    if (m.memory != null) notes.push(`~${m.memory} GB`);
    if (m.needs_token) notes.push("needs a token");
    else if (!m.downloaded) notes.push("downloads on first use");
    return { label: m.name, note: notes.join(" · ") };
  }

  function renderControls() {
    const signature = JSON.stringify([state.databases, state.database, state.backend, state.models, state.local_model, state.loaded_model,
      state.ejecting, busy(), state.chunks_only]);
    if (signature === controlsSignature) return;
    controlsSignature = signature;
    const db = currentDatabase();
    const local = state.backend === "Local Model";
    const model = currentModel();
    const dbOption = db ? { label: db.name, note: db.model || "" } : { label: state.databases.length ? "Choose a database" : "No databases yet" };
    let modelField = "";
    if (local) {
      const loaded = state.loaded_model;
      const eject = state.ejecting
        ? '<button type="button" class="btn ghost small eject" disabled><span class="spinner"></span>Ejecting…</button>'
        : loaded
          ? `<button type="button" class="btn ghost small eject" data-act="eject" data-tip-text="Unload ${esc(loaded)} to free the memory it uses"${busy() ? " disabled" : ""}>${EJECT}Eject</button>`
          : "";
      const status = loaded ? `<span class="loaded-note">${esc(loaded)} is loaded</span>` : "";
      modelField = '<div class="field">'
        + `<div class="field-label"><span>Model</span>${status}</div>`
        + `<div class="select-row">${VDB.selectButton("query.model", model ? modelOption(model) : { label: state.models.length ? "Choose a model" : "None available" }, busy() || state.ejecting || !state.models.length)}${eject}</div>`
        + "</div>";
    } else {
      modelField = '<div class="field">'
        + '<div class="field-label"><span>Connection</span></div>'
        + `<button type="button" class="btn ghost settings-btn" data-act="backend_settings">${ICONS.chip}Backend settings</button>`
        + "</div>";
    }
    $("#ask-controls").innerHTML = `<div class="control-grid${state.chunks_only ? " chunks-mode" : ""}">`
      + '<div class="field">'
      + '<div class="field-label"><span>Database</span></div>'
      + VDB.selectButton("query.database", dbOption, busy() || !state.databases.length)
      + "</div>"
      + '<div class="field backend-field">'
      + '<div class="field-label"><span>Answered by</span></div>'
      + VDB.selectButton("query.backend", { label: state.backend }, busy())
      + "</div>"
      + `<div class="model-slot">${modelField}</div>`
      + "</div>";
  }

  function stat(label, value) {
    return `<span class="stat"><span class="stat-label">${esc(label)}</span><b>${esc(value)}</b></span>`;
  }

  function renderNotes() {
    const signature = JSON.stringify([state.settings, state.readiness, state.notice, state.chunks_only, state.backend, currentDatabase(), state.cpu_only, state.databases.length]);
    if (signature === notesSignature) return;
    notesSignature = signature;
    const s = state.settings;
    let html = '<div class="settings-line">'
      + stat("Contexts", s.contexts != null ? fmt.int(s.contexts) : "—")
      + stat("Similarity", s.similarity != null ? String(s.similarity) : "—")
      + stat("Search term", s.search_term || "none")
      + stat("File type", s.document_types)
      + stat("Device", String(s.device).toUpperCase())
      + '<button type="button" class="link settings-link" data-tab="Settings">Change on the Settings tab</button>'
      + "</div>";
    const db = currentDatabase();
    if (!state.databases.length) {
      html += `<div class="status info">${ICONS.info}<div class="status-body">There are no databases to query yet. <button type="button" class="link" data-tab="Create Database">Create one on the Create Database tab</button></div></div>`;
    } else if (db && db.model_missing) {
      html += `<div class="status warn">${ICONS.warn}<div class="status-body">The embedding model this database was created with, <b>${esc(db.model)}</b>, is not downloaded, so the database can't be searched. <button type="button" class="link" data-tab="Models">Download it on the Models tab</button></div></div>`;
    }
    if (!state.chunks_only && state.readiness) {
      const button = state.readiness.action
        ? ` <button type="button" class="link" data-act="fix_readiness">${esc(state.readiness.label)}</button>`
        : "";
      html += `<div class="status warn">${ICONS.warn}<div class="status-body">${esc(state.readiness.message)}${button}</div></div>`;
    }
    if (!state.chunks_only && state.cpu_only && (state.backend === "Local Model" || state.backend === "LM Studio")) {
      html += `<div class="status info">${ICONS.info}<div class="status-body">Without a supported NVIDIA GPU only small local models are offered, and they run slowly on the CPU. LM Studio is recommended for larger or faster models.</div></div>`;
    }
    const n = state.notice;
    if (n) {
      const icon = { ok: ICONS.check, warn: ICONS.warn, error: ICONS.error }[n.kind] || ICONS.info;
      html += `<div class="status ${n.kind}">${icon}<div class="status-body">${esc(n.message)}</div>`
        + `<button type="button" class="icon-btn btn ghost small dismiss" data-act="dismiss_notice" data-tip-text="Dismiss">${ICONS.close}</button></div>`;
    }
    $("#ask-notes").innerHTML = html;
  }

  function micHTML() {
    if (state.voice === "recording") {
      return `<button type="button" class="btn danger small mic" data-act="stop_recording"><span class="rec-dot"></span>Stop recording<span class="elapsed" data-since="${state.voice_started}">${VDB.elapsed(Date.now() / 1000 - state.voice_started)}</span></button>`;
    }
    if (state.voice === "transcribing") {
      return '<button type="button" class="btn ghost small mic" disabled><span class="spinner"></span>Transcribing…</button>';
    }
    return `<button type="button" class="btn ghost small mic" data-act="record" data-tip-text="Record your question with the microphone; it is transcribed into the box">${ICONS.mic}Speak a question</button>`;
  }

  function renderActions() {
    const reason = blocker();
    const signature = JSON.stringify([reason, state.chunks_only, state.voice, state.voice_started, busy(), askError]);
    if (signature === actionsSignature) return;
    actionsSignature = signature;
    const ready = !reason;
    const label = busy() ? '<span class="spinner on-primary"></span>Answering…' : `${SEND}${state.chunks_only ? "Search" : "Ask"}`;
    const tip = !ready && reason !== "busy" ? ` data-tip-text="${esc(reason)}"` : "";
    $("#ask-actions").innerHTML = '<div class="actions-row">'
      + `<button type="button" class="switch" role="switch" aria-checked="${state.chunks_only}" data-switch="chunks_only"${busy() ? " disabled" : ""}><span class="track"></span><span>Chunks only</span></button>`
      + micHTML()
      + `<span class="ask-hint">${askError ? `<span class="ask-error">${esc(askError)}</span>` : state.chunks_only ? "Ctrl+Enter to search" : "Ctrl+Enter to ask"}</span>`
      + `<button type="button" class="btn primary ask-btn" data-act="ask"${ready ? "" : " disabled"}${tip}>${label}</button>`
      + "</div>";
  }

  function render() {
    renderHead();
    renderTurns();
    renderControls();
    renderNotes();
    renderActions();
    if (state.transcript && state.transcript.id !== lastTranscript) {
      lastTranscript = state.transcript.id;
      const box = $("#question");
      const text = state.transcript.text;
      box.value = box.value.trim() ? `${box.value.replace(/\s+$/, "")} ${text}` : text;
      box.focus();
      box.setSelectionRange(box.value.length, box.value.length);
      askError = "";
      renderActions();
    }
    if (state.focus !== lastFocus) {
      lastFocus = state.focus;
      $("#question").focus();
    }
  }

  function ask() {
    if (blocker()) return;
    const box = $("#question");
    const question = box.value.trim();
    askError = "";
    VDB.tooltip.hide();
    VDB.call("ask", { question }).then((result) => {
      if (result.ok) {
        box.value = "";
      } else if (result.error) {
        askError = result.error;
      }
      actionsSignature = "";
      renderActions();
    });
  }

  function openDatabaseSelect() {
    VDB.openSelect("query.database", {
      title: "Database",
      search: state.databases.length > 8,
      placeholder: "Filter databases",
      options: state.databases.map((d) => ({ value: d.name, label: d.name, note: d.model || "" })),
      value: state.database,
      onPick: (name) => {
        if (name !== state.database) VDB.call("select_database", { name });
      },
    });
  }

  function openBackendSelect() {
    VDB.openSelect("query.backend", {
      title: "Answered by",
      options: state.backends.map((b) => ({ value: b, label: b, note: b === "Local Model" ? "runs on this computer" : b === "Kobold" || b === "LM Studio" ? "local server" : "online service" })),
      value: state.backend,
      onPick: (name) => {
        if (name !== state.backend) VDB.call("select_backend", { name });
      },
    });
  }

  function openModelSelect() {
    VDB.openSelect("query.model", {
      title: "Local model",
      options: state.models.map((m) => ({ value: m.name, ...modelOption(m) })),
      value: state.local_model,
      onPick: (name) => {
        if (name !== state.local_model) VDB.call("select_model", { name });
      },
    });
  }

  function showCopied(id) {
    clearTimeout(copied.timer);
    copied.id = id;
    renderTurns();
    copied.timer = setTimeout(() => {
      copied.id = null;
      renderTurns();
    }, 1600);
  }

  function act(button) {
    const action = button.dataset.act;
    const turn = button.dataset.turn ? Number(button.dataset.turn) : null;
    if (action === "ask") ask();
    else if (action === "copy") VDB.call("copy", { turn_id: turn }).then((r) => r.ok && showCopied(turn));
    else if (action === "speak") VDB.call("speak", { turn_id: turn });
    else if (action === "clear") {
      expanded.clear();
      VDB.call("clear");
    } else if (action === "eject" || action === "stop_speaking" || action === "record" || action === "stop_recording"
               || action === "backend_settings" || action === "fix_readiness" || action === "dismiss_notice") {
      VDB.call(action);
    }
  }

  function bind() {
    document.addEventListener("click", (e) => {
      const tab = e.target.closest("[data-tab]");
      if (tab) {
        VDB.call("open_tab", { name: tab.dataset.tab });
        return;
      }
      const select = e.target.closest("[data-select]");
      if (select && !select.disabled) {
        const id = select.dataset.select;
        if (id === "query.database") openDatabaseSelect();
        else if (id === "query.backend") openBackendSelect();
        else if (id === "query.model") openModelSelect();
        return;
      }
      const sw = e.target.closest('[data-switch="chunks_only"]');
      if (sw && !sw.disabled) {
        VDB.call("set_chunks_only", { value: sw.getAttribute("aria-checked") !== "true" });
        return;
      }
      const open = e.target.closest("[data-open]");
      if (open) {
        VDB.call("open_file", { path: open.dataset.open });
        return;
      }
      const reveal = e.target.closest("[data-reveal]");
      if (reveal) {
        VDB.call("reveal_file", { path: reveal.dataset.reveal });
        return;
      }
      const more = e.target.closest("[data-more]");
      if (more) {
        const key = more.dataset.more;
        if (expanded.has(key)) expanded.delete(key);
        else expanded.add(key);
        renderTurns();
        return;
      }
      const button = e.target.closest("[data-act]");
      if (button && !button.disabled) act(button);
    });
    const box = $("#question");
    box.addEventListener("input", () => {
      if (askError) askError = "";
      renderActions();
    });
    box.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        ask();
      }
    });
  }

  bind();

  window.TabApp = {
    init(payload) {
      state = payload;
      lastFocus = payload.focus;
      lastTranscript = payload.transcript ? payload.transcript.id : 0;
      render();
    },
    setState(payload) {
      state = payload;
      render();
    },
  };
})();
