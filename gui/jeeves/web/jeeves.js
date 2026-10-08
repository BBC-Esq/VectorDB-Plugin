"use strict";

(function () {
  const { ICONS, esc, $ } = VDB;

  const SEND = '<svg viewBox="0 0 16 16"><path d="M2.4 2.6 14 8 2.4 13.4l1.9-5.4L2.4 2.6Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M4.3 8h5.2" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
  const EJECT = '<svg viewBox="0 0 16 16"><path d="M8 3 13 9H3L8 3Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M3 12.4h10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';

  let state = null;
  let plaqueDone = false;
  let headSignature = "";
  let modelSignature = "";
  let promptSignature = "";
  let notesSignature = "";
  let voiceSignature = "";
  const messageSignatures = new Map();
  const openSources = new Set();
  let askError = "";
  let poemIndex = 0;

  function blocked() {
    if (!state.models.length) return "No chat models are available on this computer";
    if (state.loading) return "Jeeves is getting ready…";
    if (!state.loaded) return "Please choose a model above to start";
    if (state.db_error) return "The user manual database could not be loaded";
    if (state.prompt) return "Please answer Jeeves first";
    if (state.phase !== "idle") return "Jeeves is answering…";
    return null;
  }

  function avatar() {
    return state.image ? `<span class="avatar" style="background-image:url('${esc(state.image)}')"></span>` : '<span class="avatar plain">J</span>';
  }

  function bodyHTML(message) {
    if (message.kind === "answer") {
      return `<div class="md${message.streaming ? " streaming" : ""}">${message.text ? Markdown.render(message.text) : '<p><span class="typing"><i></i><i></i><i></i></span></p>'}</div>`;
    }
    return `<div class="plain${message.kind === "poem" ? " poem" : ""}${message.streaming ? " streaming" : ""}">${esc(message.text)}</div>`;
  }

  function sourceListHTML(message) {
    return '<div class="sources">' + message.sources.map((s, i) => '<div class="src">'
      + '<div class="src-head">'
      + `<span class="src-n">${i + 1}</span>`
      + `<button type="button" class="src-title" data-open="${esc(s.path)}" data-tip-text="${esc(s.path)}">${esc(s.title)}</button>`
      + (s.score != null ? `<span class="src-score">score ${s.score.toFixed(3)}</span>` : "")
      + "</div>"
      + `<div class="src-text">${esc(s.text)}</div>`
      + "</div>").join("") + "</div>";
  }

  function footHTML(message) {
    if (message.role !== "jeeves" || message.streaming || !["answer", "poem"].includes(message.kind) || !message.text.trim()) return "";
    let speak = "";
    if (state.speaking === message.id) {
      speak = `<button type="button" class="btn danger small" data-act="stop_speaking">${ICONS.stop}Cancel Playback</button>`;
    } else if (state.tts) {
      const busy = state.speaking != null;
      speak = `<button type="button" class="btn ghost small" data-act="speak" data-message="${message.id}"${busy ? " disabled" : ""}>${ICONS.speaker}Speak Response</button>`;
    }
    let toggle = "";
    let list = "";
    if (message.sources.length) {
      const open = openSources.has(message.id);
      toggle = `<button type="button" class="link src-toggle" data-sources="${message.id}">${open ? "Hide" : "Show"} sources (${message.sources.length})</button>`;
      if (open) list = sourceListHTML(message);
    }
    if (!speak && !toggle) return "";
    return `<div class="msg-foot"><div class="msg-actions">${speak}${toggle}</div>${list}</div>`;
  }

  function messageHTML(message) {
    if (message.role === "user") {
      return `<div class="bubble user-bubble">${esc(message.text)}</div>`;
    }
    return `${avatar()}<div class="bubble jeeves-bubble">${bodyHTML(message)}${footHTML(message)}</div>`;
  }

  function atBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }

  function renderChat() {
    const chat = $("#chat");
    const stick = atBottom(chat);
    const ids = new Set(state.messages.map((m) => m.id));
    for (const el of [...chat.querySelectorAll(".msg")]) {
      if (!ids.has(Number(el.dataset.id))) {
        el.remove();
        messageSignatures.delete(Number(el.dataset.id));
      }
    }
    let added = false;
    let previous = null;
    for (const message of state.messages) {
      const signature = JSON.stringify([message, state.speaking, state.tts, openSources.has(message.id), state.image]);
      let el = chat.querySelector(`.msg[data-id="${message.id}"]`);
      if (!el) {
        el = document.createElement("div");
        el.dataset.id = String(message.id);
        if (previous && previous.nextSibling) chat.insertBefore(el, previous.nextSibling);
        else chat.appendChild(el);
        added = true;
      }
      if (messageSignatures.get(message.id) !== signature) {
        messageSignatures.set(message.id, signature);
        el.className = `msg ${message.role} kind-${message.kind}`;
        el.innerHTML = messageHTML(message);
      }
      previous = el;
    }
    let status = $("#chat-status");
    const statusText = state.phase === "searching" ? "Jeeves is looking through the user guide…" : "";
    if (statusText) {
      if (!status) {
        chat.insertAdjacentHTML("beforeend", '<div class="chat-status" id="chat-status"></div>');
        status = $("#chat-status");
      }
      status.innerHTML = `${avatar()}<div class="bubble jeeves-bubble"><span class="typing"><i></i><i></i><i></i></span><span class="status-text">${esc(statusText)}</span></div>`;
      chat.appendChild(status);
      added = true;
    } else if (status) {
      status.remove();
    }
    if (added || stick) chat.scrollTop = chat.scrollHeight;
  }

  function renderPlaque() {
    if (plaqueDone || !state) return;
    plaqueDone = true;
    $("#plaque").innerHTML = state.image
      ? `<img src="${esc(state.image)}" alt="Ask Jeeves" draggable="false">`
      : '<div class="plaque-text">Ask Jeeves</div>';
  }

  function modelOption(m) {
    const notes = [];
    if (m.memory != null) notes.push(`~${m.memory} GB`);
    if (!m.downloaded) notes.push("downloads on first use");
    return { label: m.name, note: notes.join(" · ") };
  }

  function renderModel() {
    const signature = JSON.stringify([state.models, state.model, state.loaded, state.loading, state.phase]);
    if (signature === modelSignature) return;
    modelSignature = signature;
    const current = state.models.find((m) => m.name === state.model);
    const option = current ? modelOption(current) : { label: state.models.length ? "Please choose a model..." : "No chat models are available on this computer" };
    let status = "";
    if (state.loading) {
      status = `<span class="model-status"><span class="spinner"></span>Loading ${esc(state.loading)} ... (the first time, this downloads the model -- please wait.)</span>`;
    } else if (state.loaded) {
      status = `<span class="model-status ready">${ICONS.check}${esc(state.loaded)} is ready</span>`;
    }
    const busy = state.phase !== "idle" || Boolean(state.loading);
    const eject = state.loaded
      ? `<button type="button" class="btn ghost small eject" data-act="eject" data-tip-text="Unload ${esc(state.loaded)} to free the memory it uses"${busy ? " disabled" : ""}>${EJECT}Eject</button>`
      : "";
    $("#model-row").innerHTML = '<span class="field-label">Model</span>'
      + `<div class="model-select">${VDB.selectButton("jeeves.model", option, busy || !state.models.length)}</div>`
      + eject + status;
  }

  function renderHead() {
    const signature = JSON.stringify([state.messages.length, state.phase]);
    if (signature === headSignature) return;
    headSignature = signature;
    const many = state.messages.length > 1;
    $("#chat-head").innerHTML = many
      ? `<div class="chat-head"><button type="button" class="btn ghost small" data-act="clear"${state.phase !== "idle" ? " disabled" : ""}>${ICONS.close}Clear</button></div>`
      : "";
  }

  function renderPrompt() {
    const signature = JSON.stringify([state.prompt, state.poems, poemIndex]);
    if (signature === promptSignature) return;
    promptSignature = signature;
    const p = state.prompt;
    let html = "";
    if (p && p.type === "offer") {
      html = '<div class="prompt-row">'
        + '<button type="button" class="btn primary small" data-act="poem_yes">Yes, please</button>'
        + '<button type="button" class="btn ghost small" data-act="poem_no">No, thank you</button>'
        + "</div>";
    } else if (p && p.type === "choose") {
      poemIndex = Math.min(poemIndex, Math.max(0, state.poems.length - 1));
      html = '<div class="prompt-row">'
        + `<div class="poem-select">${VDB.selectButton("jeeves.poem", { label: state.poems[poemIndex] || "" })}</div>`
        + '<button type="button" class="btn primary small" data-act="poem_recite">Recite</button>'
        + '<button type="button" class="btn ghost small" data-act="poem_cancel">Never mind</button>'
        + "</div>";
    }
    $("#prompt").innerHTML = html;
  }

  function renderNotes() {
    const signature = JSON.stringify([state.notice, state.db_error, askError]);
    if (signature === notesSignature) return;
    notesSignature = signature;
    let html = "";
    if (state.db_error) {
      html += `<div class="status error">${ICONS.error}<div class="status-body">The user manual database could not be loaded, so Jeeves cannot answer questions. ${esc(state.db_error)}</div></div>`;
    }
    const n = state.notice;
    if (n) {
      const icon = { warn: ICONS.warn, error: ICONS.error }[n.kind] || ICONS.info;
      html += `<div class="status ${n.kind}">${icon}<div class="status-body">${esc(n.message)}</div>`
        + `<button type="button" class="icon-btn btn ghost small dismiss" data-act="dismiss_notice" data-tip-text="Dismiss">${ICONS.close}</button></div>`;
    }
    if (askError) {
      html += `<div class="status warn">${ICONS.warn}<div class="status-body">${esc(askError)}</div></div>`;
    }
    $("#notes").innerHTML = html;
  }

  function renderVoice() {
    const signature = JSON.stringify([state.tts, state.voice, state.speed, state.speaking]);
    if (signature === voiceSignature) return;
    voiceSignature = signature;
    const off = !state.tts || state.speaking != null;
    const tip = state.tts ? "" : ' data-tip-text="Text-to-speech is not available. Please check if KokoroTTS is properly installed."';
    $("#voice-row").innerHTML = `<div class="voice-row"${tip}>`
      + '<span class="field-label">Voice</span>'
      + `<div class="mini-select">${VDB.selectButton("jeeves.voice", { label: state.voice }, off)}</div>`
      + '<span class="field-label">Speed</span>'
      + `<div class="mini-select">${VDB.selectButton("jeeves.speed", { label: state.speed }, off)}</div>`
      + "</div>";
  }

  function renderInput() {
    const reason = blocked();
    const input = $("#message");
    input.disabled = Boolean(reason);
    input.placeholder = reason || "Type your message here...";
    $("#ask-box").classList.toggle("disabled", Boolean(reason));
    const button = $("#ask-btn");
    button.disabled = Boolean(reason) || !input.value.trim();
    button.innerHTML = state.phase === "answering" || state.phase === "searching" ? '<span class="spinner on-primary"></span>Asking…' : `${SEND}Ask`;
  }

  function render() {
    renderPlaque();
    renderModel();
    renderHead();
    renderChat();
    renderPrompt();
    renderNotes();
    renderVoice();
    renderInput();
  }

  function ask() {
    const input = $("#message");
    const text = input.value.trim();
    if (!text || blocked()) return;
    askError = "";
    VDB.call("ask", { text }).then((result) => {
      if (result.ok) input.value = "";
      else if (result.error) askError = result.error;
      notesSignature = "";
      render();
    });
  }

  function choose(id, title, options, value, onPick, search = false) {
    VDB.openSelect(id, { title, options, value, onPick, search, placeholder: "Filter" });
  }

  function bind() {
    document.addEventListener("click", (e) => {
      const select = e.target.closest("[data-select]");
      if (select && !select.disabled) {
        const id = select.dataset.select;
        if (id === "jeeves.model") {
          choose(id, "Model", state.models.map((m) => ({ value: m.name, ...modelOption(m) })), state.model, (name) => {
            if (name !== state.model || !state.loaded) VDB.call("select_model", { name });
          });
        } else if (id === "jeeves.voice") {
          choose(id, "Voice", state.voices.map((v) => ({ value: v, label: v })), state.voice, (value) => VDB.call("set_voice", { value }));
        } else if (id === "jeeves.speed") {
          choose(id, "Speed", state.speeds.map((v) => ({ value: v, label: v })), state.speed, (value) => VDB.call("set_speed", { value }));
        } else if (id === "jeeves.poem") {
          choose(id, "Which poem?", state.poems.map((label, i) => ({ value: i, label })), poemIndex, (value) => {
            poemIndex = value;
            promptSignature = "";
            renderPrompt();
          }, state.poems.length > 8);
        }
        return;
      }
      const toggle = e.target.closest("[data-sources]");
      if (toggle) {
        const id = Number(toggle.dataset.sources);
        if (openSources.has(id)) openSources.delete(id);
        else openSources.add(id);
        renderChat();
        return;
      }
      const source = e.target.closest("[data-open]");
      if (source) {
        VDB.call("open_source", { path: source.dataset.open });
        return;
      }
      const button = e.target.closest("[data-act]");
      if (!button || button.disabled) return;
      const action = button.dataset.act;
      if (action === "ask") ask();
      else if (action === "speak") VDB.call("speak", { message_id: Number(button.dataset.message) });
      else if (action === "poem_yes") VDB.call("poem_answer", { yes: true });
      else if (action === "poem_no") VDB.call("poem_answer", { yes: false });
      else if (action === "poem_recite") VDB.call("poem_recite", { index: poemIndex });
      else if (action === "clear") {
        openSources.clear();
        VDB.call("clear");
      } else VDB.call(action);
    });
    const input = $("#message");
    input.addEventListener("input", () => {
      if (askError) {
        askError = "";
        renderNotes();
      }
      renderInput();
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        ask();
      }
    });
  }

  bind();

  window.TabApp = {
    init(payload) {
      state = payload;
      render();
      $("#message").focus();
    },
    setState(payload) {
      const wasBlocked = state && blocked();
      state = payload;
      render();
      if (wasBlocked && !blocked()) $("#message").focus();
    },
  };
})();
