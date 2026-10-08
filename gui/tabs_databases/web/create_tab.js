"use strict";

(function () {
  const { ICONS, esc, fmt, $ } = VDB;

  const LOCAL_ACTIONS = new Set(["clear_selection", "remove_selected", "toggle_log", "create"]);
  const DEVICE_LABELS = { cuda: "GPU", cpu: "CPU", mps: "Apple GPU" };

  let state = null;
  let nameDraft = "";
  let formSignature = "";
  let actionSignature = "";
  let logOpen = false;
  let logLines = [];
  let logTimer = 0;
  let lastResult = "";

  function busy() {
    return Boolean(state && ["scanning", "building", "saving"].includes(state.build.phase));
  }

  function currentModel() {
    return state.models.find((m) => m.path === state.model) || null;
  }

  function nameProblem(name) {
    if (!name) return null;
    if (name.length < 3) return "Use at least 3 characters.";
    if (name === "null" || name === "none") return "That name is reserved. Choose another one.";
    if (name.length > state.name_limit) return `Use at most ${state.name_limit} characters.`;
    if (state.existing.includes(name)) return "A database with this name already exists.";
    return null;
  }

  function blocker() {
    if (busy()) return "busy";
    if (state.files.staging) return "Files are still being added to the list.";
    if (state.transcribing) return "A transcription is running on the Tools tab. Create the database after it finishes so the transcript is included.";
    if (!state.files.count) return "Add files to the list first.";
    if (!state.model) return state.models.length ? "Choose an embedding model." : "Download an embedding model on the Models tab first.";
    if (!nameDraft) return "Enter a name for the database.";
    return nameProblem(nameDraft);
  }

  function statHTML(label, value) {
    return `<span class="stat"><span class="stat-label">${esc(label)}</span><b>${esc(value ?? "—")}</b></span>`;
  }

  function nameFieldHTML() {
    const problem = busy() ? null : nameProblem(nameDraft);
    const message = problem
      ? `<div class="field-error" id="name-message">${esc(problem)}</div>`
      : '<div class="field-hint" id="name-message">Lowercase letters, numbers, hyphens and underscores</div>';
    return '<div class="field">'
      + '<div class="field-label"><span>Database name</span></div>'
      + `<label class="input${problem ? " invalid" : ""}${busy() ? " disabled" : ""}"><input id="db-name" type="text" value="${esc(nameDraft)}" maxlength="${state.name_limit}" placeholder="for example contracts_2024" autocomplete="off" spellcheck="false"${busy() ? " disabled" : ""}></label>`
      + message + "</div>";
  }

  function modelFieldHTML() {
    const model = currentModel();
    let hint = "";
    if (!state.models.length) {
      hint = '<div class="field-hint">No embedding models are downloaded yet. Get one on the <button type="button" class="link" data-tab="Models">Models tab</button>.</div>';
    } else if (model && model.known) {
      hint = `<div class="field-hint">${esc(model.vendor)} · ${fmt.int(model.dimensions)} dimensions · reads up to ${fmt.int(model.max_sequence)} tokens</div>`;
    } else if (model) {
      hint = '<div class="field-hint">A model folder this program does not list</div>';
    } else {
      hint = '<div class="field-hint">Turns each chunk of text into a vector</div>';
    }
    const option = model ? { label: model.name } : { label: state.models.length ? "Choose a model" : "No models downloaded" };
    return '<div class="field">'
      + '<div class="field-label"><span>Embedding model</span></div>'
      + VDB.selectButton("create.model", option, busy() || !state.models.length)
      + hint + "</div>";
  }

  function settingsHTML() {
    const s = state.settings;
    const precision = s.precision || "—";
    return '<div class="settings-line">'
      + statHTML("Chunk size", s.chunk_size != null ? fmt.int(s.chunk_size) : null)
      + statHTML("Overlap", s.chunk_overlap != null ? fmt.int(s.chunk_overlap) : null)
      + statHTML("Precision", precision)
      + statHTML("Device", DEVICE_LABELS[s.device] || s.device)
      + statHTML("Pipeline", s.preset ? s.preset[0].toUpperCase() + s.preset.slice(1) : null)
      + '<button type="button" class="link settings-link" data-tab="Settings">Change on the Settings tab</button>'
      + "</div>";
  }

  function extrasHTML() {
    const parts = [];
    const pdfs = (state.files.summary || {}).pdf || 0;
    if (pdfs) {
      const on = state.check_pdfs;
      parts.push('<div class="option-row">'
        + `<button type="button" class="switch" role="switch" aria-checked="${on}" data-switch="check_pdfs"${busy() ? " disabled" : ""}><span class="track"></span>`
        + `<span>Check the ${fmt.int(pdfs)} PDF${pdfs === 1 ? "" : "s"} for missing text first</span></button>`
        + '<span class="option-note">Strongly recommended. Scanned PDFs without a text layer would add nothing to the database.</span></div>');
    }
    const images = (state.files.summary || {}).image || 0;
    if (images && state.settings.vision) {
      const kind = state.settings.vision_needs_gpu ? "warn" : "info";
      const text = state.settings.vision_needs_gpu
        ? `The ${fmt.int(images)} image${images === 1 ? "" : "s"} need a vision model that runs on this computer; ${esc(state.settings.vision)} requires a supported NVIDIA GPU.`
        : `The ${fmt.int(images)} image${images === 1 ? "" : "s"} will be described by <b>${esc(state.settings.vision)}</b>.`;
      parts.push(`<div class="status ${kind}">${kind === "warn" ? ICONS.warn : ICONS.info}<div class="status-body">${text} <button type="button" class="link" data-tab="Settings">Choose a vision model</button></div></div>`);
    }
    if (state.settings.cpu_warning) {
      parts.push(`<div class="status warn">${ICONS.warn}<div class="status-body">A GPU is available, but databases are set to be created on the CPU, which is much slower. <button type="button" class="link" data-tab="Settings">Change the device</button></div></div>`);
    }
    return parts.join("");
  }

  function captureFocus() {
    const el = document.activeElement;
    if (el && el.id === "db-name") return { start: el.selectionStart, end: el.selectionEnd };
    return null;
  }

  function renderForm() {
    const signature = JSON.stringify([state.models, state.model, state.settings, state.check_pdfs, state.files.summary,
      state.name_limit, state.existing, busy()]);
    if (signature === formSignature) return;
    formSignature = signature;
    const focus = captureFocus();
    $("#create-form").innerHTML = `<h2 class="section-title">${ICONS.database}<span>New Database</span></h2>`
      + `<div class="form-grid">${nameFieldHTML()}${modelFieldHTML()}</div>`
      + settingsHTML()
      + extrasHTML();
    if (focus) {
      const input = $("#db-name");
      if (input && !input.disabled) {
        input.focus();
        input.setSelectionRange(focus.start, focus.end);
      }
    }
  }

  function updateNameMessage() {
    const problem = busy() ? null : nameProblem(nameDraft);
    const label = $("#db-name").closest(".input");
    label.classList.toggle("invalid", Boolean(problem));
    const message = $("#name-message");
    message.className = problem ? "field-error" : "field-hint";
    message.textContent = problem || "Lowercase letters, numbers, hyphens and underscores";
  }

  function stepperHTML(progress) {
    return '<div class="stepper">' + progress.stages.map((s) => {
      const icon = s.status === "done" ? ICONS.check : s.status === "running" ? '<span class="spinner"></span>' : '<span class="dot"></span>';
      return `<div class="step ${s.status}">${icon}<span>${esc(s.label)}</span></div>`;
    }).join("") + "</div>";
  }

  function detailText(p) {
    if (!p) return "Starting…";
    const docs = p.documents != null ? `${fmt.int(p.documents)} document${p.documents === 1 ? "" : "s"}` : "";
    const chunks = p.chunks != null ? `${fmt.int(p.chunks)} chunks` : "";
    switch (p.stage) {
      case "extract": return docs ? `Read ${docs}` : "Reading the files…";
      case "media": return "Describing images and adding transcripts…";
      case "split": return chunks ? `Split into ${chunks}` : "Splitting the text into chunks…";
      case "tokenize": return `Tokenizing ${chunks || "the chunks"}…`;
      case "embed": return p.batches ? `Embedding ${chunks || "chunks"} · batch ${fmt.int(p.batch || 0)} of ${fmt.int(p.batches)}` : `Embedding ${chunks || "the chunks"}…`;
      case "write": return "Saving the database…";
      case "done": return "Finishing up…";
      default: return "Starting…";
    }
  }

  function logHTML() {
    if (!logOpen) return "";
    const text = logLines.length ? logLines.map(esc).join("\n") : "Waiting for output…";
    return `<pre class="build-log" id="build-log">${text}</pre>`;
  }

  function buildingHTML(b) {
    const p = b.progress;
    const determinate = p && p.stage === "embed" && p.batches;
    const width = determinate ? Math.min(100, ((p.batch || 0) / p.batches) * 100) : 0;
    const bar = determinate
      ? `<div class="progress"><i style="width:${width.toFixed(1)}%"></i></div>`
      : '<div class="progress indeterminate"><i></i></div>';
    const cancel = b.cancelling
      ? '<span class="status-note">Cancelling…</span>'
      : `<button type="button" class="btn ghost small" data-act="cancel_build">${ICONS.stop}Cancel</button>`;
    return '<div class="build-panel">'
      + `<div class="build-head"><span class="build-title">Creating <b>${esc(b.name)}</b> with ${esc(b.model || "")}</span>`
      + `<span class="elapsed" data-since="${b.started}">${VDB.elapsed(Date.now() / 1000 - b.started)}</span></div>`
      + (p ? stepperHTML(p) : "")
      + bar
      + `<div class="build-detail"><span>${esc(detailText(p))}</span>`
      + `<button type="button" class="link" data-act="toggle_log">${logOpen ? "Hide log" : "Show log"}</button>${cancel}</div>`
      + (p && p.line ? `<div class="build-line">${esc(p.line)}</div>` : "")
      + logHTML()
      + "</div>";
  }

  function scanningHTML(b) {
    const s = b.scan || { done: 0, total: 0 };
    const width = s.total ? (s.done / s.total) * 100 : 0;
    const cancel = b.cancelling
      ? '<span class="status-note">Cancelling…</span>'
      : `<button type="button" class="btn ghost small" data-act="cancel_build">${ICONS.stop}Cancel</button>`;
    return '<div class="build-panel">'
      + `<div class="build-head"><span class="build-title">Checking PDFs for missing text before creating <b>${esc(b.name)}</b></span>`
      + `<span class="elapsed" data-since="${b.started}">${VDB.elapsed(Date.now() / 1000 - b.started)}</span></div>`
      + `<div class="progress${s.total ? "" : " indeterminate"}"><i style="width:${width.toFixed(1)}%"></i></div>`
      + `<div class="build-detail"><span>${s.total ? `${fmt.int(s.done)} of ${fmt.int(s.total)} PDFs checked` : "Looking for PDFs…"}</span>${cancel}</div>`
      + "</div>";
  }

  function resultHTML(r) {
    const close = `<button type="button" class="icon-btn btn ghost small dismiss" data-act="dismiss_result" data-tip-text="Dismiss">${ICONS.close}</button>`;
    if (r.kind === "ok") {
      const parts = [];
      if (r.documents != null) parts.push(`${fmt.int(r.documents)} document${r.documents === 1 ? "" : "s"}`);
      if (r.chunks != null) parts.push(`${fmt.int(r.chunks)} chunks`);
      const from = parts.length ? ` from ${parts.join(", ")}` : "";
      let notes = "";
      if (r.backup === "running") notes += '<div class="status-note"><span class="spinner inline"></span>Backing up the new database…</div>';
      else if (r.backup === "failed") notes += `<div class="status-note warn">The backup copy could not be made: ${esc(r.backup_error || "unknown error")}</div>`;
      if (r.not_added && r.not_added.length) {
        notes += `<details class="notice-details" open><summary>${fmt.int(r.not_added.length)} file${r.not_added.length === 1 ? " was" : "s were"} not fully added</summary>`
          + `<div class="notice-list">${r.not_added.slice(0, 200).map((e) => `<div>${esc(e)}</div>`).join("")}</div></details>`;
      }
      const kind = r.not_added && r.not_added.length ? "warn" : "ok";
      return `<div class="status ${kind} result">${kind === "ok" ? ICONS.check : ICONS.warn}<div class="status-body">`
        + `Created <b>${esc(r.name)}</b> in ${VDB.elapsed(r.seconds)}${from}. `
        + '<button type="button" class="link" data-tab="Query Database">Query it on the Query Database tab</button>'
        + (state.files.count === 0 ? '<div class="status-note">The file list was cleared, ready for your next database.</div>' : "")
        + `${notes}</div>${close}</div>`;
    }
    if (r.kind === "ocr") {
      const names = r.names.slice(0, 8).map((n) => `<div>${esc(n)}</div>`).join("") + (r.names.length > 8 ? `<div>…and ${fmt.int(r.names.length - 8)} more</div>` : "");
      return `<div class="status warn result">${ICONS.warn}<div class="status-body">`
        + `${fmt.int(r.names.length)} PDF${r.names.length === 1 ? " has" : "s have"} no text layer and need${r.names.length === 1 ? "s" : ""} OCR first, so no database was created.`
        + `<div class="notice-list compact">${names}</div>`
        + '<div class="result-actions">'
        + `<button type="button" class="btn ghost small" data-act="open_report">${ICONS.file}Open the full list</button>`
        + `<button type="button" class="btn ghost small" data-act="remove_ocr_files">${ICONS.close}Remove them from the list</button>`
        + '<span class="status-note">or run OCR on them on the <button type="button" class="link" data-tab="Tools">Tools tab</button></span>'
        + `</div></div>${close}</div>`;
    }
    if (r.kind === "cancelled" || r.kind === "info") {
      return `<div class="status info result">${ICONS.info}<div class="status-body">${esc(r.message)}</div>${close}</div>`;
    }
    const log = r.log ? ` <button type="button" class="link" data-act="toggle_log">${logOpen ? "Hide log" : "Show log"}</button>` : "";
    return `<div class="status error result">${ICONS.error}<div class="status-body">${esc(r.message)}${log}${r.log ? logHTML() : ""}</div>${close}</div>`;
  }

  function idleHTML() {
    const reason = blocker();
    const ready = !reason;
    const files = state.files.count;
    const message = ready
      ? `Ready to create <b>${esc(nameDraft)}</b> from ${fmt.int(files)} file${files === 1 ? "" : "s"}.`
      : esc(reason);
    return '<div class="create-row">'
      + `<button type="button" class="btn primary create-btn" data-act="create"${ready ? "" : " disabled"}>${ICONS.build}Create Database</button>`
      + `<span class="create-reason${ready ? " ready" : ""}">${message}</span></div>`;
  }

  function renderAction() {
    const b = state.build;
    const signature = JSON.stringify([b, blocker(), nameDraft, logOpen, logOpen ? logLines.length : 0, logOpen ? logLines[logLines.length - 1] : ""]);
    if (signature === actionSignature) return;
    actionSignature = signature;
    const keepScroll = $("#build-log");
    const atBottom = keepScroll ? keepScroll.scrollTop + keepScroll.clientHeight >= keepScroll.scrollHeight - 4 : true;
    let html = "";
    if (b.phase === "scanning") html = scanningHTML(b);
    else if (b.phase === "building") html = buildingHTML(b);
    else {
      if (b.result) html += resultHTML(b.result);
      if (b.phase === "saving" && !(b.result && b.result.kind === "ok")) {
        html += `<div class="status"><span class="spinner"></span><div class="status-body">Backing up the new database…</div></div>`;
      }
      html += idleHTML();
    }
    $("#create-action").innerHTML = html;
    const log = $("#build-log");
    if (log && atBottom) log.scrollTop = log.scrollHeight;
  }

  function pollLog() {
    clearTimeout(logTimer);
    if (!logOpen) return;
    VDB.call("log").then((result) => {
      logLines = result.lines || [];
      renderAction();
      if (logOpen && state.build.phase === "building") logTimer = setTimeout(pollLog, 1000);
    });
  }

  function render() {
    CreateFiles.update(state.files, state.kinds, busy());
    renderForm();
    renderAction();
    const result = state.build.result;
    const key = result ? JSON.stringify([result.kind, result.name]) : "";
    if (key !== lastResult) {
      lastResult = key;
      if (result && result.kind === "ok") {
        nameDraft = "";
        formSignature = "";
        renderForm();
        renderAction();
      }
    }
  }

  function create() {
    if (blocker()) return;
    VDB.tooltip.hide();
    logOpen = false;
    logLines = [];
    VDB.call("create", { name: nameDraft });
  }

  function openModelSelect() {
    VDB.openSelect("create.model", {
      title: "Embedding model",
      options: state.models.map((m) => ({
        value: m.path,
        label: m.name,
        note: m.known ? `${m.vendor} · ${fmt.int(m.dimensions)} dims` : "custom",
      })),
      value: state.model,
      onPick: (path) => {
        if (path !== state.model) VDB.call("select_model", { path });
      },
    });
  }

  function bind() {
    document.addEventListener("click", (e) => {
      const tab = e.target.closest("[data-tab]");
      if (tab) {
        VDB.call("open_tab", { name: tab.dataset.tab });
        return;
      }
      const select = e.target.closest('[data-select="create.model"]');
      if (select && !select.disabled) {
        openModelSelect();
        return;
      }
      const toggle = e.target.closest('[data-switch="check_pdfs"]');
      if (toggle && !toggle.disabled) {
        VDB.call("set_check_pdfs", { value: toggle.getAttribute("aria-checked") !== "true" });
        return;
      }
      const act = e.target.closest("[data-act]");
      if (!act || act.disabled) return;
      const action = act.dataset.act;
      if (action === "create") create();
      else if (action === "toggle_log") {
        logOpen = !logOpen;
        if (logOpen) pollLog();
        else clearTimeout(logTimer);
        actionSignature = "";
        renderAction();
      } else if (!LOCAL_ACTIONS.has(action)) {
        VDB.call(action);
      }
    });
    document.addEventListener("input", (e) => {
      if (e.target.id !== "db-name") return;
      const input = e.target;
      const caret = input.selectionStart;
      const raw = input.value;
      const tidy = (text) => text.toLowerCase().replace(/\s/g, "_").replace(/[^a-z0-9_-]/g, "");
      const clean = tidy(raw).slice(0, state.name_limit);
      if (clean !== raw) {
        const position = Math.min(clean.length, tidy(raw.slice(0, caret)).length);
        input.value = clean;
        input.setSelectionRange(position, position);
      }
      nameDraft = clean;
      updateNameMessage();
      renderAction();
    });
    document.addEventListener("keydown", (e) => {
      if (e.target.id === "db-name" && e.key === "Enter") {
        e.preventDefault();
        create();
      }
    });
  }

  CreateFiles.init();
  bind();

  window.TabApp = {
    init(payload) {
      state = payload;
      render();
    },
    setState(payload) {
      const wasBuilding = state && state.build.phase === "building";
      state = payload;
      render();
      if (logOpen && !wasBuilding && state.build.phase === "building") pollLog();
      if (logOpen && wasBuilding && state.build.phase !== "building") pollLog();
    },
  };
})();
