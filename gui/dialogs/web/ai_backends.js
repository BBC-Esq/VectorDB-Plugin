"use strict";

(function () {
  const { ICONS, esc, $ } = VDB;

  const SAVED_MS = 1800;
  const CONFIRM_MS = 3500;
  const ORDER = ["chatgpt", "lmstudio", "minimax", "kobold"];
  const NAMES = { chatgpt: "ChatGPT", lmstudio: "LM Studio", minimax: "MiniMax", kobold: "Kobold" };
  const ONLINE = new Set(["chatgpt", "minimax"]);
  const TESTABLE = new Set(["chatgpt", "lmstudio", "kobold"]);
  const KEY_BACKEND = { "openai.api_key": "chatgpt", "minimax.api_key": "minimax" };
  const LEVELS = { none: "None", low: "Low", medium: "Medium", high: "High", xhigh: "Extra high" };
  const PORT_RE = /:(\d{1,5})(?=\/)/;
  const FOCUS_ATTRS = ["key", "value", "select", "backend", "reveal", "remove", "check"];

  const KEY_ICON = '<svg viewBox="0 0 16 16"><circle cx="5.3" cy="10.7" r="2.8" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="m7.3 8.7 5.9-5.9M11.3 4.7l1.7 1.7M9.6 6.4l1.3 1.3" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
  const EYE_OFF = '<svg viewBox="0 0 16 16"><path d="M1.6 8s2.4-4.6 6.4-4.6S14.4 8 14.4 8s-2.4 4.6-6.4 4.6S1.6 8 1.6 8Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><circle cx="8" cy="8" r="2.1" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="m2.6 2.6 10.8 10.8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
  const PLUG = '<svg viewBox="0 0 16 16"><path d="M5.9 1.8v3M10.1 1.8v3M4.1 4.8h7.8v2.4a3.9 3.9 0 0 1-7.8 0V4.8ZM8 11.1v3.1" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  let state = null;
  let selected = "chatgpt";
  let queue = Promise.resolve();
  let fadeTimer = 0;
  let closing = false;
  const drafts = {};
  const errors = {};
  const savedAt = {};
  const revealed = {};
  const shown = new Set();
  const confirmRemove = {};

  function hostOf(url) {
    try {
      return new URL(url).host || url;
    } catch (e) {
      return url;
    }
  }

  function navStatus(backend) {
    const data = state[backend];
    if (ONLINE.has(backend)) return data.key_set ? { text: "API key saved", kind: "ok" } : { text: "Needs an API key", kind: "warn" };
    if (backend === "lmstudio") {
      if (data.missing) return { text: "Needs a port", kind: "warn" };
      if (data.malformed) return { text: "Address needs fixing", kind: "warn" };
      return { text: hostOf(data.connection), kind: "" };
    }
    return { text: hostOf(data.address), kind: "" };
  }

  function renderNav() {
    $("#nav").innerHTML = ORDER.map((backend) => {
      const status = navStatus(backend);
      const on = backend === selected;
      return `<button type="button" class="nav-item${on ? " on" : ""}" role="tab" aria-selected="${on}" data-backend="${backend}">`
        + `<span class="nav-icon">${ONLINE.has(backend) ? ICONS.globe : ICONS.chip}</span>`
        + `<span class="nav-text"><span class="nav-name">${NAMES[backend]}</span>`
        + `<span class="nav-status${status.kind ? ` ${status.kind}` : ""}">${esc(status.text)}</span></span></button>`;
    }).join("");
  }

  function labelHTML(key, label) {
    const info = state.help[key] ? `<span class="info" data-tip="help" data-help="${esc(key)}">${ICONS.info}</span>` : "";
    const recent = savedAt[key] && Date.now() - savedAt[key] < SAVED_MS;
    return `<div class="field-label"><span>${esc(label)}</span>${info}<span class="saved${recent ? " show" : ""}" data-saved="${esc(key)}">${ICONS.check}Saved</span></div>`;
  }

  function link(url, text) {
    return `<a href="${esc(url)}">${esc(text)}</a>`;
  }

  function previewAddress(text) {
    const lm = state.lmstudio;
    const base = lm.connection || lm.default_connection;
    return /^\d{1,5}$/.test(text) && PORT_RE.test(base) ? base.replace(PORT_RE, `:${Number(text)}`) : base;
  }

  function hintFor(key) {
    if (key === "openai.api_key") return { html: `Get a key at ${link(state.chatgpt.keys_url, "platform.openai.com/api-keys")}` };
    if (key === "minimax.api_key") return { html: `Get a key at ${link(state.minimax.keys_url, "platform.minimax.io")}` };
    if (key === "server.port") {
      const lm = state.lmstudio;
      if (lm.malformed) return { html: esc(`The saved address, ${lm.connection}, has no port to change. Fix it in config.yaml.`), warn: true };
      if (lm.missing && !(key in drafts)) return { html: esc(`Not set yet. LM Studio uses ${lm.default_port} unless you changed it.`), warn: true };
      const text = key in drafts ? drafts[key].trim() : lm.port;
      return { html: `Address: ${esc(previewAddress(text))}` };
    }
    return { html: "" };
  }

  function messageHTML(key) {
    if (errors[key]) return `<div class="field-error">${esc(errors[key])}</div>`;
    const hint = hintFor(key);
    return `<div class="field-hint${hint.warn ? " warn" : ""}">${hint.html}</div>`;
  }

  function field(key, label, control, span) {
    return `<div class="field span-${span}" data-field="${esc(key)}">${labelHTML(key, label)}${control}${messageHTML(key)}</div>`;
  }

  function segmentedHTML(key, options, current) {
    const buttons = options.map((value) => {
      const on = value === current;
      return `<button type="button" class="${on ? "on" : ""}" role="radio" aria-checked="${on}" data-key="${esc(key)}" data-value="${esc(value)}">${esc(LEVELS[value] || value)}</button>`;
    }).join("");
    return `<div class="seg full" role="radiogroup">${buttons}</div>`;
  }

  function switchHTML(key, on) {
    return `<button type="button" class="switch" role="switch" aria-checked="${Boolean(on)}" data-key="${esc(key)}"><span class="track"></span><span>${on ? "On" : "Off"}</span></button>`;
  }

  function secretHTML(key, data, placeholder) {
    const visible = shown.has(key);
    const text = key in drafts ? drafts[key] : "";
    const hint = data.key_set ? `•••••••••••••••••••• ${data.key_hint}`.trim() : placeholder;
    const confirming = confirmRemove[key] && Date.now() - confirmRemove[key] < CONFIRM_MS;
    let remove = "";
    if (data.key_set) {
      remove = confirming
        ? `<button type="button" class="btn danger" data-remove="${esc(key)}">Click again to remove</button>`
        : `<button type="button" class="btn ghost" data-remove="${esc(key)}" data-tip-text="Delete the saved key from config.yaml">Remove</button>`;
    }
    const tip = visible ? "Hide the key" : (data.key_set || text ? "Show the key" : "Show what you type");
    return `<div class="key-row"><label class="input secret${errors[key] ? " invalid" : ""}">${KEY_ICON}`
      + `<input type="${visible ? "text" : "password"}" data-key="${esc(key)}" data-kind="secret" value="${esc(text)}" placeholder="${esc(hint)}" autocomplete="off" spellcheck="false">`
      + `<button type="button" class="reveal" data-reveal="${esc(key)}" data-tip-text="${tip}">${visible ? EYE_OFF : ICONS.eye}</button></label>${remove}</div>`;
  }

  function portHTML() {
    const key = "server.port";
    const lm = state.lmstudio;
    const text = key in drafts ? drafts[key] : lm.port;
    return `<label class="input${errors[key] ? " invalid" : ""}"><input type="text" inputmode="numeric" data-key="${key}" data-kind="port" value="${esc(text)}" placeholder="${esc(lm.default_port)}" autocomplete="off" spellcheck="false"></label>`;
  }

  function price(value) {
    if (!(value > 0)) return '<div class="price-value none">—</div>';
    return `<div class="price-value">$${value >= 0.1 ? value.toFixed(2) : value.toFixed(3)}</div>`;
  }

  function pricesHTML(pricing) {
    const cell = (label, value) => `<div class="price"><div class="price-label">${label}</div>${price(value)}</div>`;
    return `<div class="prices">${cell("Input", pricing.input)}${cell("Cached input", pricing.cached)}${cell("Output", pricing.output)}</div>`;
  }

  function testButton(backend, blocked) {
    const test = state[backend].test;
    const running = Boolean(test && test.status === "running");
    const tip = blocked ? ` data-tip-text="${esc(blocked)}"` : "";
    const inner = running ? '<span class="spinner muted"></span>Testing…' : `${PLUG}Test connection`;
    return `<span class="panel-action"${tip}><button type="button" class="btn ghost small" data-check="${backend}"${running || blocked ? " disabled" : ""}>${inner}</button></span>`;
  }

  function resultHTML(backend) {
    const test = state[backend].test;
    if (!test) return "";
    if (test.status === "running") return '<div class="result status info"><span class="spinner muted"></span><div>Testing the connection…</div></div>';
    const icon = test.status === "ok" ? ICONS.check : test.status === "warn" ? ICONS.warn : ICONS.error;
    return `<div class="result status ${test.status}">${icon}<div>${esc(test.message)}</div></div>`;
  }

  function panelHTML(backend, summary, fields, action = "") {
    const icon = ONLINE.has(backend) ? ICONS.globe : ICONS.chip;
    return `<section class="card panel" data-panel="${backend}"><div class="panel-head"><div class="panel-heading">`
      + `<h2 class="panel-title">${icon}${NAMES[backend]}</h2><p class="panel-sub">${esc(summary)}</p></div>${action}</div>`
      + `${TESTABLE.has(backend) ? resultHTML(backend) : ""}<div class="fields">${fields.join("")}</div></section>`;
  }

  function chatgptPanel() {
    const data = state.chatgpt;
    const model = data.models.find((m) => m.value === data.model) || data.models[0];
    const fields = [
      field("openai.api_key", "API key", secretHTML("openai.api_key", data, "Paste your OpenAI API key (sk-...)"), 4),
      field("openai.model", "Model", VDB.selectButton("openai.model", { label: model.label, note: model.note }), data.show_verbosity ? 2 : 4),
    ];
    if (data.show_verbosity) fields.push(field("openai.verbosity", "Verbosity", segmentedHTML("openai.verbosity", data.verbosities, data.verbosity), 2));
    if (data.show_reasoning) fields.push(field("openai.reasoning_effort", "Reasoning effort", segmentedHTML("openai.reasoning_effort", data.reasonings, data.reasoning), 4));
    fields.push(`<div class="field span-4">${labelHTML("openai.pricing", "Price per million tokens")}${pricesHTML(data.pricing)}</div>`);
    return panelHTML(
      "chatgpt",
      "OpenAI's models answer over the internet, and OpenAI charges your account for each answer at the prices below.",
      fields,
      testButton("chatgpt", data.key_set ? "" : "Enter an API key first"),
    );
  }

  function lmstudioPanel() {
    const data = state.lmstudio;
    const fields = [
      field("server.port", "Port", portHTML(), 2),
      field("server.show_thinking", "Show thinking", switchHTML("server.show_thinking", data.show_thinking), 2),
    ];
    const blocked = data.missing ? "Enter the port first" : data.malformed ? "Fix the saved address first" : "";
    return panelHTML(
      "lmstudio",
      "Answers come from the model loaded in LM Studio, through its local server on this computer.",
      fields,
      testButton("lmstudio", blocked),
    );
  }

  function minimaxPanel() {
    const data = state.minimax;
    const models = data.models.map((m) => `<code>${esc(m)}</code>`).join("");
    const fields = [
      field("minimax.api_key", "API key", secretHTML("minimax.api_key", data, "Paste your MiniMax API key"), 4),
      `<div class="field span-4">${labelHTML("minimax.models", "Models")}<div class="stat-value">${models}</div>`
        + '<div class="field-hint">Choose one in the Query Database tab\'s backend menu.</div></div>',
    ];
    return panelHTML("minimax", "MiniMax's models answer over the internet, and MiniMax charges your account for each answer.", fields);
  }

  function koboldPanel() {
    const data = state.kobold;
    const fields = [
      `<div class="field span-4">${labelHTML("kobold.address", "Address")}<div class="stat-value"><code>${esc(data.address)}</code></div>`
        + '<div class="field-hint">KoboldCpp\'s standard port. Start KoboldCpp and load a model before you ask a question.</div></div>',
    ];
    return panelHTML(
      "kobold",
      "Answers come from the model loaded in KoboldCpp, through its local server on this computer.",
      fields,
      testButton("kobold", ""),
    );
  }

  const PANELS = { chatgpt: chatgptPanel, lmstudio: lmstudioPanel, minimax: minimaxPanel, kobold: koboldPanel };

  function attrName(name) {
    return name.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);
  }

  function captureFocus() {
    const active = document.activeElement;
    if (!active || !active.dataset) return null;
    const attrs = FOCUS_ATTRS.filter((name) => active.dataset[name] != null).map((name) => [name, active.dataset[name]]);
    if (!attrs.length) return null;
    return { tag: active.tagName.toLowerCase(), attrs, input: active.tagName === "INPUT", start: active.selectionStart, end: active.selectionEnd };
  }

  function restoreFocus(focus) {
    if (!focus) return;
    const selector = focus.tag + focus.attrs.map(([name, value]) => `[data-${attrName(name)}="${CSS.escape(value)}"]`).join("");
    const element = document.querySelector(selector);
    if (!element || element.disabled) return;
    element.focus({ preventScroll: true });
    if (focus.input && focus.start != null) {
      try {
        element.setSelectionRange(focus.start, focus.end);
      } catch (e) {
        return;
      }
    }
  }

  function focusInput(key) {
    const input = document.querySelector(`input[data-key="${CSS.escape(key)}"]`);
    if (!input) return;
    input.focus({ preventScroll: true });
    const end = input.value.length;
    try {
      input.setSelectionRange(end, end);
    } catch (e) {
      return;
    }
  }

  function scheduleFade() {
    clearTimeout(fadeTimer);
    const now = Date.now();
    const pending = Object.values(savedAt).filter((t) => now - t < SAVED_MS);
    if (!pending.length) return;
    fadeTimer = setTimeout(() => {
      for (const [key, time] of Object.entries(savedAt)) {
        if (Date.now() - time >= SAVED_MS) {
          const badge = document.querySelector(`[data-saved="${CSS.escape(key)}"]`);
          if (badge) badge.classList.remove("show");
        }
      }
      scheduleFade();
    }, SAVED_MS - (now - Math.min(...pending)) + 20);
  }

  function render() {
    const focus = captureFocus();
    renderNav();
    const notice = state.error ? `<div class="notice error">${ICONS.error}<div>${esc(state.error)}</div></div>` : "";
    $("#main").innerHTML = notice + PANELS[selected]();
    restoreFocus(focus);
    scheduleFade();
  }

  function updateMessage(key) {
    const element = document.querySelector(`[data-field="${CSS.escape(key)}"]`);
    if (!element) return;
    const input = element.querySelector(".input");
    if (input) input.classList.toggle("invalid", Boolean(errors[key]));
    const message = element.querySelector(".field-hint, .field-error");
    if (message) message.outerHTML = messageHTML(key);
  }

  function forgetSecret(key) {
    delete drafts[key];
    delete revealed[key];
    shown.delete(key);
  }

  function commit(key, value) {
    VDB.tooltip.hide();
    queue = queue.then(() => VDB.call("apply", { key, value })).then((result) => {
      if (result.state) state = result.state;
      if (result.ok) {
        delete errors[key];
        if (KEY_BACKEND[key]) forgetSecret(key);
        else delete drafts[key];
        if (result.changed) savedAt[key] = Date.now();
      } else {
        errors[key] = result.error || "The setting could not be saved.";
      }
      render();
    });
    return queue;
  }

  function portError(text) {
    if (!/^\d{1,5}$/.test(text) || Number(text) < 1 || Number(text) > 65535) return "Port must be a number between 1 and 65535.";
    return "";
  }

  function commitInput(input) {
    const key = input.dataset.key;
    if (!(key in drafts)) return;
    const text = drafts[key].trim();
    if (input.dataset.kind === "port") {
      if (text === "") {
        revert(key);
        return;
      }
      const error = portError(text);
      if (error) {
        errors[key] = error;
        updateMessage(key);
        return;
      }
      if (!state.lmstudio.missing && Number(text) === Number(state.lmstudio.port)) {
        revert(key);
        return;
      }
      commit(key, text);
      return;
    }
    if (key in revealed && text === revealed[key]) return;
    if (text === "") {
      revert(key);
      return;
    }
    commit(key, text);
  }

  function revert(key) {
    if (KEY_BACKEND[key]) forgetSecret(key);
    else delete drafts[key];
    delete errors[key];
    render();
  }

  function onInput(input) {
    const key = input.dataset.key;
    drafts[key] = input.value;
    if (input.dataset.kind === "port") {
      const text = input.value.trim();
      const error = text === "" ? "" : portError(text);
      if (error) errors[key] = error;
      else delete errors[key];
    } else {
      delete errors[key];
    }
    updateMessage(key);
  }

  function step(input, direction) {
    const current = Number(input.value.trim() || state.lmstudio.port || state.lmstudio.default_port);
    const next = Math.min(65535, Math.max(1, (Number.isFinite(current) ? current : 1234) + direction));
    input.value = String(next);
    onInput(input);
  }

  async function reveal(key) {
    if (shown.has(key)) {
      shown.delete(key);
      if (key in revealed && drafts[key] === revealed[key]) {
        delete drafts[key];
        delete revealed[key];
      }
      render();
      focusInput(key);
      return;
    }
    const backend = KEY_BACKEND[key];
    if (!(key in drafts) && state[backend].key_set) {
      const result = await VDB.call("reveal", { backend });
      if (result.error) {
        errors[key] = result.error;
        render();
        return;
      }
      revealed[key] = result.key || "";
      drafts[key] = revealed[key];
    }
    shown.add(key);
    render();
    focusInput(key);
  }

  function removeKey(key) {
    const now = Date.now();
    if (!(confirmRemove[key] && now - confirmRemove[key] < CONFIRM_MS)) {
      confirmRemove[key] = now;
      render();
      setTimeout(() => {
        if (confirmRemove[key] === now) {
          delete confirmRemove[key];
          render();
        }
      }, CONFIRM_MS);
      return;
    }
    delete confirmRemove[key];
    forgetSecret(key);
    commit(key, "");
  }

  function runCheck(backend) {
    VDB.tooltip.hide();
    queue = queue.then(() => VDB.call("check", { backend })).then((result) => {
      if (result.state) state = result.state;
      render();
    });
  }

  function select(backend) {
    if (!ORDER.includes(backend) || backend === selected) return;
    const active = document.activeElement;
    if (active && active.matches && active.matches("input[data-key]")) commitInput(active);
    selected = backend;
    VDB.popover.close();
    VDB.tooltip.hide();
    render();
    VDB.call("select", { backend });
  }

  function openModelSelect() {
    const data = state.chatgpt;
    VDB.openSelect("openai.model", {
      title: "OpenAI model",
      options: data.models.map((m) => ({ value: m.value, label: m.label, note: m.note })),
      value: data.model,
      onPick: (value) => {
        if (value !== state.chatgpt.model) commit("openai.model", value);
      },
    });
  }

  function segmentValue(key) {
    if (key === "openai.verbosity") return state.chatgpt.verbosity;
    if (key === "openai.reasoning_effort") return state.chatgpt.reasoning;
    return null;
  }

  function requestClose() {
    if (closing) return;
    closing = true;
    const active = document.activeElement;
    if (active && active.matches && active.matches("input[data-key]")) commitInput(active);
    queue.then(() => VDB.call("close"));
  }

  function tipContent(element) {
    if (element.dataset.tip !== "help") return "";
    return esc(state.help[element.dataset.help] || "");
  }

  function bindEvents() {
    const nav = $("#nav");
    nav.addEventListener("click", (e) => {
      const item = e.target.closest("[data-backend]");
      if (item) select(item.dataset.backend);
    });
    nav.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      e.preventDefault();
      const index = ORDER.indexOf(selected);
      const next = ORDER[(index + (e.key === "ArrowDown" ? 1 : -1) + ORDER.length) % ORDER.length];
      select(next);
      const item = document.querySelector(`[data-backend="${next}"]`);
      if (item) item.focus();
    });
    const main = $("#main");
    main.addEventListener("click", (e) => {
      const segment = e.target.closest(".seg button[data-key]");
      if (segment) {
        if (segment.dataset.value !== segmentValue(segment.dataset.key)) commit(segment.dataset.key, segment.dataset.value);
        return;
      }
      const toggle = e.target.closest(".switch[data-key]");
      if (toggle) {
        commit(toggle.dataset.key, toggle.getAttribute("aria-checked") !== "true");
        return;
      }
      if (e.target.closest(".select[data-select]")) {
        openModelSelect();
        return;
      }
      const eye = e.target.closest("[data-reveal]");
      if (eye) {
        e.preventDefault();
        reveal(eye.dataset.reveal);
        return;
      }
      const remove = e.target.closest("[data-remove]");
      if (remove) {
        removeKey(remove.dataset.remove);
        return;
      }
      const check = e.target.closest("[data-check]");
      if (check && !check.disabled) runCheck(check.dataset.check);
    });
    main.addEventListener("input", (e) => {
      const input = e.target.closest("input[data-key]");
      if (input) onInput(input);
    });
    main.addEventListener("keydown", (e) => {
      const input = e.target.closest("input[data-key]");
      if (!input) return;
      if (e.key === "Enter") {
        e.preventDefault();
        commitInput(input);
      } else if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        revert(input.dataset.key);
      } else if ((e.key === "ArrowUp" || e.key === "ArrowDown") && input.dataset.kind === "port") {
        e.preventDefault();
        step(input, e.key === "ArrowUp" ? 1 : -1);
      }
    });
    main.addEventListener("focusout", (e) => {
      const input = e.target.closest("input[data-key]");
      if (!input || !input.isConnected) return;
      const next = e.relatedTarget;
      if (next && next.dataset && next.dataset.reveal === input.dataset.key) return;
      commitInput(input);
    });
    $("#close-btn").addEventListener("click", requestClose);
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape" || e.defaultPrevented || VDB.popover.key || VDB.overlay.isOpen()) return;
      e.preventDefault();
      requestClose();
    });
  }

  VDB.tooltip.provider = tipContent;
  bindEvents();

  window.TabApp = {
    init(payload) {
      state = payload;
      selected = payload.selected;
      render();
    },
    setState(payload) {
      state = payload;
      VDB.tooltip.hide();
      render();
      VDB.popover.close();
    },
    requestClose,
  };
})();
