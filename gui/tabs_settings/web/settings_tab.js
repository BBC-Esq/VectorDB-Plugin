"use strict";

(function () {
  const { ICONS, esc, fmt, $ } = VDB;

  const SAVED_MS = 1800;
  const TWO_COLUMN_TTS = new Set(["bark", "whisperspeech", "kokoro", "kyutaipocket"]);

  let state = null;
  const errors = {};
  const drafts = {};
  const savedAt = {};
  let queue = Promise.resolve();
  let fadeTimer = 0;

  function currentBackend() {
    return state.tts.backends.find((b) => b.key === state.tts.current) || state.tts.backends[0];
  }

  function currentVision() {
    return state.vision.models.find((m) => m.name === state.vision.current) || state.vision.models[0];
  }

  function vramText(model) {
    return String(model.vram || "").replace(/gb/i, "GB");
  }

  function fieldDef(key) {
    for (const section of state.sections) {
      for (const field of section.fields || []) {
        if (field.key === key) return field;
      }
    }
    return { key };
  }

  function helpFor(key) {
    return state.help[key] || "";
  }

  function parseNumber(field, text) {
    const clean = String(text ?? "").trim().replace(/[,\s_]/g, "");
    if (clean === "") return { empty: true };
    if (field.decimals === 0 ? !/^-?\d+$/.test(clean) : !/^-?(\d+\.?\d*|\.\d+)$/.test(clean)) {
      return { error: field.error };
    }
    const value = Number(clean);
    if (value < field.min || value > field.max) return { error: field.error };
    return { value };
  }

  function currentNumber(key) {
    const text = key in drafts ? drafts[key] : state.values[key];
    return parseNumber(fieldDef(key), text).value;
  }

  function hintFor(field) {
    if (field.key === "create.chunk_size") {
      const size = currentNumber("create.chunk_size");
      if (size == null) return {};
      const low = Math.round(size / 4);
      const high = Math.round(size / 3);
      const range = `≈ ${fmt.int(low)}–${fmt.int(high)} tokens`;
      if (!state.embedding) return { text: range };
      return { text: `${range} (limit ${fmt.int(state.embedding.max_sequence)})`, warn: high > state.embedding.max_sequence };
    }
    if (field.key === "create.chunk_overlap") {
      const size = currentNumber("create.chunk_size");
      const overlap = currentNumber("create.chunk_overlap");
      if (!size || overlap == null) return {};
      return { text: `${Math.round((overlap / size) * 100)}% of chunk size` };
    }
    if (field.key === "create.half" && field.disabled) return { text: "Needs a supported NVIDIA GPU" };
    return {};
  }

  function messageHTML(field) {
    if (errors[field.key]) return `<div class="field-error">${esc(errors[field.key])}</div>`;
    const hint = hintFor(field);
    return `<div class="field-hint${hint.warn ? " warn" : ""}">${hint.text ? esc(hint.text) : ""}</div>`;
  }

  function labelHTML(key, label) {
    const info = helpFor(key) ? `<span class="info" data-tip="help" data-help="${esc(key)}">${ICONS.info}</span>` : "";
    const recent = savedAt[key] && Date.now() - savedAt[key] < SAVED_MS;
    return `<div class="field-label"><span>${esc(label)}</span>${info}<span class="saved${recent ? " show" : ""}" data-saved="${esc(key)}">${ICONS.check}Saved</span></div>`;
  }

  function segmentedHTML(key, options, current) {
    const buttons = options.map((o) => {
      const on = String(o.value) === String(current);
      return `<button type="button" class="${on ? "on" : ""}" role="radio" aria-checked="${on}" data-key="${esc(key)}" data-value="${esc(o.value)}">${esc(o.label)}</button>`;
    }).join("");
    return `<div class="seg full" role="radiogroup">${buttons}</div>`;
  }

  function numberHTML(field) {
    const saved = state.values[field.key];
    const text = field.key in drafts ? drafts[field.key] : (saved == null ? "" : String(saved));
    const unit = field.unit ? `<span class="unit">${esc(field.unit)}</span>` : "";
    return `<label class="input${errors[field.key] ? " invalid" : ""}"><input type="text" inputmode="decimal" data-key="${esc(field.key)}" data-kind="number" value="${esc(text)}" placeholder="${esc(field.placeholder || "")}" autocomplete="off" spellcheck="false">${unit}</label>`;
  }

  function textHTML(field) {
    const text = field.key in drafts ? drafts[field.key] : String(state.values[field.key] || "");
    return `<label class="input${errors[field.key] ? " invalid" : ""}${text ? " has-text" : ""}"><input type="text" data-key="${esc(field.key)}" data-kind="text" value="${esc(text)}" placeholder="${esc(field.placeholder || "")}" autocomplete="off" spellcheck="false"><button type="button" class="clear" data-clear="${esc(field.key)}" data-tip-text="Clear">${ICONS.close}</button></label>`;
  }

  function switchHTML(key, on, text, disabled, extra = "") {
    return `<button type="button" class="switch" role="switch" aria-checked="${Boolean(on)}" data-key="${esc(key)}"${extra}${disabled ? " disabled" : ""}><span class="track"></span><span>${esc(text)}</span></button>`;
  }

  function fieldHTML(field) {
    const span = field.span ? ` span-${field.span}` : "";
    let control = "";
    if (field.type === "segmented") control = segmentedHTML(field.key, field.options, state.values[field.key]);
    else if (field.type === "number") control = numberHTML(field);
    else if (field.type === "text") control = textHTML(field);
    else if (field.type === "toggle") control = switchHTML(field.key, state.values[field.key], field.disabled ? "Not available" : field.text || "", field.disabled);
    return `<div class="field${span}" data-field="${esc(field.key)}">${labelHTML(field.key, field.label)}${control}${messageHTML(field)}</div>`;
  }

  function speedText(speed) {
    return speed >= 10 ? speed.toFixed(0) : speed.toFixed(1);
  }

  function ttsCardHTML(backend) {
    const card = backend.card;
    const bench = card.benchmark;
    const memory = card.device === "cpu" ? "RAM" : "VRAM";
    const where = card.device === "cpu" ? "on the CPU" : "on the GPU";
    const chips = card.online
      ? '<span class="chip online" data-tip-text="Sends the text to Google and plays the speech it returns. Needs an internet connection.">Online</span>'
      : [
        bench ? `<span class="chip speed" data-tip-text="Seconds of speech made per second of work. Above 1× keeps up with playback.">${speedText(bench.speed)}× real time</span>` : "",
        bench ? `<span class="chip vram" data-tip-text="${memory} used while speaking">${esc(fmt.size(bench.memory_mb))} ${memory}</span>` : "",
        card.dtype ? `<span class="chip prec-${esc(card.dtype)}" data-tip-text="Runs in ${esc(card.dtype)} ${where}">${esc(fmt.precision(card.dtype))}</span>` : "",
        card.requires_cuda ? '<span class="chip gpu" data-tip-text="Needs a supported NVIDIA GPU">GPU only</span>' : "",
      ].join("");
    const name = card.url
      ? `<a class="name" href="${esc(card.url)}" draggable="false" data-tip-text="Open the model page on Hugging Face">${esc(card.name)}${ICONS.external}</a>`
      : `<span class="name">${esc(card.name)}</span>`;
    return '<article class="card model-card tts-card">'
      + `<div class="card-head">${name}</div>`
      + `<div class="chips">${chips}</div></article>`;
  }

  function ttsSegmentHTML(backend, name, label) {
    const control = backend[name];
    const key = `tts.${name}`;
    const buttons = control.options.map((o) => {
      const on = String(o.value) === String(control.value);
      const disabled = control.disabled || o.disabled;
      return `<button type="button" class="${on ? "on" : ""}" role="radio" aria-checked="${on}" data-key="${esc(key)}" data-value="${esc(o.value)}" data-tts-option="${esc(name)}"${disabled ? " disabled" : ""}>${esc(o.label)}</button>`;
    }).join("");
    const tip = control.hint ? ` data-tip-text="${esc(control.hint)}"` : "";
    const error = errors[key] ? `<div class="field-error">${esc(errors[key])}</div>` : "";
    return `<div class="field" data-field="${esc(key)}">${labelHTML(key, label)}<div class="seg full" role="radiogroup"${tip}>${buttons}</div>${error}</div>`;
  }

  function ttsFields() {
    const backend = currentBackend();
    const controls = [
      `<div class="field" data-field="tts.backend">${labelHTML("tts.backend", "Backend")}`
        + `${VDB.selectButton("tts.backend", { label: backend.label, note: backend.note })}${messageHTML({ key: "tts.backend" })}</div>`,
    ];
    if (backend.device) controls.push(ttsSegmentHTML(backend, "device", "Device"));
    if (backend.precision) controls.push(ttsSegmentHTML(backend, "precision", "Precision"));
    for (const extra of backend.extras) {
      if (!extra.visible) continue;
      const key = `tts.${backend.key}.${extra.name}`;
      const option = extra.options.find((o) => o.value === extra.value) || { label: extra.value };
      controls.push(`<div class="field" data-field="${esc(key)}">${labelHTML(key, extra.label)}${VDB.selectButton(key, option)}${messageHTML({ key })}</div>`);
    }
    if (TWO_COLUMN_TTS.has(backend.key) && controls.length > 2) {
      const rows = Math.ceil(controls.length / 2);
      return `<div class="field-stack two" style="grid-template-rows: repeat(${rows}, auto)">${controls.join("")}</div>`
        + `<div class="field span-2">${ttsCardHTML(backend)}</div>`;
    }
    return `<div class="field-stack">${controls.join("")}</div><div class="field span-3">${ttsCardHTML(backend)}</div>`;
  }

  function specHTML(label, value) {
    return `<div class="spec"><span class="spec-label">${esc(label)}</span><span class="spec-value">${esc(value || "—")}</span></div>`;
  }

  function visionCardHTML(model) {
    const where = state.cuda ? "on this GPU" : "on the CPU";
    const chips = [
      model.size ? `<span class="chip params" data-tip-text="Number of parameters">${esc(model.size.toUpperCase())} params</span>` : "",
      model.vram ? `<span class="chip vram" data-tip-text="Approximate memory used while describing images">${esc(vramText(model))} VRAM</span>` : "",
      model.speed ? `<span class="chip speed" data-tip-text="Average speed when describing an image">${fmt.int(Math.round(model.speed))} chars/s</span>` : "",
      model.precision ? `<span class="chip prec-${esc(model.precision)}" data-tip-text="Runs in ${esc(model.precision)} ${where}">${esc(fmt.precision(model.precision))}</span>` : "",
      model.quant && model.quant !== "n/a" ? `<span class="chip quant" data-tip-text="Loaded with ${esc(model.quant)} quantization to use less memory">${esc(model.quant)}</span>` : "",
      model.requires_cuda ? '<span class="chip gpu" data-tip-text="Needs a supported NVIDIA GPU">GPU only</span>' : "",
    ].join("");
    const name = model.url
      ? `<a class="name" href="${esc(model.url)}" draggable="false" data-tip-text="Open the model page on Hugging Face">${esc(model.name)}${ICONS.external}</a>`
      : `<span class="name">${esc(model.name)}</span>`;
    return '<article class="card model-card vision-card">'
      + `<div class="card-head">${name}</div>`
      + `<div class="chips">${chips}</div>`
      + '<div class="specs">'
      + specHTML("Vision component", model.vision_component)
      + specHTML("Chat component", model.chat_component)
      + specHTML("Average description", model.avg_length ? `${fmt.int(model.avg_length)} characters` : "")
      + "</div></article>";
  }

  function visionFields() {
    const model = currentVision();
    if (!model) return "";
    return `<div class="field" data-field="vision.model">${labelHTML("vision.model", "Model")}`
      + `${VDB.selectButton("vision.model", { label: model.name, note: model.size })}${messageHTML({ key: "vision.model" })}</div>`
      + `<div class="field span-3">${visionCardHTML(model)}</div>`;
  }

  function sectionHTML(section) {
    let body;
    if (section.custom === "tts") body = ttsFields();
    else if (section.custom === "vision") body = visionFields();
    else body = section.fields.map(fieldHTML).join("");
    return `<section class="card section" data-section="${esc(section.id)}"><h2 class="section-title">${ICONS[section.icon] || ""}${esc(section.title)}</h2><div class="fields">${body}</div></section>`;
  }

  function renderHardware() {
    $("#hardware").innerHTML = state.cuda ? `${ICONS.chip}<b>${esc(state.gpu || "GPU")}</b>` : `${ICONS.chip}<b>CPU</b> only`;
  }

  function captureFocus() {
    const active = document.activeElement;
    if (!active || !active.dataset || !(active.dataset.key || active.dataset.select)) return null;
    return {
      key: active.dataset.key,
      select: active.dataset.select,
      value: active.dataset.value,
      input: active.tagName === "INPUT",
      start: active.selectionStart,
      end: active.selectionEnd,
    };
  }

  function restoreFocus(focus) {
    if (!focus) return;
    let element = null;
    if (focus.select) element = document.querySelector(`[data-select="${CSS.escape(focus.select)}"]`);
    else if (focus.value != null) element = document.querySelector(`[data-key="${CSS.escape(focus.key)}"][data-value="${CSS.escape(focus.value)}"]`);
    else element = document.querySelector(`[data-key="${CSS.escape(focus.key)}"]`);
    if (!element) return;
    element.focus();
    if (focus.input && focus.start != null) element.setSelectionRange(focus.start, focus.end);
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
    renderHardware();
    const notice = state.error ? `<div class="notice error">${ICONS.info}<div>${esc(state.error)}</div></div>` : "";
    $("#main").innerHTML = notice + state.sections.map(sectionHTML).join("");
    restoreFocus(focus);
    scheduleFade();
  }

  function updateMessage(key) {
    const element = document.querySelector(`[data-field="${CSS.escape(key)}"]`);
    if (!element) return;
    const input = element.querySelector(".input");
    if (input) input.classList.toggle("invalid", Boolean(errors[key]));
    const message = element.querySelector(".field-hint, .field-error");
    if (message) message.outerHTML = messageHTML(fieldDef(key));
  }

  function commit(key, value, fieldKey = key) {
    VDB.tooltip.hide();
    queue = queue.then(() => VDB.call("apply", { key, value })).then((result) => {
      if (result.state) state = result.state;
      if (result.ok) {
        delete errors[fieldKey];
        delete drafts[fieldKey];
        if (result.changed) savedAt[fieldKey] = Date.now();
      } else {
        errors[fieldKey] = result.error || "The setting could not be saved.";
      }
      render();
    });
    return queue;
  }

  function sameNumber(a, b) {
    return b != null && b !== "" && Number(a) === Number(b);
  }

  function commitInput(input) {
    const key = input.dataset.key;
    if (!(key in drafts)) return;
    const text = drafts[key];
    if (input.dataset.kind === "number") {
      const parsed = parseNumber(fieldDef(key), text);
      if (parsed.empty) {
        revert(key);
        return;
      }
      if (parsed.error) {
        errors[key] = parsed.error;
        updateMessage(key);
        return;
      }
      if (sameNumber(parsed.value, state.values[key])) {
        revert(key);
        return;
      }
      commit(key, parsed.value);
    } else {
      if (text.trim() === String(state.values[key] || "")) {
        revert(key);
        return;
      }
      commit(key, text);
    }
  }

  function revert(key) {
    delete drafts[key];
    delete errors[key];
    render();
  }

  function onInput(input) {
    const key = input.dataset.key;
    drafts[key] = input.value;
    if (input.dataset.kind === "number") {
      const parsed = parseNumber(fieldDef(key), input.value);
      if (parsed.error) errors[key] = parsed.error;
      else delete errors[key];
    } else {
      delete errors[key];
      input.parentElement.classList.toggle("has-text", input.value.length > 0);
    }
    updateMessage(key);
    if (key === "create.chunk_size") updateMessage("create.chunk_overlap");
  }

  function step(input, direction) {
    const field = fieldDef(input.dataset.key);
    const parsed = parseNumber(field, input.value);
    const base = parsed.value ?? currentNumber(field.key) ?? field.min;
    let next = base + direction * field.step;
    next = Math.min(field.max, Math.max(field.min, next));
    input.value = field.decimals ? next.toFixed(field.decimals) : String(Math.round(next));
    onInput(input);
  }

  function openSelectFor(id) {
    if (id === "tts.backend") {
      const options = state.tts.backends.map((b) => ({ value: b.key, label: b.label, note: b.note }));
      VDB.openSelect(id, {
        title: "Text to speech backend",
        options,
        value: state.tts.current,
        onPick: (value) => {
          if (value !== state.tts.current) commit("tts.backend", value);
        },
      });
    } else if (id === "vision.model") {
      const options = state.vision.models.map((m) => ({ value: m.name, label: m.name, note: `${m.size} · ${vramText(m)}` }));
      VDB.openSelect(id, {
        title: "Vision model",
        options,
        value: state.vision.current,
        onPick: (value) => {
          if (value !== state.vision.current) commit("vision.model", value);
        },
      });
    } else if (id.startsWith("tts.")) {
      const [, backend, name] = id.split(".");
      const extra = currentBackend().extras.find((x) => x.name === name);
      if (!extra) return;
      VDB.openSelect(id, {
        title: extra.label,
        options: extra.options,
        value: extra.value,
        onPick: (value) => {
          if (value !== extra.value) commit("tts.option", { backend, name, value }, id);
        },
      });
    }
  }

  function onSwitch(button) {
    commit(button.dataset.key, button.getAttribute("aria-checked") !== "true");
  }

  function tipContent(element) {
    if (element.dataset.tip === "hardware") {
      return state.cuda
        ? `<div class="tt-title">${esc(state.gpu || "NVIDIA GPU")}</div>Half precision and the GPU text to speech backends and vision models are available.`
        : '<div class="tt-title">CPU only</div>No supported NVIDIA GPU was found, so Half precision is unavailable and GPU-only options are hidden.';
    }
    if (element.dataset.tip !== "help") return "";
    const key = element.dataset.help;
    if (key === "create.half" && !state.cuda) return esc(state.help["create.half.unavailable"]);
    let html = esc(helpFor(key));
    if (key === "create.chunk_size" && state.embedding) {
      html += `<div class="tt-note">Your selected embedding model, ${esc(state.embedding.name)}, reads up to ${fmt.int(state.embedding.max_sequence)} tokens.</div>`;
    }
    return html;
  }

  function bindEvents() {
    const main = $("#main");
    main.addEventListener("click", (e) => {
      const ttsSegment = e.target.closest(".seg button[data-tts-option]");
      if (ttsSegment) {
        const backend = currentBackend();
        const name = ttsSegment.dataset.ttsOption;
        if (!ttsSegment.disabled && ttsSegment.dataset.value !== String(backend[name].value)) {
          commit("tts.option", { backend: backend.key, name, value: ttsSegment.dataset.value }, `tts.${name}`);
        }
        return;
      }
      const segment = e.target.closest(".seg button[data-key]");
      if (segment) {
        if (segment.dataset.value !== String(state.values[segment.dataset.key])) commit(segment.dataset.key, segment.dataset.value);
        return;
      }
      const toggle = e.target.closest(".switch[data-key]");
      if (toggle) {
        if (!toggle.disabled) onSwitch(toggle);
        return;
      }
      const select = e.target.closest(".select[data-select]");
      if (select) {
        openSelectFor(select.dataset.select);
        return;
      }
      const clear = e.target.closest("[data-clear]");
      if (clear) {
        e.preventDefault();
        const key = clear.dataset.clear;
        if (String(state.values[key] || "") === "" && !(key in drafts)) return;
        drafts[key] = "";
        commit(key, "");
      }
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
        revert(input.dataset.key);
      } else if ((e.key === "ArrowUp" || e.key === "ArrowDown") && input.dataset.kind === "number") {
        e.preventDefault();
        step(input, e.key === "ArrowUp" ? 1 : -1);
      }
    });
    main.addEventListener("focusout", (e) => {
      const input = e.target.closest("input[data-key]");
      if (!input || !input.isConnected) return;
      const next = e.relatedTarget;
      if (next && next.dataset && next.dataset.clear === input.dataset.key) return;
      commitInput(input);
    });
  }

  VDB.tooltip.provider = tipContent;
  bindEvents();

  window.TabApp = {
    init(payload) {
      state = payload;
      render();
    },
    setState(payload) {
      state = payload;
      VDB.tooltip.hide();
      render();
      VDB.popover.close();
    },
  };
})();
