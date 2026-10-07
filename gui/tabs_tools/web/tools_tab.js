"use strict";

(function () {
  const { ICONS, esc } = VDB;

  const tools = {};
  const rendered = {};
  let state = null;

  function register(name, tool) {
    tools[name] = tool;
  }

  function data(name) {
    const tool = tools[name];
    return tool.select ? tool.select(state) : state[name];
  }

  function call(action, args = {}) {
    VDB.tooltip.hide();
    return VDB.call(action, args);
  }

  const FOCUS_ATTRS = ["key", "select", "act"];

  function captureFocus(root) {
    const active = document.activeElement;
    if (!active || !root.contains(active) || !active.dataset) return null;
    const attr = FOCUS_ATTRS.find((a) => active.dataset[a]);
    if (!attr) return null;
    return {
      attr,
      value: active.dataset[attr],
      arg: active.dataset.arg ?? null,
      input: active.tagName === "INPUT",
      start: active.selectionStart,
      end: active.selectionEnd,
    };
  }

  function restoreFocus(root, focus) {
    if (!focus) return;
    const name = { key: "data-key", select: "data-select", act: "data-act" }[focus.attr];
    let selector = `[${name}="${CSS.escape(focus.value)}"]`;
    if (focus.arg != null) selector += `[data-arg="${CSS.escape(focus.arg)}"]`;
    const element = root.querySelector(selector);
    if (!element || element.disabled) return;
    element.focus();
    if (focus.input && focus.start != null) element.setSelectionRange(focus.start, focus.end);
  }

  function render(name, force = false) {
    const root = document.querySelector(`[data-tool="${name}"]`);
    if (!root) return;
    const slice = data(name);
    const signature = JSON.stringify(slice);
    if (!force && rendered[name] === signature) return;
    rendered[name] = signature;
    if (VDB.tooltip.target && root.contains(VDB.tooltip.target)) VDB.tooltip.hide();
    const focus = captureFocus(root);
    root.innerHTML = tools[name].render(slice, state);
    restoreFocus(root, focus);
    if (VDB.popover.key) {
      const anchor = VDB.popover.anchor();
      if (anchor && root.contains(anchor)) {
        if (anchor.disabled) VDB.popover.close();
        else {
          anchor.classList.add("open");
          VDB.popover.position();
        }
      }
    }
  }

  function renderAll(force = false) {
    for (const name of Object.keys(tools)) render(name, force);
  }

  function toolFor(element) {
    const root = element.closest("[data-tool]");
    return root ? { name: root.dataset.tool, tool: tools[root.dataset.tool] } : null;
  }

  function onClick(e) {
    const target = e.target.closest("[data-act], [data-select]");
    if (!target || target.disabled) return;
    const owner = toolFor(target);
    if (!owner) return;
    if (target.dataset.select) {
      if (owner.tool.openSelect) owner.tool.openSelect(target.dataset.select, data(owner.name), state);
      return;
    }
    const action = target.dataset.act;
    const local = owner.tool.actions && owner.tool.actions[action];
    if (local) {
      local(target, data(owner.name), state);
      return;
    }
    call(action, target.dataset.arg != null ? { name: target.dataset.arg } : {});
  }

  function dispatch(kind) {
    return (e) => {
      const input = e.target.closest && e.target.closest("input[data-key]");
      if (!input) return;
      const owner = toolFor(input);
      if (owner && owner.tool[kind]) owner.tool[kind](input, e, data(owner.name), state);
    };
  }

  function tipContent(element) {
    const owner = toolFor(element);
    if (owner && owner.tool.tip) return owner.tool.tip(element, data(owner.name), state) || "";
    return "";
  }

  const html = {
    header(icon, title, description, aside = "") {
      const desc = description ? `<span class="tool-desc">${esc(description)}</span>` : "";
      const side = aside ? `<span class="tool-aside">${aside}</span>` : "";
      return `<h2 class="tool-title">${ICONS[icon] || ""}<span class="tool-name">${esc(title)}</span>${desc}${side}</h2>`;
    },
    label(text, tip) {
      const info = tip ? `<span class="info" data-tip-text="${esc(tip)}">${ICONS.info}</span>` : "";
      return `<div class="field-label"><span>${esc(text)}</span>${info}</div>`;
    },
    field(label, control, { tip = "", cls = "", message = "" } = {}) {
      return `<div class="field${cls ? ` ${cls}` : ""}">${label ? html.label(label, tip) : ""}${control}${message}</div>`;
    },
    segmented(act, options, current, disabled = false) {
      const buttons = options.map((o) => {
        const on = String(o.value) === String(current);
        const off = disabled || o.disabled;
        const tip = o.tip ? ` data-tip-text="${esc(o.tip)}"` : "";
        return `<button type="button" class="${on ? "on" : ""}" role="radio" aria-checked="${on}" data-act="${esc(act)}" data-arg="${esc(o.value)}"${tip}${off ? " disabled" : ""}>${esc(o.label)}</button>`;
      }).join("");
      return `<div class="seg full" role="radiogroup">${buttons}</div>`;
    },
    file(act, file, placeholder, { icon = "file", note = "", disabled = false } = {}) {
      const name = file ? `<span class="file-name">${esc(file.name)}</span>` : `<span class="file-name">${esc(placeholder)}</span>`;
      const extra = note ? `<span class="file-note">${esc(note)}</span>` : "";
      const tip = file ? ` data-tip-text="${esc(file.path)}"` : "";
      return `<button type="button" class="file${file ? " set" : ""}" data-act="${esc(act)}"${tip}${disabled ? " disabled" : ""}>${ICONS[icon]}${name}${extra}</button>`;
    },
    action(act, label, { icon = "play", kind = "primary", disabled = false, tip = "" } = {}) {
      const tipAttr = tip ? ` data-tip-text="${esc(tip)}"` : "";
      return `<button type="button" class="btn ${kind} action" data-act="${esc(act)}"${tipAttr}${disabled ? " disabled" : ""}>${ICONS[icon] || ""}${esc(label)}</button>`;
    },
    since(started) {
      if (!started) return "";
      return `<span class="elapsed" data-since="${started}">${VDB.elapsed(Date.now() / 1000 - started)}</span>`;
    },
    status(kind, body) {
      const icon = { run: '<span class="spinner"></span>', ok: ICONS.check, warn: ICONS.warn, error: ICONS.error, info: ICONS.info }[kind] || "";
      return `<div class="status ${kind === "run" ? "" : kind}">${icon}<div class="status-body">${body}</div></div>`;
    },
    link(act, label, arg) {
      const argAttr = arg != null ? ` data-arg="${esc(arg)}"` : "";
      return `<button type="button" class="link" data-act="${esc(act)}"${argAttr}>${esc(label)}</button>`;
    },
  };

  document.addEventListener("click", onClick);
  document.addEventListener("input", dispatch("onInput"));
  document.addEventListener("keydown", dispatch("onKey"));
  document.addEventListener("focusout", dispatch("onBlur"));
  VDB.tooltip.provider = tipContent;

  window.Tools = { register, call, render, html, data };

  window.TabApp = {
    init(payload) {
      state = payload;
      renderAll(true);
    },
    setState(payload) {
      state = payload;
      renderAll();
    },
  };
})();
