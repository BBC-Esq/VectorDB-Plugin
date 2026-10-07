"use strict";

(function () {
  const ICONS = {
    search: '<svg viewBox="0 0 16 16"><circle cx="7" cy="7" r="4.6" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m10.6 10.6 3.4 3.4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
    close: '<svg viewBox="0 0 16 16"><path d="m4.5 4.5 7 7m0-7-7 7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
    download: '<svg viewBox="0 0 16 16"><path d="M8 2.5v7.6M4.8 7.2 8 10.4l3.2-3.2" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M3 13h10" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
    check: '<svg viewBox="0 0 16 16"><path d="m3.5 8.4 3 3 6-6.6" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    external: '<svg viewBox="0 0 16 16"><path d="M9.5 2.75h3.75V6.5M13 3 7.6 8.4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 9.6v2.65c0 .55-.45 1-1 1H3.75c-.55 0-1-.45-1-1V5c0-.55.45-1 1-1H6.4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    caret: '<svg viewBox="0 0 16 16"><path d="m4.5 6.25 3.5 3.5 3.5-3.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    cards: '<svg viewBox="0 0 16 16"><rect x="2" y="2" width="5" height="5" rx="1.3" fill="none" stroke="currentColor" stroke-width="1.4"/><rect x="9" y="2" width="5" height="5" rx="1.3" fill="none" stroke="currentColor" stroke-width="1.4"/><rect x="2" y="9" width="5" height="5" rx="1.3" fill="none" stroke="currentColor" stroke-width="1.4"/><rect x="9" y="9" width="5" height="5" rx="1.3" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>',
    list: '<svg viewBox="0 0 16 16"><path d="M2.5 4h11M2.5 8h11M2.5 12h11" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
    info: '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M8 7.3v3.9" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="8" cy="4.9" r=".95" fill="currentColor"/></svg>',
    chip: '<svg viewBox="0 0 16 16"><rect x="3.5" y="3.5" width="9" height="9" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4"/><rect x="6" y="6" width="4" height="4" rx=".6" fill="currentColor"/><path d="M6 1.5v2M10 1.5v2M6 12.5v2M10 12.5v2M1.5 6h2M1.5 10h2M12.5 6h2M12.5 10h2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>',
    sort: '<svg viewBox="0 0 16 16"><path d="M5 3v10M2.5 10.5 5 13l2.5-2.5M11 13V3M8.5 5.5 11 3l2.5 2.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    database: '<svg viewBox="0 0 16 16"><ellipse cx="8" cy="3.9" rx="5" ry="1.9" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M3 3.9v8.2c0 1.05 2.24 1.9 5 1.9s5-.85 5-1.9V3.9M3 8c0 1.05 2.24 1.9 5 1.9s5-.85 5-1.9" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>',
    build: '<svg viewBox="0 0 16 16"><path d="M8 1.8 13.5 4.9v6.2L8 14.2 2.5 11.1V4.9L8 1.8Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M2.7 5 8 8l5.3-3M8 8v6" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>',
    speaker: '<svg viewBox="0 0 16 16"><path d="M2.5 6.1h2.6L8.6 3v10L5.1 9.9H2.5V6.1Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M11 5.6a3.3 3.3 0 0 1 0 4.8M12.8 3.9a5.7 5.7 0 0 1 0 8.2" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>',
    eye: '<svg viewBox="0 0 16 16"><path d="M1.6 8s2.4-4.6 6.4-4.6S14.4 8 14.4 8s-2.4 4.6-6.4 4.6S1.6 8 1.6 8Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><circle cx="8" cy="8" r="2.1" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>',
  };

  function esc(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  function trimZeros(text) {
    return text.includes(".") ? text.replace(/\.?0+$/, "") : text;
  }

  const fmt = {
    int: (n) => Number(n).toLocaleString("en-US"),
    tokens: (n) => (n >= 1024 && n % 1024 === 0 ? `${n / 1024}K` : fmt.int(n)),
    size: (mb) => (mb >= 1000 ? `${trimZeros((mb / 1000).toFixed(mb >= 10000 ? 1 : 2))} GB` : `${mb} MB`),
    params: (m) => (m >= 1000 ? `${trimZeros((m / 1000).toFixed(2))}B` : `${trimZeros(m.toFixed(1))}M`),
    score: (v) => (v == null ? "—" : v.toFixed(1)),
    precision: (p) => ({ float32: "fp32", bfloat16: "bf16", float16: "fp16" })[p] || p,
  };

  const $ = (selector, root = document) => root.querySelector(selector);

  function applyTheme(theme) {
    const root = document.documentElement;
    for (const [key, value] of Object.entries(theme.colors)) {
      root.style.setProperty(`--${key.replace(/_/g, "-")}`, value);
    }
    root.style.setProperty("--accent", theme.accent);
    root.style.setProperty("--gold", theme.gold);
    root.style.setProperty("--danger", theme.danger);
    root.style.setProperty("--on-control", theme.on_control);
    root.style.setProperty("--on-control-hover", theme.on_control_hover);
    root.dataset.scheme = theme.scheme;
    root.style.colorScheme = theme.scheme;
  }

  const bridgeReady = new Promise((resolve) => {
    if (window.QWebChannel && window.qt && window.qt.webChannelTransport) {
      new QWebChannel(window.qt.webChannelTransport, (channel) => resolve(channel.objects.bridge));
    } else {
      resolve(null);
    }
  });

  function call(name, args = {}) {
    return bridgeReady.then((bridge) => new Promise((resolve) => {
      if (!bridge) {
        resolve({ error: "Not connected to the program." });
        return;
      }
      bridge.call(name, JSON.stringify(args), (result) => resolve(JSON.parse(result || "{}")));
    }));
  }

  function fillIcons(root = document) {
    for (const slot of root.querySelectorAll("[data-icon]")) {
      slot.outerHTML = ICONS[slot.dataset.icon];
    }
  }

  const tooltip = {
    element: null,
    target: null,
    timer: 0,
    provider: () => "",
    contentFor(element) {
      if (element.dataset.tipText) return esc(element.dataset.tipText);
      return this.provider(element) || "";
    },
    show(element) {
      const html = this.contentFor(element);
      if (!html) return;
      const tip = this.element;
      tip.innerHTML = html;
      tip.style.left = "0px";
      tip.style.top = "0px";
      const rect = element.getBoundingClientRect();
      const width = tip.offsetWidth;
      const height = tip.offsetHeight;
      let top = rect.bottom + 7;
      if (top + height > window.innerHeight - 6) top = Math.max(6, rect.top - height - 7);
      const left = Math.max(6, Math.min(rect.left + rect.width / 2 - width / 2, window.innerWidth - width - 6));
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
      tip.classList.add("show");
    },
    hide() {
      clearTimeout(this.timer);
      this.target = null;
      if (this.element) this.element.classList.remove("show");
    },
    install(provider) {
      this.provider = provider || this.provider;
      this.element = document.createElement("div");
      this.element.id = "tooltip";
      this.element.className = "tooltip";
      this.element.setAttribute("role", "tooltip");
      document.body.appendChild(this.element);
      document.addEventListener("mouseover", (e) => {
        const element = e.target.closest("[data-tip], [data-tip-text]");
        if (element === this.target) return;
        this.hide();
        if (!element || e.target.closest(".popover")) return;
        this.target = element;
        this.timer = setTimeout(() => this.show(element), 350);
      });
      document.addEventListener("mousedown", () => this.hide(), true);
      document.addEventListener("scroll", () => this.hide(), true);
      document.documentElement.addEventListener("mouseleave", () => this.hide());
      window.addEventListener("resize", () => this.hide());
    },
  };

  const popover = {
    element: null,
    key: null,
    options: null,
    install() {
      this.element = document.createElement("div");
      this.element.id = "popover";
      this.element.className = "popover";
      this.element.setAttribute("role", "dialog");
      document.body.appendChild(this.element);
      this.element.addEventListener("click", (e) => {
        if (this.options && this.options.click) this.options.click(e);
      });
      document.addEventListener("mousedown", (e) => {
        if (!this.key || e.target.closest(".popover")) return;
        const anchor = this.anchor();
        if (anchor && anchor.contains(e.target)) return;
        this.close();
      });
      document.addEventListener("keydown", (e) => {
        if (!this.key) return;
        if (e.key === "Escape") {
          e.preventDefault();
          const anchor = this.anchor();
          this.close();
          if (anchor) anchor.focus();
        } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
          e.preventDefault();
          this.moveHighlight(e.key === "ArrowDown" ? 1 : -1);
        } else if (e.key === "Enter") {
          const current = this.element.querySelector(".opt.kbd");
          if (current) {
            e.preventDefault();
            current.click();
          }
        }
      });
      window.addEventListener("resize", () => this.position());
    },
    anchor() {
      return this.options ? this.options.anchor() : null;
    },
    isOpen(key) {
      return this.key === key;
    },
    open(key, options) {
      tooltip.hide();
      this.key = key;
      this.options = options;
      this.render();
      this.element.classList.add("show");
      this.position();
      if (options.onOpen) options.onOpen();
    },
    toggle(key, options) {
      if (this.key === key) this.close();
      else this.open(key, options);
    },
    render() {
      if (!this.key) return;
      this.element.innerHTML = this.options.render();
    },
    position() {
      if (!this.key) return;
      const anchor = this.anchor();
      if (!anchor) return;
      const rect = anchor.getBoundingClientRect();
      const popover = this.element;
      popover.style.minWidth = `${Math.max(250, this.options.matchWidth ? rect.width : 0)}px`;
      const width = popover.offsetWidth;
      const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8));
      const below = window.innerHeight - rect.bottom - 16;
      const above = rect.top - 16;
      popover.style.left = `${left}px`;
      if (below >= Math.min(popover.scrollHeight, 260) || below >= above) {
        popover.style.top = `${rect.bottom + 6}px`;
        popover.style.maxHeight = `${Math.max(140, below)}px`;
      } else {
        const height = Math.min(popover.scrollHeight, above);
        popover.style.top = `${rect.top - 6 - height}px`;
        popover.style.maxHeight = `${Math.max(140, above)}px`;
      }
    },
    moveHighlight(step) {
      const options = [...this.element.querySelectorAll(".opt:not(:disabled)")];
      if (!options.length) return;
      const index = options.findIndex((o) => o.classList.contains("kbd"));
      const next = options[(index + step + options.length) % options.length];
      options.forEach((o) => o.classList.remove("kbd"));
      next.classList.add("kbd");
      next.scrollIntoView({ block: "nearest" });
    },
    close() {
      if (!this.key) return;
      const onClose = this.options && this.options.onClose;
      this.key = null;
      this.options = null;
      this.element.classList.remove("show");
      if (onClose) onClose();
    },
  };

  function radioOption(attrs, label, selected, count, disabled = false) {
    const classes = ["opt", selected ? "on" : "", count === 0 ? "zero" : ""].filter(Boolean).join(" ");
    const countHTML = count == null ? "" : `<span class="n">${count}</span>`;
    return `<button type="button" class="${classes}" ${attrs}${disabled ? " disabled" : ""}><span class="mark">${selected ? ICONS.check : ""}</span><span class="label">${label}</span>${countHTML}</button>`;
  }

  function checkOption(attrs, label, selected, count) {
    const classes = ["opt", selected ? "on" : "", count === 0 ? "zero" : ""].filter(Boolean).join(" ");
    return `<button type="button" class="${classes}" ${attrs}><span class="box">${ICONS.check}</span><span class="label">${label}</span><span class="n">${count}</span></button>`;
  }

  function selectButton(id, option, disabled = false) {
    const label = option ? esc(option.label) : "";
    const note = option && option.note ? `<span class="select-note">${esc(option.note)}</span>` : "";
    return `<button type="button" class="select" data-select="${esc(id)}"${disabled ? " disabled" : ""}><span class="select-text">${label}</span>${note}${ICONS.caret}</button>`;
  }

  function openSelect(id, { title, options, value, onPick }) {
    const anchor = () => document.querySelector(`[data-select="${CSS.escape(id)}"]`);
    popover.toggle(`select:${id}`, {
      anchor,
      matchWidth: true,
      render: () => {
        const head = title ? `<div class="pop-title">${esc(title)}</div>` : "";
        return head + options.map((o, i) => radioOption(
          `data-index="${i}"`,
          `${esc(o.label)}${o.note ? ` <span class="n">${esc(o.note)}</span>` : ""}`,
          o.value === value,
          null,
          Boolean(o.disabled),
        )).join("");
      },
      click: (e) => {
        const button = e.target.closest(".opt[data-index]");
        if (!button || button.disabled) return;
        const option = options[Number(button.dataset.index)];
        popover.close();
        const target = anchor();
        if (target) target.focus();
        onPick(option.value);
      },
      onOpen: () => {
        const selected = popover.element.querySelector(".opt.on") || popover.element.querySelector(".opt:not(:disabled)");
        if (selected) {
          selected.classList.add("kbd");
          selected.scrollIntoView({ block: "nearest" });
        }
        const target = anchor();
        if (target) target.classList.add("open");
      },
      onClose: () => {
        const target = anchor();
        if (target) target.classList.remove("open");
      },
    });
  }

  window.VDB = {
    ICONS, esc, fmt, $, applyTheme, call, fillIcons, tooltip, popover,
    radioOption, checkOption, selectButton, openSelect,
  };

  document.addEventListener("DOMContentLoaded", () => {
    fillIcons();
    tooltip.install();
    popover.install();
  });
})();
