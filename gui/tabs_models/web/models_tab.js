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
  };

  const PRECISION_SHORT = { float32: "fp32", bfloat16: "bf16", float16: "fp16" };
  const OPS = { ge: "≥", eq: "=", le: "≤" };
  const OP_WORDS = { ge: "At least", eq: "Exactly", le: "At most" };
  const SIZE_STEPS = [500, 1000, 2000, 4000, 8000, 16000];
  const STATUS = [["any", "Any"], ["downloaded", "Downloaded"], ["missing", "Not downloaded"]];
  const FILTERS = [
    { key: "dims", label: "Dimensions" },
    { key: "ctx", label: "Context" },
    { key: "size", label: "Size" },
    { key: "precision", label: "Precision" },
    { key: "scores", label: "Scores" },
    { key: "status", label: "Status" },
  ];
  const LIST_LABELS = {
    english_retrieval: "Eng",
    multilingual_retrieval: "Multi",
    rteb_legal: "Legal",
    mteb_code: "Code",
    code_information_retrieval: "Code IR",
  };

  const state = {
    data: null,
    byId: new Map(),
    bench: new Map(),
    query: "",
    dims: { op: "ge", value: null },
    ctx: { op: "ge", value: null },
    size: null,
    precision: new Set(),
    scores: new Set(),
    status: "any",
    sort: "vendor",
    view: "cards",
  };

  let bridge = null;
  let openMenu = null;
  let tipTimer = 0;
  let tipElement = null;

  const $ = (selector) => document.querySelector(selector);

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
    precision: (p) => PRECISION_SHORT[p] || p,
  };

  function connectBridge() {
    if (window.QWebChannel && window.qt && window.qt.webChannelTransport) {
      new QWebChannel(window.qt.webChannelTransport, (channel) => {
        bridge = channel.objects.bridge;
      });
    }
  }

  function applyTheme(theme) {
    const root = document.documentElement;
    for (const [key, value] of Object.entries(theme.colors)) {
      root.style.setProperty(`--${key.replace(/_/g, "-")}`, value);
    }
    root.style.setProperty("--accent", theme.accent);
    root.style.setProperty("--gold", theme.gold);
    root.style.setProperty("--on-control", theme.on_control);
    root.style.setProperty("--on-control-hover", theme.on_control_hover);
    root.dataset.scheme = theme.scheme;
    root.style.colorScheme = theme.scheme;
  }

  function setData(payload) {
    state.data = payload;
    state.byId = new Map(payload.models.map((m) => [m.id, m]));
    state.bench = new Map(payload.benchmarks.map((b) => [b.key, b]));
    renderAll();
  }

  function sortOptions() {
    return [
      { key: "vendor", label: "Vendor", short: "Vendor" },
      { key: "name", label: "Name (A to Z)", short: "Name" },
      ...state.data.benchmarks.map((b) => ({ key: b.key, label: `${b.title} (best first)`, short: b.label })),
      { key: "dimensions", label: "Dimensions (most first)", short: "Dimensions" },
      { key: "max_sequence", label: "Context (longest first)", short: "Context" },
      { key: "size_mb", label: "Size (smallest first)", short: "Size" },
      { key: "parameters_m", label: "Parameters (fewest first)", short: "Parameters" },
    ];
  }

  function comparator(key) {
    if (key === "name") {
      return (a, b) => a.name.localeCompare(b.name, "en", { sensitivity: "base", numeric: true });
    }
    if (state.bench.has(key)) {
      return (a, b) => {
        const x = a.scores[key];
        const y = b.scores[key];
        if (x == null || y == null) {
          return x == null && y == null ? a.order - b.order : x == null ? 1 : -1;
        }
        return y - x || a.order - b.order;
      };
    }
    if (key === "dimensions" || key === "max_sequence") {
      return (a, b) => b[key] - a[key] || a.order - b.order;
    }
    if (key === "size_mb" || key === "parameters_m") {
      return (a, b) => a[key] - b[key] || a.order - b.order;
    }
    return (a, b) => a.order - b.order;
  }

  function compareValue(value, op, target) {
    if (op === "ge") return value >= target;
    if (op === "le") return value <= target;
    return value === target;
  }

  function matches(m, s) {
    if (s.query) {
      const haystack = `${m.name} ${m.vendor} ${m.repo_id} ${m.license.label} ${m.license.key}`.toLowerCase();
      if (!s.query.split(/\s+/).every((term) => haystack.includes(term))) return false;
    }
    if (s.dims.value != null && !compareValue(m.dimensions, s.dims.op, s.dims.value)) return false;
    if (s.ctx.value != null && !compareValue(m.max_sequence, s.ctx.op, s.ctx.value)) return false;
    if (s.size != null && m.size_mb > s.size) return false;
    if (s.precision.size && !s.precision.has(m.precision.native)) return false;
    for (const key of s.scores) {
      if (m.scores[key] == null) return false;
    }
    if (s.status === "downloaded" && !m.downloaded) return false;
    if (s.status === "missing" && m.downloaded) return false;
    return true;
  }

  function countWith(patch) {
    const s = { ...state, ...patch };
    return state.data.models.filter((m) => matches(m, s)).length;
  }

  function filterValue(key) {
    switch (key) {
      case "dims":
      case "ctx":
        return state[key].value == null ? "" : `${OPS[state[key].op]} ${fmt.int(state[key].value)}`;
      case "size":
        return state.size == null ? "" : `≤ ${fmt.size(state.size)}`;
      case "precision":
        return [...state.precision].map(fmt.precision).join(", ");
      case "scores":
        return state.scores.size > 2
          ? `${state.scores.size} selected`
          : [...state.scores].map((k) => state.bench.get(k).label).join(" + ");
      case "status":
        return { any: "", downloaded: "Downloaded", missing: "Not downloaded" }[state.status];
      default:
        return "";
    }
  }

  function resetFilters() {
    state.dims = { op: state.dims.op, value: null };
    state.ctx = { op: state.ctx.op, value: null };
    state.size = null;
    state.precision = new Set();
    state.scores = new Set();
    state.status = "any";
  }

  function renderAll() {
    renderFilters();
    renderSortButton();
    renderHardware();
    renderViewToggle();
    renderMain();
  }

  function refresh(scrollTop) {
    renderFilters();
    renderSortButton();
    renderMain();
    if (scrollTop) $("#main").scrollTop = 0;
    if (openMenu) positionPopover();
  }

  function renderFilters() {
    const pills = FILTERS.map((f) => {
      const value = filterValue(f.key);
      const classes = ["filter", value ? "active" : "", openMenu === f.key ? "open" : ""].filter(Boolean).join(" ");
      const valueHTML = value ? ` <span class="value">${esc(value)}</span>` : "";
      return `<button type="button" class="${classes}" data-filter="${f.key}">${f.label}${valueHTML}${ICONS.caret}</button>`;
    });
    if (FILTERS.some((f) => filterValue(f.key))) {
      pills.push(`<button type="button" class="clear-filters" id="clearFilters">${ICONS.close}Clear filters</button>`);
    }
    $("#filters").innerHTML = pills.join("");
  }

  function renderSortButton() {
    const option = sortOptions().find((o) => o.key === state.sort) || sortOptions()[0];
    $("#sortButton").innerHTML = `${ICONS.sort}<span class="menu-label">Sort:</span>${esc(option.short)}${ICONS.caret}`;
    $("#sortButton").classList.toggle("open", openMenu === "sort");
  }

  function renderHardware() {
    const hw = state.data.hardware;
    let text;
    if (hw.cpu_only) text = "<b>CPU</b> only";
    else if (hw.device === "cpu") text = "<b>CPU</b> (from Settings)";
    else text = `<b>${esc(hw.gpu_short || "GPU")}</b> · Half ${hw.half ? "on" : "off"}`;
    $("#hardware").innerHTML = `${ICONS.chip}${text}`;
  }

  function renderViewToggle() {
    document.body.classList.toggle("view-cards", state.view === "cards");
    document.body.classList.toggle("view-list", state.view === "list");
    for (const button of document.querySelectorAll("#viewToggle button")) {
      button.classList.toggle("on", button.dataset.view === state.view);
    }
  }

  function renderCount(shown) {
    const total = state.data.models.length;
    $("#count").innerHTML = shown === total ? `${total} models` : `<b>${shown}</b> of ${total}`;
  }

  function renderMain() {
    const models = state.data.models.filter((m) => matches(m, state)).sort(comparator(state.sort));
    renderCount(models.length);
    const notice = state.data.cpu_note
      ? `<div class="notice">${ICONS.info}<div>${esc(state.data.cpu_note)}</div></div>`
      : "";
    let body;
    if (!models.length) body = emptyHTML();
    else if (state.view === "list") body = listHTML(models);
    else body = cardsHTML(models);
    $("#main").innerHTML = notice + body;
  }

  function emptyHTML() {
    return `<div class="empty"><div class="empty-title">No models match</div>
      <div>Try removing a filter or changing the search.</div>
      <button type="button" class="btn" id="resetAll">Clear search and filters</button></div>`;
  }

  function cardsHTML(models) {
    return `<div class="grid">${models.map(cardHTML).join("")}</div>`;
  }

  function cardHTML(m) {
    const sep = '<span class="sep">·</span>';
    return `
      <article class="card" data-id="${esc(m.id)}">
        <div class="card-head">
          <a class="name" href="${esc(m.url)}" draggable="false" data-tip="link">${esc(m.name)}${ICONS.external}</a>
          ${downloadButton(m, false)}
          <div class="meta"><span class="vendor">${esc(m.vendor)}</span>${sep}<span data-tip="license">${esc(m.license.label)}</span>${sep}<span data-tip="params">${fmt.params(m.parameters_m)} params</span>${sep}<span data-tip="size">${fmt.size(m.size_mb)}</span></div>
        </div>
        <div class="chips">${chipsHTML(m)}</div>
        <div class="bench">${state.data.benchmarks.map((b) => scoreHTML(m, b)).join("")}</div>
      </article>`;
  }

  function downloadButton(m, compact) {
    const busy = state.data.downloading;
    if (busy === m.id) {
      const label = compact ? "" : "Downloading…";
      return `<button type="button" class="btn dl busy${compact ? " icon-btn" : ""}" data-tip="download" disabled><span class="spinner"></span>${label}</button>`;
    }
    const classes = ["btn", "dl", m.downloaded ? "done" : "", compact ? "icon-btn" : ""].filter(Boolean).join(" ");
    const icon = m.downloaded ? ICONS.check : ICONS.download;
    const label = compact ? "" : m.downloaded ? "Downloaded" : "Download";
    return `<button type="button" class="${classes}" data-tip="download"${busy ? " disabled" : ""}>${icon}${label}</button>`;
  }

  function precisionChip(m) {
    const p = m.precision;
    const label = p.native === p.current
      ? fmt.precision(p.current)
      : `${fmt.precision(p.native)}<span class="arrow">→</span>${fmt.precision(p.current)}`;
    return `<span class="chip prec-${esc(p.current)}" data-tip="precision">${label}</span>`;
  }

  function chipsHTML(m) {
    return [
      `<span class="chip dims" data-tip="dims">${fmt.int(m.dimensions)} dims</span>`,
      `<span class="chip ctx" data-tip="ctx">${fmt.tokens(m.max_sequence)} tokens</span>`,
      precisionChip(m),
      m.requires_cuda ? `<span class="chip gpu" data-tip="gpu">GPU only</span>` : "",
      m.custom_code ? `<span class="chip code" data-tip="code">Custom code</span>` : "",
    ].join("");
  }

  function scoreHTML(m, b) {
    const value = m.scores[b.key];
    const sorted = state.sort === b.key ? " sorted" : "";
    const label = `<span class="score-label">${esc(b.label)}</span>`;
    if (value == null) {
      return `<div class="score na${sorted}" data-tip="score" data-bench="${b.key}">${label}<span class="score-value">—</span><span class="bar"></span></div>`;
    }
    const best = value === b.max ? " best" : "";
    const width = b.max > b.min ? 14 + (86 * (value - b.min)) / (b.max - b.min) : 100;
    return `<div class="score${best}${sorted}" data-tip="score" data-bench="${b.key}">${label}<span class="score-value">${fmt.score(value)}</span><span class="bar"><i style="width:${width.toFixed(1)}%"></i></span></div>`;
  }

  function listColumns() {
    return [
      { key: "name", label: "Model", cls: "l", sort: "name" },
      { key: "dimensions", label: "Dims", width: 46, sort: "dimensions", tip: "Embedding dimensions" },
      { key: "max_sequence", label: "Context", width: 56, sort: "max_sequence", tip: "Max sequence (tokens)" },
      { key: "parameters_m", label: "Params", width: 52, sort: "parameters_m", tip: "Parameters" },
      { key: "size_mb", label: "Size", width: 60, sort: "size_mb", tip: "Download size" },
      { key: "precision", label: "Precision", width: 82, tip: "Native precision → precision used on this computer" },
      ...state.data.benchmarks.map((b) => ({
        key: b.key, label: LIST_LABELS[b.key] || b.label, width: 50, sort: b.key, tip: b.title, bench: true,
      })),
      { key: "download", label: "", width: 42 },
    ];
  }

  function listHTML(models) {
    const columns = listColumns();
    const colgroup = columns.map((c) => (c.width ? `<col style="width:${c.width}px">` : "<col>")).join("");
    const head = columns.map((c) => {
      const classes = [c.cls || "", c.sort ? "sortable" : "", c.sort && state.sort === c.sort ? "sorted" : ""].filter(Boolean).join(" ");
      const sortAttr = c.sort ? ` data-sort="${c.sort}"` : "";
      const tipAttr = c.tip ? ` data-tip-text="${esc(c.tip)}"` : "";
      return `<th class="${classes}"${sortAttr}${tipAttr}>${esc(c.label)}</th>`;
    }).join("");
    let rows = "";
    let vendor = null;
    for (const m of models) {
      if (state.sort === "vendor" && m.vendor !== vendor) {
        vendor = m.vendor;
        rows += `<tr><td class="group-row" colspan="${columns.length}">${esc(vendor)}</td></tr>`;
      }
      rows += rowHTML(m, columns);
    }
    return `<table class="list"><colgroup>${colgroup}</colgroup><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
  }

  function rowHTML(m, columns) {
    const cells = columns.map((c) => {
      switch (c.key) {
        case "name": {
          const vendor = state.sort !== "vendor" ? `<span class="vendor-inline">${esc(m.vendor)}</span>` : "";
          return `<td class="l"><a class="name" href="${esc(m.url)}" draggable="false" data-tip="link">${esc(m.name)}${ICONS.external}</a>${vendor}</td>`;
        }
        case "dimensions":
          return `<td data-tip="dims">${fmt.int(m.dimensions)}</td>`;
        case "max_sequence":
          return `<td data-tip="ctx">${fmt.int(m.max_sequence)}</td>`;
        case "parameters_m":
          return `<td data-tip="params">${fmt.params(m.parameters_m)}</td>`;
        case "size_mb":
          return `<td data-tip="size">${fmt.size(m.size_mb)}</td>`;
        case "precision":
          return `<td>${precisionChip(m)}</td>`;
        case "download":
          return `<td>${downloadButton(m, true)}</td>`;
        default: {
          const b = state.bench.get(c.key);
          const value = m.scores[c.key];
          if (value == null) {
            return `<td class="na" data-tip="score" data-bench="${c.key}">—</td>`;
          }
          const t = b.max > b.min ? (value - b.min) / (b.max - b.min) : 1;
          const classes = [value === b.max ? "best" : "", state.sort === c.key ? "sorted-col" : ""].filter(Boolean).join(" ");
          const tint = Math.round(6 + t * 30);
          return `<td class="${classes}" style="background: color-mix(in srgb, var(--accent) ${tint}%, transparent)" data-tip="score" data-bench="${c.key}">${fmt.score(value)}</td>`;
        }
      }
    }).join("");
    return `<tr class="model" data-id="${esc(m.id)}">${cells}</tr>`;
  }

  function radioOption(attrs, label, selected, count) {
    const classes = ["opt", selected ? "on" : "", count === 0 ? "zero" : ""].filter(Boolean).join(" ");
    const countHTML = count == null ? "" : `<span class="n">${count}</span>`;
    return `<button type="button" class="${classes}" ${attrs}><span class="mark">${selected ? ICONS.check : ""}</span><span class="label">${label}</span>${countHTML}</button>`;
  }

  function checkOption(attrs, label, selected, count) {
    const classes = ["opt", selected ? "on" : "", count === 0 ? "zero" : ""].filter(Boolean).join(" ");
    return `<button type="button" class="${classes}" ${attrs}><span class="box">${ICONS.check}</span><span class="label">${label}</span><span class="n">${count}</span></button>`;
  }

  function numericMenu(key) {
    const field = key === "dims" ? "dimensions" : "max_sequence";
    const title = key === "dims" ? "Embedding dimensions" : "Max sequence (tokens)";
    const current = state[key];
    const values = [...new Set(state.data.models.map((m) => m[field]))].sort((a, b) => a - b);
    const segment = ["ge", "eq", "le"].map((op) =>
      `<button type="button" class="${current.op === op ? "on" : ""}" data-op="${op}">${OP_WORDS[op]}</button>`).join("");
    const options = [radioOption('data-value=""', "Any", current.value == null, countWith({ [key]: { op: current.op, value: null } }))]
      .concat(values.map((v) => radioOption(
        `data-value="${v}"`, `${OPS[current.op]} ${fmt.int(v)}`, current.value === v,
        countWith({ [key]: { op: current.op, value: v } }),
      )));
    return `<div class="pop-title">${title}</div><div class="seg pop-seg">${segment}</div>${options.join("")}`;
  }

  function sizeMenu() {
    const options = [radioOption('data-value=""', "Any size", state.size == null, countWith({ size: null }))]
      .concat(SIZE_STEPS.map((v) => radioOption(`data-value="${v}"`, `Up to ${fmt.size(v)}`, state.size === v, countWith({ size: v }))));
    return `<div class="pop-title">Download size</div>${options.join("")}`;
  }

  function precisionMenu() {
    const natives = [...new Set(state.data.models.map((m) => m.precision.native))];
    const options = natives.map((p) => checkOption(
      `data-value="${esc(p)}"`, `${esc(p)} <span class="n">(${fmt.precision(p)})</span>`, state.precision.has(p),
      countWith({ precision: new Set([p]) }),
    ));
    return `<div class="pop-title">Native precision</div>${options.join("")}`;
  }

  function scoresMenu() {
    const options = state.data.benchmarks.map((b) => {
      const next = new Set(state.scores);
      next.add(b.key);
      return checkOption(`data-value="${b.key}"`, esc(b.title), state.scores.has(b.key), countWith({ scores: next }));
    });
    return `<div class="pop-title">Has a benchmark score for</div>${options.join("")}`;
  }

  function statusMenu() {
    const options = STATUS.map(([value, label]) =>
      radioOption(`data-value="${value}"`, label, state.status === value, countWith({ status: value })));
    return `<div class="pop-title">Download status</div>${options.join("")}`;
  }

  function sortMenu() {
    const options = new Map(sortOptions().map((o) => [o.key, o]));
    const sections = [
      ["vendor", "name"],
      state.data.benchmarks.map((b) => b.key),
      ["dimensions", "max_sequence", "size_mb", "parameters_m"],
    ];
    const html = sections.map((keys) => keys.map((k) =>
      radioOption(`data-sort="${k}"`, esc(options.get(k).label), state.sort === k, null)).join(""));
    return `<div class="pop-title">Sort by</div>${html.join('<div class="pop-sep"></div>')}`;
  }

  function renderPopover() {
    const builders = {
      sort: sortMenu, dims: () => numericMenu("dims"), ctx: () => numericMenu("ctx"),
      size: sizeMenu, precision: precisionMenu, scores: scoresMenu, status: statusMenu,
    };
    const popover = $("#popover");
    popover.innerHTML = builders[openMenu]();
    popover.classList.add("show");
  }

  function menuAnchor() {
    return openMenu === "sort" ? $("#sortButton") : document.querySelector(`[data-filter="${openMenu}"]`);
  }

  function positionPopover() {
    const anchor = menuAnchor();
    const popover = $("#popover");
    if (!anchor) return;
    const rect = anchor.getBoundingClientRect();
    const width = popover.offsetWidth;
    const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8));
    popover.style.left = `${left}px`;
    popover.style.top = `${rect.bottom + 6}px`;
    popover.style.maxHeight = `${Math.max(160, window.innerHeight - rect.bottom - 16)}px`;
    popover.style.overflowY = "auto";
  }

  function toggleMenu(key) {
    hideTip();
    if (openMenu === key) {
      closeMenu();
      return;
    }
    openMenu = key;
    renderFilters();
    renderSortButton();
    renderPopover();
    positionPopover();
  }

  function closeMenu() {
    if (!openMenu) return;
    openMenu = null;
    $("#popover").classList.remove("show");
    renderFilters();
    renderSortButton();
  }

  function onPopoverClick(event) {
    const key = openMenu;
    const opButton = event.target.closest("[data-op]");
    if (opButton) {
      state[key] = { ...state[key], op: opButton.dataset.op };
      refresh(true);
      renderPopover();
      return;
    }
    const sortButton = event.target.closest("[data-sort]");
    if (sortButton) {
      state.sort = sortButton.dataset.sort;
      closeMenu();
      refresh(true);
      return;
    }
    const option = event.target.closest(".opt[data-value]");
    if (!option) return;
    const raw = option.dataset.value;
    if (key === "dims" || key === "ctx") {
      state[key] = { ...state[key], value: raw === "" ? null : Number(raw) };
      closeMenu();
    } else if (key === "size") {
      state.size = raw === "" ? null : Number(raw);
      closeMenu();
    } else if (key === "status") {
      state.status = raw;
      closeMenu();
    } else {
      const set = new Set(state[key]);
      if (set.has(raw)) set.delete(raw);
      else set.add(raw);
      state[key] = set;
    }
    refresh(true);
    if (openMenu) renderPopover();
  }

  function modelFor(element) {
    const holder = element.closest("[data-id]");
    return holder ? state.byId.get(holder.dataset.id) : null;
  }

  function precisionTip(m) {
    const hw = state.data.hardware;
    const p = m.precision;
    const current = '<td class="tt-muted">← current</td>';
    const rows = [`<tr><td>Native</td><td>${esc(p.native)}</td><td></td></tr>`];
    if (hw.device === "cuda") {
      rows.push(`<tr class="${hw.half ? "" : "current"}"><td>Half off</td><td>${esc(p.half_off)}</td>${hw.half ? "<td></td>" : current}</tr>`);
      rows.push(`<tr class="${hw.half ? "current" : ""}"><td>Half on</td><td>${esc(p.half_on)}</td>${hw.half ? current : "<td></td>"}</tr>`);
      rows.push("<tr><td>CPU</td><td>float32</td><td></td></tr>");
    } else {
      rows.push(`<tr class="current"><td>CPU</td><td>float32</td>${current}</tr>`);
    }
    let where;
    if (hw.device === "cuda") {
      where = `For ${esc(hw.gpu_short || hw.gpu)} (compute ${esc(hw.cc)}). The Half checkbox is in the Settings tab.`;
    } else if (hw.cpu_only) {
      where = "No supported GPU, so every model runs in float32 and the Half checkbox is unavailable.";
    } else {
      where = "The Settings tab creates databases on the CPU, so every model runs in float32.";
    }
    const guard = p.fp16_guarded
      ? '<div class="tt-note">This model does not work correctly in float16, so on this GPU it runs in float32 even with Half on.</div>'
      : "";
    return `<div class="tt-title">Precision</div><table>${rows.join("")}</table><div class="tt-note">${where}</div>${guard}`;
  }

  function scoreTip(m, key) {
    const b = state.bench.get(key);
    const value = m.scores[key];
    if (value == null) {
      return `<div class="tt-title">${esc(b.title)}</div>No published score for this model.<div class="tt-note">${esc(b.about)}</div>`;
    }
    const rank = b.ranking.indexOf(m.id) + 1;
    const leader = state.byId.get(b.ranking[0]);
    const lead = rank === 1 ? "The best score of these models." : `Best: ${esc(leader.name)} (${b.max.toFixed(2)}).`;
    return `<div class="tt-title">${esc(b.title)}: ${value.toFixed(2)}</div>Rank ${rank} of ${b.count} models with a score. ${lead}<div class="tt-note">${esc(b.about)} Higher is better.</div>`;
  }

  function hardwareTip() {
    const hw = state.data.hardware;
    if (hw.cpu_only) {
      return '<div class="tt-title">CPU only</div>Models that need a GPU are hidden, and every model runs in float32.';
    }
    if (hw.device === "cpu") {
      return '<div class="tt-title">Creating databases on the CPU</div>The Settings tab is set to the CPU, so the precision chips show float32.';
    }
    return `<div class="tt-title">${esc(hw.gpu)}</div>Compute capability ${esc(hw.cc)} · Half is ${hw.half ? "on" : "off"}.<div class="tt-note">The precision chips show what each model will run in on this GPU with the current Half setting.</div>`;
  }

  function tipContent(element) {
    if (element.dataset.tipText) return esc(element.dataset.tipText);
    const kind = element.dataset.tip;
    if (kind === "hardware") return hardwareTip();
    const m = modelFor(element);
    if (!m) return "";
    switch (kind) {
      case "link":
        return `Open <b>huggingface.co/${esc(m.repo_id)}</b> in your browser`;
      case "license":
        return `<div class="tt-title">${esc(m.license.label)}</div>${esc(m.license.title)}`;
      case "dims":
        return `<div class="tt-title">${fmt.int(m.dimensions)} dimensions</div>Each chunk becomes a vector of ${fmt.int(m.dimensions)} numbers. More dimensions can capture more detail but take more space.`;
      case "ctx":
        return `<div class="tt-title">Max sequence: ${fmt.int(m.max_sequence)} tokens</div>The longest chunk the model reads at once; anything longer is cut off. A token is roughly 3-4 characters.`;
      case "params":
        return `<div class="tt-title">${fmt.params(m.parameters_m)} parameters</div>Bigger models are usually more accurate but slower.`;
      case "size":
        return `<div class="tt-title">${fmt.int(m.size_mb)} MB download</div>Saved to the Models/vector folder.`;
      case "precision":
        return precisionTip(m);
      case "gpu":
        return '<div class="tt-title">Needs an NVIDIA GPU</div>Hidden in CPU-only mode because it is far too slow on a CPU.';
      case "code":
        return '<div class="tt-title">Custom code</div>Runs model code from its repository, so its .py files are downloaded too.';
      case "score":
        return scoreTip(m, element.dataset.bench);
      case "download":
        if (state.data.downloading === m.id) {
          return '<div class="tt-title">Downloading</div>The files are being saved to the Models/vector folder.';
        }
        if (state.data.downloading) {
          return '<div class="tt-title">Please wait</div>Another model is downloading.';
        }
        return m.downloaded
          ? '<div class="tt-title">Downloaded</div>Click to download it again.'
          : `<div class="tt-title">Download</div>Saves the model to the Models/vector folder (${fmt.size(m.size_mb)}).`;
      default:
        return "";
    }
  }

  function showTip(element) {
    const html = tipContent(element);
    if (!html) return;
    const tip = $("#tooltip");
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
  }

  function hideTip() {
    clearTimeout(tipTimer);
    tipElement = null;
    $("#tooltip").classList.remove("show");
  }

  function requestDownload(id) {
    if (!bridge || state.data.downloading || !state.byId.has(id)) return;
    hideTip();
    bridge.requestDownload(id);
  }

  function setQuery(text) {
    $("#query").value = text;
    state.query = text.trim().toLowerCase();
    $(".search").classList.toggle("has-text", text.length > 0);
    refresh(true);
  }

  function bindEvents() {
    $("#query").addEventListener("input", (e) => setQuery(e.target.value));
    $("#clearQuery").addEventListener("click", (e) => {
      e.preventDefault();
      setQuery("");
      $("#query").focus();
    });
    $("#sortButton").addEventListener("click", () => toggleMenu("sort"));
    $("#viewToggle").addEventListener("click", (e) => {
      const button = e.target.closest("[data-view]");
      if (!button || button.dataset.view === state.view) return;
      state.view = button.dataset.view;
      renderViewToggle();
      refresh(true);
    });
    $("#filters").addEventListener("click", (e) => {
      if (e.target.closest("#clearFilters")) {
        resetFilters();
        closeMenu();
        refresh(true);
        return;
      }
      const pill = e.target.closest("[data-filter]");
      if (pill) toggleMenu(pill.dataset.filter);
    });
    $("#popover").addEventListener("click", onPopoverClick);
    $("#main").addEventListener("click", (e) => {
      const download = e.target.closest("button.dl");
      if (download) {
        requestDownload(download.closest("[data-id]").dataset.id);
        return;
      }
      const header = e.target.closest("th[data-sort]");
      if (header) {
        state.sort = header.dataset.sort;
        refresh(false);
        return;
      }
      if (e.target.closest("#resetAll")) {
        resetFilters();
        setQuery("");
      }
    });
    $("#main").addEventListener("scroll", hideTip, { passive: true });

    document.addEventListener("mousedown", (e) => {
      hideTip();
      if (!openMenu) return;
      if (e.target.closest("#popover, [data-filter], #sortButton")) return;
      closeMenu();
    });
    document.addEventListener("mouseover", (e) => {
      const element = e.target.closest("[data-tip], [data-tip-text]");
      if (element === tipElement) return;
      hideTip();
      if (!element || e.target.closest("#popover")) return;
      tipElement = element;
      tipTimer = setTimeout(() => showTip(element), 350);
    });
    document.documentElement.addEventListener("mouseleave", hideTip);
    document.addEventListener("keydown", (e) => {
      const query = $("#query");
      if (e.key === "Escape") {
        if (openMenu) closeMenu();
        else if (document.activeElement === query && query.value) setQuery("");
        return;
      }
      if ((e.ctrlKey && e.key.toLowerCase() === "f") || (e.key === "/" && document.activeElement !== query)) {
        e.preventDefault();
        query.focus();
        query.select();
      }
    });
    window.addEventListener("resize", () => {
      hideTip();
      if (openMenu) positionPopover();
    });
  }

  for (const slot of document.querySelectorAll("[data-icon]")) {
    slot.outerHTML = ICONS[slot.dataset.icon];
  }
  bindEvents();
  connectBridge();

  window.ModelsTab = {
    init(payload, theme) {
      applyTheme(theme);
      setData(payload);
    },
    applyTheme,
    setData(payload) {
      const top = $("#main").scrollTop;
      hideTip();
      setData(payload);
      $("#main").scrollTop = top;
      if (openMenu) renderPopover();
    },
  };
})();
