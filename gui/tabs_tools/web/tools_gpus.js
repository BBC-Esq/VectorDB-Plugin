"use strict";

(function () {
  const { ICONS, esc, fmt } = VDB;
  const { html } = Tools;

  const SORTS = {
    cores: { label: "CUDA cores", key: (r) => r.cores, descending: true },
    vram: { label: "VRAM", key: (r) => r.vram, descending: true },
    cc: { label: "Compute capability", key: (r) => Number(r.cc), descending: true },
    name: { label: "Name", key: (r) => r.name.toLowerCase(), descending: false },
  };

  let range = { min: 8, max: 16 };
  let error = null;
  let rows = [];
  let filter = "";
  let sort = "cores";
  let local = null;

  function sizeOption(gb) {
    return { value: gb, label: `${gb} GB` };
  }

  function render(g) {
    local = g.local;
    const message = error ? `<div class="field-error">${esc(error)}</div>` : "";
    const aside = g.local ? `Yours: <b>${esc(g.local)}</b>` : "";
    return html.header("chart", "Compare GPUs", "", aside)
      + '<div class="line">'
      + html.field("Least VRAM", VDB.selectButton("gpus.min", sizeOption(range.min)), { cls: "grow" })
      + html.field("Most VRAM", VDB.selectButton("gpus.max", sizeOption(range.max)), { cls: "grow" })
      + html.action("gpus_open", "Compare", { icon: "chart" })
      + "</div>"
      + message;
  }

  function sorted() {
    const terms = filter.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const spec = SORTS[sort];
    const shown = rows.filter((r) => terms.every((t) => `${r.name} ${r.arch}`.toLowerCase().includes(t)));
    return shown.sort((a, b) => {
      const x = spec.key(a);
      const y = spec.key(b);
      if (x === y) return b.cores - a.cores;
      return (x < y ? -1 : 1) * (spec.descending ? -1 : 1);
    });
  }

  function listHTML() {
    const top = Math.max(1, ...rows.map((r) => r.cores));
    const shown = sorted();
    if (!shown.length) return '<div class="pop-empty">No GPUs match.</div>';
    return shown.map((r) => {
      const mine = local && r.name.toLowerCase() === local.toLowerCase();
      const pct = (r.cores / top) * 100;
      return `<div class="gpu-row${mine ? " mine" : ""}">`
        + `<span class="gpu-name">${esc(r.name)}${mine ? ' <span class="chip">Your GPU</span>' : ""}</span>`
        + `<span class="gpu-arch">${esc(r.arch)}${r.released ? ` <span class="gpu-sub">${esc(r.released)}</span>` : ""}</span>`
        + `<span class="gpu-cc">${esc(r.cc)}</span>`
        + `<span class="gpu-vram">${fmt.int(r.vram)} GB <span class="gpu-sub">${esc(r.memory)}</span></span>`
        + `<span class="gpu-cores"><span class="gpu-bar"><i style="width:${pct.toFixed(1)}%"></i></span><span class="gpu-num">${fmt.int(r.cores)}</span></span>`
        + "</div>";
    }).join("");
  }

  function headHTML() {
    const cell = (key, label) => {
      const on = sort === key;
      return `<button type="button" class="gpu-sort${on ? " on" : ""}" data-sort="${key}">${esc(label)}${on ? ICONS.caret : ""}</button>`;
    };
    return `<div class="gpu-row head">${cell("name", "GPU")}<span>Architecture</span>${cell("cc", "Compute")}${cell("vram", "VRAM")}${cell("cores", "CUDA cores")}</div>`;
  }

  function countText() {
    const shown = sorted().length;
    return shown === rows.length ? `${fmt.int(rows.length)} GPUs` : `${fmt.int(shown)} of ${fmt.int(rows.length)} GPUs`;
  }

  function refreshList(panel) {
    panel.querySelector(".gpu-head").innerHTML = headHTML();
    panel.querySelector(".gpu-list").innerHTML = listHTML();
    panel.querySelector(".gpu-count").textContent = countText();
  }

  function openOverlay() {
    filter = "";
    sort = "cores";
    const body = '<div class="gpu-tools">'
      + `<label class="pop-search gpu-search">${ICONS.search}<input type="text" data-gpu-search placeholder="Filter, for example 4090 or Ada" autocomplete="off" spellcheck="false"></label>`
      + `<span class="gpu-count">${countText()}</span></div>`
      + '<div class="gpu-note">Bars compare CUDA cores. bfloat16 needs compute capability 8.0 or newer (Ampere and later).</div>'
      + `<div class="gpu-head">${headHTML()}</div><div class="gpu-list">${listHTML()}</div>`;
    VDB.overlay.open(`GPUs with ${range.min}–${range.max} GB of VRAM`, body);
    const panel = document.querySelector("#overlay .overlay-panel");
    const input = panel.querySelector("[data-gpu-search]");
    input.addEventListener("input", () => {
      filter = input.value;
      refreshList(panel);
    });
    panel.addEventListener("click", (e) => {
      const button = e.target.closest("[data-sort]");
      if (!button) return;
      sort = button.dataset.sort;
      refreshList(panel);
    });
    input.focus();
  }

  Tools.register("gpus", {
    select: (state) => state.gpu,
    render,
    openSelect(id, g) {
      const which = id === "gpus.min" ? "min" : "max";
      VDB.openSelect(id, {
        title: which === "min" ? "Least VRAM" : "Most VRAM",
        options: g.sizes.map(sizeOption),
        value: range[which],
        onPick: (value) => {
          range = { ...range, [which]: value };
          error = range.min > range.max ? "Minimum V-RAM value cannot exceed maximum V-RAM value." : null;
          Tools.render("gpus", true);
        },
      });
    },
    actions: {
      gpus_open() {
        if (range.min > range.max) {
          error = "Minimum V-RAM value cannot exceed maximum V-RAM value.";
          Tools.render("gpus", true);
          return;
        }
        Tools.call("gpus", { min_vram: range.min, max_vram: range.max }).then((result) => {
          if (result.error) {
            error = result.error;
            Tools.render("gpus", true);
            return;
          }
          error = null;
          rows = result.rows || [];
          openOverlay();
        });
      },
    },
  });
})();
