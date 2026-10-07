"use strict";

(function () {
  const { ICONS, esc, fmt } = VDB;
  const { html } = Tools;

  const IMAGES_TIP = "Describes the images you added on the Create Database tab with the vision model chosen on the Settings tab, "
    + "so you can check the descriptions before creating the database.";
  const COMPARE_TIP = "Describes one image with each selected model, one model at a time, and saves a comparison of their "
    + "descriptions, lengths and speeds. Larger models can take several minutes.";

  let picked = null;

  function selectedCount(s) {
    return (picked || s.selected).length;
  }

  function modelsLabel(s) {
    const available = s.models.filter((m) => m.available).length;
    const count = selectedCount(s);
    if (!count) return "No models";
    if (count === available) return available === 1 ? "1 model" : `All ${available} models`;
    return `${count} of ${available} models`;
  }

  function cellHTML(p, row) {
    const icon = {
      pending: ICONS.play,
      running: '<span class="spinner"></span>',
      done: ICONS.check,
      failed: ICONS.error,
      skipped: ICONS.stop,
    }[p.status];
    let metric = "waiting";
    if (p.status === "running") metric = html.since(p.since);
    else if (p.status === "done" && row) metric = `${fmt.int(row.chars)} characters · ${row.seconds.toFixed(1)} s`;
    else if (p.status === "done") metric = `${p.seconds.toFixed(1)} s`;
    else if (p.status === "failed") metric = "failed";
    else if (p.status === "skipped") metric = "skipped";
    return `<div class="model-cell ${p.status}">${icon}<span class="model-name">${esc(p.name)}</span><span class="model-metric">${metric}</span></div>`;
  }

  function gridHTML(progress, rows) {
    const byName = Object.fromEntries((rows || []).map((r) => [r.name, r]));
    return `<div class="model-grid">${progress.map((p) => cellHTML(p, byName[p.name])).join("")}</div>`;
  }

  function resultHTML(s) {
    const r = s.result;
    if (!r) return "";
    if (!r.ok) return html.status("error", esc(r.message));
    const open = html.link("vision_open", "Open results");
    const openError = r.open_error ? `<div class="status-note error">${esc(r.open_error)}</div>` : "";
    if (r.kind === "summarize") {
      const count = `${fmt.int(r.count)} image${r.count === 1 ? "" : "s"}`;
      return html.status("ok", `Described <b>${count}</b> with ${esc(r.model)} in ${r.seconds.toFixed(1)} s. `
        + `Average ${fmt.int(r.average)} characters. ${open}`
        + `<div class="status-note">The longest description has ${fmt.int(r.longest)} characters. Keep the chunk size above this so each description stays in one chunk.</div>${openError}`);
    }
    const failed = r.rows.filter((row) => row.failed).length;
    const head = r.cancelled ? "Comparison cancelled." : "Comparison finished.";
    const note = failed ? `<div class="status-note error">${failed} model${failed === 1 ? "" : "s"} failed. The results file shows the error.</div>` : "";
    return html.status("ok", `${head} ${open}${note}${openError}`) + gridHTML(s.progress, r.rows);
  }

  function runningHTML(s) {
    if (s.task === "summarize") {
      const count = `${fmt.int(s.images)} image${s.images === 1 ? "" : "s"}`;
      return html.status("run", `Describing <b>${count}</b> with ${esc(s.chosen)} ${html.since(s.started)}`);
    }
    if (s.task === "compare") {
      const cancel = s.cancelling
        ? '<span class="status-note">Stopping after the current model…</span>'
        : `<button type="button" class="btn ghost small" data-act="vision_cancel">${ICONS.stop}Cancel</button>`;
      return `<div class="run-line">${html.status("run", `Comparing ${s.progress.length} model${s.progress.length === 1 ? "" : "s"} on <b>${esc(s.file.name)}</b> ${html.since(s.started)}`)}${cancel}</div>`
        + gridHTML(s.progress, null);
    }
    return "";
  }

  function render(s) {
    const busy = Boolean(s.task);
    const count = `<b>${fmt.int(s.images)}</b> image${s.images === 1 ? "" : "s"}`;
    const imagesText = s.images
      ? `${count} on the Create Database tab · <b>${esc(s.chosen)}</b>`
      : `No images on the Create Database tab yet · <b>${esc(s.chosen)}</b>`;
    const summarizeTip = s.images ? "" : "Add images on the Create Database tab first";
    return html.header("eye", "Test Vision Models", "Check image descriptions before adding images to a database")
      + '<div class="line">'
      + html.field("Your images and chosen model", `<div class="stat-box">${imagesText}</div>`, { tip: IMAGES_TIP, cls: "grow" })
      + html.action("vision_summarize", "Describe", { icon: "image", disabled: busy || !s.images, tip: summarizeTip })
      + "</div>"
      + '<div class="line">'
      + html.field("One image, several models", html.file("vision_choose", s.file, "Choose an image…", { icon: "image", disabled: busy }), { tip: COMPARE_TIP, cls: "grow" })
      + html.field("Models", VDB.selectButton("vision.models", { label: modelsLabel(s) }, busy), { cls: "fixed models" })
      + html.action("vision_compare", "Compare", { icon: "chart", disabled: busy || !s.file || !s.selected.length })
      + "</div>"
      + (busy ? runningHTML(s) : resultHTML(s));
  }

  function popoverHTML(s) {
    const chosen = new Set(picked || s.selected);
    const options = s.models.map((m) => {
      const on = chosen.has(m.name) && m.available;
      return `<button type="button" class="opt${on ? " on" : ""}" data-model="${esc(m.name)}"${m.available ? "" : " disabled"}>`
        + `<span class="box">${ICONS.check}</span><span class="label">${esc(m.name)}</span>`
        + `<span class="n">${esc(m.available ? m.vram : "requires GPU")}</span></button>`;
    }).join("");
    return '<div class="pop-title">Models to compare</div>'
      + '<div class="pop-actions"><button type="button" class="btn ghost small" data-pick="all">Select all</button>'
      + '<button type="button" class="btn ghost small" data-pick="none">Clear</button></div>'
      + `<div class="pop-options">${options}</div>`;
  }

  function commitPicked() {
    if (!picked) return;
    const names = picked;
    Tools.call("vision_models", { names });
  }

  Tools.register("vision", {
    render,
    openSelect(id, s) {
      const anchor = () => document.querySelector('[data-select="vision.models"]');
      VDB.popover.toggle("vision.models", {
        anchor,
        matchWidth: false,
        render: () => popoverHTML(Tools.data("vision")),
        click: (e) => {
          const current = Tools.data("vision");
          const list = new Set(picked || current.selected);
          const pick = e.target.closest("[data-pick]");
          const option = e.target.closest(".opt[data-model]");
          if (pick) {
            if (pick.dataset.pick === "all") current.models.filter((m) => m.available).forEach((m) => list.add(m.name));
            else list.clear();
          } else if (option && !option.disabled) {
            const name = option.dataset.model;
            if (list.has(name)) list.delete(name);
            else list.add(name);
          } else {
            return;
          }
          picked = current.models.map((m) => m.name).filter((name) => list.has(name));
          VDB.popover.render();
          const button = anchor();
          if (button) button.querySelector(".select-text").textContent = modelsLabel(current);
          commitPicked();
        },
        onOpen: () => {
          picked = null;
          const button = anchor();
          if (button) button.classList.add("open");
        },
        onClose: () => {
          picked = null;
          const button = anchor();
          if (button) button.classList.remove("open");
          Tools.render("vision", true);
        },
      });
    },
  });
})();
