"use strict";

(function () {
  const { esc, fmt } = VDB;
  const { html } = Tools;

  const ENGINE_TIP = "RapidOCR is the default and also reports quality notes when it finishes. Tesseract is an alternative engine.";

  function pagesText(n) {
    return `${fmt.int(n)} page${n === 1 ? "" : "s"}`;
  }

  function runningHTML(s) {
    const p = s.progress || { done: 0, total: 0 };
    const known = p.total > 0 && p.done > 0;
    const width = known ? Math.min(100, (p.done / p.total) * 100) : 0;
    const bar = known
      ? `<div class="progress"><i style="width:${width.toFixed(1)}%"></i></div>`
      : '<div class="progress indeterminate"><i></i></div>';
    const count = known ? `Page <b>${fmt.int(p.done)}</b> of ${fmt.int(p.total)}` : "Preparing pages…";
    return `<div class="run-block">${bar}${html.status("run", `Reading <b>${esc(s.file.name)}</b> · ${count} ${html.since(s.started)}`)}</div>`;
  }

  function resultHTML(s) {
    const r = s.result;
    if (!r) return "";
    if (!r.ok) return html.status("error", esc(r.message));
    const saved = r.output
      ? `Saved as <b>${esc(r.output_name)}</b> next to the original. ${html.link("ocr_open", "Open PDF")}`
      : `The output file ${esc(r.output_name)} was not found.`;
    const notes = [
      ...r.warnings.map((w) => html.status("warn", esc(w))),
      ...r.infos.map((i) => html.status("info", esc(i))),
    ].join("");
    return html.status("ok", `Finished in ${esc(r.time)}. ${saved}`) + (notes ? `<div class="notes">${notes}</div>` : "");
  }

  function render(s) {
    const engines = s.engines.map((e) => ({ value: e.value, label: e.label }));
    const note = s.file && s.file.pages ? pagesText(s.file.pages) : "";
    return html.header("scan", "Optical Character Recognition", "Adds a searchable text layer to scanned PDFs")
      + '<div class="line">'
      + html.field("Engine", html.segmented("ocr_engine", engines, s.engine, s.running), { tip: ENGINE_TIP, cls: "fixed engine" })
      + html.field("Scanned PDF", html.file("ocr_choose", s.file, "Choose a PDF…", { icon: "file", note, disabled: s.running }), { cls: "grow" })
      + html.action("ocr_start", "Run OCR", { icon: "scan", disabled: s.running || !s.file })
      + "</div>"
      + (s.running ? runningHTML(s) : resultHTML(s));
  }

  Tools.register("ocr", {
    render,
    actions: {
      ocr_engine(button, s) {
        if (button.dataset.arg !== s.engine) Tools.call("ocr_engine", { value: button.dataset.arg });
      },
    },
  });
})();
