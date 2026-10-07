"use strict";

(function () {
  const { esc } = VDB;
  const { html } = Tools;

  const MODEL_TIP = "Distil models use about 70% of the memory of their full-size versions with little loss in quality. "
    + "Whisper large-v3 turbo is much faster than large-v3 with a small loss in quality.";
  const BATCH_TIP = "How many audio segments are transcribed at once. Higher is faster but uses more memory. "
    + "The User Guide lists suggested values.";
  const PRECISION_TIPS = {
    bfloat16: "Needs an NVIDIA GPU with compute capability 8.0 or newer",
    float16: "Needs a supported NVIDIA GPU",
  };

  let draft = null;
  let error = null;

  function batchHTML(s) {
    const text = draft ?? String(s.batch);
    return `<label class="input${error ? " invalid" : ""}${s.running ? " disabled" : ""}"><input type="text" inputmode="numeric" data-key="transcribe.batch" value="${esc(text)}" autocomplete="off" spellcheck="false"${s.running ? " disabled" : ""}></label>`;
  }

  function statusHTML(s) {
    if (s.running) {
      return html.status("run", `Transcribing <b>${esc(s.file.name)}</b> with ${esc(s.model)} (${esc(s.precision)}) ${html.since(s.started)}`);
    }
    if (s.result) return html.status(s.result.ok ? "ok" : "error", esc(s.result.message));
    return "";
  }

  function render(s) {
    const precisions = s.precisions.map((p) => ({
      value: p.value,
      label: p.value,
      disabled: !p.available,
      tip: p.available ? "" : PRECISION_TIPS[p.value],
    }));
    const message = error ? `<div class="field-error">${esc(error)}</div>` : "";
    return html.header("mic", "Transcribe Audio", "Turns a recording into text that your next database will include")
      + '<div class="line">'
      + html.field("Model", VDB.selectButton("transcribe.model", { label: s.model || "No models available" }, s.running), { tip: MODEL_TIP, cls: "grow" })
      + html.field("Precision", html.segmented("transcribe_precision", precisions, s.precision, s.running), { cls: "fixed precision" })
      + html.field("Batch size", batchHTML(s), { tip: BATCH_TIP, cls: "fixed batch", message })
      + "</div>"
      + '<div class="line">'
      + html.file("transcribe_choose", s.file, "Choose an audio file…", { icon: "file", disabled: s.running })
      + html.action("transcribe_start", "Transcribe", { disabled: s.running || !s.file || !s.model })
      + "</div>"
      + statusHTML(s);
  }

  function rerender() {
    Tools.render("transcribe", true);
  }

  function commitBatch(input, s) {
    if (draft == null) return;
    const text = draft.trim();
    if (text === "" || text === String(s.batch)) {
      draft = null;
      error = null;
      rerender();
      return;
    }
    if (!/^\d+$/.test(text) || Number(text) < 1 || Number(text) > 150) {
      error = "Use a whole number from 1 to 150.";
      rerender();
      return;
    }
    Tools.call("transcribe_batch", { value: Number(text) }).then((result) => {
      if (result.error) error = result.error;
      else {
        draft = null;
        error = null;
      }
      rerender();
    });
  }

  Tools.register("transcribe", {
    render,
    openSelect(id, s) {
      VDB.openSelect(id, {
        title: "Whisper model",
        options: s.models.map((name) => ({ value: name, label: name })),
        value: s.model,
        onPick: (value) => {
          if (value !== s.model) Tools.call("transcribe_model", { value });
        },
      });
    },
    actions: {
      transcribe_precision(button, s) {
        if (button.dataset.arg !== s.precision) Tools.call("transcribe_precision", { value: button.dataset.arg });
      },
    },
    onInput(input) {
      draft = input.value;
      const text = draft.trim();
      error = text === "" || (/^\d+$/.test(text) && Number(text) >= 1 && Number(text) <= 150) ? null : "Use a whole number from 1 to 150.";
      const field = input.closest(".field");
      field.querySelector(".input").classList.toggle("invalid", Boolean(error));
      const old = field.querySelector(".field-error");
      if (old) old.remove();
      if (error) field.insertAdjacentHTML("beforeend", `<div class="field-error">${esc(error)}</div>`);
    },
    onKey(input, e, s) {
      if (e.key === "Enter") {
        e.preventDefault();
        commitBatch(input, s);
      } else if (e.key === "Escape") {
        e.preventDefault();
        draft = null;
        error = null;
        rerender();
      } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        const base = /^\d+$/.test(input.value.trim()) ? Number(input.value.trim()) : s.batch;
        const next = Math.min(150, Math.max(1, base + (e.key === "ArrowUp" ? 1 : -1)));
        input.value = String(next);
        this.onInput(input);
      }
    },
    onBlur(input, e, s) {
      if (input.isConnected) commitBatch(input, s);
    },
  });
})();
