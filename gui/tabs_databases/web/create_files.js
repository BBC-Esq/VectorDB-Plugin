"use strict";

(function () {
  const { ICONS, esc, fmt, $ } = VDB;

  const ROW = 30;
  const OVERSCAN = 8;
  const HUES = { pdf: 4, word: 214, text: 190, web: 268, email: 36, sheet: 140, image: 320, transcript: 168, other: 220 };

  let rows = [];
  let version = -1;
  let loading = false;
  let shown = [];
  let filter = "all";
  let query = "";
  let selected = new Set();
  let anchor = null;
  let active = -1;
  let files = { count: 0, summary: {}, staging: null, notice: null };
  let kinds = [];
  let busy = false;
  let frame = 0;
  let headSignature = "";
  let footSignature = "";
  let listSignature = "";
  let lastOpen = { name: null, at: 0 };

  const list = () => $("#files-list");

  function ext(name) {
    const dot = name.lastIndexOf(".");
    return dot > 0 ? name.slice(dot + 1).toUpperCase() : "FILE";
  }

  function applyFilter() {
    const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    shown = rows.filter((r) => (filter === "all" || r.kind === filter)
      && terms.every((t) => r.name.toLowerCase().includes(t)));
    const visible = new Set(shown.map((r) => r.name));
    for (const name of [...selected]) if (!visible.has(name)) selected.delete(name);
    if (active >= shown.length) active = shown.length - 1;
  }

  function rowHTML(r, index) {
    const on = selected.has(r.name);
    const tip = r.target && r.target !== r.name ? ` data-tip-text="${esc(r.target)}"` : "";
    return `<div class="file-row${on ? " on" : ""}${index === active ? " active" : ""}" role="option" aria-selected="${on}" data-index="${index}" style="top:${index * ROW}px"${tip}>`
      + `<span class="file-kind" style="--h:${HUES[r.kind] ?? 220}">${esc(ext(r.name))}</span>`
      + `<span class="file-label">${esc(r.name)}</span>`
      + (busy ? "" : `<button type="button" class="file-remove" data-remove="${index}" data-tip-text="Remove from the list" tabindex="-1">${ICONS.close}</button>`)
      + "</div>";
  }

  function paint() {
    frame = 0;
    const el = list();
    if (!el) return;
    el.querySelector(".file-spacer").style.height = `${shown.length * ROW}px`;
    const first = Math.max(0, Math.floor(el.scrollTop / ROW) - OVERSCAN);
    const last = Math.min(shown.length, Math.ceil((el.scrollTop + el.clientHeight) / ROW) + OVERSCAN);
    let html = "";
    for (let i = first; i < last; i += 1) html += rowHTML(shown[i], i);
    el.querySelector(".file-rows").innerHTML = html;
    const empty = el.querySelector(".file-empty");
    if (!rows.length && !loading) {
      empty.innerHTML = '<div class="empty-title">No files yet</div>'
        + "<div>Add files or a whole folder. They are listed here until you create the database.</div>";
      empty.hidden = false;
    } else if (!shown.length && rows.length) {
      empty.innerHTML = '<div class="empty-title">No matching files</div>';
      empty.hidden = false;
    } else {
      empty.hidden = true;
    }
  }

  function schedulePaint() {
    if (!frame) frame = requestAnimationFrame(paint);
  }

  function chipHTML(key, label, count) {
    const on = filter === key;
    return `<button type="button" class="kind-chip${on ? " on" : ""}" data-filter="${esc(key)}"${count ? "" : " disabled"}>`
      + `${esc(label)}<span class="n">${fmt.int(count)}</span></button>`;
  }

  function renderHead() {
    const summary = files.summary || {};
    const chips = kinds.filter((k) => summary[k.key]).map((k) => chipHTML(k.key, k.label, summary[k.key])).join("");
    const signature = JSON.stringify([files.count, summary, filter, busy, Boolean(files.staging), kinds.length]);
    if (signature === headSignature) return;
    headSignature = signature;
    const disabled = busy || Boolean(files.staging);
    const searchValue = query;
    const hadFocus = document.activeElement && document.activeElement.id === "file-search";
    $("#files-head").innerHTML = '<div class="files-title-row">'
      + `<h2 class="section-title">${ICONS.file}<span>Files to Add</span><span class="title-count">${fmt.int(files.count)} file${files.count === 1 ? "" : "s"}</span></h2>`
      + `<button type="button" class="btn small" data-act="add_files"${disabled ? " disabled" : ""}>${ICONS.file}Add Files</button>`
      + `<button type="button" class="btn small" data-act="add_folder"${disabled ? " disabled" : ""}>${ICONS.folder}Add Folder</button>`
      + "</div>"
      + (files.count
        ? '<div class="files-tools">'
          + `<div class="kind-chips">${chipHTML("all", "All", files.count)}${chips}</div>`
          + `<label class="search file-search${searchValue ? " has-text" : ""}">${ICONS.search}<input id="file-search" type="text" placeholder="Filter by name" value="${esc(searchValue)}" autocomplete="off" spellcheck="false"><button type="button" class="clear" data-clear-search data-tip-text="Clear">${ICONS.close}</button></label>`
          + "</div>"
        : "");
    if (hadFocus) {
      const input = $("#file-search");
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
    }
  }

  function noticeHTML(notice) {
    if (!notice) return "";
    const icon = { ok: ICONS.check, warn: ICONS.warn, error: ICONS.error, info: ICONS.info }[notice.kind] || ICONS.info;
    let more = "";
    if (notice.skipped && notice.skipped.length) {
      const names = notice.skipped.slice(0, 6).map(esc).join(", ") + (notice.skipped.length > 6 ? ", …" : "");
      more += `<div class="status-note">Skipped because of their file type: ${names}. Audio files can be transcribed on the `
        + '<button type="button" class="link" data-tab="Tools">Tools tab</button>.</div>';
    }
    if (notice.error_count || (notice.errors && notice.errors.length)) {
      const count = notice.error_count || notice.errors.length;
      more += `<details class="notice-details"><summary>${fmt.int(count)} file${count === 1 ? "" : "s"} could not be added</summary>`
        + `<div class="notice-list">${notice.errors.map((e) => `<div>${esc(e)}</div>`).join("")}</div></details>`;
    }
    return `<div class="status ${notice.kind === "ok" ? "ok" : notice.kind}">${icon}<div class="status-body">${esc(notice.message)}${more}</div>`
      + `<button type="button" class="icon-btn btn ghost small dismiss" data-act="dismiss_notice" data-tip-text="Dismiss">${ICONS.close}</button></div>`;
  }

  function renderFoot() {
    const count = selected.size;
    const signature = JSON.stringify([count, files.staging, files.notice, busy, shown.length, rows.length]);
    if (signature === footSignature) return;
    footSignature = signature;
    let html = "";
    if (files.staging) {
      const s = files.staging;
      html += '<div class="staging">'
        + `<div class="progress${s.percent ? "" : " indeterminate"}"><i style="width:${s.percent || 0}%"></i></div>`
        + `<div class="staging-row">${VDB.ICONS.info}<span>${s.cancelling ? "Stopping…" : `Adding ${fmt.int(s.total)} file${s.total === 1 ? "" : "s"} to the list…`}</span>`
        + `<span class="staging-pct">${s.percent ? `${s.percent}%` : ""}</span>`
        + `<button type="button" class="btn ghost small" data-act="cancel_staging"${s.cancelling ? " disabled" : ""}>${ICONS.stop}Cancel</button></div></div>`;
    }
    html += noticeHTML(files.notice);
    if (count) {
      html += '<div class="selection-row">'
        + `<span>${fmt.int(count)} selected</span>`
        + `<button type="button" class="btn ghost small" data-act="clear_selection">Clear selection</button>`
        + `<button type="button" class="btn danger small" data-act="remove_selected"${busy ? " disabled" : ""}>${ICONS.close}Remove ${fmt.int(count)} file${count === 1 ? "" : "s"}</button>`
        + "</div>";
    } else if (busy) {
      html += '<div class="hint-row">The list is locked while the database is being created.</div>';
    } else if (rows.length && !files.staging && !files.notice) {
      html += '<div class="hint-row">Double-click a file to open it. Select files and press Delete to remove them from the list.</div>';
    }
    $("#files-foot").innerHTML = html;
  }

  function refreshAll() {
    applyFilter();
    renderHead();
    renderFoot();
    const signature = JSON.stringify([version, filter, query, busy, shown.length, rows.length]);
    if (signature !== listSignature) {
      listSignature = signature;
      schedulePaint();
    }
  }

  function loadRows() {
    if (loading) return;
    loading = true;
    VDB.call("files").then((result) => {
      loading = false;
      rows = result.rows || [];
      version = result.version;
      const names = new Set(rows.map((r) => r.name));
      for (const name of [...selected]) if (!names.has(name)) selected.delete(name);
      refreshAll();
      if (files.version > version) loadRows();
    });
  }

  function update(nextFiles, nextKinds, nextBusy) {
    files = nextFiles;
    kinds = nextKinds;
    if (nextBusy !== busy) {
      $("#files-card").classList.toggle("locked", nextBusy);
      footSignature = "";
    }
    busy = nextBusy;
    if (files.version > version) loadRows();
    refreshAll();
  }

  function selectIndex(index, e) {
    const r = shown[index];
    if (!r) return;
    if (e && e.shiftKey && anchor != null) {
      const from = Math.min(anchor, index);
      const to = Math.max(anchor, index);
      if (!e.ctrlKey) selected.clear();
      for (let i = from; i <= to; i += 1) selected.add(shown[i].name);
    } else if (e && e.ctrlKey) {
      if (selected.has(r.name)) selected.delete(r.name);
      else selected.add(r.name);
      anchor = index;
    } else {
      selected = new Set([r.name]);
      anchor = index;
    }
    active = index;
    renderFoot();
    updateRows();
  }

  function updateRows() {
    for (const row of list().querySelectorAll(".file-row")) {
      const index = Number(row.dataset.index);
      const r = shown[index];
      const on = Boolean(r && selected.has(r.name));
      row.classList.toggle("on", on);
      row.classList.toggle("active", index === active);
      row.setAttribute("aria-selected", String(on));
    }
  }

  function openIndex(index) {
    const r = shown[index];
    if (!r) return;
    const now = Date.now();
    if (lastOpen.name === r.name && now - lastOpen.at < 700) return;
    lastOpen = { name: r.name, at: now };
    VDB.call("open", { name: r.name });
  }

  function ensureVisible(index) {
    const el = list();
    const top = index * ROW;
    if (top < el.scrollTop) el.scrollTop = top;
    else if (top + ROW > el.scrollTop + el.clientHeight) el.scrollTop = top + ROW - el.clientHeight;
  }

  function removeNames(names) {
    if (!names.length || busy) return;
    VDB.call("remove", { names }).then(() => {
      for (const name of names) selected.delete(name);
      renderFoot();
    });
  }

  function bind() {
    const el = list();
    el.addEventListener("scroll", schedulePaint);
    window.addEventListener("resize", schedulePaint);
    el.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || e.target.closest(".file-remove")) return;
      const row = e.target.closest(".file-row");
      if (!row) return;
      e.preventDefault();
      el.focus();
      const index = Number(row.dataset.index);
      if (e.detail === 2 && !e.shiftKey && !e.ctrlKey) {
        openIndex(index);
        return;
      }
      selectIndex(index, e);
    });
    el.addEventListener("click", (e) => {
      const remove = e.target.closest("[data-remove]");
      if (remove) {
        const r = shown[Number(remove.dataset.remove)];
        if (r) removeNames([r.name]);
      }
    });
    el.addEventListener("keydown", (e) => {
      if (!shown.length) return;
      if ((e.key === "a" || e.key === "A") && e.ctrlKey) {
        e.preventDefault();
        selected = new Set(shown.map((r) => r.name));
        renderFoot();
        updateRows();
      } else if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        removeNames([...selected]);
      } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const next = Math.max(0, Math.min(shown.length - 1, (active < 0 ? -1 : active) + (e.key === "ArrowDown" ? 1 : -1)));
        selectIndex(next, { shiftKey: e.shiftKey, ctrlKey: false });
        ensureVisible(next);
      } else if (e.key === "Enter" && active >= 0) {
        e.preventDefault();
        openIndex(active);
      } else if (e.key === "Escape") {
        selected.clear();
        renderFoot();
        updateRows();
      }
    });
    $("#files-card").addEventListener("input", (e) => {
      if (e.target.id !== "file-search") return;
      query = e.target.value;
      e.target.parentElement.classList.toggle("has-text", query.length > 0);
      list().scrollTop = 0;
      refreshAll();
    });
    $("#files-card").addEventListener("click", (e) => {
      const chip = e.target.closest("[data-filter]");
      if (chip && !chip.disabled) {
        filter = chip.dataset.filter;
        headSignature = "";
        refreshAll();
        list().scrollTop = 0;
        return;
      }
      if (e.target.closest("[data-clear-search]")) {
        query = "";
        headSignature = "";
        refreshAll();
        return;
      }
      const act = e.target.closest("[data-act]");
      if (!act || act.disabled) return;
      const action = act.dataset.act;
      if (action === "clear_selection") {
        selected.clear();
        renderFoot();
        updateRows();
      } else if (action === "remove_selected") {
        removeNames([...selected]);
      }
    });
  }

  window.CreateFiles = {
    init: bind,
    update,
    selectedCount: () => selected.size,
  };
})();
