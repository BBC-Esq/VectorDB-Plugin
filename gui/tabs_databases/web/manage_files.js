"use strict";

(function () {
  const { ICONS, esc, fmt, $ } = VDB;

  const ROW = 30;
  const OVERSCAN = 8;
  const HUES = { pdf: 4, word: 214, text: 190, web: 268, email: 36, sheet: 140, image: 320, audio: 168, other: 220 };

  let database = null;
  let version = -1;
  let files = null;
  let kinds = [];
  let rows = [];
  let dirs = [];
  let missing = new Set();
  let checked = false;
  let loading = false;
  let shown = [];
  let filter = "all";
  let query = "";
  let active = -1;
  let frame = 0;
  let headSignature = "";
  let footSignature = "";
  let listSignature = "";
  let lastOpen = { index: null, at: 0 };
  let message = null;
  let missingFor = -1;

  const list = () => $("#files-list");

  function ext(name) {
    const dot = name.lastIndexOf(".");
    return dot > 0 ? name.slice(dot + 1).toUpperCase() : "FILE";
  }

  function pathOf(row) {
    if (row[3] < 0) return row[4] || "";
    const dir = dirs[row[3]] || "";
    const sep = dir.includes("\\") ? "\\" : "/";
    return dir.endsWith(sep) ? dir + row[0] : dir + sep + row[0];
  }

  function applyFilter() {
    const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    shown = [];
    for (let i = 0; i < rows.length; i += 1) {
      const r = rows[i];
      if (filter === "missing" ? !missing.has(i) : filter !== "all" && r[1] !== filter) continue;
      const name = r[0].toLowerCase();
      if (terms.every((t) => name.includes(t))) shown.push(i);
    }
    if (active >= shown.length) active = shown.length - 1;
  }

  function rowHTML(position) {
    const index = shown[position];
    const r = rows[index];
    const gone = missing.has(index);
    const chunks = `${fmt.int(r[2])} chunk${r[2] === 1 ? "" : "s"}`;
    return `<div class="file-row${position === active ? " active on" : ""}${gone ? " gone" : ""}" role="option" aria-selected="${position === active}" data-position="${position}" style="top:${position * ROW}px" data-tip-text="${esc(pathOf(r))}">`
      + `<span class="file-kind" style="--h:${HUES[r[1]] ?? 220}">${esc(ext(r[0]))}</span>`
      + `<span class="file-label">${esc(r[0])}</span>`
      + (gone ? '<span class="file-gone">Not found</span>' : "")
      + `<span class="file-chunks">${chunks}</span>`
      + `<button type="button" class="file-action" data-reveal="${position}" data-tip-text="Show in folder" tabindex="-1">${ICONS.folder}</button>`
      + "</div>";
  }

  function emptyHTML() {
    if (!files) return "";
    if (files.status === "loading" || loading) return '<div class="file-wait"><span class="spinner"></span><span>Reading the file list…</span></div>';
    if (files.status === "error") return `<div class="empty-title">The file list could not be read</div><div>${esc(files.error || "")}</div>`;
    if (!rows.length) return '<div class="empty-title">This database has no files</div>';
    if (!shown.length) return '<div class="empty-title">No matching files</div>';
    return "";
  }

  function paint() {
    frame = 0;
    const el = list();
    if (!el) return;
    el.querySelector(".file-spacer").style.height = `${shown.length * ROW}px`;
    const first = Math.max(0, Math.floor(el.scrollTop / ROW) - OVERSCAN);
    const last = Math.min(shown.length, Math.ceil((el.scrollTop + el.clientHeight) / ROW) + OVERSCAN);
    let html = "";
    for (let i = first; i < last; i += 1) html += rowHTML(i);
    el.querySelector(".file-rows").innerHTML = html;
    const empty = el.querySelector(".file-empty");
    const content = emptyHTML();
    empty.innerHTML = content;
    empty.hidden = !content;
  }

  function schedulePaint() {
    if (!frame) frame = requestAnimationFrame(paint);
  }

  function chipHTML(key, label, count, extra = "") {
    const on = filter === key;
    return `<button type="button" class="kind-chip${on ? " on" : ""}${extra}" data-filter="${esc(key)}"${count ? "" : " disabled"}>`
      + `${esc(label)}<span class="n">${fmt.int(count)}</span></button>`;
  }

  function renderHead() {
    const summary = (files && files.summary) || {};
    const count = files && files.count != null ? files.count : null;
    const signature = JSON.stringify([database, count, summary, filter, missing.size, kinds.length, Boolean(files)]);
    if (signature === headSignature) return;
    headSignature = signature;
    const head = $("#files-head");
    if (!files) {
      head.innerHTML = "";
      return;
    }
    const chips = kinds.filter((k) => summary[k.key]).map((k) => chipHTML(k.key, k.label, summary[k.key])).join("");
    const gone = missing.size ? chipHTML("missing", "Not found", missing.size, " warn") : "";
    const hadFocus = document.activeElement && document.activeElement.id === "file-search";
    head.innerHTML = '<div class="files-title-row">'
      + `<h3 class="sub-title">${ICONS.file}<span>Files</span>${count != null ? `<span class="title-count">${fmt.int(count)}</span>` : ""}</h3>`
      + "</div>"
      + (count
        ? '<div class="files-tools">'
          + `<div class="kind-chips">${chipHTML("all", "All", count)}${chips}${gone}</div>`
          + `<label class="search file-search${query ? " has-text" : ""}">${ICONS.search}<input id="file-search" type="text" placeholder="Filter by name" value="${esc(query)}" autocomplete="off" spellcheck="false"><button type="button" class="clear" data-clear-search data-tip-text="Clear">${ICONS.close}</button></label>`
          + "</div>"
        : "");
    if (hadFocus) {
      const input = $("#file-search");
      if (input) {
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
      }
    }
  }

  function renderFoot() {
    const signature = JSON.stringify([database, message, rows.length, missing.size, checked, Boolean(files)]);
    if (signature === footSignature) return;
    footSignature = signature;
    let html = "";
    if (message) {
      html += `<div class="status ${message.kind}">${message.kind === "warn" ? ICONS.warn : ICONS.error}<div class="status-body">${message.html}</div>`
        + `<button type="button" class="icon-btn btn ghost small dismiss" data-act="dismiss_message" data-tip-text="Dismiss">${ICONS.close}</button></div>`;
    }
    if (files && rows.length) {
      html += '<div class="hint-row">Double-click a file to open it. Point at a name to see where the file is stored.</div>';
      if (missing.size) {
        const one = missing.size === 1;
        html += `<div class="hint-row warn">${ICONS.warn}<span>${fmt.int(missing.size)} file${one ? " is" : "s are"} no longer where ${one ? "it was" : "they were"} when the database was created, so ${one ? "it" : "they"} can't be opened. The database still searches ${one ? "its" : "their"} text.</span></div>`;
      } else if (!checked) {
        html += '<div class="hint-row"><span class="spinner inline muted"></span>Checking that the files are still there…</div>';
      }
    }
    $("#files-foot").innerHTML = html;
  }

  function refreshAll() {
    applyFilter();
    renderHead();
    renderFoot();
    const signature = JSON.stringify([database, version, filter, query, missing.size, shown.length, rows.length, files && files.status, loading]);
    if (signature !== listSignature) {
      listSignature = signature;
      schedulePaint();
    }
  }

  function reset(name) {
    database = name;
    version = -1;
    rows = [];
    dirs = [];
    missing = new Set();
    checked = false;
    shown = [];
    filter = "all";
    query = "";
    active = -1;
    message = null;
    missingFor = -1;
    headSignature = "";
    footSignature = "";
    const el = list();
    if (el) el.scrollTop = 0;
  }

  function loadRows(target) {
    loading = true;
    VDB.call("files", { name: target.name, version: target.version }).then((result) => {
      loading = false;
      const same = files && files.name === target.name && files.version === target.version;
      if (same && !result.stale) {
        const data = JSON.parse(result.payload);
        rows = data.rows || [];
        dirs = data.dirs || [];
        version = target.version;
      }
      if (!same && files && files.status === "ready" && files.version !== version) loadRows(files);
      refreshAll();
      maybeLoadMissing();
    });
  }

  function maybeLoadMissing() {
    if (files && files.status === "ready" && files.checked && version === files.version && missingFor !== files.version) {
      missingFor = files.version;
      loadMissing(files);
    }
  }

  function loadMissing(target) {
    VDB.call("missing", { name: target.name, version: target.version }).then((result) => {
      if (result.stale || !files || files.name !== target.name || version !== target.version) return;
      missing = new Set(result.missing || []);
      checked = true;
      if (filter === "missing" && !missing.size) filter = "all";
      headSignature = "";
      refreshAll();
    });
  }

  function update(nextFiles, nextKinds) {
    kinds = nextKinds;
    const name = nextFiles ? nextFiles.name : null;
    if (name !== database) reset(name);
    files = nextFiles;
    $("#files-section").hidden = !files;
    if (files && files.status === "ready" && files.version !== version && !loading) loadRows(files);
    else maybeLoadMissing();
    refreshAll();
  }

  function setActive(position) {
    active = position;
    for (const row of list().querySelectorAll(".file-row")) {
      const on = Number(row.dataset.position) === active;
      row.classList.toggle("on", on);
      row.classList.toggle("active", on);
      row.setAttribute("aria-selected", String(on));
    }
  }

  function ensureVisible(position) {
    const el = list();
    const top = position * ROW;
    if (top < el.scrollTop) el.scrollTop = top;
    else if (top + ROW > el.scrollTop + el.clientHeight) el.scrollTop = top + ROW - el.clientHeight;
  }

  function showMissing(index, result) {
    const r = rows[index];
    const where = result.path ? ` It was at <span class="path">${esc(result.path)}</span>.` : "";
    const folder = result.opened_folder ? " Its folder was opened instead." : "";
    message = { kind: "warn", html: `<b>${esc(r ? r[0] : "This file")}</b> was moved or deleted after the database was created.${where}${folder}` };
    if (r && !missing.has(index)) {
      missing.add(index);
      headSignature = "";
    }
    refreshAll();
  }

  function openPosition(position) {
    const index = shown[position];
    if (index == null) return;
    const now = Date.now();
    if (lastOpen.index === index && now - lastOpen.at < 700) return;
    lastOpen = { index, at: now };
    VDB.call("open_file", { version, index }).then((result) => {
      if (result.missing) showMissing(index, result);
    });
  }

  function revealPosition(position) {
    const index = shown[position];
    if (index == null) return;
    VDB.call("reveal_file", { version, index }).then((result) => {
      if (result.missing) showMissing(index, result);
    });
  }

  function bind() {
    const el = list();
    el.addEventListener("scroll", schedulePaint);
    window.addEventListener("resize", schedulePaint);
    el.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || e.target.closest(".file-action")) return;
      const row = e.target.closest(".file-row");
      if (!row) return;
      e.preventDefault();
      el.focus();
      const position = Number(row.dataset.position);
      if (e.detail === 2) {
        openPosition(position);
        return;
      }
      setActive(position);
    });
    el.addEventListener("click", (e) => {
      const reveal = e.target.closest("[data-reveal]");
      if (reveal) revealPosition(Number(reveal.dataset.reveal));
    });
    el.addEventListener("keydown", (e) => {
      if (!shown.length) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "Home" || e.key === "End") {
        e.preventDefault();
        let next;
        if (e.key === "Home") next = 0;
        else if (e.key === "End") next = shown.length - 1;
        else next = Math.max(0, Math.min(shown.length - 1, (active < 0 ? -1 : active) + (e.key === "ArrowDown" ? 1 : -1)));
        setActive(next);
        ensureVisible(next);
      } else if (e.key === "Enter" && active >= 0) {
        e.preventDefault();
        openPosition(active);
      }
    });
    $("#files-section").addEventListener("input", (e) => {
      if (e.target.id !== "file-search") return;
      query = e.target.value;
      e.target.parentElement.classList.toggle("has-text", query.length > 0);
      active = -1;
      list().scrollTop = 0;
      refreshAll();
    });
    $("#files-section").addEventListener("click", (e) => {
      const chip = e.target.closest("[data-filter]");
      if (chip && !chip.disabled) {
        filter = chip.dataset.filter;
        active = -1;
        headSignature = "";
        list().scrollTop = 0;
        refreshAll();
        return;
      }
      if (e.target.closest("[data-clear-search]")) {
        query = "";
        active = -1;
        headSignature = "";
        refreshAll();
        return;
      }
      const act = e.target.closest('[data-act="dismiss_message"]');
      if (act) {
        message = null;
        refreshAll();
      }
    });
  }

  window.ManageFiles = { init: bind, update };
})();
