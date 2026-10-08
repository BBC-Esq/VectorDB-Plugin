"use strict";

(function () {
  const { ICONS, esc, fmt, $ } = VDB;

  const TRASH = '<svg viewBox="0 0 16 16"><path d="M2.8 4.2h10.4M6.3 4.2V2.8h3.4v1.4M4.2 4.2l.7 8.9c.05.6.55 1.1 1.15 1.1h3.9c.6 0 1.1-.5 1.15-1.1l.7-8.9" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/><path d="M6.7 6.8v4.6M9.3 6.8v4.6" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>';
  const COLUMNS = [
    { key: "name", label: "Name" },
    { key: "model", label: "Embedding model", cls: "col-model" },
    { key: "files", label: "Files", num: true },
    { key: "chunks", label: "Chunks", num: true },
    { key: "size", label: "Size", num: true },
    { key: "created", label: "Created", cls: "col-created" },
  ];
  const TASK_LABELS = { delete: "Deleting", backup: "Backing up", restore: "Restoring" };
  const FILTER_AT = 9;

  let state = null;
  let selected = null;
  let pendingSelect = null;
  let sort = { key: "name", dir: 1 };
  let confirmName = null;
  let dbQuery = "";
  let headSignature = "";
  let tableSignature = "";
  let footSignature = "";
  let detailSignature = "";

  function bytes(n) {
    if (n == null) return null;
    const units = ["B", "KB", "MB", "GB", "TB"];
    let value = n;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    if (unit === 0) return `${value} B`;
    return `${value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${units[unit]}`;
  }

  function day(seconds) {
    if (!seconds) return null;
    return new Date(seconds * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  }

  function moment(seconds) {
    if (!seconds) return null;
    return new Date(seconds * 1000).toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
  }

  function sortValue(db, key) {
    const info = db.info || {};
    switch (key) {
      case "name": return db.name.toLowerCase();
      case "model": return db.model ? db.model.name.toLowerCase() : null;
      case "files": return info.files ?? null;
      case "chunks": return info.chunks ?? null;
      case "size": return info.size ?? null;
      case "created": return info.created ?? null;
      default: return null;
    }
  }

  function sortedDatabases() {
    const terms = dbQuery.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const list = state.databases.filter((db) => terms.every((t) => db.name.toLowerCase().includes(t)));
    return list.sort((a, b) => {
      const x = sortValue(a, sort.key);
      const y = sortValue(b, sort.key);
      if (x == null && y == null) return a.name.localeCompare(b.name);
      if (x == null) return 1;
      if (y == null) return -1;
      if (x < y) return -sort.dir;
      if (x > y) return sort.dir;
      return a.name.localeCompare(b.name);
    });
  }

  function current() {
    return state.databases.find((db) => db.name === selected) || null;
  }

  function taskFor(name) {
    return state.task && state.task.name === name ? state.task : null;
  }

  function badgeHTML(db) {
    const task = taskFor(db.name);
    if (task) return `<span class="badge accent"><span class="spinner tiny"></span>${esc(TASK_LABELS[task.kind])}</span>`;
    switch (db.status) {
      case "building": return '<span class="badge accent"><span class="spinner tiny"></span>Being created</span>';
      case "missing": return '<span class="badge danger">Folder missing</span>';
      case "leftover": return '<span class="badge warn">Leftover folder</span>';
      case "incomplete": return '<span class="badge warn">Incomplete</span>';
      default: return db.backup ? "" : '<span class="badge muted" data-tip-text="There is no backup copy of this database yet.">No backup</span>';
    }
  }

  function modelCell(db) {
    if (!db.model) return '<span class="faint">—</span>';
    const warn = db.model.downloaded ? "" : `<span class="model-warn" data-tip-text="This model is not downloaded, so the database can't be searched.">${ICONS.warn}</span>`;
    return `${warn}<span class="ellipsis">${esc(db.model.name)}</span>`;
  }

  function numCell(db, value, text) {
    if (db.loading) return '<span class="faint">…</span>';
    return value == null ? '<span class="faint">—</span>' : esc(text ?? fmt.int(value));
  }

  function renderHead() {
    const total = state.databases.reduce((sum, db) => sum + ((db.info && db.info.size) || 0), 0);
    const loading = state.databases.some((db) => db.loading);
    const many = state.databases.length >= FILTER_AT;
    const signature = JSON.stringify([state.databases.length, total, loading, many]);
    if (signature === headSignature) return;
    headSignature = signature;
    const count = state.databases.length;
    const hadFocus = document.activeElement && document.activeElement.id === "db-search";
    $("#dbs-head").innerHTML = '<div class="dbs-title-row">'
      + `<h2 class="section-title">${ICONS.database}<span>Databases</span><span class="title-count">${fmt.int(count)}</span></h2>`
      + (many ? `<label class="search db-search${dbQuery ? " has-text" : ""}">${ICONS.search}<input id="db-search" type="text" placeholder="Filter databases" value="${esc(dbQuery)}" autocomplete="off" spellcheck="false"><button type="button" class="clear" data-clear-db-search data-tip-text="Clear">${ICONS.close}</button></label>` : "")
      + (count ? `<span class="dbs-total">${loading ? "Measuring…" : `${esc(bytes(total))} in all`}</span>` : "")
      + "</div>";
    if (hadFocus) {
      const input = $("#db-search");
      if (input) {
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
      }
    }
  }

  function headerHTML() {
    return '<div class="db-row db-header" role="presentation">' + COLUMNS.map((c) => {
      const on = sort.key === c.key;
      const arrow = on ? `<span class="sort-arrow">${sort.dir > 0 ? "▲" : "▼"}</span>` : "";
      return `<button type="button" class="db-col${c.num ? " num" : ""}${c.cls ? ` ${c.cls}` : ""}${on ? " on" : ""}" data-sort="${c.key}">${esc(c.label)}${arrow}</button>`;
    }).join("") + "</div>";
  }

  function rowHTML(db) {
    const info = db.info || {};
    const on = db.name === selected;
    return `<div class="db-row${on ? " on" : ""} status-${db.status}" role="option" aria-selected="${on}" data-db="${esc(db.name)}">`
      + `<span class="db-name"><span class="db-label">${esc(db.name)}</span>${badgeHTML(db)}</span>`
      + `<span class="db-model col-model">${modelCell(db)}</span>`
      + `<span class="num">${numCell(db, info.files)}</span>`
      + `<span class="num">${numCell(db, info.chunks)}</span>`
      + `<span class="num">${numCell(db, info.size, bytes(info.size))}</span>`
      + `<span class="db-date col-created">${db.loading ? '<span class="faint">…</span>' : esc(day(info.created) || "—")}</span>`
      + "</div>";
  }

  function renderTable() {
    const list = sortedDatabases();
    const signature = JSON.stringify([list, selected, sort, state.task, dbQuery]);
    if (signature === tableSignature) return;
    tableSignature = signature;
    const table = $("#dbs-table");
    if (!state.databases.length) {
      table.innerHTML = '<div class="db-empty"><div class="empty-title">No databases yet</div>'
        + '<div>Databases you create are listed here, with their files, size and backup.</div>'
        + `<button type="button" class="btn primary" data-tab="Create Database">${ICONS.build}Create a database</button></div>`;
      return;
    }
    table.innerHTML = headerHTML()
      + (list.length ? list.map(rowHTML).join("") : '<div class="db-none">No databases match the filter.</div>');
  }

  function renderFoot() {
    const signature = JSON.stringify([state.notice, state.config_error]);
    if (signature === footSignature) return;
    footSignature = signature;
    let html = "";
    if (state.config_error) {
      html += `<div class="status error">${ICONS.error}<div class="status-body">${esc(state.config_error)} The list may be incomplete, and nothing can be deleted until the file is fixed.</div></div>`;
    }
    const n = state.notice;
    if (n) {
      const icon = { ok: ICONS.check, warn: ICONS.warn, error: ICONS.error }[n.kind] || ICONS.info;
      const details = n.details && n.details.length
        ? `<div class="notice-list">${n.details.map((d) => `<div>${esc(d)}</div>`).join("")}</div>`
        : "";
      html += `<div class="status ${n.kind}">${icon}<div class="status-body">${esc(n.message)}${details}</div>`
        + `<button type="button" class="icon-btn btn ghost small dismiss" data-act="dismiss_notice" data-tip-text="Dismiss">${ICONS.close}</button></div>`;
    }
    $("#dbs-foot").innerHTML = html;
  }

  function tile(label, value, extra = "") {
    return `<div class="tile${extra}"><div class="tile-label">${esc(label)}</div><div class="tile-value">${value}</div></div>`;
  }

  function plain(value) {
    return value == null || value === "" ? '<span class="faint">—</span>' : esc(value);
  }

  function statsHTML(db) {
    if (db.status === "building") return "";
    const info = db.info || {};
    const wait = db.loading ? '<span class="faint">…</span>' : null;
    const number = (value) => wait || plain(value != null ? fmt.int(value) : null);
    const model = db.model
      ? `<span class="ellipsis">${esc(db.model.name)}</span>${db.model.vendor ? `<span class="tile-note">${esc(db.model.vendor)}</span>` : ""}`
      : plain(null);
    const backup = db.backup
      ? `<span class="ok-text">${ICONS.check}Backed up</span>`
      : '<span class="warn-text">Not backed up</span>';
    const tiles = [];
    if (db.registered) tiles.push(tile("Embedding model", model, " wide"));
    if (db.status !== "missing" && db.status !== "leftover") tiles.push(tile("Dimensions", number(info.dimensions)));
    if (db.registered) {
      tiles.push(tile("Chunk size", plain(db.chunk_size != null ? fmt.int(db.chunk_size) : null)));
      tiles.push(tile("Overlap", plain(db.chunk_overlap != null ? fmt.int(db.chunk_overlap) : null)));
    }
    if (db.status !== "missing" && db.status !== "leftover") {
      tiles.push(tile("Files", number(info.files)));
      tiles.push(tile("Chunks", number(info.chunks)));
    }
    if (db.status !== "missing") {
      tiles.push(tile("Size on disk", wait || plain(bytes(info.size))));
      tiles.push(tile("Created", wait || plain(moment(info.created)), " date"));
    }
    tiles.push(tile("Backup", backup));
    return `<div class="tiles">${tiles.join("")}</div>`;
  }

  function warningsHTML(db) {
    const parts = [];
    const tabLink = (tab, text) => `<button type="button" class="link" data-tab="${esc(tab)}">${esc(text)}</button>`;
    if (db.status === "building") {
      parts.push(`<div class="status info">${ICONS.info}<div class="status-body">This database is being created right now. ${tabLink("Create Database", "Follow it on the Create Database tab")}</div></div>`);
    } else if (db.status === "missing") {
      const restore = db.backup
        ? " A backup copy exists, so you can restore it."
        : " There is no backup copy, so it can only be removed from the list.";
      parts.push(`<div class="status error">${ICONS.error}<div class="status-body">The folder that held this database is missing, so it can't be searched.${restore}</div></div>`);
    } else if (db.status === "leftover") {
      parts.push(`<div class="status warn">${ICONS.warn}<div class="status-body">This folder in Vector_DB isn't a database the program knows about, so it can't be searched. It is most likely left over from a database that was never finished. Deleting it frees the space and the name.</div></div>`);
    } else if (db.status === "incomplete") {
      parts.push(`<div class="status warn">${ICONS.warn}<div class="status-body">${esc((db.info && db.info.problem) || "This database is incomplete.")} Searching it may not work. Deleting it lets you create it again.</div></div>`);
    }
    if (db.registered && db.model && !db.model.downloaded && db.status !== "missing") {
      parts.push(`<div class="status warn">${ICONS.warn}<div class="status-body">The embedding model this database was created with, <b>${esc(db.model.name)}</b>, is not downloaded, so the database can't be searched until you download it again. ${tabLink("Models", "Go to the Models tab")}</div></div>`);
    }
    return parts.join("");
  }

  function confirmHTML(db) {
    const info = db.info || {};
    let title;
    let text;
    if (db.status === "missing") {
      title = `Remove <b>${esc(db.name)}</b> from the list?`;
      text = db.backup ? "Its folder is already gone. Its backup copy will be deleted too." : "Its folder is already gone.";
    } else if (db.status === "leftover") {
      title = `Delete the leftover folder <b>${esc(db.name)}</b>?`;
      text = `This deletes the folder${info.size ? ` (${esc(bytes(info.size))})` : ""}${db.backup ? " and its backup copy" : ""}.`;
    } else {
      const facts = [];
      if (info.files != null) facts.push(`${fmt.int(info.files)} file${info.files === 1 ? "" : "s"}`);
      if (info.chunks != null) facts.push(`${fmt.int(info.chunks)} chunks`);
      if (info.size != null) facts.push(bytes(info.size));
      title = `Delete <b>${esc(db.name)}</b> permanently?`;
      text = `This deletes the database${facts.length ? ` (${esc(facts.join(", "))})` : ""}${db.backup ? " and its backup copy" : ""}. Your original files are not touched.`;
    }
    return `<div class="confirm" role="alertdialog" aria-label="Confirm delete">${ICONS.warn}<div class="confirm-body">`
      + `<div class="confirm-title">${title}</div><div class="confirm-text">${text} This can't be undone.</div>`
      + '<div class="confirm-actions">'
      + `<button type="button" class="btn danger" data-act="confirm_delete">${TRASH}${db.status === "missing" ? "Remove from list" : "Delete permanently"}</button>`
      + '<button type="button" class="btn ghost" data-act="cancel_delete" id="cancel-delete">Cancel</button>'
      + "</div></div></div>";
  }

  function actionsHTML(db) {
    const task = taskFor(db.name);
    if (task) {
      return `<div class="detail-task"><span class="spinner"></span><span>${esc(TASK_LABELS[task.kind])} ${esc(db.name)}…</span></div>`;
    }
    if (db.status === "building") return "";
    const blocked = state.blocked;
    const off = blocked ? ` disabled data-tip-text="${esc(blocked)}"` : "";
    const buttons = [];
    if (db.status === "ready" || db.status === "incomplete") {
      buttons.push(`<button type="button" class="btn ghost small" data-act="search">${ICONS.search}Search it</button>`);
    }
    if (db.status === "ready" && !db.backup) {
      buttons.push(`<button type="button" class="btn ghost small" data-act="backup"${off}>${ICONS.archive}Back up</button>`);
    }
    if (db.status === "missing" && db.backup) {
      buttons.push(`<button type="button" class="btn ghost small" data-act="restore"${off}>${ICONS.restore}Restore from backup</button>`);
    }
    const label = db.status === "missing" ? "Remove…" : "Delete…";
    buttons.push(`<button type="button" class="btn danger small" data-act="delete"${off}${confirmName === db.name ? " disabled" : ""}>${TRASH}${label}</button>`);
    return `<div class="detail-actions">${buttons.join("")}</div>`;
  }

  function renderDetail() {
    const db = current();
    $("#detail-card").hidden = !db;
    if (!db) {
      detailSignature = "";
      ManageFiles.update(null, state.kinds);
      return;
    }
    if (confirmName && confirmName !== db.name) confirmName = null;
    const signature = JSON.stringify([db, state.task, state.blocked, confirmName]);
    if (signature !== detailSignature) {
      detailSignature = signature;
      const hadCancelFocus = document.activeElement && document.activeElement.id === "cancel-delete";
      $("#detail-head").innerHTML = '<div class="detail-title-row">'
        + `<h2 class="section-title detail-name">${ICONS.database}<span class="ellipsis">${esc(db.name)}</span>${badgeHTML(db)}</h2>`
        + actionsHTML(db)
        + "</div>";
      $("#detail-body").innerHTML = (confirmName === db.name ? confirmHTML(db) : "")
        + statsHTML(db)
        + warningsHTML(db);
      if (confirmName === db.name && hadCancelFocus) $("#cancel-delete").focus();
    }
    ManageFiles.update(state.files && state.files.name === db.name ? state.files : null, state.kinds);
  }

  function render() {
    if (pendingSelect && state.selected !== pendingSelect && state.databases.some((db) => db.name === pendingSelect)) {
      selected = pendingSelect;
    } else {
      pendingSelect = null;
      selected = state.selected;
    }
    renderHead();
    renderTable();
    renderFoot();
    renderDetail();
  }

  function select(name) {
    if (!name || name === selected) return;
    selected = name;
    pendingSelect = name;
    confirmName = null;
    tableSignature = "";
    renderTable();
    renderDetail();
    VDB.call("select", { name }).then((result) => {
      if (result.error && pendingSelect === name) {
        pendingSelect = null;
        render();
      }
    });
  }

  function moveSelection(step) {
    const list = sortedDatabases();
    if (!list.length) return;
    const index = list.findIndex((db) => db.name === selected);
    const next = list[Math.max(0, Math.min(list.length - 1, (index < 0 ? -1 : index) + step))];
    select(next.name);
    const row = document.querySelector(`.db-row[data-db="${CSS.escape(next.name)}"]`);
    if (row) row.scrollIntoView({ block: "nearest" });
  }

  function nextAfter(name) {
    const list = sortedDatabases();
    const index = list.findIndex((db) => db.name === name);
    const rest = list.filter((db) => db.name !== name);
    if (!rest.length) return null;
    return rest[Math.min(index, rest.length - 1)].name;
  }

  function askDelete() {
    const db = current();
    if (!db || state.blocked || taskFor(db.name) || db.status === "building") return;
    confirmName = db.name;
    detailSignature = "";
    renderDetail();
    const cancel = $("#cancel-delete");
    if (cancel) cancel.focus();
  }

  function act(action) {
    if (action === "dismiss_notice") {
      VDB.call("dismiss_notice");
      return;
    }
    const db = current();
    if (!db) return;
    if (action === "delete") askDelete();
    else if (action === "cancel_delete") {
      confirmName = null;
      detailSignature = "";
      renderDetail();
    } else if (action === "confirm_delete") {
      const name = confirmName;
      confirmName = null;
      VDB.call("delete", { name, next_name: nextAfter(name) });
    } else if (action === "search") VDB.call("search_database", { name: db.name });
    else if (action === "backup") VDB.call("backup", { name: db.name });
    else if (action === "restore") VDB.call("restore", { name: db.name });
  }

  function bind() {
    document.addEventListener("click", (e) => {
      const tab = e.target.closest("[data-tab]");
      if (tab) {
        VDB.call("open_tab", { name: tab.dataset.tab });
        return;
      }
      const column = e.target.closest("[data-sort]");
      if (column) {
        const key = column.dataset.sort;
        sort = sort.key === key ? { key, dir: -sort.dir } : { key, dir: key === "name" || key === "model" ? 1 : -1 };
        tableSignature = "";
        renderTable();
        return;
      }
      if (e.target.closest("[data-clear-db-search]")) {
        dbQuery = "";
        headSignature = "";
        tableSignature = "";
        renderHead();
        renderTable();
        return;
      }
      const action = e.target.closest("[data-act]");
      if (action && !action.disabled && !action.closest("#files-section")) {
        act(action.dataset.act);
      }
    });
    $("#dbs-table").addEventListener("mousedown", (e) => {
      const row = e.target.closest(".db-row[data-db]");
      if (!row || e.button !== 0) return;
      e.preventDefault();
      $("#dbs-table").focus();
      select(row.dataset.db);
    });
    $("#dbs-table").addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        moveSelection(e.key === "ArrowDown" ? 1 : -1);
      } else if (e.key === "Delete") {
        e.preventDefault();
        askDelete();
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && confirmName && !VDB.popover.key) {
        e.preventDefault();
        act("cancel_delete");
      }
    });
    document.addEventListener("input", (e) => {
      if (e.target.id !== "db-search") return;
      dbQuery = e.target.value;
      e.target.parentElement.classList.toggle("has-text", dbQuery.length > 0);
      tableSignature = "";
      renderTable();
    });
  }

  ManageFiles.init();
  bind();

  window.TabApp = {
    init(payload) {
      state = payload;
      render();
    },
    setState(payload) {
      state = payload;
      render();
    },
  };
})();
