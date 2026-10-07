"use strict";

(function () {
  const { esc, fmt } = VDB;
  const { html } = Tools;

  const BACKUP_TIP = "Copies every database to the Vector_DB_Backup folder, replacing the previous backup. "
    + "New databases are also backed up automatically when you create them.";
  const RESTORE_TIP = "Replaces your current databases with the ones in the backup.";

  function summaryHTML(s) {
    if (!s.databases && !s.has_backup) return html.status("info", "No databases yet.");
    const count = `<b>${fmt.int(s.databases)}</b> database${s.databases === 1 ? "" : "s"}`;
    if (!s.has_backup) return html.status("warn", `${count} · no backup yet`);
    if (!s.missing.length) return html.status("ok", `${count} · all backed up`);
    const names = s.missing.slice(0, 3).map(esc).join(", ") + (s.missing.length > 3 ? ", …" : "");
    return html.status("warn", `${count} · ${s.missing.length} not in the backup: ${names}`);
  }

  function render(s) {
    let line;
    if (s.task) {
      line = html.status("run", `${s.task === "backup" ? "Backing up the databases" : "Restoring the backup"}… ${html.since(s.started)}`);
    } else if (s.result) {
      line = html.status(s.result.ok ? "ok" : "error", esc(s.result.message));
    } else {
      line = summaryHTML(s);
    }
    const busy = Boolean(s.task);
    return html.header("archive", "Database Backup", "")
      + `<div class="summary">${line}</div>`
      + '<div class="line buttons">'
      + html.action("backup", "Back Up", { icon: "archive", disabled: busy || !s.databases, tip: s.databases ? BACKUP_TIP : "There are no databases to back up" })
      + html.action("restore", "Restore", { icon: "restore", kind: "danger", disabled: busy || !s.has_backup, tip: s.has_backup ? RESTORE_TIP : "There is no backup to restore" })
      + "</div>";
  }

  Tools.register("backups", { render });
})();
