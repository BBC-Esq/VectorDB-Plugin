"use strict";

(function () {
  const { ICONS, esc, fmt } = VDB;
  const { html } = Tools;

  function pages(n) {
    return `${fmt.int(n)} page${n === 1 ? "" : "s"}`;
  }

  function rowHTML(row) {
    let icon = "";
    let text = "";
    let kind = "";
    if (row.status === "starting") {
      icon = '<span class="spinner"></span>';
      text = `Starting… ${html.since(row.started)}`;
    } else if (row.status === "scraping") {
      icon = '<span class="spinner"></span>';
      text = `Scraping… <b>${pages(row.pages)}</b> ${html.since(row.started)}`;
    } else if (row.status === "cancelling") {
      icon = '<span class="spinner muted"></span>';
      text = `Cancelling… ${pages(row.pages)} saved`;
    } else if (row.status === "completed") {
      icon = ICONS.check;
      kind = "ok";
      text = `Completed · <b>${pages(row.pages)}</b>`;
    } else if (row.status === "cancelled") {
      icon = ICONS.stop;
      kind = "muted";
      text = `Cancelled · ${pages(row.pages)} saved`;
    } else if (row.status === "rate_limited") {
      icon = ICONS.warn;
      kind = "warn";
      text = `The site is limiting requests. ${pages(row.pages)} saved; scrape it again and choose Resume to continue.`;
    }
    const busy = row.running;
    const buttons = [
      busy && row.status !== "cancelling"
        ? `<button type="button" class="btn ghost small" data-act="scrape_cancel" data-arg="${esc(row.name)}">${ICONS.stop}Cancel</button>` : "",
      `<button type="button" class="btn ghost small" data-act="scrape_open" data-arg="${esc(row.name)}" data-tip-text="Open the folder with the scraped pages">${ICONS.folder}Open</button>`,
      busy ? "" : `<button type="button" class="icon-btn btn ghost small" data-act="scrape_dismiss" data-arg="${esc(row.name)}" data-tip-text="Remove from this list">${ICONS.close}</button>`,
    ].join("");
    return `<div class="job ${kind}"><span class="job-icon">${icon}</span><span class="job-name">${esc(row.name)}</span>`
      + `<span class="job-status">${text}</span><span class="job-actions">${buttons}</span></div>`;
  }

  function render(s) {
    const current = s.docs.find((d) => d.name === s.selected);
    const option = current ? { label: current.name, note: current.scraped ? "scraped" : "" } : { label: "Choose documentation" };
    const busy = s.rows.some((r) => r.name === s.selected && r.running);
    const full = s.active >= s.limit;
    const scraped = s.docs.filter((d) => d.scraped).length;
    const tip = full ? `At most ${s.limit} scrapes can run at the same time` : busy ? "This documentation is already being scraped" : "";
    const aside = s.active
      ? `<b>${s.active}</b> of ${s.limit} running`
      : `${fmt.int(s.docs.length)} sources · <b>${fmt.int(scraped)}</b> scraped`;
    const jobs = s.rows.length ? `<div class="jobs">${s.rows.map(rowHTML).join("")}</div>` : "";
    return html.header("globe", "Scrape Documentation", "Saves online documentation to add to a database", aside)
      + '<div class="line">'
      + html.field("Documentation", VDB.selectButton("scrape.doc", option), { cls: "grow" })
      + html.action("scrape_start", "Scrape", { icon: "download", disabled: !current || busy || full, tip })
      + "</div>"
      + jobs;
  }

  Tools.register("scrape", {
    render,
    openSelect(id, s) {
      VDB.openSelect(id, {
        title: "Documentation",
        options: s.docs.map((d) => ({ value: d.name, label: d.name, note: d.scraped ? "scraped" : "" })),
        value: s.selected,
        search: true,
        placeholder: "Type to filter, for example torch",
        onPick: (value) => {
          if (value !== s.selected) Tools.call("scrape_select", { name: value });
        },
      });
    },
    actions: {
      scrape_start(button, s) {
        Tools.call("scrape_start", { name: s.selected });
      },
    },
  });
})();
