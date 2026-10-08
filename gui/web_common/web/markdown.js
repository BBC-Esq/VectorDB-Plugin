"use strict";

(function () {
  const esc = (text) => String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  const FENCE = /^\s*(```|~~~)(.*)$/;
  const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
  const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
  const BULLET = /^(\s*)([-*+])\s+(.*)$/;
  const NUMBER = /^(\s*)(\d+)[.)]\s+(.*)$/;
  const QUOTE = /^\s*>\s?(.*)$/;
  const TABLE_ROW = /^\s*\|.*\|\s*$/;
  const TABLE_RULE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

  function inline(text) {
    const codes = [];
    let s = String(text).replace(/`([^`]+)`/g, (_, code) => {
      codes.push(code);
      return `\u0000${codes.length - 1}\u0000`;
    });
    s = esc(s);
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) => `<a href="${url}">${label}</a>`);
    s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?'"])/g, (_, lead, url) => `${lead}<a href="${url}">${url}</a>`);
    s = s.replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/__(?=\S)([\s\S]*?\S)__/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*\w])\*(?=\S)([^*]*?\S)\*(?!\*)/g, "$1<em>$2</em>");
    s = s.replace(/(^|[^_\w])_(?=\S)([^_]*?\S)_(?![_\w])/g, "$1<em>$2</em>");
    s = s.replace(/~~(?=\S)([\s\S]*?\S)~~/g, "<del>$1</del>");
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${esc(codes[Number(i)])}</code>`);
  }

  function splitRow(line) {
    let row = line.trim();
    if (row.startsWith("|")) row = row.slice(1);
    if (row.endsWith("|")) row = row.slice(0, -1);
    return row.split("|").map((cell) => cell.trim());
  }

  function table(lines) {
    const head = splitRow(lines[0]);
    const aligns = splitRow(lines[1]).map((cell) => (cell.startsWith(":") && cell.endsWith(":") ? "center" : cell.endsWith(":") ? "right" : ""));
    const cell = (tag, text, i) => `<${tag}${aligns[i] ? ` style="text-align:${aligns[i]}"` : ""}>${inline(text)}</${tag}>`;
    let html = "<div class=\"md-table\"><table><thead><tr>" + head.map((h, i) => cell("th", h, i)).join("") + "</tr></thead><tbody>";
    for (const line of lines.slice(2)) {
      const cells = splitRow(line);
      html += "<tr>" + head.map((_, i) => cell("td", cells[i] ?? "", i)).join("") + "</tr>";
    }
    return html + "</tbody></table></div>";
  }

  function list(lines) {
    const root = { items: [], ordered: false, start: 1, indent: null };
    const stack = [root];
    for (const line of lines) {
      const match = line.match(BULLET) || line.match(NUMBER);
      if (!match) {
        const top = stack[stack.length - 1];
        const last = top.items[top.items.length - 1];
        if (last) last.text += " " + line.trim();
        continue;
      }
      const indent = match[1].replace(/\t/g, "    ").length;
      const ordered = /\d/.test(match[2]);
      while (stack.length > 1 && indent < stack[stack.length - 1].indent) stack.pop();
      let top = stack[stack.length - 1];
      if (top.indent === null) {
        top.indent = indent;
        top.ordered = ordered;
        top.start = ordered ? Number(match[2]) : 1;
      } else if (indent >= top.indent + 2 && top.items.length) {
        const child = { items: [], ordered, start: ordered ? Number(match[2]) : 1, indent };
        top.items[top.items.length - 1].children = child;
        stack.push(child);
        top = child;
      }
      top.items.push({ text: match[3], children: null });
    }
    const render = (node) => {
      const tag = node.ordered ? "ol" : "ul";
      const start = node.ordered && node.start !== 1 ? ` start="${node.start}"` : "";
      return `<${tag}${start}>` + node.items.map((item) => `<li>${inline(item.text)}${item.children ? render(item.children) : ""}</li>`).join("") + `</${tag}>`;
    };
    return render(root);
  }

  function blocks(lines) {
    let html = "";
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) {
        i += 1;
        continue;
      }
      const fence = line.match(FENCE);
      if (fence) {
        const marker = fence[1];
        const code = [];
        i += 1;
        while (i < lines.length && !lines[i].trim().startsWith(marker)) {
          code.push(lines[i]);
          i += 1;
        }
        i += 1;
        html += `<pre><code>${esc(code.join("\n"))}</code></pre>`;
        continue;
      }
      const heading = line.match(HEADING);
      if (heading) {
        const level = Math.min(6, heading[1].length + 2);
        html += `<h${level}>${inline(heading[2])}</h${level}>`;
        i += 1;
        continue;
      }
      if (RULE.test(line)) {
        html += "<hr>";
        i += 1;
        continue;
      }
      if (TABLE_ROW.test(line) && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1])) {
        const rows = [line, lines[i + 1]];
        i += 2;
        while (i < lines.length && TABLE_ROW.test(lines[i])) {
          rows.push(lines[i]);
          i += 1;
        }
        html += table(rows);
        continue;
      }
      if (QUOTE.test(line)) {
        const quoted = [];
        while (i < lines.length && QUOTE.test(lines[i])) {
          quoted.push(lines[i].match(QUOTE)[1]);
          i += 1;
        }
        html += `<blockquote>${blocks(quoted)}</blockquote>`;
        continue;
      }
      if (BULLET.test(line) || NUMBER.test(line)) {
        const items = [];
        while (i < lines.length && lines[i].trim() && (BULLET.test(lines[i]) || NUMBER.test(lines[i]) || /^\s{2,}\S/.test(lines[i]))) {
          items.push(lines[i]);
          i += 1;
        }
        html += list(items);
        continue;
      }
      const para = [];
      while (i < lines.length && lines[i].trim() && !FENCE.test(lines[i]) && !HEADING.test(lines[i]) && !RULE.test(lines[i])
             && !QUOTE.test(lines[i]) && !BULLET.test(lines[i]) && !NUMBER.test(lines[i])
             && !(TABLE_ROW.test(lines[i]) && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1]))) {
        para.push(lines[i]);
        i += 1;
      }
      html += `<p>${para.map(inline).join("<br>")}</p>`;
    }
    return html;
  }

  window.Markdown = {
    render(text) {
      return blocks(String(text || "").replace(/\r\n?/g, "\n").split("\n"));
    },
  };
})();
