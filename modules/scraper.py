import os
import re
import json
import asyncio
import textwrap
import aiofiles
import markdown
import yaml
from bs4 import BeautifulSoup
from copy import deepcopy
import hashlib
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit
from PySide6.QtCore import Signal, QObject
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError

from core.constants import PROJECT_ROOT


_VERSION_SUFFIX_RE = re.compile(r"^v?\d+(\.\d+)*$")

_CRUFT_TAGS = ("script", "style", "nav", "footer", "svg")
_CRUFT_CLASSES = (
    "toctree-wrapper",
    "related",
    "sphinxsidebar",
    "footer",
    "edit-this-page",
    "md-source-file",
    "prev-next-area",
    "prev-next-bottom",
    "prev-next-top",
    "bd-toc",
    "bd-sidebar-secondary",
    "feedback-widget",
    "pagination-links",
    "footer-version",
    "try_examples_button_container",
    "try_examples_outer_iframe",
    "sidemenu",
    "banner",
    "sr-only",
)
_CRUFT_IDS = (
    "indices-and-tables",
    "search",
    "footerDisclaimer",
)


def _strip_trailing_version(path: str) -> str:
    parts = path.rsplit("-", 1)
    if len(parts) == 2 and _VERSION_SUFFIX_RE.match(parts[1]):
        return parts[0]
    return path


def _path_within(path: str, base: str) -> bool:
    base = base.rstrip("/")
    return path == base or path.startswith(base + "/")


_CRUFT_HEADINGS = {"copyright", "copyrights", "license", "licenses"}


def _strip_embedded_cruft(content):
    for tag_name in _CRUFT_TAGS:
        for el in content.find_all(tag_name):
            el.decompose()
    for cls in _CRUFT_CLASSES:
        for el in content.find_all(class_=cls):
            el.decompose()
    for cruft_id in _CRUFT_IDS:
        for el in content.find_all(id=cruft_id):
            el.decompose()
    for section in content.find_all("section"):
        h = section.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        if h and h.get_text(strip=True).lower() in _CRUFT_HEADINGS:
            section.decompose()
    _NAV_LABELS = {"prev", "previous", "next"}
    for p in content.find_all("p"):
        anchors = p.find_all("a")
        if not anchors:
            continue
        anchor_text = " ".join(a.get_text(strip=True).lower() for a in anchors).split()
        if anchor_text and all(w in _NAV_LABELS for w in anchor_text):
            non_anchor_text = "".join(
                str(c) for c in p.contents if getattr(c, "name", None) != "a"
            ).strip()
            if not non_anchor_text:
                p.decompose()
    for el in content.find_all(True):
        for attr in [a for a in el.attrs if a.lower().startswith("on")]:
            del el[attr]
    return content


class BaseScraper:
    def __init__(self, url, folder):
        self.url = url
        self.folder = folder
        self.save_dir = os.path.join(
            str(PROJECT_ROOT),
            "Scraped_Documentation",
            folder,
        )

    def process_html(self, soup):
        main_content = self.extract_main_content(soup)
        if main_content:
            cleaned = _strip_embedded_cruft(deepcopy(main_content))
            new_soup = BeautifulSoup("<html><body></body></html>", "lxml")
            new_soup.body.append(cleaned)
            return new_soup
        if type(self) is BaseScraper:
            return soup
        return BeautifulSoup("<html><body></body></html>", "lxml")

    def extract_main_content(self, soup):
        return None


SCRAPER_SELECTORS = {
    "HuggingfaceScraper": ("div", {"class_": "prose-doc prose relative mx-auto max-w-4xl break-words"}),
    "ReadthedocsScraper": ("div", {"class_": "rst-content"}),
    "PyTorchScraper": ("article", {"id": "pytorch-article"}),
    "TileDBScraper": ("main", {"id": "content"}),
    "RstContentScraper": ("div", {"class_": "rst-content"}),
    "FuroThemeScraper": ("article", {"id": "furo-main-content"}),
    "PydataThemeScraper": ("article", {"class_": "bd-article"}),
    "FastcoreScraper": ("main", {"id": "quarto-document-content", "class_": "content"}),
    "RtdThemeScraper": ("div", {"attrs": {"itemprop": "articleBody"}}),
    "BodyRoleMainScraper": ("div", {"class_": "body", "attrs": {"role": "main"}}),
    "ArticleMdContentInnerMdTypesetScraper": ("article", {"class_": "md-content__inner md-typeset"}),
    "DivClassDocumentScraper": ("div", {"class_": "document"}),
    "MainIdMainContentRoleMainScraper": ("main", {"id": "main-content", "attrs": {"role": "main"}}),
    "DivIdMainContentRoleMainScraper": ("div", {"id": "main-content", "attrs": {"role": "main"}}),
    "MainScraper": ("main", {}),
    "DivClassThemeDocMarkdownMarkdownScraper": ("div", {"class_": ["theme-doc-markdown", "markdown"]}),
    "DivIdContentScraper": ("div", {"id": "content"}),
    "DivIdDocsBodyScraper": ("div", {"id": "docs-body"}),
    "DivClassTdContentScraper": ("div", {"class_": "td-content"}),
    "BodyScraper": ("body", {}),
    "ArticleRoleMainScraper": ("article", {"attrs": {"role": "main"}}),
    "ArticleClassMainContent8zFCHScraper": ("article", {"class_": "main_content__8zFCH"}),
}


class SelectorScraper(BaseScraper):
    def __init__(self, url, folder, selector_key):
        super().__init__(url, folder)
        tag, kwargs = SCRAPER_SELECTORS[selector_key]
        self._tag = tag
        kwargs = dict(kwargs)
        attrs = kwargs.pop("attrs", None)
        if attrs:
            kwargs["attrs"] = attrs
        self._kwargs = kwargs

    def extract_main_content(self, soup):
        return soup.find(self._tag, **self._kwargs)


class HuggingfacePacedScraper(SelectorScraper):
    follows_rate_limit_header = True

    def __init__(self, url, folder):
        super().__init__(url, folder, "HuggingfaceScraper")


class PymupdfScraper(BaseScraper):
    def extract_main_content(self, soup):
        article_container = soup.find("div", class_="article-container")
        if article_container:
            return article_container.find("section")
        return None


_MINTLIFY_MDX_COMPONENTS = {
    "Tabs": "UNWRAP",
    "Tab": "HEADING",
    "CodeGroup": "UNWRAP",
    "CardGroup": "UNWRAP",
    "Card": "HEADING",
    "Steps": "UNWRAP",
    "Step": "HEADING",
    "AccordionGroup": "UNWRAP",
    "Accordion": "HEADING",
    "Frame": "UNWRAP",
    "Tooltip": "UNWRAP",
    "Note": "QUOTE",
    "Warning": "QUOTE",
    "Tip": "QUOTE",
    "Info": "QUOTE",
    "Caution": "QUOTE",
}

_MINTLIFY_DOC_INDEX_RE = re.compile(
    r"^> ## Documentation Index\n(?:>.*\n)+\n*",
    flags=re.MULTILINE,
)
_MINTLIFY_FENCE_OPEN_RE = re.compile(r"^(\s*)```(\S+)(\s+.*)?$")
_MINTLIFY_FENCE_KV_ATTR_RE = re.compile(r"\s+\w+=(?:\{[^}]*\}|\S+)")
_MINTLIFY_TITLE_ATTR_RE = re.compile(r'\btitle="([^"]*)"')
_MINTLIFY_FENCE_TOGGLE_RE = re.compile(r"^\s*```")
_MINTLIFY_INLINE_CODE_RE = re.compile(r"(`+[^`]*`+)")
_MINTLIFY_LEFTOVER_TAG_RE = re.compile(r"</?[A-Z][A-Za-z0-9]*(?:\s[^>]*?)?/?>")


def _mintlify_open_indent(text, start):
    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[line_start:start]
    return len(prefix) if not prefix.strip() else 0


def _mintlify_dedent(block, width):
    if width <= 0:
        return textwrap.dedent(block)
    out = []
    for line in block.split("\n"):
        i = 0
        while i < width and i < len(line) and line[i] == " ":
            i += 1
        out.append(line[i:])
    return "\n".join(out)


def _mintlify_is_inline(m):
    if "\n" in m.group(0):
        return False
    line_start = m.string.rfind("\n", 0, m.start()) + 1
    return bool(m.string[line_start:m.start()].strip())


def _mintlify_unwrap(md, name):
    pat = re.compile(rf"<{name}(\s[^>]*)?>(.*?)</{name}>", re.DOTALL)
    while True:
        def repl(m):
            if _mintlify_is_inline(m):
                return m.group(2).strip()
            inner = _mintlify_dedent(m.group(2), _mintlify_open_indent(m.string, m.start())).strip("\n")
            return "\n\n" + inner + "\n\n"
        new = pat.sub(repl, md)
        if new == md:
            return new
        md = new


def _mintlify_heading(md, name):
    pat = re.compile(rf"<{name}(\s[^>]*)?>(.*?)</{name}>", re.DOTALL)
    while True:
        def repl(m):
            attrs = m.group(1) or ""
            inner = _mintlify_dedent(m.group(2), _mintlify_open_indent(m.string, m.start())).strip("\n")
            tm = _MINTLIFY_TITLE_ATTR_RE.search(attrs)
            if tm:
                title = tm.group(1).replace("#", r"\#")
                return f"\n\n## {title}\n\n{inner}\n\n"
            return f"\n\n{inner}\n\n"
        new = pat.sub(repl, md)
        if new == md:
            return new
        md = new


def _mintlify_quote(md, name):
    pat = re.compile(rf"<{name}(\s[^>]*)?>(.*?)</{name}>", re.DOTALL)
    while True:
        def repl(m):
            inner = _mintlify_dedent(m.group(2), _mintlify_open_indent(m.string, m.start())).strip("\n")
            inner = _mintlify_normalize_fences(inner)
            body = markdown.markdown(
                inner,
                extensions=["fenced_code", "tables", "attr_list"],
                output_format="html",
            )
            return f"\n\n<blockquote>\n{body}\n</blockquote>\n\n"
        new = pat.sub(repl, md)
        if new == md:
            return new
        md = new


def _mintlify_normalize_fences(md_text):
    out = []
    for line in md_text.split("\n"):
        m = _MINTLIFY_FENCE_OPEN_RE.match(line)
        if not m:
            out.append(line)
            continue
        indent, lang, rest = m.group(1), m.group(2), (m.group(3) or "")
        label = _MINTLIFY_FENCE_KV_ATTR_RE.sub("", rest).strip()
        if label:
            out.append(f"{indent}**{label}**")
            out.append("")
        out.append(f"{indent}```{lang}")
    return "\n".join(out)


def _mintlify_strip_leftover_tags(md_text):
    out = []
    in_fence = False
    for line in md_text.split("\n"):
        if _MINTLIFY_FENCE_TOGGLE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        parts = _MINTLIFY_INLINE_CODE_RE.split(line)
        for i in range(0, len(parts), 2):
            parts[i] = _MINTLIFY_LEFTOVER_TAG_RE.sub("", parts[i])
        out.append("".join(parts))
    return "\n".join(out)


_MINTLIFY_ALERT_RE = re.compile(
    r"^(\s*>\s*)\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION|INFO|DANGER|SUCCESS|CHECK)\][ \t]*(.*)$",
    re.IGNORECASE,
)
_OPENAPI_BLOCK_RE = re.compile(
    r"^## OpenAPI[ \t]*\n+(`{3,})ya?ml[ \t]+\S+[ \t]+(\w+)[ \t]+(\S+)[ \t]*\n(.*?)\n\1[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_OPENAPI_SPEC_SUFFIXES = (".yaml", ".yml", ".json")
_MINTLIFY_NON_PAGE_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp", ".avif", ".pdf", ".zip", ".gz", ".tgz",
    ".mp4", ".webm", ".mov", ".mp3", ".wav", ".css", ".js", ".woff", ".woff2", ".ttf", ".txt", ".csv", ".xml",
)
_OPENAPI_HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
_OPENAPI_FENCE_LANGS = {"curl": "bash", "shell": "bash", "sh": "bash", "c#": "csharp", "node": "javascript", "js": "javascript"}
_OPENAPI_MAX_DEPTH = 6
_OPENAPI_MAX_EXAMPLE_CHARS = 3000
_OPENAPI_MAX_INLINE_EXAMPLE_CHARS = 120
_MARKDOWN_EXTENSIONS = ["fenced_code", "tables", "attr_list"]


def _mintlify_strip_injected_headers(md_text):
    lines = md_text.split("\n")
    i = 0
    while True:
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines) and lines[i].startswith("> ## "):
            while i < len(lines) and lines[i].startswith(">"):
                i += 1
            continue
        return "\n".join(lines[i:])


def _mintlify_convert_alerts(md_text):
    lines = md_text.split("\n")
    out = []
    in_fence = False
    for i, line in enumerate(lines):
        if line is None:
            continue
        if _MINTLIFY_FENCE_TOGGLE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            m = _MINTLIFY_ALERT_RE.match(line)
            if m:
                label = m.group(2).capitalize()
                rest = m.group(3).strip()
                if not rest:
                    j = i + 1
                    while j < len(lines) and lines[j] is not None and lines[j].strip() == ">":
                        j += 1
                    if j < len(lines) and lines[j] is not None and re.fullmatch(
                        rf"\s*>\s*\**{label}\**:?\s*", lines[j], flags=re.IGNORECASE
                    ):
                        lines[j] = None
                line = f"{m.group(1)}**{label}:**" + (f" {rest}" if rest else "")
        out.append(line)
    return "\n".join(out)


_HTML_TAG_NAMES = frozenset((
    "a", "abbr", "b", "blockquote", "br", "caption", "center", "code", "col", "colgroup", "dd", "del", "details",
    "div", "dl", "dt", "em", "figcaption", "figure", "font", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img",
    "ins", "kbd", "li", "mark", "ol", "p", "pre", "s", "small", "span", "strong", "sub", "summary", "sup", "table",
    "tbody", "td", "tfoot", "th", "thead", "tr", "u", "ul",
))
_ANGLE_TAG_RE = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9_-]*)([^<>]*)>")


def _escape_placeholders(text):
    def repl(m):
        whole = m.group(0)
        if m.group(2).lower() in _HTML_TAG_NAMES or "://" in whole or "@" in whole:
            return whole
        return "&lt;" + whole[1:-1] + "&gt;"

    out = []
    in_fence = False
    for line in str(text).split("\n"):
        if _MINTLIFY_FENCE_TOGGLE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            parts = _MINTLIFY_INLINE_CODE_RE.split(line)
            for k in range(0, len(parts), 2):
                parts[k] = _ANGLE_TAG_RE.sub(repl, parts[k])
            line = "".join(parts)
        out.append(line)
    return "\n".join(out)


def _block(text):
    return _escape_placeholders(str(text).strip())


def _fenced(code, lang=""):
    code = str(code).rstrip("\n")
    longest = max((len(run) for run in re.findall(r"`{3,}", code)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}{lang}", code, fence]


_MD_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_MD_HEADING_RE = re.compile(r"^(#{1,6})(\s)")


def _flatten_md_tables(text):
    out, table = [], []

    def flush():
        rows = [[cell.strip() for cell in row.strip().strip("|").split("|")]
                for row in table if not _MD_TABLE_SEPARATOR_RE.match(row)]
        if len(rows) >= 2:
            header, sentences = rows[0], []
            for row in rows[1:]:
                if len(header) == 2 and len(row) >= 2:
                    sentence = f"{row[0]}: {row[1].rstrip('.')}"
                else:
                    sentence = "; ".join(f"{h}: {c}" for h, c in zip(header, row) if c).rstrip(".")
                if sentence:
                    sentences.append(sentence + ".")
            out.append(" ".join(sentences))
        else:
            out.extend(table)
        table.clear()

    for line in str(text).split("\n"):
        if line.strip().startswith("|"):
            table.append(line)
            continue
        flush()
        out.append(line)
    flush()
    return "\n".join(out)


def _demote_headings(text, levels):
    out = []
    in_fence = False
    for line in str(text).split("\n"):
        if _MINTLIFY_FENCE_TOGGLE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            line = _MD_HEADING_RE.sub(lambda m: "#" * min(6, len(m.group(1)) + levels) + m.group(2), line, count=1)
        out.append(line)
    return "\n".join(out)


def _inline(text):
    return _escape_placeholders(" ".join(_flatten_md_tables(text).split()))


def _normalized(text):
    return " ".join(str(text).split()).lower()


class _OpenApiRenderer:
    def __init__(self, spec):
        self.spec = spec if isinstance(spec, dict) else {}

    def _lookup(self, ref):
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return None
        node = self.spec
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def resolve(self, node):
        name = None
        seen = set()
        while isinstance(node, dict) and "$ref" in node:
            ref = node["$ref"]
            if ref in seen:
                break
            seen.add(ref)
            target = self._lookup(ref)
            if not isinstance(target, dict):
                break
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            name = ref.rsplit("/", 1)[-1]
            node = {**target, **siblings} if siblings else target
        return (node if isinstance(node, dict) else {}), name

    def merge(self, node, depth=0):
        schema, name = self.resolve(node)
        if "allOf" not in schema or depth > _OPENAPI_MAX_DEPTH:
            return schema, name
        merged = {k: v for k, v in schema.items() if k not in ("allOf", "properties", "required")}
        properties = {}
        required = []
        for part in schema.get("allOf") or []:
            sub, _ = self.merge(part, depth + 1)
            properties.update(sub.get("properties") or {})
            required.extend(sub.get("required") or [])
            for key, value in sub.items():
                if key not in ("properties", "required") and key not in merged:
                    merged[key] = value
        properties.update(schema.get("properties") or {})
        required.extend(schema.get("required") or [])
        if properties:
            merged["properties"] = properties
        if required:
            merged["required"] = list(dict.fromkeys(required))
        return merged, name

    @staticmethod
    def _is_array(schema):
        types = schema.get("type")
        return types == "array" or (isinstance(types, list) and "array" in types) or "items" in schema

    def _has_fields(self, node, depth=0):
        if depth > _OPENAPI_MAX_DEPTH:
            return False
        schema, _ = self.merge(node)
        if schema.get("properties"):
            return True
        if self._is_array(schema):
            return self._has_fields(schema.get("items") or {}, depth + 1)
        options = schema.get("oneOf") or schema.get("anyOf") or []
        if any(self._has_fields(option, depth + 1) for option in options):
            return True
        additional = schema.get("additionalProperties")
        return isinstance(additional, dict) and self._has_fields(additional, depth + 1)

    def type_label(self, node, depth=0):
        schema, name = self.merge(node)
        if depth > _OPENAPI_MAX_DEPTH:
            return name or "object"
        for key in ("oneOf", "anyOf"):
            options = schema.get(key)
            if options:
                labels = [self.type_label(option, depth + 1) for option in options]
                return " or ".join(dict.fromkeys(label for label in labels if label))
        types = schema.get("type")
        if isinstance(types, list):
            return " or ".join(dict.fromkeys(self._single_type(schema, t, name, depth) for t in types))
        return self._single_type(schema, types, name, depth)

    def _single_type(self, schema, type_name, name, depth):
        if type_name == "array" or (type_name is None and "items" in schema):
            return f"array of {self.type_label(schema.get('items') or {}, depth + 1)}"
        if type_name == "object" or (type_name is None and "properties" in schema):
            additional = schema.get("additionalProperties")
            if not schema.get("properties") and isinstance(additional, dict):
                return f"map of {self.type_label(additional, depth + 1)}"
            return f"{name} object" if name else "object"
        if type_name is None:
            if "const" in schema:
                return type(schema["const"]).__name__.replace("str", "string")
            return name or "any"
        fmt = schema.get("format")
        return f"{type_name} ({fmt})" if fmt else str(type_name)

    @staticmethod
    def _value(value):
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    @staticmethod
    def constraints(schema):
        notes = []
        for key, label in (("deprecated", "deprecated"), ("readOnly", "read-only"), ("writeOnly", "write-only")):
            if schema.get(key):
                notes.append(label)
        for key, label in (("minLength", "min length"), ("maxLength", "max length"), ("minimum", "minimum"),
                           ("maximum", "maximum"), ("minItems", "min items"), ("maxItems", "max items")):
            if key in schema:
                notes.append(f"{label} {schema[key]}")
        if "pattern" in schema:
            notes.append(f"pattern `{schema['pattern']}`")
        return notes

    def extras(self, schema):
        parts = []
        if "const" in schema:
            parts.append(f"Always `{self._value(schema['const'])}`.")
        if schema.get("enum"):
            parts.append("Allowed values: " + ", ".join(f"`{self._value(v)}`" for v in schema["enum"]) + ".")
        default = None
        for key in ("default", "x-default"):
            if key in schema:
                default = schema[key]
                parts.append(f"Default: `{self._value(default)}`.")
                break
        example = schema.get("example")
        if example is None and isinstance(schema.get("examples"), list) and schema["examples"]:
            example = schema["examples"][0]
        if example is not None and example != default and not isinstance(example, (dict, list)):
            text = _inline(self._value(example))
            if text and len(text) <= _OPENAPI_MAX_INLINE_EXAMPLE_CHARS:
                parts.append(f"Example: `{text}`.")
        return parts

    def field(self, prop, node, is_required, level, expanded, depth):
        schema, _ = self.merge(node)
        details = [self.type_label(node), "required" if is_required else "optional"] + self.constraints(schema)
        sentences = []
        description = _inline(schema.get("description") or schema.get("title") or "")
        if description:
            sentences.append(description)
        sentences.extend(self.extras(schema))
        if self._is_array(schema) and "enum" not in schema:
            items, _ = self.merge(schema.get("items") or {})
            if items.get("enum"):
                sentences.extend(s for s in self.extras(items) if s.startswith("Allowed"))
        line = f"{'    ' * level}- `{prop}` ({', '.join(details)})"
        if sentences:
            line += ": " + " ".join(sentences)
        return [line] + self.children(node, level + 1, expanded, depth + 1)

    def field_lines(self, schema, level, expanded, depth=0):
        required = set(schema.get("required") or [])
        lines = []
        for prop, sub in (schema.get("properties") or {}).items():
            lines.extend(self.field(prop, sub, prop in required, level, expanded, depth))
        return lines

    def children(self, node, level, expanded, depth):
        if depth > _OPENAPI_MAX_DEPTH:
            return []
        schema, name = self.merge(node)
        indent = "    " * level
        if self._is_array(schema):
            return self.children(schema.get("items") or {}, level, expanded, depth)
        options = schema.get("oneOf") or schema.get("anyOf")
        if options:
            with_fields = [option for option in options if self._has_fields(option)]
            if len(with_fields) == 1:
                return self.children(with_fields[0], level, expanded, depth)
            lines = []
            for option in with_fields:
                option_schema, option_name = self.merge(option)
                label = option_name or option_schema.get("title") or "object"
                description = _inline(option_schema.get("description") or "")
                lines.append(f"{indent}- One option, `{label}`" + (f": {description}" if description else ""))
                lines.extend(self.children(option, level + 1, expanded, depth + 1))
            return lines
        if schema.get("properties"):
            if name:
                if name in expanded:
                    return [f"{indent}- The fields of `{name}` are listed above."]
                expanded.add(name)
            return self.field_lines(schema, level, expanded, depth)
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict) and self._has_fields(additional):
            return self.children(additional, level, expanded, depth)
        return []

    def body_lines(self, node, expanded):
        schema, name = self.merge(node)
        lines = [f"Type: {self.type_label(node)}."]
        description = _block(schema.get("description") or "")
        if description:
            lines += ["", description]
        if schema.get("properties"):
            if name:
                expanded.add(name)
            fields = self.field_lines(schema, 0, expanded)
        else:
            fields = self.children(node, 0, expanded, 1)
        if fields:
            lines += ["", *fields]
        return lines

    def media_example(self, media):
        value = None
        if "example" in media:
            value = media["example"]
        elif isinstance(media.get("examples"), dict) and media["examples"]:
            first, _ = self.resolve(next(iter(media["examples"].values())))
            value = first.get("value")
        else:
            schema, _ = self.merge(media.get("schema") or {})
            if isinstance(schema.get("example"), (dict, list)):
                value = schema["example"]
        if value is None:
            return None
        text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
        if len(text) > _OPENAPI_MAX_EXAMPLE_CHARS:
            text = text[:_OPENAPI_MAX_EXAMPLE_CHARS].rstrip() + "\n..."
        return text

    def parameter_line(self, param):
        schema = param.get("schema") or {}
        merged, _ = self.merge(schema)
        details = [param.get("in") or "parameter", self.type_label(schema),
                   "required" if param.get("required") else "optional"] + self.constraints(merged)
        if param.get("deprecated") and "deprecated" not in details:
            details.append("deprecated")
        sentences = []
        description = _inline(param.get("description") or merged.get("description") or "")
        if description:
            sentences.append(description)
        sentences.extend(self.extras({**merged, **({"example": param["example"]} if "example" in param else {})}))
        line = f"- `{param.get('name')}` ({', '.join(details)})"
        if sentences:
            line += ": " + " ".join(sentences)
        return line

    def base_url(self, op, path_item):
        servers = op.get("servers") or path_item.get("servers") or self.spec.get("servers") or []
        if servers and isinstance(servers[0], dict):
            return str(servers[0].get("url") or "").rstrip("/")
        return ""

    def webhook(self, name, page_text="", title="Webhook"):
        item = (self.spec.get("webhooks") or {}).get(name)
        if not isinstance(item, dict):
            return ""
        for method in _OPENAPI_HTTP_METHODS:
            if isinstance(item.get(method), dict):
                return self.operation(method, name, page_text, path_item=item, title=title)
        return ""

    def operation(self, method, path, page_text="", path_item=None, title="Endpoint"):
        is_webhook = path_item is not None
        if path_item is None:
            path_item = (self.spec.get("paths") or {}).get(path) or {}
        op = path_item.get(method)
        if not isinstance(op, dict):
            raise KeyError(f"{method.upper()} {path} is not in the OpenAPI block")
        if is_webhook:
            target = f"Event `{path}`, sent as an HTTP `{method.upper()}` request to your webhook URL."
        else:
            target = f"`{method.upper()} {self.base_url(op, path_item)}{path}`"
        out = [f"## {title}", "", target, ""]
        if op.get("deprecated"):
            out += ["**Deprecated.**", ""]
        description = str(op.get("description") or "").strip()
        if description and _normalized(description) not in _normalized(page_text):
            out += [_block(description), ""]

        params = {}
        for raw in list(path_item.get("parameters") or []) + list(op.get("parameters") or []):
            param, _ = self.resolve(raw)
            if param.get("name"):
                params[(param.get("name"), param.get("in"))] = param
        if params:
            out += ["### Parameters", ""] + [self.parameter_line(p) for p in params.values()] + [""]

        body, _ = self.resolve(op.get("requestBody") or {})
        content = body.get("content") or {}
        if content:
            content_type, media = next(iter(content.items()))
            note = f"Content type: `{content_type}`, {'required' if body.get('required') else 'optional'}."
            body_description = _block(body.get("description") or "")
            out += [f"### {'Payload' if is_webhook else 'Request body'}", "",
                    note + (f" {body_description}" if body_description else ""), ""]
            if media.get("schema"):
                out += self.body_lines(media["schema"], set()) + [""]
            example = self.media_example(media)
            if example:
                out += ["Example request body:", "", *_fenced(example, "json"), ""]

        responses = op.get("responses") or {}
        if responses:
            out += ["### Responses", ""]
            for code, raw in responses.items():
                response, _ = self.resolve(raw)
                out += [f"#### {code}", ""]
                response_description = _block(response.get("description") or "")
                if response_description:
                    out += [response_description, ""]
                media_types = response.get("content") or {}
                if media_types:
                    media = next(iter(media_types.values()))
                    if media.get("schema"):
                        out += self.body_lines(media["schema"], set()) + [""]
                    if str(code).startswith("2"):
                        example = self.media_example(media)
                        if example:
                            out += ["Example response:", "", *_fenced(example, "json"), ""]

        samples = op.get("x-codeSamples") or op.get("x-code-samples") or []
        if samples:
            out += ["### Code samples", ""]
            for sample in samples:
                if not isinstance(sample, dict) or not sample.get("source"):
                    continue
                label = sample.get("label") or sample.get("lang") or "Example"
                lang = str(sample.get("lang") or "").lower()
                out += [f"**{label}**", "", *_fenced(sample["source"], _OPENAPI_FENCE_LANGS.get(lang, lang)), ""]

        external = op.get("externalDocs")
        if isinstance(external, dict) and external.get("url"):
            out += [f"See also: [{external.get('description') or external['url']}]({external['url']})", ""]
        return "\n".join(out)

    def overview(self):
        info = self.spec.get("info") or {}
        out = [f"# {info.get('title') or 'API reference'}", ""]
        if info.get("version"):
            out += [f"Version: `{info['version']}`", ""]
        if info.get("description"):
            out += [_block(info["description"]), ""]
        base = self.base_url({}, {})
        if base:
            out += [f"Base URL: `{base}`", ""]
        schemes = ((self.spec.get("components") or {}).get("securitySchemes") or {})
        auth = []
        for scheme_name, raw in schemes.items():
            scheme, _ = self.resolve(raw)
            where = " ".join(str(scheme.get(k)) for k in ("type", "in", "name", "scheme") if scheme.get(k))
            text = f"- `{scheme_name}` ({where})" if where else f"- `{scheme_name}`"
            if scheme.get("description"):
                text += f": {_inline(scheme['description'])}"
            auth.append(text)
        if auth:
            out += ["## Authentication", "", *auth, ""]
        for tag in self.spec.get("tags") or []:
            if not isinstance(tag, dict) or not tag.get("name"):
                continue
            out += [f"## {tag['name']}", ""]
            if tag.get("description"):
                out += [_demote_headings(_block(tag["description"]), 2), ""]
            external = tag.get("externalDocs")
            if isinstance(external, dict) and external.get("url"):
                out += [f"See also: [{external.get('description') or external['url']}]({external['url']})", ""]
        endpoints = []
        for path, item in (self.spec.get("paths") or {}).items():
            if not isinstance(item, dict):
                continue
            for method, op in item.items():
                if method.lower() in _OPENAPI_HTTP_METHODS and isinstance(op, dict):
                    summary = _inline(op.get("summary") or op.get("operationId") or "")
                    endpoints.append(f"- `{method.upper()} {path}`" + (f": {summary}" if summary else ""))
        if endpoints:
            out += ["## Endpoints", "", *endpoints, ""]
        for name in self.spec.get("webhooks") or {}:
            rendered = self.webhook(name, title=f"Webhook: {name}")
            if rendered:
                out += [rendered, ""]
        return "\n".join(out)


def _render_openapi_blocks(md_text):
    def repl(m):
        try:
            renderer = _OpenApiRenderer(yaml.safe_load(m.group(4)))
            method, target, page_text = m.group(2).lower(), m.group(3), m.string[:m.start()]
            if method == "webhook":
                return renderer.webhook(target, page_text)
            return renderer.operation(method, target, page_text)
        except Exception:
            return m.group(0)
    return _OPENAPI_BLOCK_RE.sub(repl, md_text)


def render_openapi_spec_overview(text):
    spec = yaml.safe_load(text)
    if not isinstance(spec, dict) or not ("openapi" in spec or "swagger" in spec):
        raise ValueError("not an OpenAPI document")
    md_text = _mintlify_convert_alerts(_OpenApiRenderer(spec).overview())
    return markdown.markdown(md_text, extensions=_MARKDOWN_EXTENSIONS, output_format="html")


def render_mintlify_markdown(md_text):
    md_text = _mintlify_strip_injected_headers(md_text)
    md_text = _MINTLIFY_DOC_INDEX_RE.sub("", md_text, count=1)
    for name, action in _MINTLIFY_MDX_COMPONENTS.items():
        if action == "UNWRAP":
            md_text = _mintlify_unwrap(md_text, name)
        elif action == "HEADING":
            md_text = _mintlify_heading(md_text, name)
        elif action == "QUOTE":
            md_text = _mintlify_quote(md_text, name)
    md_text = _mintlify_strip_leftover_tags(md_text)
    md_text = _mintlify_convert_alerts(md_text)
    md_text = _render_openapi_blocks(md_text)
    md_text = _mintlify_normalize_fences(md_text)
    return markdown.markdown(
        md_text,
        extensions=_MARKDOWN_EXTENSIONS,
        output_format="html",
    )


class MintlifyScraper(BaseScraper):
    async def collect_seed_urls(self, session):
        parsed_seed = urlparse(self.url)
        seed_prefix = parsed_seed.path.rstrip("/")
        llms_url = f"{parsed_seed.scheme}://{parsed_seed.netloc}/llms.txt"
        try:
            resp = await session.get(llms_url, timeout=30, allow_redirects=True)
        except Exception:
            return []
        if resp.status_code != 200:
            return []
        urls = []
        for line in resp.text.split("\n"):
            m = re.search(r"\((https?://[^)\s]+\.md)\)", line)
            if not m:
                continue
            base_url = m.group(1)[:-3]
            p = urlparse(base_url)
            if p.netloc != parsed_seed.netloc:
                continue
            if seed_prefix and not p.path.startswith(seed_prefix):
                continue
            urls.append(base_url)
        return urls

    def canonical_url(self, url):
        return url.rstrip("/")

    def fetch_url_for(self, url):
        u = url.rstrip("/")
        if u.endswith(".md") or u.lower().endswith(_OPENAPI_SPEC_SUFFIXES):
            return u
        parts = urlsplit(u)
        if not parts.path:
            return urlunsplit((parts.scheme, parts.netloc, "/index.md", "", ""))
        return u + ".md"

    def page_url_for(self, final_url):
        u = str(final_url).split("#", 1)[0].split("?", 1)[0].rstrip("/")
        return u[:-3] if u.endswith(".md") else u

    def wants_url(self, url):
        return not urlsplit(url).path.rstrip("/").lower().endswith(_MINTLIFY_NON_PAGE_SUFFIXES)

    def transform_response(self, text, url):
        head = text.lstrip()[:200].lower()
        if head.startswith("<!doctype") or head.startswith("<html"):
            return text
        if url.rstrip("/").lower().endswith(_OPENAPI_SPEC_SUFFIXES):
            return f"<div>{render_openapi_spec_overview(text)}</div>"
        rendered = render_mintlify_markdown(text)
        return f"<div>{rendered}</div>"

    def extract_main_content(self, soup):
        div = soup.find("div")
        if div is not None:
            return div
        return soup.body if soup.body else soup


class DivIdContentSecondScraper(BaseScraper):
    def extract_main_content(self, soup):
        content_divs = soup.find_all("div", id="content")
        if len(content_divs) >= 2:
            return content_divs[1]
        return None


class PropCacheScraper(BaseScraper):
    def __init__(self, url, folder):
        super().__init__(url, folder)

        if self.url.rstrip("/").endswith("propcache.aio-libs.org"):
            self.url = urljoin(self.url, "en/latest/")

        if not self.url.endswith("/"):
            self.url += "/"

        self.base_url = self.url

    def extract_main_content(self, soup):
        return soup.find("div", class_="body", attrs={"role": "main"})


class FileDownloader(BaseScraper):

    def extract_main_content(self, soup):
        return None

    async def save_file(self, content: bytes, url: str, save_dir: str):
        from pathlib import Path

        basename = Path(url).name or "download"
        filename = os.path.join(save_dir, basename)

        async with aiofiles.open(filename, "wb") as f:
            await f.write(content)


class ScraperRegistry:
    _special_scrapers = {
        "BaseScraper": BaseScraper,
        "PymupdfScraper": PymupdfScraper,
        "DivIdContentSecondScraper": DivIdContentSecondScraper,
        "PropCacheScraper": PropCacheScraper,
        "MintlifyScraper": MintlifyScraper,
        "HuggingfacePacedScraper": HuggingfacePacedScraper,
        "FileDownloader": FileDownloader,
    }

    @classmethod
    def get_scraper(cls, scraper_name):
        if scraper_name in cls._special_scrapers:
            return cls._special_scrapers[scraper_name]
        if scraper_name in SCRAPER_SELECTORS:
            key = scraper_name
            return lambda url, folder: SelectorScraper(url, folder, key)
        return BaseScraper


def _parse_rate_limit(value):
    if not value:
        return None
    remaining = re.search(r"\br=(\d+)", value)
    reset_in = re.search(r"\bt=(\d+)", value)
    if not remaining or not reset_in:
        return None
    return int(remaining.group(1)), int(reset_in.group(1))


class _RateLimitPacer:
    SAFETY_MARGIN = 2
    UNKNOWN_SPACING = 1.0
    RESET_GRACE = 1.0
    FALLBACK_WAIT = 60

    def __init__(self):
        self._lock = asyncio.Lock()
        self._remaining = None
        self._reset_at = 0.0
        self._next_at = 0.0

    async def wait_turn(self):
        loop = asyncio.get_running_loop()
        async with self._lock:
            while True:
                now = loop.time()
                if self._remaining is not None and now >= self._reset_at:
                    self._remaining = None
                if self._remaining is not None and self._remaining <= self.SAFETY_MARGIN:
                    await asyncio.sleep(self._reset_at - now + self.RESET_GRACE)
                    continue
                if self._next_at > now:
                    await asyncio.sleep(self._next_at - now)
                    continue
                break
            if self._remaining is None:
                self._next_at = now + self.UNKNOWN_SPACING
            else:
                self._next_at = now + (self._reset_at - now) / (self._remaining - self.SAFETY_MARGIN)
                self._remaining -= 1

    def observe(self, response):
        limit = _parse_rate_limit(response.headers.get("ratelimit"))
        if limit is None:
            return
        remaining, reset_in = limit
        reset_at = asyncio.get_running_loop().time() + reset_in
        if self._remaining is not None and abs(reset_at - self._reset_at) < 2:
            remaining = min(remaining, self._remaining)
        self._remaining = remaining
        self._reset_at = reset_at

    def hold(self, response):
        limit = _parse_rate_limit(response.headers.get("ratelimit"))
        if limit is not None:
            wait = limit[1]
        else:
            retry_after = str(response.headers.get("retry-after", ""))
            wait = int(retry_after) if retry_after.isdigit() else self.FALLBACK_WAIT
        now = asyncio.get_running_loop().time()
        already_held = (
            self._remaining is not None
            and self._remaining <= self.SAFETY_MARGIN
            and self._reset_at > now
        )
        self._remaining = 0
        self._reset_at = max(self._reset_at, now + wait) if already_held else now + wait
        return not already_held


class ScraperWorker(QObject):
    status_updated = Signal(str, str)
    scraping_finished = Signal(str, bool, bool)

    RATE_LIMIT_THRESHOLD = 5
    MAX_WIN_PATH = 250

    def __init__(self, url, folder, scraper_class=BaseScraper, name="", resume=False):
        super().__init__()
        self.url = url
        self.folder = folder
        self.name = name
        self.scraper = scraper_class(url, folder)
        self.save_dir = self.scraper.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.stats = {"scraped": 0}
        self._loop = None
        self._task = None
        self._cancelled = False
        self._rate_limited = False
        self._429s_since_last_success = 0
        self.resume = resume
        self._log_lock = None
        self._pacer = _RateLimitPacer() if getattr(self.scraper, "follows_rate_limit_header", False) else None

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._task = self._loop.create_task(self.crawl_domain())
            try:
                self._loop.run_until_complete(self._task)
            except asyncio.CancelledError:
                pass
        finally:
            if not self._cancelled and not self._rate_limited:
                self._finalize_clean_run()
            self.cleanup()
            self._loop.close()
            self.scraping_finished.emit(self.name, self._cancelled, self._rate_limited)

    def _finalize_clean_run(self):
        try:
            for fname in os.listdir(self.save_dir):
                if fname.endswith(".links.json"):
                    try:
                        os.remove(os.path.join(self.save_dir, fname))
                    except Exception:
                        pass
        except Exception:
            pass
        log_file = os.path.join(self.save_dir, "failed_urls.log")
        try:
            if os.path.exists(log_file) and os.path.getsize(log_file) == 0:
                os.remove(log_file)
        except Exception:
            pass

    def cancel(self):
        self._cancelled = True
        if self._loop and self._task and not self._task.done():
            self._loop.call_soon_threadsafe(self._task.cancel)

    def count_saved_files(self):
        return len([f for f in os.listdir(self.save_dir) if f.endswith(".html")])

    def _emit_count(self):
        self.status_updated.emit(self.name, str(self.stats["scraped"]))

    def _filename_key(self, url):
        if hasattr(self.scraper, "canonical_url"):
            return self.scraper.canonical_url(url)
        return url

    def _html_path(self, url):
        name = self.sanitize_filename(self._filename_key(url))
        path = os.path.join(self.save_dir, name + ".html")
        if os.path.exists(path) and os.path.basename(os.path.realpath(path)) != name + ".html":
            suffix = "_" + hashlib.md5(name.encode()).hexdigest()[:8]
            allowed = max(1, self.MAX_WIN_PATH - len(self.save_dir) - len(os.sep) - len(".html") - len(suffix))
            path = os.path.join(self.save_dir, name[:allowed] + suffix + ".html")
        return path

    async def crawl_domain(
        self,
        max_concurrent_requests: int = 20,
        batch_size: int = 50,
        page_limit: int = 5_000,
    ):
        parsed_url = urlparse(self.url)
        acceptable_domain = parsed_url.netloc
        acceptable_domain_extension = parsed_url.path.rstrip("/")

        log_file = os.path.join(self.save_dir, "failed_urls.log")

        self.stats["scraped"] = self.count_saved_files()

        semaphore = asyncio.BoundedSemaphore(max_concurrent_requests)
        visited = set()

        if self.resume:
            to_visit = self._build_resume_queue(log_file)
        else:
            to_visit = [self.url]

        async def process_batch(batch_urls, session):
            pending = [
                (u, self.fetch(
                    session,
                    u,
                    acceptable_domain,
                    semaphore,
                    self.save_dir,
                    log_file,
                    acceptable_domain_extension,
                ))
                for u in batch_urls
                if u not in visited
            ]
            urls_for_tasks = [u for u, _ in pending]
            tasks = [t for _, t in pending]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            visited.update(batch_urls)
            out = []
            for url, r in zip(urls_for_tasks, results):
                if isinstance(r, set):
                    out.append(r)
                elif isinstance(r, Exception):
                    print(f"Scrape task for {url} raised {type(r).__name__}: {r}")
                    try:
                        await self.log_failed_url(url, log_file)
                    except Exception:
                        pass
            return out

        async with AsyncSession(impersonate="chrome") as session:
            if hasattr(self.scraper, "collect_seed_urls"):
                try:
                    extra = await self.scraper.collect_seed_urls(session)
                    if extra:
                        already = set(to_visit)
                        for u in extra:
                            if u not in already:
                                to_visit.append(u)
                                already.add(u)
                except Exception as e:
                    print(f"collect_seed_urls failed: {type(e).__name__}: {e}")

            while to_visit:
                if self._cancelled or self._rate_limited:
                    break
                current_batch = to_visit[:batch_size]
                to_visit = to_visit[batch_size:]

                for new_links in await process_batch(current_batch, session):
                    new_to_visit = new_links - visited
                    to_visit.extend(new_to_visit)

                if self._rate_limited:
                    break

                await asyncio.sleep(0.2)

                if len(visited) >= page_limit:
                    break

        return visited

    def _build_resume_queue(self, log_file):
        candidates = set()
        try:
            for fname in os.listdir(self.save_dir):
                if fname.endswith(".links.json"):
                    try:
                        with open(os.path.join(self.save_dir, fname), "r", encoding="utf-8") as f:
                            for link in json.load(f):
                                if isinstance(link, str):
                                    candidates.add(link)
                    except Exception:
                        pass
        except Exception:
            pass
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            candidates.add(line)
            except Exception:
                pass
            try:
                os.remove(log_file)
            except Exception:
                pass
        candidates.add(self.url)
        return list(candidates)

    async def fetch(
        self,
        session,
        url,
        base_domain,
        semaphore,
        save_dir,
        log_file,
        acceptable_domain_extension,
        retries: int = 3,
    ):
        if os.path.exists(self._html_path(url)):
            return set()

        fetch_url = (
            self.scraper.fetch_url_for(url)
            if hasattr(self.scraper, "fetch_url_for")
            else url
        )
        has_response_transform = hasattr(self.scraper, "transform_response")

        async with semaphore:
            for attempt in range(1, retries + 1):
                if self._cancelled:
                    return set()
                if self._rate_limited:
                    await self.log_failed_url(url, log_file)
                    return set()
                try:
                    response = await self._get(session, fetch_url)
                except (asyncio.TimeoutError, RequestsError, OSError):
                    if attempt == retries:
                        await self.log_failed_url(url, log_file)
                        self._emit_count()
                    await asyncio.sleep(2)
                    continue

                if response.status_code == 429:
                    self._429s_since_last_success += 1
                    if self._429s_since_last_success >= self.RATE_LIMIT_THRESHOLD:
                        self._rate_limited = True
                    await self.log_failed_url(url, log_file)
                    self._emit_count()
                    return set()

                if response.status_code != 200:
                    await self.log_failed_url(url, log_file)
                    self._emit_count()
                    return set()

                self._429s_since_last_success = 0

                save_url = url
                if hasattr(self.scraper, "page_url_for"):
                    final_page = self.scraper.page_url_for(getattr(response, "url", None) or fetch_url)
                    if final_page and self._filename_key(final_page) != self._filename_key(url):
                        if os.path.exists(self._html_path(final_page)):
                            self._emit_count()
                            return set()
                        save_url = final_page

                content_type = response.headers.get("content-type", "").lower()
                if not has_response_transform and "text/html" not in content_type:
                    self._emit_count()
                    return set()

                html = response.text
                if has_response_transform:
                    try:
                        html = self.scraper.transform_response(html, save_url)
                    except Exception as e:
                        print(f"transform_response failed for {url}: {type(e).__name__}: {e}")
                        await self.log_failed_url(url, log_file)
                        self._emit_count()
                        return set()

                try:
                    links = self.extract_links(
                        html, save_url, base_domain, acceptable_domain_extension
                    )
                    created = await self.save_html(html, save_url, save_dir, links=links)
                except Exception:
                    await self.log_failed_url(url, log_file)
                    self._emit_count()
                    return set()
                if created:
                    self.stats["scraped"] += 1
                self._emit_count()
                return links
        return set()

    async def _get(self, session, fetch_url):
        if self._pacer is None:
            return await session.get(fetch_url, timeout=30, allow_redirects=True)
        while True:
            await self._pacer.wait_turn()
            response = await session.get(fetch_url, timeout=30, allow_redirects=True)
            if response.status_code != 429:
                self._pacer.observe(response)
                return response
            if self._pacer.hold(response):
                self._429s_since_last_success += 1
                if self._429s_since_last_success >= self.RATE_LIMIT_THRESHOLD:
                    self._rate_limited = True
            if self._rate_limited or self._cancelled:
                return response

    async def save_html(self, content, url, save_dir, links=None):
        filename = self._html_path(url)
        soup = BeautifulSoup(content, "lxml")
        processed_soup = self.scraper.process_html(soup)

        source_link = processed_soup.new_tag("a", href=url)
        source_link.string = "Original Source"

        if processed_soup.body:
            processed_soup.body.insert(0, source_link)
        elif processed_soup.html:
            new_body = processed_soup.new_tag("body")
            new_body.insert(0, source_link)
            processed_soup.html.insert(0, new_body)
        else:
            new_html = processed_soup.new_tag("html")
            new_body = processed_soup.new_tag("body")
            new_body.insert(0, source_link)
            new_html.insert(0, new_body)
            processed_soup.insert(0, new_html)

        created = False
        while not created:
            try:
                async with aiofiles.open(filename, "x", encoding="utf-8") as f:
                    await f.write(str(processed_soup))
                created = True
            except FileExistsError:
                retry = self._html_path(url)
                if retry == filename:
                    break
                filename = retry

        if links:
            sidecar = filename[:-5] + ".links.json"
            tmp = sidecar + ".tmp"
            try:
                async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(sorted(links)))
                await asyncio.to_thread(os.replace, tmp, sidecar)
            except Exception:
                try:
                    await asyncio.to_thread(os.remove, tmp)
                except Exception:
                    pass

        return created

    def sanitize_filename(self, url: str) -> str:
        original_url = url

        base_url = url.split("?", 1)[0].split("#", 1)[0]

        for open_br, close_br in ("[]", "()"):
            while open_br in base_url and close_br in base_url:
                start, end = base_url.find(open_br), base_url.find(close_br)
                if 0 <= start < end:
                    base_url = base_url[:start] + base_url[end + 1 :]

        filename = (
            base_url.replace("https://", "")
            .replace("http://", "")
            .replace("/", "_")
            .replace("\\", "_")
        )
        for ch in '<>:"|?*':
            filename = filename.replace(ch, "_")
        if filename.lower().endswith(".html"):
            filename = filename[:-5]

        reserved = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
        if filename.strip(" .").lower() in reserved:
            filename = f"file_{filename}"

        need_hash = ("?" in original_url or "#" in original_url)

        full_path = os.path.join(self.save_dir, filename + ".html")
        if need_hash or len(full_path) > self.MAX_WIN_PATH:
            allowed = self.MAX_WIN_PATH - len(self.save_dir) - len(os.sep) - len(".html") - 9
            allowed = max(1, allowed)
            filename = (
                filename[:allowed]
                + "_"
                + hashlib.md5(original_url.encode()).hexdigest()[:8]
            )

        return filename.rstrip(". ")

    async def log_failed_url(self, url, log_file):
        if self._log_lock is None:
            self._log_lock = asyncio.Lock()
        async with self._log_lock:
            async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
                await f.write(url + "\n")

    def extract_links(
        self,
        html,
        base_url,
        base_domain,
        acceptable_domain_extension,
    ):
        soup = BeautifulSoup(html, "lxml")
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].replace("&amp;num;", "#")
            if href.startswith("www."):
                href = "https://" + href
            elif href.startswith("https/"):
                href = "https://" + href[len("https/"):]
            elif href.startswith("http/"):
                href = "http://" + href[len("http/"):]
            url = (
                urljoin(f"https://{base_domain}", href)
                if href.startswith("/")
                else urljoin(base_url, href)
            )
            p = urlsplit(url)
            canon = urlunsplit((p.scheme, p.netloc, p.path, "", ""))
            if self.is_valid_url(
                canon, base_domain, acceptable_domain_extension
            ):
                links.add(canon)
        return links

    def is_valid_url(self, url, base_domain, acceptable_domain_extension):
        def strip_www(netloc: str) -> str:
            return netloc[4:] if netloc.startswith("www.") else netloc

        parsed = urlparse(url)
        if strip_www(parsed.netloc) != strip_www(base_domain):
            return False
        if hasattr(self.scraper, "wants_url") and not self.scraper.wants_url(url):
            return False

        if acceptable_domain_extension:
            path = parsed.path
            if _path_within(path, acceptable_domain_extension):
                return True
            return _path_within(
                _strip_trailing_version(path),
                _strip_trailing_version(acceptable_domain_extension),
            )
        return True

    def cleanup(self):
        pass
