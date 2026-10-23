import asyncio
import os
import re
import pytest
import yaml
from bs4 import BeautifulSoup

import modules.scraper as scraper_module
from modules.scraper import (
    BaseScraper,
    MintlifyScraper,
    SelectorScraper,
    ScraperRegistry,
    ScraperWorker,
    SCRAPER_SELECTORS,
    _mintlify_convert_alerts,
    _mintlify_strip_injected_headers,
    render_openapi_spec_overview,
)


class TestProcessHtml:

    def test_returns_soup(self):
        from bs4 import BeautifulSoup
        html = "<html><body><p>Hello world</p></body></html>"
        soup = BeautifulSoup(html, 'html.parser')

        scraper = object.__new__(BaseScraper)
        scraper.url = "http://example.com"
        scraper.folder = "/tmp"
        result = scraper.process_html(soup)
        assert result is not None
        assert 'Hello world' in result.get_text()


class TestSelectorScraper:

    def test_finds_content_by_selector(self):
        from bs4 import BeautifulSoup
        html = '<html><body><main id="content">Target text</main><div>Other</div></body></html>'
        soup = BeautifulSoup(html, 'html.parser')

        scraper = SelectorScraper("http://example.com", "/tmp", "TileDBScraper")
        result = scraper.extract_main_content(soup)
        assert result is not None
        assert 'Target text' in result.get_text()

    def test_returns_none_no_match(self):
        from bs4 import BeautifulSoup
        html = '<html><body><p>No matching element</p></body></html>'
        soup = BeautifulSoup(html, 'html.parser')

        scraper = SelectorScraper("http://example.com", "/tmp", "TileDBScraper")
        result = scraper.extract_main_content(soup)
        assert result is None


class TestScraperRegistry:

    def test_has_special_scrapers(self):
        assert hasattr(ScraperRegistry, '_special_scrapers')
        assert isinstance(ScraperRegistry._special_scrapers, dict)

    def test_get_scraper_known(self):
        scraper_cls = ScraperRegistry.get_scraper("BaseScraper")
        assert scraper_cls is BaseScraper

    def test_get_scraper_selector(self):
        factory = ScraperRegistry.get_scraper("TileDBScraper")
        assert callable(factory)

    def test_get_scraper_unknown_returns_base(self):
        scraper_cls = ScraperRegistry.get_scraper("NonexistentScraper")
        assert scraper_cls is BaseScraper


class TestScraperSelectors:

    def test_not_empty(self):
        assert len(SCRAPER_SELECTORS) > 0

    def test_values_are_tuples(self):
        for key, value in SCRAPER_SELECTORS.items():
            assert isinstance(value, tuple), f"SCRAPER_SELECTORS['{key}'] is not a tuple"
            assert len(value) == 2, f"SCRAPER_SELECTORS['{key}'] doesn't have 2 elements"

    def test_first_element_is_tag(self):
        for key, (tag, _) in SCRAPER_SELECTORS.items():
            assert isinstance(tag, str), f"SCRAPER_SELECTORS['{key}'] tag is not a string"


class TestDocumentationEntries:

    def test_every_entry_names_a_known_scraper(self):
        from core.constants import scrape_documentation
        known = set(SCRAPER_SELECTORS) | set(ScraperRegistry._special_scrapers)
        unknown = {n: e.get("scraper_class") for n, e in scrape_documentation.items()
                   if e.get("scraper_class", "BaseScraper") not in known}
        assert unknown == {}

    def test_entries_have_unique_folders_and_web_urls(self):
        from core.constants import scrape_documentation
        folders = [e["folder"] for e in scrape_documentation.values()]
        assert len(folders) == len(set(folders))
        assert all(e["URL"].startswith("https://") for e in scrape_documentation.values())


class TestUrlValidation:

    def test_valid_same_domain(self):
        from urllib.parse import urlparse
        base = "https://docs.example.com/guide"
        url = "https://docs.example.com/guide/page1"
        base_domain = urlparse(base).netloc.replace('www.', '')
        url_domain = urlparse(url).netloc.replace('www.', '')
        assert base_domain == url_domain

    def test_different_domain(self):
        from urllib.parse import urlparse
        base = "https://docs.example.com"
        url = "https://other-site.org/page"
        base_domain = urlparse(base).netloc.replace('www.', '')
        url_domain = urlparse(url).netloc.replace('www.', '')
        assert base_domain != url_domain

    def test_www_stripped(self):
        from urllib.parse import urlparse
        url1 = "https://www.example.com/page"
        url2 = "https://example.com/page"
        d1 = urlparse(url1).netloc.replace('www.', '')
        d2 = urlparse(url2).netloc.replace('www.', '')
        assert d1 == d2


class TestFilenameSanitization:

    def test_protocol_stripped(self):
        url = "https://docs.example.com/page"
        sanitized = url.replace('https://', '').replace('http://', '')
        assert not sanitized.startswith('http')

    def test_slashes_replaced(self):
        path = "docs/guide/page"
        sanitized = path.replace('/', '_')
        assert '/' not in sanitized

    def test_special_chars_replaced(self):
        filename = 'file<>:"|?*name'
        sanitized = re.sub(r'[<>:"|?*]', '_', filename)
        assert all(c not in sanitized for c in '<>:"|?*')


SPEC = {
    "openapi": "3.1.0",
    "info": {"title": "Example API", "version": "1"},
    "servers": [{"url": "https://api.example.com"}],
    "tags": [{"name": "Widgets", "description": "TAG-ONLY TEXT repeated on every page"}],
    "paths": {"/widgets": {"post": {
        "summary": "Create a widget",
        "description": "Creates a widget.",
        "parameters": [{"$ref": "#/components/parameters/Version"}],
        "requestBody": {"required": True, "content": {"application/json": {
            "schema": {"$ref": "#/components/schemas/WidgetCreate"}}}},
        "responses": {"201": {"description": "Widget created", "content": {"application/json": {
            "schema": {"$ref": "#/components/schemas/Widget"}}}}},
        "x-codeSamples": [{"lang": "cURL", "source": "curl https://api.example.com/widgets -H \"Authorization: Token <API_TOKEN>\""}],
    }}},
    "components": {
        "parameters": {"Version": {"name": "API-VERSION", "in": "header", "description": "Version to use.",
                                   "schema": {"type": "string", "default": "2024-01-01"}}},
        "schemas": {
            "Part": {"type": "object", "properties": {"sku": {"type": "string", "description": "Part number."}}},
            "WidgetBase": {"type": "object", "required": ["name"], "properties": {
                "name": {"type": "string", "maxLength": 50, "description": "Name <WIDGET_NAME>."},
                "color": {"type": "string", "enum": ["red", "blue"]},
            }},
            "WidgetCreate": {"allOf": [
                {"$ref": "#/components/schemas/WidgetBase"},
                {"type": "object", "properties": {
                    "main_part": {"$ref": "#/components/schemas/Part"},
                    "spare_parts": {"type": "array", "items": {"$ref": "#/components/schemas/Part"}},
                }},
            ]},
            "Widget": {"allOf": [
                {"$ref": "#/components/schemas/WidgetBase"},
                {"type": "object", "properties": {"object_id": {
                    "type": "string", "description": "| Carrier | Rule |\n|:---|:---|\n| FedEx | Max 35 |"}}},
            ]},
        },
    },
}
HEADERS = "> ## Documentation Index\n> Fetch the index.\n\n> ## Agent Instructions\n> Use the MCP server.\n\n"


def _page(fence_target, spec=SPEC):
    return (HEADERS + "# Create a widget\n\n> Creates a widget.\n\n## OpenAPI\n\n````yaml /spec.yaml "
            + fence_target + "\n" + yaml.safe_dump(spec, sort_keys=False) + "````\n\n## Related topics\n\n- [Widgets](/widgets.md)\n")


def _text(html):
    return " ".join(BeautifulSoup(html, "lxml").get_text(" ").split())


class _FakeResponse:
    def __init__(self, url):
        self.url = url
        self.status_code = 200
        self.headers = {"content-type": "text/markdown"}
        self.text = "# Start\n\nWelcome."


class _FakeSession:
    def __init__(self, final_url):
        self.final_url = final_url

    async def get(self, url, timeout=30, allow_redirects=True):
        return _FakeResponse(self.final_url)


class TestMintlifyScraper:

    def test_injected_header_blocks_are_removed(self):
        out = _mintlify_strip_injected_headers(HEADERS + "# Title\n\n> A quote that stays.\n")
        assert out.startswith("# Title")
        assert "Agent Instructions" not in out and "A quote that stays." in out

    def test_alerts_get_one_label_and_code_is_untouched(self):
        md = "> [!NOTE]\n>\n> **Note**\n>\n> Use the test token.\n\n```text\n> [!NOTE]\n```"
        out = _mintlify_convert_alerts(md)
        assert out.count("**Note:**") == 1
        assert "> **Note**\n" not in out and "Use the test token." in out
        assert "```text\n> [!NOTE]\n```" in out

    def test_openapi_block_becomes_readable_reference(self):
        html = MintlifyScraper("https://docs.example.com/", "x").transform_response(
            _page("post /widgets"), "https://docs.example.com/widgets/create")
        text = _text(html)
        assert "openapi:" not in html and "TAG-ONLY TEXT" not in text
        assert "Documentation Index" not in text and "Use the MCP server" not in text
        assert "POST https://api.example.com/widgets" in text
        assert "API-VERSION (header, string, optional): Version to use. Default: 2024-01-01 ." in text
        assert "name (string, required, max length 50): Name <WIDGET_NAME>." in text
        assert "Allowed values: red , blue ." in text
        assert "spare_parts (array of Part object, optional)" in text
        assert "The fields of Part are listed above." in text
        assert text.count("Creates a widget.") == 1
        assert "Widget created" in text and "FedEx: Max 35." in text
        assert "<API_TOKEN>" in text and "Related topics" in text

    def test_webhook_block_without_webhook_is_dropped(self):
        text = _text(MintlifyScraper("https://docs.example.com/", "x").transform_response(
            _page("webhook widget_updated"), "https://docs.example.com/widgets/updated"))
        assert "Create a widget" in text and "TAG-ONLY TEXT" not in text and "openapi" not in text

    def test_spec_file_becomes_one_overview_page(self):
        spec = {**SPEC, "tags": [{"name": "Widgets", "description": "# Inner\n\nTAG TEXT"}],
                "webhooks": {"widget_updated": {"post": {
                    "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Widget"}}}},
                    "responses": {"200": {"description": "OK"}}}}}}
        html = render_openapi_spec_overview(yaml.safe_dump(spec))
        text = _text(html)
        assert "<h2>Widgets</h2>" in html and "<h3>Inner</h3>" in html
        assert "POST /widgets : Create a widget" in text
        assert "Webhook: widget_updated" in text and "Event widget_updated" in text
        with pytest.raises(ValueError):
            render_openapi_spec_overview("just: a yaml file")

    def test_url_rules(self):
        s = MintlifyScraper("https://docs.example.com/", "x")
        assert s.fetch_url_for("https://docs.example.com/") == "https://docs.example.com/index.md"
        assert s.fetch_url_for("https://docs.example.com/guides/start") == "https://docs.example.com/guides/start.md"
        assert s.fetch_url_for("https://docs.example.com/spec/api.yaml") == "https://docs.example.com/spec/api.yaml"
        assert s.page_url_for("https://docs.example.com/guides/start.md") == "https://docs.example.com/guides/start"
        assert not s.wants_url("https://docs.example.com/images/label.png")
        assert s.wants_url("https://docs.example.com/changelog/v1.2.0")

    def test_redirect_is_saved_once_under_the_final_page(self, tmp_path, monkeypatch):
        monkeypatch.setattr(scraper_module, "PROJECT_ROOT", tmp_path)
        worker = ScraperWorker("https://docs.example.com/", "example", MintlifyScraper, name="Example")

        def fetch(url):
            return asyncio.run(worker.fetch(_FakeSession("https://docs.example.com/guides/start.md"), url,
                                            "docs.example.com", asyncio.BoundedSemaphore(1), worker.save_dir,
                                            str(tmp_path / "failed.log"), ""))

        fetch("https://docs.example.com/old/start")
        final_file = worker._html_path("https://docs.example.com/guides/start")
        assert os.path.exists(final_file)
        assert not os.path.exists(worker._html_path("https://docs.example.com/old/start"))
        with open(final_file, encoding="utf-8") as handle:
            assert 'href="https://docs.example.com/guides/start"' in handle.read()
        assert fetch("https://docs.example.com/older/start") == set()
        assert [f for f in os.listdir(worker.save_dir) if f.endswith(".html")] == [os.path.basename(final_file)]
