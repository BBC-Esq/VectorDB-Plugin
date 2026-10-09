import re
import pytest

from modules.scraper import (
    BaseScraper,
    SelectorScraper,
    ScraperRegistry,
    SCRAPER_SELECTORS,
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
