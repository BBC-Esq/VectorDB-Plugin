import os
import re
import json
import asyncio
import aiofiles
from bs4 import BeautifulSoup
from copy import deepcopy
import hashlib
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit
from PySide6.QtCore import Signal, QObject
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError

from core.constants import PROJECT_ROOT


_VERSION_SUFFIX_RE = re.compile(r"^v?\d+(\.\d+)*$")

# Cruft commonly embedded INSIDE the main article element by Sphinx-style themes.
# Stripped post-extraction so the saved HTML is closer to "vector-DB-ready" content.
_CRUFT_TAGS = ("script", "style", "nav", "footer", "svg")
_CRUFT_CLASSES = (
    "toctree-wrapper",      # Sphinx project TOC tree, often dumped at bottom of index pages
    "related",              # Sphinx prev/next bar
    "sphinxsidebar",        # classic Sphinx sidebar, when scoped inside the content
    "footer",               # Sphinx classic theme div.footer (copyright/build info)
    "edit-this-page",       # MkDocs Material "Edit this page"
    "md-source-file",       # MkDocs Material source-file metadata
    "prev-next-area",       # Pydata-theme prev/next nav
    "prev-next-bottom",
    "prev-next-top",
    "bd-toc",               # Pydata "On this page" sidebar
    "bd-sidebar-secondary",
    "feedback-widget",      # Pydantic.dev "Was this page helpful?"
    "pagination-links",     # Pydantic.dev / generic prev-next pagination
    "footer-version",       # PyMuPDF "This documentation covers all versions..."
    "try_examples_button_container",  # SciPy "Try it in your browser! / Open in Tab"
    "try_examples_outer_iframe",      # SciPy interactive-examples sandbox iframe
    "sidemenu",             # lxml.de project nav inside div.document
    "banner",               # lxml.de donation banner ("Like the tool? Help making it better!")
)
_CRUFT_IDS = (
    "indices-and-tables",   # Sphinx auto-generated bottom-of-index "Index/ModIndex/Search" stub
    "search",
    "footerDisclaimer",     # PyMuPDF "This software is provided AS-IS..."
)


def _strip_trailing_version(path: str) -> str:
    """Strip a trailing '-vX.Y.Z' / '-1.2.3' from a URL path.

    Used by is_valid_url so that, e.g., a seed of /foo-v1.2/ still matches /foo-v1.3/.
    Does NOT strip non-version suffixes (so /array-api-compat does not become /array-api).
    """
    parts = path.rsplit("-", 1)
    if len(parts) == 2 and _VERSION_SUFFIX_RE.match(parts[1]):
        return parts[0]
    return path


_CRUFT_HEADINGS = {"copyright", "copyrights", "license", "licenses"}


def _strip_embedded_cruft(content):
    """Remove TOC trees, nav, scripts, etc. that sit INSIDE the extracted main element."""
    for tag_name in _CRUFT_TAGS:
        for el in content.find_all(tag_name):
            el.decompose()
    for cls in _CRUFT_CLASSES:
        for el in content.find_all(class_=cls):
            el.decompose()
    for cruft_id in _CRUFT_IDS:
        for el in content.find_all(id=cruft_id):
            el.decompose()
    # Strip <section> whose first heading is a boilerplate heading like "Copyright" or "License".
    for section in content.find_all("section"):
        h = section.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        if h and h.get_text(strip=True).lower() in _CRUFT_HEADINGS:
            section.decompose()
    # Strip <p> elements whose only meaningful content is Prev/Next pagination anchors
    # (e.g. ruamel.yaml's "<p><a>Prev</a> <a>Next</a></p>" with no class/id to target).
    _NAV_LABELS = {"prev", "previous", "next"}
    for p in content.find_all("p"):
        anchors = p.find_all("a")
        if not anchors:
            continue
        anchor_text = " ".join(a.get_text(strip=True).lower() for a in anchors).split()
        if anchor_text and all(w in _NAV_LABELS for w in anchor_text):
            # Confirm there's no extra non-anchor text in the paragraph.
            non_anchor_text = "".join(
                str(c) for c in p.contents if getattr(c, "name", None) != "a"
            ).strip()
            if not non_anchor_text:
                p.decompose()
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
        # Fallback differs based on whether the user configured a scraper_class:
        #   - BaseScraper directly (no scraper_class set) => preserve full page
        #   - any subclass (selector configured but missed) => save empty stub
        #     so a misconfigured selector is visible as a tiny file instead of
        #     silently saving the full untrimmed page with TOC/nav cruft.
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


class PymupdfScraper(BaseScraper):
    def extract_main_content(self, soup):
        article_container = soup.find("div", class_="article-container")
        if article_container:
            return article_container.find("section")
        return None


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


class ScraperWorker(QObject):
    status_updated = Signal(str, str)
    scraping_finished = Signal(str, bool, bool)

    RATE_LIMIT_THRESHOLD = 5

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
        filename = os.path.join(save_dir, self.sanitize_filename(url) + ".html")
        if os.path.exists(filename):
            return set()

        async with semaphore:
            for attempt in range(1, retries + 1):
                if self._rate_limited or self._cancelled:
                    return set()
                try:
                    response = await session.get(url, timeout=30, allow_redirects=True)
                except (asyncio.TimeoutError, RequestsError, Exception):
                    if attempt == retries:
                        await self.log_failed_url(url, log_file)
                        self.stats["scraped"] = self.count_saved_files()
                        self.status_updated.emit(self.name, str(self.stats["scraped"]))
                    await asyncio.sleep(2)
                    continue

                if response.status_code == 429:
                    self._429s_since_last_success += 1
                    if self._429s_since_last_success >= self.RATE_LIMIT_THRESHOLD:
                        self._rate_limited = True
                    await self.log_failed_url(url, log_file)
                    self.stats["scraped"] = self.count_saved_files()
                    self.status_updated.emit(self.name, str(self.stats["scraped"]))
                    return set()

                if response.status_code != 200:
                    await self.log_failed_url(url, log_file)
                    self.stats["scraped"] = self.count_saved_files()
                    self.status_updated.emit(self.name, str(self.stats["scraped"]))
                    return set()

                self._429s_since_last_success = 0

                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type:
                    self.stats["scraped"] = self.count_saved_files()
                    self.status_updated.emit(self.name, str(self.stats["scraped"]))
                    return set()

                html = response.text
                try:
                    links = self.extract_links(
                        html, url, base_domain, acceptable_domain_extension
                    )
                    await self.save_html(html, url, save_dir, links=links)
                except Exception:
                    await self.log_failed_url(url, log_file)
                    self.stats["scraped"] = self.count_saved_files()
                    self.status_updated.emit(self.name, str(self.stats["scraped"]))
                    return set()
                self.stats["scraped"] = self.count_saved_files()
                self.status_updated.emit(self.name, str(self.stats["scraped"]))
                return links
        return set()

    async def save_html(self, content, url, save_dir, links=None):
        filename = os.path.join(save_dir, self.sanitize_filename(url) + ".html")
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

        try:
            async with aiofiles.open(filename, "x", encoding="utf-8") as f:
                await f.write(str(processed_soup))
        except FileExistsError:
            pass

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

        MAX_WIN_PATH = 250
        full_path = os.path.join(self.save_dir, filename + ".html")
        if need_hash or len(full_path) > MAX_WIN_PATH:
            allowed = MAX_WIN_PATH - len(self.save_dir) - len(os.sep) - len(".html") - 9
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

        if acceptable_domain_extension:
            base_no_version = _strip_trailing_version(acceptable_domain_extension)
            return (
                parsed.path.startswith(acceptable_domain_extension) or
                parsed.path.startswith(base_no_version)
            )
        return True

    def cleanup(self):
        pass
