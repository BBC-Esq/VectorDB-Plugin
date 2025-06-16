import os
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from copy import deepcopy
import hashlib
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit
from PySide6.QtCore import Signal, QObject
from charset_normalizer import detect


class BaseScraper:
    def __init__(self, url, folder):
        self.url = url
        self.folder = folder
        self.save_dir = os.path.join(
            os.path.dirname(__file__),
            "Scraped_Documentation",
            folder,
        )

    def process_html(self, soup):
        main_content = self.extract_main_content(soup)
        if main_content:
            new_soup = BeautifulSoup("<html><body></body></html>", "lxml")
            new_soup.body.append(deepcopy(main_content))
            return new_soup
        return soup

    def extract_main_content(self, soup):
        return None


class HuggingfaceScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find(
            "div",
            class_="prose-doc prose relative mx-auto max-w-4xl break-words",
        )


class ReadthedocsScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("div", class_="rst-content")


class PyTorchScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("article", id="pytorch-article", class_="pytorch-article", attrs={"itemprop": "articleBody"})


class TileDBScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("main", {"id": "content"})


class RstContentScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("div", class_="rst-content")


class FuroThemeScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("article", id="furo-main-content")


class PydataThemeScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("article", class_="bd-article")


class FastcoreScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("main", id="quarto-document-content", class_="content")


class RtdThemeScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("div", attrs={"itemprop": "articleBody"})


class BodyRoleMainScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("div", class_="body", attrs={"role": "main"})


class ArticleMdContentInnerMdTypesetScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("article", class_="md-content__inner md-typeset")


class DivClassDocumentScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("div", class_="document")


class MainIdMainContentRoleMainScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("main", id="main-content", attrs={"role": "main"})


class DivIdMainContentRoleMainScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("div", id="main-content", attrs={"role": "main"})


class MainScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("main")


class DivClassThemeDocMarkdownMarkdownScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("div", class_=["theme-doc-markdown", "markdown"])


class DivClassTdContentScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("div", class_="td-content")


class PymupdfScraper(BaseScraper):
    def extract_main_content(self, soup):
        article_container = soup.find("div", class_="article-container")
        if article_container:
            return article_container.find("section")
        return None


class BodyScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("body")


class ArticleRoleMainScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("article", attrs={"role": "main"})


class DivIdContentSecondScraper(BaseScraper):
    def extract_main_content(self, soup):
        content_divs = soup.find_all("div", id="content")
        if len(content_divs) >= 2:
            return content_divs[1]  # Return the second one (index 1)
        return None


class ArticleClassMainContent8zFCHScraper(BaseScraper):
    def extract_main_content(self, soup):
        return soup.find("article", class_="main_content__8zFCH")


class PropCacheScraper(ReadthedocsScraper):# only one thus far that modifies the crawling behavior
    """Special-case scraper for https://propcache.aio-libs.org/ .

    The bare domain 302-redirects to /en/latest/, so we force that
    versioned path before link extraction to keep urljoin() correct.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If user passed the root URL, rewrite it to the docs root.
        if self.url.rstrip("/").endswith("propcache.aio-libs.org"):
            self.url = urljoin(self.url, "en/latest/")

        # Always keep a trailing “/” for consistent joins.
        if not self.url.endswith("/"):
            self.url += "/"

        self.base_url = self.url

    def extract_main_content(self, soup):
        return soup.find("div", class_="body", attrs={"role": "main"})


class FileDownloader(BaseScraper):
    """Save any non-HTML asset (e.g. YAML) exactly as delivered."""

    def extract_main_content(self, soup):
        return None

    async def save_file(self, content: bytes, url: str, save_dir: str):
        from pathlib import Path

        basename = Path(url).name or "download"
        filename = os.path.join(save_dir, basename)

        async with aiofiles.open(filename, "wb") as f:
            await f.write(content)


class ScraperRegistry:
    _scrapers = {
        "BaseScraper": BaseScraper,
        "HuggingfaceScraper": HuggingfaceScraper,
        "ReadthedocsScraper": ReadthedocsScraper,
        "PyTorchScraper": PyTorchScraper,
        "TileDBScraper": TileDBScraper,
        "PropCacheScraper": PropCacheScraper,
        "FuroThemeScraper": FuroThemeScraper,
        "RstContentScraper": RstContentScraper,
        "PydataThemeScraper": PydataThemeScraper,
        "FastcoreScraper": FastcoreScraper,
        "RtdThemeScraper": RtdThemeScraper,
        "BodyRoleMainScraper": BodyRoleMainScraper,
        "ArticleMdContentInnerMdTypesetScraper": ArticleMdContentInnerMdTypesetScraper,
        "DivClassDocumentScraper": DivClassDocumentScraper,
        "MainIdMainContentRoleMainScraper": MainIdMainContentRoleMainScraper,
        "DivIdMainContentRoleMainScraper": DivIdMainContentRoleMainScraper,
        "MainScraper": MainScraper,
        "DivClassThemeDocMarkdownMarkdownScraper": DivClassThemeDocMarkdownMarkdownScraper,
        "DivClassTdContentScraper": DivClassTdContentScraper,
        "PymupdfScraper": PymupdfScraper,
        "BodyScraper": BodyScraper,
        "ArticleRoleMainScraper": ArticleRoleMainScraper,
        "DivIdContentSecondScraper": DivIdContentSecondScraper,
        "ArticleClassMainContent8zFCHScraper": ArticleClassMainContent8zFCHScraper,
        "FileDownloader": FileDownloader,
    }

    @classmethod
    def get_scraper(cls, scraper_name):
        return cls._scrapers.get(scraper_name, BaseScraper)


class ScraperWorker(QObject):
    status_updated = Signal(str)
    scraping_finished = Signal()

    def __init__(self, url, folder, scraper_class=BaseScraper):
        super().__init__()
        self.url = url
        self.folder = folder
        self.scraper = scraper_class(url, folder)
        self.save_dir = self.scraper.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.stats = {"scraped": 0}

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.crawl_domain())
        finally:
            self.cleanup()
            loop.close()

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
        to_visit = [self.url]
        visited = set()

        async def process_batch(batch_urls, session):
            tasks = [
                self.fetch(
                    session,
                    u,
                    acceptable_domain,
                    semaphore,
                    self.save_dir,
                    log_file,
                    acceptable_domain_extension,
                )
                for u in batch_urls
                if u not in visited
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            visited.update(batch_urls)
            return [r for r in results if isinstance(r, set)]

        async with aiohttp.ClientSession() as session:
            while to_visit:
                current_batch = to_visit[:batch_size]
                to_visit = to_visit[batch_size:]

                for new_links in await process_batch(current_batch, session):
                    new_to_visit = new_links - visited
                    to_visit.extend(new_to_visit)

                await asyncio.sleep(0.2)

                if len(visited) >= page_limit:
                    break

        self.scraping_finished.emit()
        return visited

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

        fallback_encodings = [
            "latin-1",
            "iso-8859-1",
            "cp1252",
            "windows-1252",
            "ascii",
        ]
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept-Charset": "utf-8, iso-8859-1;q=0.8, *;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        async with semaphore:
            for attempt in range(1, retries + 1):
                try:
                    # Add timeout to prevent hanging
                    timeout = aiohttp.ClientTimeout(total=30)
                    async with session.get(url, headers=headers, timeout=timeout) as response:
                        if response.status == 200:
                            content_type = response.headers.get("content-type", "").lower()
                            if "text/html" in content_type:
                                try:
                                    html = await response.text(encoding="utf-8")
                                except UnicodeDecodeError:
                                    raw = await response.read()

                                    encodings_to_try = []
                                    try:
                                        detected = detect(raw).get("encoding")
                                        if detected:
                                            encodings_to_try.append(detected)
                                    except Exception:
                                        pass

                                    encodings_to_try.extend(
                                        [e for e in fallback_encodings if e not in encodings_to_try]
                                    )

                                    for enc in encodings_to_try:
                                        try:
                                            html = raw.decode(enc)
                                            break
                                        except UnicodeDecodeError:
                                            continue
                                    else:
                                        html = raw.decode("utf-8", errors="ignore")

                                await self.save_html(html, url, save_dir)
                                self.stats["scraped"] = self.count_saved_files()
                                self.status_updated.emit(
                                    str(self.stats["scraped"])
                                )
                                return self.extract_links(
                                    html,
                                    url,
                                    base_domain,
                                    acceptable_domain_extension,
                                )

                            self.stats["scraped"] = self.count_saved_files()
                            self.status_updated.emit(str(self.stats["scraped"]))
                            return set()
                        else:
                            await self.log_failed_url(url, log_file)
                            self.stats["scraped"] = self.count_saved_files()
                            self.status_updated.emit(str(self.stats["scraped"]))
                            return set()
                except asyncio.TimeoutError:
                    if attempt == retries:
                        await self.log_failed_url(url, log_file)
                        self.stats["scraped"] = self.count_saved_files()
                        self.status_updated.emit(str(self.stats["scraped"]))
                    await asyncio.sleep(2)
                except UnicodeDecodeError:
                    if attempt == retries:
                        await self.log_failed_url(url, log_file)
                        self.stats["scraped"] = self.count_saved_files()
                        self.status_updated.emit(str(self.stats["scraped"]))
                    await asyncio.sleep(2)
                except Exception:
                    if attempt == retries:
                        await self.log_failed_url(url, log_file)
                        self.stats["scraped"] = self.count_saved_files()
                        self.status_updated.emit(str(self.stats["scraped"]))
                    await asyncio.sleep(2)
        return set()

    async def save_html(self, content, url, save_dir):
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
        async with aiofiles.open(log_file, "a") as f:
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

        # Allow either “…/scipy-<version>” *or* “…/scipy/”
        if acceptable_domain_extension:
            base_no_version = acceptable_domain_extension.rsplit('-', 1)[0]
            return (
                parsed.path.startswith(acceptable_domain_extension) or
                parsed.path.startswith(base_no_version)
            )
        return True

    def cleanup(self):
        pass
