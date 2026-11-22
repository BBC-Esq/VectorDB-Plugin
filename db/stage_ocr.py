import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("stage_ocr")
SCAN_PROGRESS = "Checking PDFs:"
READ_PROGRESS = "Reading pages with OCR:"
SAVE_EVERY = 10

PAGES_PER_TASK = 8
THREADS_PER_WORKER = 2


def worker_count(tasks):
    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or multiprocessing.cpu_count()
    except ImportError:
        cores = max(1, multiprocessing.cpu_count() // 2)
    return max(1, min((cores - 1) // THREADS_PER_WORKER, tasks))


def pdf_files(source_dir):
    return sorted((entry for entry in Path(source_dir).iterdir() if entry.suffix.lower() == ".pdf"),
                  key=lambda entry: entry.name.lower())


def inspect(entry):
    import fitz
    try:
        with fitz.open(str(entry)) as document:
            if document.needs_pass:
                return 0, "it is password protected"
            return document.page_count, None
    except Exception:
        return 0, "it could not be opened"


def tasks_for(path, pages):
    return [(path, pages[start:start + PAGES_PER_TASK]) for start in range(0, len(pages), PAGES_PER_TASK)]


def run_tasks(function, tasks, option, check, reader, label):
    from modules import ocr_reader
    if not tasks:
        return
    total = sum(len(pages) for _, pages in tasks)
    done = 0
    workers = worker_count(len(tasks))
    with ProcessPoolExecutor(workers, initializer=ocr_reader.start_worker,
                             initargs=(check, reader, THREADS_PER_WORKER)) as pool:
        futures = [pool.submit(function, path, pages, option) for path, pages in tasks]
        for future in as_completed(futures):
            result = future.result()
            done += len(result[1])
            print(f"{label} {done}/{total} pages", flush=True)
            yield result


def find_pages(source_dir, check):
    from core import ocr_cache
    from modules import ocr_reader
    check_id = ocr_reader.CHECKS[check]
    documents, problems, tasks = {}, [], []
    for entry in pdf_files(source_dir):
        count, problem = inspect(entry)
        if problem:
            problems.append({"name": entry.name, "path": str(entry), "reason": problem})
            continue
        record = ocr_cache.load(entry)
        documents[str(entry)] = {"entry": entry, "record": record, "total": count}
        if check_id not in record["checks"]:
            record["checks"][check_id] = None
            tasks.extend(tasks_for(str(entry), list(range(1, count + 1))))
    if tasks:
        if check == "detector":
            ocr_reader.ensure_models(check, None)
        flagged = {}
        for path, _, pages in run_tasks(ocr_reader.check_pages, tasks, check, check, None, SCAN_PROGRESS):
            flagged.setdefault(path, []).extend(pages)
        for path, document in documents.items():
            if document["record"]["checks"][check_id] is None:
                document["record"]["checks"][check_id] = sorted(flagged.get(path, []))
                ocr_cache.save(document["entry"], document["record"])
    for document in documents.values():
        document["pages"] = list(document["record"]["checks"][check_id])
    return documents, problems


def unread_pages(document, reader_id):
    stored = document["record"]["text"].get(reader_id) or {}
    return [page for page in document["pages"] if str(page) not in stored]


def scan(source_dir, check, reader):
    from modules import ocr_reader
    documents, problems = find_pages(source_dir, check)
    reader_id = ocr_reader.READERS[reader]
    needs = [{"name": document["entry"].name, "path": path, "pages": len(document["pages"]),
              "total": document["total"], "unread": len(unread_pages(document, reader_id))}
             for path, document in documents.items() if document["pages"]]
    return {"needs": needs, "problems": problems}


def usable_reader(reader):
    if reader != "lighton":
        return reader
    from modules import ocr_lighton
    if ocr_lighton.available():
        return reader
    print("LightOnOCR needs an NVIDIA GPU with at least 6 GB of memory, so RapidOCR reads the pages instead.", flush=True)
    return "rapidocr"


def read_with_lighton(tasks, store):
    import fitz
    from modules import ocr_lighton
    items = [(path, page) for path, pages in tasks for page in pages]
    model, processor = ocr_lighton.load()
    open_documents = {}
    done = [0]

    def render(item):
        path, page = item
        if path not in open_documents:
            open_documents[path] = fitz.open(path)
        return ocr_lighton.render(open_documents[path][page - 1])

    def on_result(item, text, cut_off):
        path, page = item
        if cut_off:
            logger.warning(f"Page {page} of {Path(path).name} was cut off at the length limit; its text may be incomplete.")
        done[0] += 1
        print(f"{READ_PROGRESS} {done[0]}/{len(items)} pages", flush=True)
        store(path, {page: text}, done[0] % SAVE_EVERY == 0)

    try:
        ocr_lighton.read_in_batches(model, processor, items, render, on_result)
    finally:
        for document in open_documents.values():
            document.close()


def read(source_dir, check, reader):
    from core import ocr_cache
    from modules import ocr_reader
    documents, _ = find_pages(source_dir, check)
    tasks = []
    if any(unread_pages(document, ocr_reader.READERS[reader]) for document in documents.values()):
        reader = usable_reader(reader)
    reader_id = ocr_reader.READERS[reader]
    for path, document in documents.items():
        tasks.extend(tasks_for(path, unread_pages(document, reader_id)))
    unsaved = set()

    def store(path, pages, save=True):
        record = documents[path]["record"]
        record["text"].setdefault(reader_id, {}).update({str(number): text for number, text in pages.items()})
        unsaved.add(path)
        if save:
            for changed in unsaved:
                ocr_cache.save(documents[changed]["entry"], documents[changed]["record"])
            unsaved.clear()

    if tasks:
        try:
            if reader == "lighton":
                read_with_lighton(tasks, store)
            else:
                ocr_reader.ensure_models(None, reader)
                for path, _, pages in run_tasks(ocr_reader.read_pages, tasks, reader, None, reader, READ_PROGRESS):
                    store(path, pages)
        finally:
            for changed in unsaved:
                ocr_cache.save(documents[changed]["entry"], documents[changed]["record"])
    output = {}
    for document in documents.values():
        stored = document["record"]["text"].get(reader_id) or {}
        pages = {str(page): stored[str(page)] for page in document["pages"] if str(page) in stored}
        if pages:
            output[os.path.normcase(os.path.realpath(document["entry"]))] = pages
    return output


def command(action, source_dir, output, check, reader):
    return [sys.executable, str(Path(__file__).resolve()), action, str(source_dir), str(output),
            "--check", check, "--reader", reader]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["scan", "read"])
    parser.add_argument("source_dir")
    parser.add_argument("output")
    parser.add_argument("--check", choices=["detector", "quick"], default="detector")
    parser.add_argument("--reader", choices=["rapidocr", "tesseract", "lighton"], default="rapidocr")
    args = parser.parse_args()

    if not Path(args.source_dir).is_dir():
        print(f"ERROR: Source directory does not exist: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    started = time.time()
    result = (scan if args.action == "scan" else read)(args.source_dir, args.check, args.reader)
    temporary = Path(args.output).with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(result, f)
    temporary.replace(args.output)
    logger.info(f"OCR {args.action} finished in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
