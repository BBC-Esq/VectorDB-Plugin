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
    results = []
    if not tasks:
        return results
    total = sum(len(pages) for _, pages in tasks)
    done = 0
    workers = worker_count(len(tasks))
    with ProcessPoolExecutor(workers, initializer=ocr_reader.start_worker,
                             initargs=(check, reader, THREADS_PER_WORKER)) as pool:
        futures = [pool.submit(function, path, pages, option) for path, pages in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += len(result[1])
            print(f"{label} {done}/{total} pages", flush=True)
    return results


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


def read(source_dir, check, reader):
    from core import ocr_cache
    from modules import ocr_reader
    documents, _ = find_pages(source_dir, check)
    reader_id = ocr_reader.READERS[reader]
    tasks = []
    for path, document in documents.items():
        tasks.extend(tasks_for(path, unread_pages(document, reader_id)))
    if tasks:
        ocr_reader.ensure_models(None, reader)
        texts = {}
        for path, _, pages in run_tasks(ocr_reader.read_pages, tasks, reader, None, reader, "Reading pages with OCR:"):
            texts.setdefault(path, {}).update(pages)
        for path, pages in texts.items():
            record = documents[path]["record"]
            record["text"].setdefault(reader_id, {}).update({str(number): text for number, text in pages.items()})
            ocr_cache.save(documents[path]["entry"], record)
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
    parser.add_argument("--reader", choices=["rapidocr", "tesseract"], default="rapidocr")
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
