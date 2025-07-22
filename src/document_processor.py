# document_processor.py

import os
import sys
import io
import logging
import warnings
from contextlib import redirect_stdout
import yaml
import math
from pathlib import Path, PurePath
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    EverNoteLoader,
    UnstructuredEPubLoader,
    UnstructuredEmailLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredRTFLoader,
    UnstructuredODTLoader,
    UnstructuredMarkdownLoader,
    BSHTMLLoader
)

from typing import Optional, Any, Iterator, Union
from langchain_community.document_loaders.blob_loaders import Blob

from langchain_community.document_loaders.parsers import PyMuPDFParser
import pymupdf

from constants import DOCUMENT_LOADERS
from extract_metadata import extract_document_metadata, add_pymupdf_page_metadata, compute_content_hash

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processor.log', mode='w')
    ]
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIRECTORY = Path(__file__).parent
SOURCE_DIRECTORY = ROOT_DIRECTORY / "Docs_for_DB"
INGEST_THREADS = max(2, os.cpu_count() - 12)


from typing import List

class FixedSizeTextSplitter:
    """Splits text into equally-sized character chunks without regex.

    Parameters
    ----------
    chunk_size : int
        Maximum characters per chunk.  Comes straight from config.yaml.
    """

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def split_documents(self, docs: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in docs:
            text = doc.page_content or ""
            for start in range(0, len(text), self.chunk_size):
                piece = text[start : start + self.chunk_size].strip()
                if not piece:
                    continue
                # shallow-copy metadata so each chunk carries origin info
                chunks.append(Document(page_content=piece, metadata=dict(doc.metadata)))
        return chunks


class CustomPyMuPDFParser(PyMuPDFParser):
    def _lazy_parse(self, blob: Blob, text_kwargs: Optional[dict[str, Any]] = None) -> Iterator[Document]:
        with PyMuPDFParser._lock:
            with blob.as_bytes_io() as file_path:
                doc = pymupdf.open(stream=file_path, filetype="pdf") if blob.data else pymupdf.open(file_path)

                full_content = []
                for page in doc:
                    page_content = self._get_page_content(doc, page, text_kwargs)
                    if page_content.strip():
                        full_content.append(f"[[page{page.number + 1}]]{page_content}")

                yield Document(
                    page_content="".join(full_content),
                    metadata=self._extract_metadata(doc, blob)
                )

class CustomPyMuPDFLoader(PyMuPDFLoader):
    def __init__(self, file_path: Union[str, PurePath], **kwargs: Any) -> None:
        super().__init__(file_path, **kwargs)
        self.parser = CustomPyMuPDFParser(
            text_kwargs=kwargs.get('text_kwargs'),
            extract_images=kwargs.get('extract_images', False)
        )

# ensure all loader class names map correctly
for ext, loader_name in DOCUMENT_LOADERS.items():
    DOCUMENT_LOADERS[ext] = globals()[loader_name]

def load_single_document(file_path: Path) -> Document:
    file_extension = file_path.suffix.lower()
    loader_class = DOCUMENT_LOADERS.get(file_extension)
    if not loader_class:
        print(f"\033[91mFailed---> {file_path.name} (extension: {file_extension})\033[0m")
        logging.error(f"Unsupported file type: {file_path.name} (extension: {file_extension})")
        return None

    loader_options = {}

    if file_extension in [".epub", ".rtf", ".odt", ".md"]:
        loader_options.update({
            "mode": "single",
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension == ".pdf":
        loader_options.update({
            "extract_images": False,
            "text_kwargs": {},  # Optional: passed to https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_text
        })
    elif file_extension in [".eml", ".msg"]:
        loader_options.update({
            "mode": "single",
            "process_attachments": False,
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension == ".html":
        loader_options.update({
            "open_encoding": "utf-8",
            "bs_kwargs": {
                "features": "lxml",  # Specify the parser to use (lxml is generally fast and lenient)
                # "parse_only": SoupStrainer("body"),  # Optionally parse only the body tag
                # "from_encoding": "utf-8",  # Specify a different input encoding if needed
            },
            "get_text_separator": "\n",  # Use newline as separator when extracting text
            # "file_path": "path/to/file.html",  # Usually set automatically by the loader
            # "open_encoding": None,  # Set to None to let BeautifulSoup detect encoding
            # "get_text_separator": " ",  # Use space instead of newline if preferred
        })
    elif file_extension in [".xlsx", ".xls", ".xlsm"]:
        loader_options.update({
            "mode": "single",
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension in [".csv", ".txt"]:
        loader_options.update({
            "encoding": "utf-8",
            "autodetect_encoding": True
        })

    try:
        if file_extension in [".epub", ".rtf", ".odt", ".md", ".eml", ".msg", ".xlsx", ".xls", ".xlsm"]:
            unstructured_kwargs = loader_options.pop("unstructured_kwargs", {})
            loader = loader_class(str(file_path), mode=loader_options.get("mode", "single"), **unstructured_kwargs)
        else:
            loader = loader_class(str(file_path), **loader_options)
        documents = loader.load()

        if not documents:
            print(f"\033[91mFailed---> {file_path.name} (No content extracted)\033[0m")
            logging.error(f"No content could be extracted from file: {file_path.name}")
            return None

        document = documents[0]

        content_hash = compute_content_hash(document.page_content)
        metadata = extract_document_metadata(file_path, content_hash)
        document.metadata.update(metadata)
        print(f"Loaded---> {file_path.name}")
        return document

    except (OSError, UnicodeDecodeError) as e:
        print(f"\033[91mFailed---> {file_path.name} (Access/encoding error)\033[0m")
        logging.error(f"File access/encoding error - File: {file_path.name} - Error: {str(e)}")
        return None
    except Exception as e:
        print(f"\033[91mFailed---> {file_path.name} (Unexpected error)\033[0m")
        logging.error(f"Unexpected error processing file: {file_path.name} - Error: {str(e)}")
        return None

def load_document_batch(filepaths, threads_per_process):
    with ThreadPoolExecutor(threads_per_process) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [future.result() for future in futures]
    return (data_list, filepaths)

def load_documents(source_dir: Path) -> list:
    valid_extensions = {ext.lower() for ext in DOCUMENT_LOADERS.keys()}
    doc_paths = [f for f in source_dir.iterdir() if f.suffix.lower() in valid_extensions]

    docs = []

    if doc_paths:
        n_workers = min(INGEST_THREADS, max(len(doc_paths), 1))

        total_cores = os.cpu_count()
        threads_per_process = 2

        with ProcessPoolExecutor(n_workers) as executor:
            chunksize = math.ceil(len(doc_paths) / n_workers)
            futures = []
            for i in range(0, len(doc_paths), chunksize):
                chunk_paths = doc_paths[i:i + chunksize]
                futures.append(executor.submit(load_document_batch, chunk_paths, threads_per_process))

            for future in as_completed(futures):
                contents, _ = future.result()
                docs.extend([d for d in contents if d is not None])

    return docs

def split_documents(documents=None, text_documents_pdf=None):
   try:
       print("\nSplitting documents into chunks.")

       with open("config.yaml", "r", encoding='utf-8') as config_file:
           config = yaml.safe_load(config_file)
           chunk_size = config["database"]["chunk_size"]
           chunk_overlap = config["database"]["chunk_overlap"]

       # instantiate text splitter
       text_splitter = FixedSizeTextSplitter(chunk_size=chunk_size)

       # text_splitter = RecursiveCharacterTextSplitter(
           # chunk_size=chunk_size,
           # chunk_overlap=chunk_overlap,
           # keep_separator=False,
       # )

       texts = []

       if documents:
           # use text splitter directly
           texts = text_splitter.split_documents(documents)

       if text_documents_pdf:
           processed_pdf_docs = []
           for doc in text_documents_pdf:
               chunked_docs = add_pymupdf_page_metadata(
                   doc,
                   chunk_size=chunk_size,
                   chunk_overlap=chunk_overlap,
               )
               processed_pdf_docs.extend(chunked_docs)
           
           texts.extend(processed_pdf_docs)

       return texts

   except Exception as e:
       logging.exception("Error during document splitting")
       logging.error(f"Error type: {type(e)}")
       raise
