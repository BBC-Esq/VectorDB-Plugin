import gc
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional
import threading
import re
import sqlite3
import torch
import yaml
import concurrent.futures
import queue
from collections import defaultdict, deque
import shutil
import random
import sys
import traceback

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import TileDB

from document_processor import load_documents, split_documents
from module_process_images import choose_image_loader
from utilities import my_cprint, get_model_native_precision, get_appropriate_dtype, supports_flash_attention, set_cuda_paths
from constants import VECTOR_MODELS

logging.basicConfig(level=logging.CRITICAL, force=True)
logger = logging.getLogger(__name__)


class BaseEmbeddingModel:
    """
    Wrapper that prepares structured kwargs for HuggingFaceEmbeddings (SentenceTransformer).

    Outer dict keys map directly to SentenceTransformer.__init__ parameters:
      - device
      - trust_remote_code
      - prompts / default_prompt_name (not used here but supported)
      - model_kwargs        (inner dict: low-level HF AutoModel args)
      - tokenizer_kwargs    (outer-level sibling)
      - config_kwargs       (outer-level sibling)

    encode_kwargs are passed to SentenceTransformer.encode().
    """

    def __init__(self, model_name, model_kwargs, encode_kwargs, is_query: bool = False):
        self.model_name = model_name
        self.model_kwargs = model_kwargs            # OUTER kwargs to SentenceTransformer
        self.encode_kwargs = encode_kwargs          # kwargs to .encode()
        self.is_query = is_query

    def prepare_kwargs(self):
        # shallow copy to avoid mutating caller's dict
        hf_embed_kw = deepcopy(self.model_kwargs)

        # Required outer-level defaults
        hf_embed_kw.setdefault("device", "cpu")
        hf_embed_kw.setdefault("trust_remote_code", True)

        # Ensure tokenizer kwargs (outer-level)
        tok_kw = hf_embed_kw.setdefault("tokenizer_kwargs", {})
        tok_kw.setdefault("trust_remote_code", True)
        tok_kw.setdefault("use_fast", True)
        tok_kw.setdefault("padding", True)
        tok_kw.setdefault("truncation", True)
        tok_kw.setdefault("return_token_type_ids", False)
        tok_kw.setdefault("model_max_length", 512)

        # Inner model_kwargs (low-level HF AutoModel args) should exist only if needed
        inner = hf_embed_kw.get("model_kwargs", {})
        # Remove meaningless None values
        inner = {k: v for k, v in inner.items() if v is not None}
        if inner:
            hf_embed_kw["model_kwargs"] = inner
        else:
            hf_embed_kw.pop("model_kwargs", None)

        return hf_embed_kw

    def prepare_encode_kwargs(self):
        if self.is_query:
            self.encode_kwargs.setdefault("batch_size", 1)
            self.encode_kwargs.setdefault("max_length", 512)
            self.encode_kwargs.setdefault("normalize_embeddings", True)
        return self.encode_kwargs

    def create(self):
        model_kwargs = self.prepare_kwargs()
        encode_kwargs = self.prepare_encode_kwargs()

        hf = HuggingFaceEmbeddings(
            model_name=self.model_name,
            show_progress=not self.is_query,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return hf


class SnowflakeEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        snow_kwargs = super().prepare_kwargs()

        # Large variant override
        if "large" in self.model_name.lower():
            tok_kw = snow_kwargs.setdefault("tokenizer_kwargs", {})
            tok_kw.update({"model_max_length": 8192})
            return snow_kwargs

        device = snow_kwargs.get("device", "").lower()
        is_cuda = device.startswith("cuda")
        use_xformers = is_cuda and supports_flash_attention()

        tok_kw = snow_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({"model_max_length": 8192})

        # Memory efficient attention flags
        snow_kwargs["config_kwargs"] = {
            "use_memory_efficient_attention": use_xformers,
            "unpad_inputs": use_xformers,
            "attn_implementation": "eager" if use_xformers else "sdpa",
        }
        return snow_kwargs


class StellaEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        stella_kwargs = super().prepare_kwargs()
        tok_kw = stella_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({"model_max_length": 512})
        return stella_kwargs

    def prepare_encode_kwargs(self):
        encode_kwargs = super().prepare_encode_kwargs()
        if self.is_query:
            encode_kwargs["prompt_name"] = "s2p_query"
        return encode_kwargs


class Stella400MEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        stella_kwargs = super().prepare_kwargs()
        device = stella_kwargs.get("device", "")
        is_cuda = device.lower().startswith("cuda")
        use_xformers = is_cuda and supports_flash_attention()

        tok_kw = stella_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({"model_max_length": 512})

        stella_kwargs["config_kwargs"] = {
            "use_memory_efficient_attention": use_xformers,
            "unpad_inputs": use_xformers,
            "attn_implementation": "eager",  # required eager
        }
        return stella_kwargs

    def prepare_encode_kwargs(self):
        encode_kwargs = super().prepare_encode_kwargs()
        if self.is_query:
            encode_kwargs["prompt_name"] = "s2p_query"
        return encode_kwargs


class BgeCodeEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        bge_kwargs = super().prepare_kwargs()
        tok_kw = bge_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({"model_max_length": 4096})
        return bge_kwargs


class AlibabaEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        ali_kwargs = super().prepare_kwargs()
        device = ali_kwargs.get("device", "").lower()
        is_cuda = device.startswith("cuda")
        use_xformers = is_cuda and supports_flash_attention()

        tok_kw = ali_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({"model_max_length": 8192})

        ali_kwargs["config_kwargs"] = {
            "use_memory_efficient_attention": use_xformers,
            "unpad_inputs": use_xformers,
            "attn_implementation": "eager" if use_xformers else "sdpa",
        }
        return ali_kwargs


class InflyAndAlibabaEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        infly_kwargs = super().prepare_kwargs()
        tok_kw = infly_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({"model_max_length": 8192})
        return infly_kwargs


class QwenEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        q_kwargs = super().prepare_kwargs()

        # Inner model kwargs only if needed
        inner = q_kwargs.setdefault("model_kwargs", {})
        inner.update({
            "attn_implementation": "sdpa",
        })

        tok_kw = q_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({
            "padding_side": "left",
            "model_max_length": 8192,
        })
        return q_kwargs

    def prepare_encode_kwargs(self):
        encode_kwargs = super().prepare_encode_kwargs()
        encode_kwargs["max_length"] = 8192
        return encode_kwargs

def create_vector_db_in_process(database_name):
    # to address potential conflict with python 3.12 and tokenizers
    import faulthandler, os
    faulthandler.enable(all_threads=True,
                        file=open(f"faulthandler_{os.getpid()}.log", "w"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # import torch
    # torch.set_num_threads(1)

    set_cuda_paths()
    create_vector_db = CreateVectorDB(database_name=database_name)
    create_vector_db.run()

def process_chunks_only_query(database_name, query, result_queue):
    try:
        query_db = QueryVectorDB.get_instance(database_name)
        contexts, metadata_list = query_db.search(query)

        formatted_contexts = []
        for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
            file_name = metadata.get('file_name', 'Unknown')
            cleaned_context = re.sub(r'\n[ \t]+\n', '\n\n', context)
            cleaned_context = re.sub(r'\n\s*\n\s*\n*', '\n\n', cleaned_context.strip())
            formatted_context = (
                f"{'-'*80}\n"
                f"CONTEXT {index} | {file_name}\n"
                f"{'-'*80}\n"
                f"{cleaned_context}\n"
            )
            formatted_contexts.append(formatted_context)

        result_queue.put("\n".join(formatted_contexts))
    except Exception as e:
        result_queue.put(f"Error querying database: {str(e)}")
    finally:
        if 'query_db' in locals():
            query_db.cleanup()


class CreateVectorDB:
    def __init__(self, database_name):
        self.ROOT_DIRECTORY = Path(__file__).resolve().parent
        self.SOURCE_DIRECTORY = self.ROOT_DIRECTORY / "Docs_for_DB"
        self.PERSIST_DIRECTORY = self.ROOT_DIRECTORY / "Vector_DB" / database_name

    def load_config(self, root_directory):
        with open(root_directory / "config.yaml", 'r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)

    @torch.inference_mode()
    def initialize_vector_model(self, embedding_model_name, config_data):
        compute_device = config_data['Compute_Device']['database_creation']
        use_half = config_data.get("database", {}).get("half", False)
        model_native_precision = get_model_native_precision(embedding_model_name, VECTOR_MODELS)
        torch_dtype = get_appropriate_dtype(compute_device, use_half, model_native_precision)

        # Outer kwargs for SentenceTransformer
        outer_model_kwargs = {
            "device": compute_device,
            "trust_remote_code": True,
            # Inner low-level HF model args ONLY if needed
            "model_kwargs": {
                "torch_dtype": torch_dtype if torch_dtype is not None else None,
                "attn_implementation": "sdpa",   # keep if supported by model
            },
            # tokenizer_kwargs / config_kwargs added by subclass overrides
        }

        # Clean inner None values
        inner = outer_model_kwargs["model_kwargs"]
        inner = {k: v for k, v in inner.items() if v is not None}
        if inner:
            outer_model_kwargs["model_kwargs"] = inner
        else:
            outer_model_kwargs.pop("model_kwargs", None)

        encode_kwargs = {
            'batch_size': 8,
            'padding': True,
            'truncation': True,
            "normalize_embeddings": True,
        }

        if compute_device.lower() == 'cpu':
            encode_kwargs['batch_size'] = 2
        else:
            batch_size_mapping = {
                'inf-retriever-v1-7b': 2,
                'Qwen3-Embedding-8B': 2,
                'Qwen3-Embedding-4B': 3,
                'inf-retriever-v1-1.5b': 3,
                'stella_en_1.5B': 4,
                'e5-base': 6,
                'e5-large': 7,
                'arctic-embed-l': 7,
                'e5-small': 10,
                'gte-large': 12,
                'Granite-30m-English': 12,
                'bge-small': 12,
                'gte-base': 14,
                'arctic-embed-m': 14,
                'stella_en_400M_v5': 20,
            }
            for key, value in batch_size_mapping.items():
                if key in embedding_model_name:
                    encode_kwargs['batch_size'] = value
                    break

        name_lower = embedding_model_name.lower()

        if "qwen3-embedding" in name_lower:
            print("Using QwenEmbedding class.")
            model = QwenEmbedding(embedding_model_name, outer_model_kwargs, encode_kwargs).create()

        elif "snowflake" in name_lower:
            print("Using SnowflakeEmbedding class.")
            model = SnowflakeEmbedding(embedding_model_name, outer_model_kwargs, encode_kwargs).create()

        elif "alibaba" in name_lower:
            print("Using AlibabaEmbedding class.")
            model = AlibabaEmbedding(embedding_model_name, outer_model_kwargs, encode_kwargs).create()

        elif "400m" in name_lower:
            print("Using Stella400MEmbedding class.")
            model = Stella400MEmbedding(embedding_model_name, outer_model_kwargs, encode_kwargs).create()

        elif "stella_en_1.5b_v5" in name_lower:
            print("Using StellaEmbedding class.")
            model = StellaEmbedding(embedding_model_name, outer_model_kwargs, encode_kwargs).create()

        elif "bge-code" in name_lower:
            print("Using BgeCodeEmbedding class.")
            model = BgeCodeEmbedding(embedding_model_name, outer_model_kwargs, encode_kwargs).create()

        elif "infly" in name_lower:
            print("Using InflyAndAlibabaEmbedding class.")
            model = InflyAndAlibabaEmbedding(embedding_model_name, outer_model_kwargs, encode_kwargs).create()

        elif "e5" in name_lower:
            print("Using BaseEmbeddingModel class with Intfloat prompt.")
            encode_kwargs["prompt"] = "passage: "
            model = BaseEmbeddingModel(embedding_model_name, outer_model_kwargs, encode_kwargs).create()

        else:
            print("Using BaseEmbeddingModel class.")
            model = BaseEmbeddingModel(embedding_model_name, outer_model_kwargs, encode_kwargs).create()

        model_name = os.path.basename(embedding_model_name)
        precision = "float32" if torch_dtype is None else str(torch_dtype).split('.')[-1]
        my_cprint(f"{model_name} ({precision}) loaded using a batch size of {encode_kwargs['batch_size']}.", "green")

        return model, encode_kwargs

    @torch.inference_mode()
    def create_database(self, texts, embeddings):

        my_cprint("\nComputing vectors...", "yellow")
        start_time = time.time()

        hash_id_mappings = []
        MAX_UINT64 = 18446744073709551615

        # atomically create the DB folder
        created_new_dir = False
        try:
            self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=False)
            created_new_dir = True
            my_cprint(f"Created directory: {self.PERSIST_DIRECTORY}", "green")
        except FileExistsError:
            raise FileExistsError(
                f"Vector database '{self.PERSIST_DIRECTORY.name}' already exists. "
                "Choose a different name or delete the existing DB first."
            )

        try:
            all_texts = []
            all_metadatas = []
            all_ids = []
            chunk_counters = defaultdict(int)

            # Process all texts and generate IDs
            for doc in texts:
                file_hash = doc.metadata.get('hash')
                chunk_counters[file_hash] += 1
                tiledb_id = str(random.randint(0, MAX_UINT64 - 1))

                all_texts.append(doc.page_content)
                all_metadatas.append(doc.metadata)
                all_ids.append(tiledb_id)
                hash_id_mappings.append((tiledb_id, file_hash))

            with open(self.ROOT_DIRECTORY / "config.yaml", 'r', encoding='utf-8') as config_file:
                config_data = yaml.safe_load(config_file)

            # create embeddings
            embedding_start_time = time.time()
            vectors = embeddings.embed_documents(all_texts)
            embedding_end_time = time.time()
            embedding_elapsed = embedding_end_time - embedding_start_time
            my_cprint(f"Embedding computation completed in {embedding_elapsed:.2f} seconds.", "cyan")

            # Build (text, embedding) tuples in correct order
            text_embed_pairs = [
                (txt, vec.tolist() if hasattr(vec, 'tolist') else list(vec))
                for txt, vec in zip(all_texts, vectors)
            ]

            # create TileDB vector store
            TileDB.from_embeddings(
                text_embeddings=text_embed_pairs,
                embedding=embeddings,
                metadatas=all_metadatas,
                ids=all_ids,
                metric="euclidean",
                index_uri=str(self.PERSIST_DIRECTORY),
                index_type="FLAT",
                allow_dangerous_deserialization=True,
            )

            my_cprint("Processed all chunks", "yellow")

            end_time = time.time()
            elapsed_time = end_time - start_time
            my_cprint(f"Database created. Elapsed time: {elapsed_time:.2f} seconds.", "green")

            return hash_id_mappings

        except Exception as e:
            # show full traceback from child process
            traceback.print_exc()
            logging.error(f"Error creating database: {str(e)}")
            if self.PERSIST_DIRECTORY.exists():
                try:
                    shutil.rmtree(self.PERSIST_DIRECTORY)
                    logging.info(f"Cleaned up failed database creation at: {self.PERSIST_DIRECTORY}")
                except Exception as cleanup_error:
                    logging.error(f"Failed to clean up database directory: {cleanup_error}")
            raise

    def create_metadata_db(self, documents, hash_id_mappings):
        if not self.PERSIST_DIRECTORY.exists():
            self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

        sqlite_db_path = self.PERSIST_DIRECTORY / "metadata.db"
        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                hash TEXT,
                file_path TEXT,
                page_content TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hash_chunk_ids (
                tiledb_id TEXT PRIMARY KEY,
                hash TEXT
            )
        ''')

        try:
            # Prepare batch data for documents
            doc_rows = [
                (
                    doc.metadata.get("file_name", ""),
                    doc.metadata.get("hash", ""),
                    doc.metadata.get("file_path", ""),
                    doc.page_content
                )
                for doc in documents
            ]
            cursor.executemany('''
                INSERT INTO document_metadata (file_name, hash, file_path, page_content)
                VALUES (?, ?, ?, ?)
            ''', doc_rows)

            # Batch insert hash–ID mappings
            cursor.executemany('''
                INSERT INTO hash_chunk_ids (tiledb_id, hash)
                VALUES (?, ?)
            ''', hash_id_mappings)

            conn.commit()
        finally:
            conn.close()

    def load_audio_documents(self, source_dir: Path = None) -> list:
        # checks the DocsforDB folder for .json files, which audio transcriptions are always saved to, and loads them
        if source_dir is None:
            source_dir = self.SOURCE_DIRECTORY
        json_paths = [f for f in source_dir.iterdir() if f.suffix.lower() == '.json']
        docs = []

        for json_path in json_paths:
            try:
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    json_str = json_file.read()
                    doc = Document.parse_raw(json_str)
                    docs.append(doc)
            except Exception as e:
                my_cprint(f"Error loading {json_path}: {e}", "red")

        return docs
    
    def clear_docs_for_db_folder(self):
        for item in self.SOURCE_DIRECTORY.iterdir():
            if item.is_file() or item.is_symlink():
                try:
                    item.unlink()
                except Exception as e:
                    print(f"Failed to delete {item}: {e}")

    @torch.inference_mode()
    def run(self):
        config_data = self.load_config(self.ROOT_DIRECTORY)
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")

        # list to hold document objects
        documents = []

        # load text document objects
        text_documents = load_documents(self.SOURCE_DIRECTORY)
        if isinstance(text_documents, list) and text_documents:
            documents.extend(text_documents)

        # separate lists for pdf and non-pdf document objects
        text_documents_pdf = [doc for doc in documents if doc.metadata.get("file_type") == ".pdf"]
        documents = [doc for doc in documents if doc.metadata.get("file_type") != ".pdf"]

        # load image descriptions
        print("Loading any images...")
        image_documents = choose_image_loader()
        if isinstance(image_documents, list) and image_documents:
            if len(image_documents) > 0:
                documents.extend(image_documents)

        # load audio transcriptions
        print("Loading any audio transcripts...")
        audio_documents = self.load_audio_documents()
        if isinstance(audio_documents, list) and audio_documents:
            documents.extend(audio_documents)

        # create a list to save pre-split text for sqliteDB
        json_docs_to_save = []
        json_docs_to_save.extend(documents)
        json_docs_to_save.extend(text_documents_pdf)

        # blank list to hold all split document objects
        texts = []

        # split document objects and add to list
        if (isinstance(documents, list) and documents) or (isinstance(text_documents_pdf, list) and text_documents_pdf):
            texts = split_documents(documents, text_documents_pdf)
            print(f"Documents split into {len(texts)} chunks.")

        del documents, text_documents_pdf
        gc.collect()

        # create db
        if isinstance(texts, list) and texts:
            embeddings, encode_kwargs = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)

            # Get hash->ID mappings and create the vector database
            hash_id_mappings = self.create_database(texts, embeddings)

            del texts
            gc.collect()

            # Pass mappings to metadata db creation
            self.create_metadata_db(json_docs_to_save, hash_id_mappings)
            del json_docs_to_save
            gc.collect()
            self.clear_docs_for_db_folder()


class QueryVectorDB:
    _instance = None
    _instance_lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Use QueryVectorDB.get_instance() instead of direct instantiation")

    def _init_once(self, selected_database):
        if not self._initialized:
            self.config = self.load_configuration()

            # upfront validation of the request
            if not selected_database:
                raise ValueError("No vector database selected.")
            if selected_database not in self.config["created_databases"]:
                raise ValueError(f'Database “{selected_database}” not found in config.')
            db_path = Path(__file__).resolve().parent / "Vector_DB" / selected_database
            if not db_path.exists():
                raise FileNotFoundError(f'Database folder “{selected_database}” is missing on disk.')

            self.selected_database = selected_database
            self.embeddings = None
            self.db = None
            self.model_name = None
            self._debug_id = id(self)
            self._initialized = True
            logging.debug(f"Created new QueryVectorDB instance {self._debug_id} for database {selected_database}")

    @classmethod
    def get_instance(cls, selected_database):
        with cls._instance_lock:
            if cls._instance is not None:
                if cls._instance.selected_database != selected_database:
                    print(f"Database changed from {cls._instance.selected_database} to {selected_database}")
                    cls._instance.cleanup()
                    cls._instance = None
                else:
                    logging.debug(f"Reusing existing instance {cls._instance._debug_id} for database {selected_database}")

            if cls._instance is None:
                cls._instance = object.__new__(cls)
                cls._instance._init_once(selected_database)

            return cls._instance

    def load_configuration(self):
        config_path = Path(__file__).resolve().parent / 'config.yaml'
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise

    @torch.inference_mode()
    def initialize_vector_model(self):
        model_path = self.config['created_databases'][self.selected_database]['model']
        self.model_name = os.path.basename(model_path)
        compute_device = self.config['Compute_Device']['database_query']

        outer_model_kwargs = {
            "device": compute_device,
            "trust_remote_code": True,
            # Add inner 'model_kwargs' only if you later decide to pass low-level model args (e.g. torch_dtype)
            # "model_kwargs": {"torch_dtype": torch.bfloat16},
        }

        encode_kwargs = {
            "batch_size": 1,
            'padding': True,
            'truncation': True,
            "normalize_embeddings": True,
        }

        mp_lower = model_path.lower()

        if "qwen3-embedding" in mp_lower:
            encode_kwargs["prompt"] = (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
            )
            embeddings = QwenEmbedding(model_path, outer_model_kwargs, encode_kwargs, is_query=True).create()

        elif "snowflake" in mp_lower:
            encode_kwargs["prompt"] = "query: "
            embeddings = SnowflakeEmbedding(model_path, outer_model_kwargs, encode_kwargs, is_query=True).create()

        elif "intfloat" in mp_lower:
            encode_kwargs["prompt"] = "query: "
            embeddings = BaseEmbeddingModel(model_path, outer_model_kwargs, encode_kwargs, is_query=True).create()

        elif "alibaba" in mp_lower:
            encode_kwargs["prompt"] = (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
            )
            embeddings = AlibabaEmbedding(model_path, outer_model_kwargs, encode_kwargs, is_query=True).create()

        elif "infly" in mp_lower:
            encode_kwargs["prompt"] = (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
            )
            embeddings = InflyAndAlibabaEmbedding(model_path, outer_model_kwargs, encode_kwargs, is_query=True).create()

        elif "bge-code" in mp_lower:
            code_instruction = "Given a question in text, retrieve relevant code that is relevant."
            encode_kwargs["prompt"] = f"<instruct>{code_instruction}\n<query>"
            embeddings = BgeCodeEmbedding(model_path, outer_model_kwargs, encode_kwargs, is_query=True).create()

        elif "stella" in mp_lower:
            encode_kwargs["prompt"] = (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
            )
            if "400m" in mp_lower:
                embeddings = Stella400MEmbedding(model_path, outer_model_kwargs, encode_kwargs, is_query=True).create()
            else:
                embeddings = StellaEmbedding(model_path, outer_model_kwargs, encode_kwargs, is_query=True).create()

        elif "bge" in mp_lower:
            encode_kwargs["prompt"] = "Represent this sentence for searching relevant passages: "
            embeddings = BaseEmbeddingModel(model_path, outer_model_kwargs, encode_kwargs, is_query=True).create()

        else:
            embeddings = BaseEmbeddingModel(model_path, outer_model_kwargs, encode_kwargs, is_query=True).create()

        return embeddings

    def initialize_database(self):
        persist_directory = Path(__file__).resolve().parent / "Vector_DB" / self.selected_database

        return TileDB.load(index_uri=str(persist_directory), embedding=self.embeddings, allow_dangerous_deserialization=True)

    @torch.inference_mode()
    def search(self, query, k: Optional[int] = None, score_threshold: Optional[float] = None):
        if not self.embeddings:
            logging.info(f"Initializing embedding model for database {self.selected_database}")
            self.embeddings = self.initialize_vector_model()

        if not self.db:
            logging.info(f"Initializing database connection for {self.selected_database}")
            self.db = self.initialize_database()

        self.config = self.load_configuration()
        document_types = self.config['database'].get('document_types', '')
        search_filter = {'document_type': document_types} if document_types else {}
        
        k = k if k is not None else int(self.config['database']['contexts'])
        score_threshold = score_threshold if score_threshold is not None else float(self.config['database']['similarity'])

        relevant_contexts = self.db.similarity_search_with_score(
            query,
            k=k,
            filter=search_filter,
            score_threshold=score_threshold
        )

        search_term = self.config['database'].get('search_term', '').lower()
        if search_term:
            filtered_contexts = [
                (doc, score) for doc, score in relevant_contexts
                if search_term in doc.page_content.lower()
            ]
        else:
            filtered_contexts = relevant_contexts

        contexts = [document.page_content for document, _ in filtered_contexts]
        metadata_list = [document.metadata for document, _ in filtered_contexts]
        scores = [score for _, score in filtered_contexts]

        for metadata, score in zip(metadata_list, scores):
            metadata['similarity_score'] = score

        return contexts, metadata_list

    def cleanup(self):
        logging.info(f"Cleaning up QueryVectorDB instance {self._debug_id} for database {self.selected_database}")

        if self.embeddings:
            logging.debug(f"Unloading embedding model for database {self.selected_database}")
            del self.embeddings
            self.embeddings = None

        if self.db:
            logging.debug(f"Closing database connection for {self.selected_database}")
            del self.db
            self.db = None

        if torch.cuda.is_available():
            logging.debug("Clearing CUDA cache")
            torch.cuda.empty_cache()

        gc.collect()
        logging.debug(f"Cleanup completed for instance {self._debug_id}")
