import gc
import logging
import warnings
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch
# NOTE: newer torch uses new namespace - torch.amp
from torch.cuda.amp import autocast # necessary to set dtype for instructor
import yaml
from InstructorEmbedding import INSTRUCTOR
from PySide6.QtCore import QDir
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import TileDB
from langchain_core.load import dumps # debugging

from document_processor import load_documents, split_documents
from module_process_images import choose_image_loader
from utilities import my_cprint

datasets_logger = logging.getLogger('datasets')
datasets_logger.setLevel(logging.WARNING)

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger().setLevel(logging.WARNING)

# debugging
# def serialize_documents_to_json(documents, file_name="split_document_objects.json"):
    # print("Saving to JSON...")
    # json_string = dumps(documents, pretty=True)

    # with open(file_name, "w") as json_file:
        # json_file.write(json_string)
            
class CreateVectorDB:
    def __init__(self, database_name):
        self.ROOT_DIRECTORY = Path(__file__).resolve().parent
        self.SOURCE_DIRECTORY = self.ROOT_DIRECTORY / "Docs_for_DB"
        self.PERSIST_DIRECTORY = self.ROOT_DIRECTORY / "Vector_DB" / database_name
        self.SAVE_JSON_DIRECTORY = self.ROOT_DIRECTORY / "Vector_DB" / database_name / "json"

    def load_config(self, root_directory):
        with open(root_directory / "config.yaml", 'r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)

    def get_appropriate_dtype(self, compute_device, use_half):
        if compute_device.lower() == 'cpu' or not use_half:
            return None
        
        # checks for nvidia gpu+cuda+pytorch
        if torch.cuda.is_available() and torch.version.cuda:
            cuda_capability = torch.cuda.get_device_capability()
            if cuda_capability[0] >= 8 and cuda_capability[1] >= 6:
                return torch.bfloat16
            else:
                return torch.float16
        else:
            return None

    @torch.inference_mode()
    def initialize_vector_model(self, embedding_model_name, config_data):
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
        compute_device = config_data['Compute_Device']['database_creation']
        
        use_half = config_data.get("database", {}).get("half", False)
        
        model_kwargs = {
            "device": compute_device, 
            "trust_remote_code": True
        }
        
        torch_dtype = self.get_appropriate_dtype(compute_device, use_half)

        encode_kwargs = {'normalize_embeddings': True, 'batch_size': 8}

        if compute_device.lower() == 'cpu':
            encode_kwargs['batch_size'] = 2
        else:
            batch_size_mapping = {
                't5-xl': 1,
                't5-large': 2,
                'instructor-xl': 2,
                't5-base': 3,
                'bge-large': 3,
                'instructor-large': 3,
                'e5-large': 3,
                'gte-large': 3,
                'instructor-base': 6,
                'mpnet': 8,
                'e5-base': 8,
                'bge-base': 8,
                'gte-base': 8,
                'e5-small': 10,
                'bge-small': 10,
                'gte-small': 10,
                'MiniLM': 30,
            }

            for key, value in batch_size_mapping.items():
                if isinstance(key, tuple):
                    if any(model_name_part in EMBEDDING_MODEL_NAME for model_name_part in key):
                        encode_kwargs['batch_size'] = value
                        break
                else:
                    if key in EMBEDDING_MODEL_NAME:
                        encode_kwargs['batch_size'] = value
                        break

        if "instructor" in embedding_model_name:
            encode_kwargs['show_progress_bar'] = True

            model = HuggingFaceInstructEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )

            # must use torch.amp. for instructor
            if torch_dtype is not None:
                model.client[0].auto_model = model.client[0].auto_model.to(torch_dtype)

        elif "bge" in embedding_model_name:
            query_instruction = config_data['embedding-models']['bge'].get('query_instruction')
            if torch_dtype is not None:
                model_kwargs["model_kwargs"] = {"torch_dtype": torch_dtype}
            encode_kwargs['show_progress_bar'] = True

            model = HuggingFaceBgeEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                query_instruction=query_instruction,
                encode_kwargs=encode_kwargs
            )

        elif "Alibaba" in embedding_model_name:
            if torch_dtype is not None:
                model_kwargs["model_kwargs"] = {"torch_dtype": torch_dtype}
            model_kwargs["tokenizer_kwargs"] = {
                "max_length": 8192,
                "padding": True,
                "truncation": True
            }
            model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                show_progress=True,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

        else:
            if torch_dtype is not None:
                model_kwargs["model_kwargs"] = {"torch_dtype": torch_dtype}
            model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                show_progress=True,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

        model_name = Path(EMBEDDING_MODEL_NAME).name
        precision = "float32" if torch_dtype is None else str(torch_dtype).split('.')[-1]
    
        my_cprint(f"{model_name} ({precision}) loaded into memory using batch size of {encode_kwargs['batch_size']}.", "green")
        
        return model, encode_kwargs

    @torch.inference_mode()
    def create_database(self, texts, embeddings):
        my_cprint("The progress bar relates to computing vectors. Afterwards, it takes a little time to insert them into the database and save to disk.\n", "yellow")

        start_time = time.time()

        if not self.PERSIST_DIRECTORY.exists():
            self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {self.PERSIST_DIRECTORY}")
        else:
            logging.info(f"Directory already exists: {self.PERSIST_DIRECTORY}")

        try:
            # separate page_content and metadata
            text_content = [doc.page_content for doc in texts]
            metadatas = [doc.metadata for doc in texts]
            
            # Get the EMBEDDING_MODEL_NAME from config
            with open(self.ROOT_DIRECTORY / "config.yaml", 'r', encoding='utf-8') as config_file:
                config_data = yaml.safe_load(config_file)
            embedding_model_name = config_data.get("EMBEDDING_MODEL_NAME", "")

            # If "intfloat" in the model's name add "passage:" to each page_content
            if "intfloat" in embedding_model_name.lower():
                text_content = [f"passage: {content}" for content in text_content]

            logging.info("Calling TileDB.from_texts()")
            db = TileDB.from_texts(
                texts=text_content,
                embedding=embeddings,
                metadatas=metadatas,
                index_uri=str(self.PERSIST_DIRECTORY),
                allow_dangerous_deserialization=True,
                metric="euclidean",
                index_type="FLAT",
            )
        except Exception as e:
            logging.error(f"Error creating database: {str(e)}")
            raise

        end_time = time.time()
        elapsed_time = end_time - start_time

        print("Database created.")
        print(f"Creating the vector database took {elapsed_time:.2f} seconds.")
        
    def save_documents_to_json(self, json_docs_to_save):
        """
        Json of all documents successfully entered into the database for purposes of populating the database management tab.
        """
        if not self.SAVE_JSON_DIRECTORY.exists():
            self.SAVE_JSON_DIRECTORY.mkdir(parents=True, exist_ok=True)

        for document in json_docs_to_save:
            document_hash = document.metadata.get('hash', None)
            if document_hash:
                json_filename = f"{document_hash}.json"
                json_file_path = self.SAVE_JSON_DIRECTORY / json_filename
                
                actual_file_path = document.metadata.get('file_path')
                if os.path.islink(actual_file_path):
                    resolved_path = os.path.realpath(actual_file_path)
                    document.metadata['file_path'] = resolved_path

                document_json = document.json(indent=4)
                
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json_file.write(document_json)
            else:
                print("Warning: Document missing 'hash' in metadata. Skipping JSON creation.")
    
    def load_audio_documents(self, source_dir: Path = None) -> list:
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

    def save_documents_to_pickle(self, documents):
        """
        Pickle all document objects incase the database creation process terminates early.  Cleared each time the program starts.
        """
        pickle_directory = self.ROOT_DIRECTORY / "pickle"
        
        if not pickle_directory.exists():
            pickle_directory.mkdir(parents=True, exist_ok=True)
        
        for file in pickle_directory.glob("*.pickle"):
            file.unlink()

        time.sleep(3)

        for i, doc in enumerate(documents):
            pickle_file_path = pickle_directory / f"document_{i}.pickle"
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(doc, pickle_file)

    
    @torch.inference_mode()
    def run(self):
        config_data = self.load_config(self.ROOT_DIRECTORY)
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
        
        # list to hold "document objects"        
        documents = []
        
        # load text document objects
        print("Loading any general files...")
        text_documents = load_documents(self.SOURCE_DIRECTORY)
        if isinstance(text_documents, list) and text_documents:
            documents.extend(text_documents)

        # separate lists for pdf and non-pdf document objects
        text_documents_pdf = [doc for doc in documents if doc.metadata.get("file_type") == ".pdf"]
        documents = [doc for doc in documents if doc.metadata.get("file_type") != ".pdf"]

        # debugging
        # print(f"Initial documents count: {len(documents) + len(text_documents_pdf)}")
        # print(f"Non-PDF documents: {[doc.metadata.get('file_type') for doc in documents]}")
        # print(f"PDF documents: {[doc.metadata.get('file_type') for doc in text_documents_pdf]}")

        # debugging
        # serialize_documents_to_json(text_documents_pdf)
        
        # load image document objects
        print("Loading any images...")
        image_documents = choose_image_loader()
        if isinstance(image_documents, list) and image_documents:
            if len(image_documents) > 0:
                documents.extend(image_documents)
        
        # load audio document objects
        print("Loading any audio transcripts...")
        audio_documents = self.load_audio_documents()
        if isinstance(audio_documents, list) and audio_documents:
            documents.extend(audio_documents)

        json_docs_to_save = []
        json_docs_to_save.extend(documents)
        json_docs_to_save.extend(text_documents_pdf)

        # list to hold all split document objects
        texts = []

        # split
        if (isinstance(documents, list) and documents) or (isinstance(text_documents_pdf, list) and text_documents_pdf):
            texts = split_documents(documents, text_documents_pdf)

        # debugging
        # serialize_documents_to_json(texts, file_name="split_document_objects.json")

        # pickle
        if isinstance(texts, list) and texts:
            self.save_documents_to_pickle(texts)

            # initialize vector model
            embeddings, encode_kwargs = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)

            # create database
            if isinstance(texts, list) and texts:
                print("Creating vector database...")
                self.create_database(texts, embeddings)
            
            self.save_documents_to_json(json_docs_to_save)
            
            # cleanup
            del embeddings.client
            del embeddings
            torch.cuda.empty_cache()
            gc.collect()
            my_cprint("Vector model removed from memory.", "red")
            
            self.clear_docs_for_db_folder()


class QueryVectorDB:
    def __init__(self, selected_database):
        self.config = self.load_configuration()
        self.selected_database = selected_database
        self.embeddings = self.initialize_vector_model()
        self.db = self.initialize_database()

    def load_configuration(self):
        config_file_path = Path(__file__).resolve().parent / "config.yaml"
        with open(config_file_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def initialize_vector_model(self):    
        model_path = self.config['created_databases'][self.selected_database]['model']
        
        compute_device = self.config['Compute_Device']['database_query']
        encode_kwargs = {'normalize_embeddings': True, 'batch_size': 1}
        
        if "instructor" in model_path:
            return HuggingFaceInstructEmbeddings(
                model_name=model_path,
                model_kwargs={"device": compute_device, "trust_remote_code": True},
                encode_kwargs=encode_kwargs,
            )
        elif "bge" in model_path:
            query_instruction = self.config['embedding-models']['bge']['query_instruction']
            return HuggingFaceBgeEmbeddings(
                model_name=model_path,
                model_kwargs={"device": compute_device, "trust_remote_code": True},
                query_instruction=query_instruction,
                encode_kwargs=encode_kwargs
            )
        elif "Alibaba" in model_path:
            return HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={
                    "device": compute_device,
                    "trust_remote_code": True,
                    "tokenizer_kwargs": {
                        "max_length": 8192,
                        "padding": True,
                        "truncation": True
                    }
                },
                # model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        else:
            return HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"device": compute_device, "trust_remote_code": True},
                encode_kwargs=encode_kwargs
            )

    def initialize_database(self):
        persist_directory = Path(__file__).resolve().parent / "Vector_DB" / self.selected_database
        
        return TileDB.load(index_uri=str(persist_directory), embedding=self.embeddings, allow_dangerous_deserialization=True)

    def is_intfloat_model(self):
        model_path = self.config['created_databases'][self.selected_database]['model']
        return "intfloat" in model_path.lower()

    def search(self, query):
        # get latest settings
        self.config = self.load_configuration()
        document_types = self.config['database'].get('document_types', '')
        search_filter = {'document_type': document_types} if document_types else {}
        score_threshold = float(self.config['database']['similarity'])
        k = int(self.config['database']['contexts'])
        
        if self.is_intfloat_model():
            query = f"query: {query}"
        
        # search
        relevant_contexts = self.db.similarity_search_with_score(
            query,
            k=k,
            filter=search_filter,
            score_threshold=score_threshold
        )
        
        search_term = self.config['database'].get('search_term', '').lower()
        filtered_contexts = [(doc, score) for doc, score in relevant_contexts if search_term in doc.page_content.lower()]
        
        contexts = [document.page_content for document, _ in filtered_contexts]
        metadata_list = [document.metadata for document, _ in filtered_contexts]
        scores = [score for _, score in filtered_contexts] # similarity scores
        
        # DEBUG
        # print(scores)
        
        return contexts, metadata_list

class DeleteDoc:
    def __init__(self, selected_database):
        self.selected_database = selected_database
        self.config = self.load_configuration()
        self.embeddings = self.initialize_vector_model()
        self.vectorstore = self.initialize_database()

    def load_configuration(self):
        config_file_path = Path(__file__).resolve().parent / "config.yaml"
        with open(config_file_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def initialize_vector_model(self):    
        model_path = self.config['created_databases'][self.selected_database]['model']
        
        compute_device = self.config['Compute_Device']['database_query']
        encode_kwargs = {'normalize_embeddings': True, 'batch_size': 1}
        
        if "instructor" in model_path:
            return HuggingFaceInstructEmbeddings(
                model_name=model_path,
                model_kwargs={"device": compute_device, "trust_remote_code": True},
                encode_kwargs=encode_kwargs,
            )
        elif "bge" in model_path:
            query_instruction = self.config['embedding-models']['bge']['query_instruction']
            return HuggingFaceBgeEmbeddings(
                model_name=model_path,
                model_kwargs={"device": compute_device, "trust_remote_code": True},
                query_instruction=query_instruction,
                encode_kwargs=encode_kwargs
            )
        elif "Alibaba" in model_path:
            return HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={
                    "device": compute_device,
                    "trust_remote_code": True,
                    "tokenizer_kwargs": {
                        "max_length": 8192,
                        "padding": True,
                        "truncation": True
                    }
                },
                encode_kwargs=encode_kwargs
            )
        else:
            return HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"device": compute_device, "trust_remote_code": True},
                encode_kwargs=encode_kwargs
            )

    def initialize_database(self):
        persist_directory = Path(__file__).resolve().parent / "Vector_DB" / self.selected_database
        
        return TileDB.load(index_uri=str(persist_directory), embedding=self.embeddings, allow_dangerous_deserialization=True)

    def delete_entries_by_hash(self, hash_value: str, batch_size: int = 1000) -> int:
        total_deleted = 0
        while True:
            results = self.vectorstore.similarity_search_with_score(
                query="",
                k=batch_size,
                filter={"hash": hash_value}
            )
            
            if not results:
                break
            
            ids_to_delete = [str(doc.metadata.get("id")) for doc, _ in results if doc.metadata.get("hash") == hash_value]
            
            if ids_to_delete:
                self.vectorstore.delete(ids=ids_to_delete)
                total_deleted += len(ids_to_delete)
            
            if len(results) < batch_size:
                break
        
        return total_deleted
