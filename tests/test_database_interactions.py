import json
import sqlite3

import pytest

from db.document_processor import Document
from db.sqlite_operations import create_metadata_db
from db.embedding_models import (
    _get_model_family,
    _get_prompt_for_family,
    _normalize_text,
)
from db.database_interactions import CreateVectorDB


class TestGetModelFamily:

    def test_qwen(self):
        assert _get_model_family("path/to/Qwen3-Embedding-4B") == "qwen"

    def test_bge(self):
        assert _get_model_family("path/to/BAAI--bge-small-en-v1.5") == "bge"

    def test_generic(self):
        assert _get_model_family("path/to/some-other-model") == "generic"

    def test_case_insensitive(self):
        assert _get_model_family("QWEN3-EMBEDDING-0.6B") == "qwen"


class TestGetPromptForFamily:

    def test_qwen_query(self):
        prompt = _get_prompt_for_family("qwen", is_query=True)
        assert "Instruct:" in prompt

    def test_bge_query(self):
        prompt = _get_prompt_for_family("bge", is_query=True)
        assert "Represent this sentence" in prompt

    def test_generic_no_prompt(self):
        assert _get_prompt_for_family("generic", is_query=True) == ""

    def test_non_query_no_prompt(self):
        assert _get_prompt_for_family("qwen", is_query=False) == ""
        assert _get_prompt_for_family("bge", is_query=False) == ""


class TestNormalizeText:

    def test_basic(self):
        result = _normalize_text("Hello World")
        assert result == "Hello World"

    def test_newlines_to_spaces(self):
        result = _normalize_text("Hello\nWorld")
        assert result == "Hello World"

    def test_tabs_to_spaces(self):
        result = _normalize_text("Hello\tWorld")
        assert result == "Hello World"

    def test_control_chars_removed(self):
        result = _normalize_text("Hello\x00World")
        assert result == "HelloWorld"

    def test_empty_returns_space(self):
        result = _normalize_text("")
        assert result == " "

    def test_whitespace_collapsed(self):
        result = _normalize_text("Hello   World")
        assert result == "Hello World"


class TestCreateMetadataDb:

    def test_creates_tables(self, tmp_path):
        persist_dir = tmp_path / "test_db"
        docs = [Document(page_content="Hello", metadata={'file_name': 'a.txt', 'hash': 'h1', 'file_path': '/a.txt'})]
        create_metadata_db(persist_dir, docs, [("id1", "h1")])

        db_path = persist_dir / "metadata.db"
        assert db_path.exists()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert 'document_metadata' in tables
        assert 'hash_chunk_ids' in tables

    def test_inserts_documents(self, tmp_path):
        persist_dir = tmp_path / "test_db"
        docs = [
            Document(page_content="Hello", metadata={'file_name': 'a.txt', 'hash': 'h1', 'file_path': '/a.txt'}),
            Document(page_content="World", metadata={'file_name': 'b.txt', 'hash': 'h2', 'file_path': '/b.txt'}),
        ]
        create_metadata_db(persist_dir, docs, [("id1", "h1"), ("id2", "h2")])

        conn = sqlite3.connect(persist_dir / "metadata.db")
        cursor = conn.cursor()
        cursor.execute("SELECT file_name FROM document_metadata")
        names = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert names == {'a.txt', 'b.txt'}

    def test_inserts_hash_mappings(self, tmp_path):
        persist_dir = tmp_path / "test_db"
        docs = [Document(page_content="Hello", metadata={'file_name': 'a.txt', 'hash': 'h1', 'file_path': '/a.txt'})]
        mappings = [("id1", "h1"), ("id2", "h1")]
        create_metadata_db(persist_dir, docs, mappings)

        conn = sqlite3.connect(persist_dir / "metadata.db")
        cursor = conn.cursor()
        cursor.execute("SELECT tiledb_id, hash FROM hash_chunk_ids")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 2


class TestLoadAudioDocuments:

    def _make_instance(self, tmp_path):
        instance = object.__new__(CreateVectorDB)
        instance.config = None
        instance.SOURCE_DIRECTORY = tmp_path / "audio_docs"
        instance.SOURCE_DIRECTORY.mkdir()
        instance.PERSIST_DIRECTORY = tmp_path / "Vector_DB" / "test_db"
        return instance

    def test_reads_json_audio(self, tmp_path):
        import json
        instance = self._make_instance(tmp_path)
        (instance.SOURCE_DIRECTORY / "audio.json").write_text(
            json.dumps({"page_content": "Transcribed text", "metadata": {"file_name": "audio.mp3"}}),
            encoding="utf-8",
        )
        docs = instance.load_audio_documents()
        assert len(docs) == 1
        assert docs[0].page_content == "Transcribed text"
        assert docs[0].metadata == {"file_name": "audio.mp3"}

    def test_empty(self, tmp_path):
        instance = self._make_instance(tmp_path)
        assert instance.load_audio_documents() == []

    def test_ignores_non_json(self, tmp_path):
        instance = self._make_instance(tmp_path)
        (instance.SOURCE_DIRECTORY / "note.txt").write_text("not audio", encoding="utf-8")
        assert instance.load_audio_documents() == []
