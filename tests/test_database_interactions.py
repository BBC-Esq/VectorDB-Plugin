import json
import pickle
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from db.document_processor import Document
from db.sqlite_operations import create_metadata_db
from db.embedding_models import (
    _get_model_family,
    _get_prompt_for_family,
    _normalize_text,
)
from db.database_interactions import CreateVectorDB, QueryVectorDB, merge_added, normalized_rows, skip_removed


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


class TestSkipRemovedChunks:

    def test_drops_removed_ids_and_keeps_the_nearest(self):
        distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        ids = np.array([5, 6, 7, 8, 9], dtype=np.uint64)
        kept_distances, kept_ids = skip_removed(distances, ids, np.array([5, 7], dtype=np.uint64), 2)
        assert kept_ids.tolist() == [6, 8]
        assert kept_distances.tolist() == pytest.approx([0.2, 0.4])

    def test_nothing_removed(self):
        distances = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        ids = np.array([5, 6, 7], dtype=np.uint64)
        _, kept_ids = skip_removed(distances, ids, np.empty(0, dtype=np.uint64), 2)
        assert kept_ids.tolist() == [5, 6]

    def test_search_asks_for_extra_results_and_returns_the_rest(self, tmp_path):
        import tiledb

        uri = str(tmp_path / "vectors")
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(tiledb.Dim(name="id", domain=(0, np.iinfo(np.uint64).max - 20000), tile=10000, dtype=np.uint64)),
            attrs=[tiledb.Attr(name="text", dtype=str, var=True), tiledb.Attr(name="metadata", dtype=str, var=True)],
            sparse=True,
        )
        tiledb.Array.create(uri, schema)
        ids = np.arange(1, 7, dtype=np.uint64)
        with tiledb.open(uri, mode="w") as array:
            array[ids] = {
                "text": np.array([f"chunk {i}" for i in ids], dtype=object),
                "metadata": np.array([json.dumps({"file_name": f"f{i}.txt"}) for i in ids], dtype=object),
            }

        class FakeIndex:
            size = 6
            asked = None

            def query(self, vectors, k):
                FakeIndex.asked = k
                return np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6][:k]]), np.array([ids[:k]], dtype=np.uint64)

        query_db = QueryVectorDB.__new__(QueryVectorDB)
        query_db.index = FakeIndex()
        query_db.removed_ids = np.array([1, 3], dtype=np.uint64)
        query_db.added_ids = np.empty(0, dtype=np.uint64)
        query_db.added_vectors = np.empty((0, 0), dtype=np.float32)
        query_db.distance_metric = "cosine"
        query_db.array_uri = uri
        results, exhausted = query_db._retrieve(np.zeros((1, 4), dtype=np.float32), 2, 0.0, tiledb)
        assert FakeIndex.asked == 4
        assert [text for text, _ in results] == ["chunk 2", "chunk 4"]
        assert exhausted is False


def _chunk_store(path, vectors):
    import tiledb

    dimensions = len(next(iter(vectors.values())))
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(tiledb.Dim(name="id", domain=(0, np.iinfo(np.uint64).max - 20000), tile=10000, dtype=np.uint64)),
        attrs=[tiledb.Attr(name="vector", dtype=np.dtype([("", np.float32)] * dimensions)),
               tiledb.Attr(name="text", dtype=str, var=True), tiledb.Attr(name="metadata", dtype=str, var=True)],
        sparse=True,
    )
    tiledb.Array.create(str(path), schema)
    ids = np.array(sorted(vectors), dtype=np.uint64)
    matrix = np.array([vectors[int(i)] for i in ids], dtype=np.float32)
    with tiledb.open(str(path), mode="w") as array:
        array[ids] = {
            "vector": matrix.view([("", np.float32)] * dimensions).reshape(-1),
            "text": np.array([f"chunk {i}" for i in ids], dtype=object),
            "metadata": np.array([json.dumps({"file_name": f"f{i}.txt"}) for i in ids], dtype=object),
        }


class TestAddedChunks:

    def test_added_chunks_are_ranked_with_the_index_results(self):
        added_vectors = normalized_rows(np.array([[0.0, 2.0], [1.0, 1.0], [-1.0, 0.0]], dtype=np.float32))
        added_ids = np.array([10, 11, 12], dtype=np.uint64)
        distances, ids = merge_added(np.array([0.1, 0.5], dtype=np.float32), np.array([1, 2], dtype=np.uint64),
                                     np.array([0.0, 3.0], dtype=np.float32), added_vectors, added_ids, 2)
        assert ids.tolist() == [10, 1, 11, 2]
        assert distances.tolist() == pytest.approx([0.0, 0.1, 1 - np.sqrt(0.5), 0.5])

    def test_nothing_added(self):
        distances, ids = merge_added(np.array([0.1], dtype=np.float32), np.array([1], dtype=np.uint64),
                                     np.array([1.0, 0.0], dtype=np.float32), np.empty((0, 0), dtype=np.float32),
                                     np.empty(0, dtype=np.uint64), 4)
        assert ids.tolist() == [1] and distances.tolist() == pytest.approx([0.1])

    def test_search_returns_added_chunks_and_skips_removed_ones(self, tmp_path):
        import tiledb

        vectors = {i: [1.0, float(i), 0.0, 0.0] for i in range(1, 5)}
        vectors.update({10: [0.0, 1.0, 0.0, 0.0], 11: [0.0, 1.0, 0.0, 0.0], 12: [1.0, 0.0, 0.0, 0.0]})
        _chunk_store(tmp_path / "vectors", vectors)

        class FakeIndex:
            size = 4

            def query(self, vectors, k):
                return np.array([[0.2, 0.3, 0.4, 0.5][:k]]), np.array([[1, 2, 3, 4][:k]], dtype=np.uint64)

        query_db = QueryVectorDB.__new__(QueryVectorDB)
        query_db.index = FakeIndex()
        query_db.removed_ids = np.array([2, 11], dtype=np.uint64)
        query_db.added_ids = np.array([10, 11, 12], dtype=np.uint64)
        query_db.added_vectors = normalized_rows(np.array([vectors[10], vectors[11], vectors[12]], dtype=np.float32))
        query_db.distance_metric = "cosine"
        query_db.array_uri = str(tmp_path / "vectors")
        results, _ = query_db._retrieve(np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32), 3, 0.0, tiledb)
        assert [text for text, _ in results] == ["chunk 10", "chunk 1", "chunk 3"]
        assert results[0][1]["similarity_score"] == pytest.approx(1.0)

    def test_added_chunks_are_loaded_from_the_chunk_store(self, tmp_path):
        import tiledb

        _chunk_store(tmp_path / "vectors", {1: [3.0, 4.0, 0.0, 0.0], 2: [0.0, 2.0, 0.0, 0.0], 3: [1.0, 0.0, 0.0, 0.0]})
        documents = [Document(page_content="a", metadata={"doc_id": 1, "file_name": "a.txt", "hash": "ha", "file_path": "C:/a.txt"})]
        create_metadata_db(tmp_path, documents, [("1", "ha", 1), ("2", "ha", 1), ("3", "ha", 1)])
        with sqlite3.connect(tmp_path / "metadata.db") as connection:
            connection.executemany("INSERT INTO added_chunks (tiledb_id) VALUES (?)", [("2",), ("3",)])
        connection.close()
        query_db = QueryVectorDB.__new__(QueryVectorDB)
        query_db.db_path = tmp_path
        query_db.array_uri = str(tmp_path / "vectors")
        query_db.removed_ids = np.array([3], dtype=np.uint64)
        query_db._load_added_chunks(tiledb)
        assert query_db.added_ids.tolist() == [2]
        assert query_db.added_vectors.tolist() == [pytest.approx([0.0, 1.0, 0.0, 0.0])]


class TestPrepareKeepDocument:

    @pytest.fixture
    def pipeline(self, tmp_path, monkeypatch):
        from db import database_interactions

        extracted = [
            ("text of the new file " * 5, {"file_name": "new.txt", "file_path": "C:/docs/new.txt", "hash": "h-new"}),
            ("text of the known file " * 5, {"file_name": "known.txt", "file_path": "C:/docs/known.txt", "hash": "h-known"}),
        ]
        seen = {"split": None, "embedded": None}

        def extract(source_directory, extracted_pkl):
            with open(extracted_pkl, "wb") as f:
                pickle.dump([(text, dict(metadata)) for text, metadata in extracted], f)

        def split(extracted_pkl, chunks_pkl, chunk_size, chunk_overlap, checkpoint_dir):
            with open(extracted_pkl, "rb") as f:
                documents = pickle.load(f)
            seen["split"] = [metadata["file_name"] for _, metadata in documents]
            with open(chunks_pkl, "wb") as f:
                pickle.dump({"texts": [text for text, _ in documents], "chunks": documents, "failed_files": []}, f)

        class Embedder:
            def embed_documents(self, texts):
                seen["embedded"] = list(texts)
                return np.ones((len(texts), 4), dtype=np.float32), list(range(len(texts)))

        monkeypatch.setattr(database_interactions, "_run_extract_with_retry", extract)
        monkeypatch.setattr(database_interactions, "_run_split_with_retry", split)
        monkeypatch.setattr(database_interactions, "get_config", lambda: SimpleNamespace())
        monkeypatch.setattr(CreateVectorDB, "initialize_vector_model", lambda self, model_path, config_data: Embedder())
        monkeypatch.setitem(sys.modules, "modules.process_images", SimpleNamespace(choose_image_loader=lambda **kwargs: []))
        return CreateVectorDB.__new__(CreateVectorDB), tmp_path, seen

    def test_documents_left_out_are_never_split_or_embedded(self, pipeline):
        builder, folder, seen = pipeline
        prepared = builder.prepare(folder, "model", 300, 50, first_document_id=7,
                                   keep_document=lambda document: document.metadata["hash"] != "h-known")
        assert seen["split"] == ["new.txt"]
        assert len(seen["embedded"]) == 1
        assert [document.metadata["doc_id"] for document in prepared.documents] == [7]
        assert [metadata["doc_id"] for metadata in prepared.metadatas] == [7]
        assert prepared.vectors.shape == (1, 4)

    def test_nothing_left_skips_the_split_and_the_model(self, pipeline):
        builder, folder, seen = pipeline
        prepared = builder.prepare(folder, "model", 300, 50, keep_document=lambda document: False)
        assert seen == {"split": None, "embedded": None}
        assert (prepared.texts, prepared.metadatas, prepared.documents, prepared.vectors.shape) == ([], [], [], (0, 0))

    def test_without_the_hook_every_document_is_kept(self, pipeline):
        builder, folder, seen = pipeline
        prepared = builder.prepare(folder, "model", 300, 50)
        assert seen["split"] == ["new.txt", "known.txt"]
        assert [document.metadata["doc_id"] for document in prepared.documents] == [1, 2]
