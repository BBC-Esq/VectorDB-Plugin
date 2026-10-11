import json
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from db import database_mutations
from db.sqlite_operations import delete_document_metadata as real_delete_metadata


class _FakeVectorArray:
    def __init__(self, vectors):
        self.vectors = vectors
        self.multi_index = self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __getitem__(self, external_ids):
        matrix = np.asarray(
            [self.vectors[int(vector_id)] for vector_id in external_ids],
            dtype=np.float32,
        )
        structured = np.ascontiguousarray(matrix).view(
            [("", np.float32)] * matrix.shape[1]
        ).reshape(-1)
        return {"id": np.asarray(external_ids, dtype=np.uint64), "vector": structured}


class _FakeFlatIndex:
    deleted = []
    restored = []
    consolidated = 0
    vacuumed = 0
    cleared = []

    def __init__(self, uri, open_for_remote_query_execution=False):
        self.uri = uri
        assert open_for_remote_query_execution

    def delete_batch(self, external_ids, timestamp=None):
        self.__class__.deleted.append((external_ids.tolist(), timestamp))

    def update_batch(self, vectors, external_ids, timestamp=None):
        assert vectors.ndim == 1 and vectors.dtype == object
        assert all(vector.ndim == 1 and vector.dtype == np.float32 for vector in vectors)
        restored = [np.asarray(vector).tolist() for vector in vectors]
        self.__class__.restored.append((external_ids.tolist(), restored, timestamp))

    def consolidate_updates(self):
        self.__class__.consolidated += 1
        return self

    def vacuum(self):
        self.__class__.vacuumed += 1

    @staticmethod
    def clear_history(uri, timestamp):
        _FakeFlatIndex.cleared.append((uri, timestamp))


class ExistingDatabaseMutationTests(unittest.TestCase):
    def setUp(self):
        _FakeFlatIndex.deleted = []
        _FakeFlatIndex.restored = []
        _FakeFlatIndex.consolidated = 0
        _FakeFlatIndex.vacuumed = 0
        _FakeFlatIndex.cleared = []
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name)
        (self.database / "vectors").mkdir()
        (self.database / "vector_index").mkdir()
        with open(self.database / "index_metadata.json", "w", encoding="utf-8") as handle:
            json.dump({"index_type": "FLAT", "num_vectors": 3}, handle)

        connection = sqlite3.connect(self.database / "metadata.db")
        connection.executescript(
            """
            CREATE TABLE document_metadata (
                id INTEGER PRIMARY KEY,
                file_name TEXT,
                hash TEXT,
                file_path TEXT,
                page_content TEXT
            );
            CREATE TABLE hash_chunk_ids (tiledb_id TEXT PRIMARY KEY, hash TEXT);
            INSERT INTO document_metadata VALUES
                (1, 'alpha.txt', 'alpha-hash', '/alpha.txt', 'alpha'),
                (2, 'beta.txt', 'beta-hash', '/beta.txt', 'beta');
            INSERT INTO hash_chunk_ids VALUES
                ('10', 'alpha-hash'), ('20', 'alpha-hash'), ('30', 'beta-hash');
            """
        )
        connection.commit()
        connection.close()

        vectors = {
            10: [1.0, 2.0],
            20: [3.0, 4.0],
            30: [5.0, 6.0],
        }
        fake_tiledb = types.ModuleType("tiledb")
        fake_tiledb.open = lambda *_args, **_kwargs: _FakeVectorArray(vectors)
        fake_vector_search = types.ModuleType("tiledb.vector_search")
        fake_vector_search.FlatIndex = _FakeFlatIndex
        fake_vector_search.Index = _FakeFlatIndex
        fake_tiledb.vector_search = fake_vector_search
        self.module_patch = patch.dict(
            sys.modules,
            {"tiledb": fake_tiledb, "tiledb.vector_search": fake_vector_search},
        )
        self.module_patch.start()

    def tearDown(self):
        self.module_patch.stop()
        self.temporary.cleanup()

    def test_removes_index_chunks_and_metadata_together(self):
        result = database_mutations.remove_document(self.database, 1)

        self.assertEqual(result["removed_documents"], 1)
        self.assertEqual(result["removed_chunks"], 2)
        self.assertEqual(result["remaining_vectors"], 1)
        self.assertEqual(_FakeFlatIndex.deleted[0][0], [10, 20])
        self.assertEqual(_FakeFlatIndex.restored, [])
        self.assertEqual(_FakeFlatIndex.consolidated, 1)
        self.assertEqual(_FakeFlatIndex.vacuumed, 1)
        self.assertEqual(len(_FakeFlatIndex.cleared), 1)

        connection = sqlite3.connect(self.database / "metadata.db")
        try:
            self.assertEqual(
                connection.execute(
                    "SELECT file_name FROM document_metadata ORDER BY file_name"
                ).fetchall(),
                [("beta.txt",)],
            )
        finally:
            connection.close()

    def test_restores_vectors_when_metadata_mutation_fails(self):
        def fail_after_delete(connection, document_id):
            real_delete_metadata(connection, document_id)
            raise RuntimeError("simulated metadata failure")

        with patch.object(
            database_mutations, "delete_document_metadata", side_effect=fail_after_delete
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated metadata failure"):
                database_mutations.remove_document(self.database, 1)

        self.assertEqual(_FakeFlatIndex.deleted[0][0], [10, 20])
        self.assertEqual(_FakeFlatIndex.restored[0][0], [10, 20])
        self.assertEqual(_FakeFlatIndex.restored[0][1], [[1.0, 2.0], [3.0, 4.0]])
        self.assertGreater(_FakeFlatIndex.restored[0][2], _FakeFlatIndex.deleted[0][1])
        self.assertEqual(_FakeFlatIndex.consolidated, 1)

        connection = sqlite3.connect(self.database / "metadata.db")
        try:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM document_metadata WHERE hash = 'alpha-hash'"
                ).fetchone()[0],
                1,
            )
        finally:
            connection.close()

    def test_identical_content_at_another_path_remains_searchable(self):
        connection = sqlite3.connect(self.database / "metadata.db")
        connection.execute("ALTER TABLE hash_chunk_ids ADD COLUMN document_id INTEGER")
        connection.execute("UPDATE hash_chunk_ids SET document_id = 1 WHERE hash = 'alpha-hash'")
        connection.execute("UPDATE hash_chunk_ids SET document_id = 2 WHERE hash = 'beta-hash'")
        connection.execute("INSERT INTO document_metadata VALUES (3, 'copy.txt', 'alpha-hash', '/copy.txt', 'alpha')")
        connection.execute("INSERT INTO hash_chunk_ids VALUES ('40', 'alpha-hash', 3)")
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
        connection.close()
        with open(self.database / "index_metadata.json", "w", encoding="utf-8") as handle:
            json.dump({"index_type": "FLAT", "num_vectors": 4}, handle)

        result = database_mutations.remove_document(self.database, 1)
        self.assertEqual(result["removed_chunks"], 2)
        self.assertEqual(_FakeFlatIndex.deleted[0][0], [10, 20])
        with sqlite3.connect(self.database / "metadata.db") as connection:
            self.assertEqual(connection.execute("SELECT file_name FROM document_metadata WHERE id = 3").fetchone(), ("copy.txt",))
            self.assertEqual(connection.execute("SELECT tiledb_id FROM hash_chunk_ids WHERE document_id = 3").fetchone(), ("40",))

    def test_failure_during_consolidation_restores_metadata_and_vectors(self):
        original = _FakeFlatIndex.consolidate_updates
        calls = 0

        def fail_once(index):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("simulated consolidation failure")
            return original(index)

        with patch.object(_FakeFlatIndex, "consolidate_updates", fail_once):
            with self.assertRaisesRegex(RuntimeError, "simulated consolidation failure"):
                database_mutations.remove_document(self.database, 1)

        self.assertEqual(_FakeFlatIndex.restored[0][0], [10, 20])
        with sqlite3.connect(self.database / "metadata.db") as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM document_metadata").fetchone()[0], 2)
        with open(self.database / "index_metadata.json", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["num_vectors"], 3)

    def test_cleanup_failure_reports_warning_after_successful_removal(self):
        with patch.object(_FakeFlatIndex, "clear_history", side_effect=RuntimeError("simulated cleanup failure")):
            result = database_mutations.remove_document(self.database, 1)

        self.assertIn("history cleanup failed", result["cleanup_warning"])
        self.assertEqual(_FakeFlatIndex.restored, [])
        self.assertEqual(_FakeFlatIndex.vacuumed, 1)
        with sqlite3.connect(self.database / "metadata.db") as connection:
            self.assertEqual(connection.execute("SELECT id FROM document_metadata").fetchall(), [(2,)])


if __name__ == "__main__":
    unittest.main()
