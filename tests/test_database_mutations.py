import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from db import database_mutations
from db.sqlite_operations import (
    DocumentNotFoundError,
    METADATA_SCHEMA_VERSION,
    delete_document_metadata,
    pending_changes,
    removed_chunk_ids,
    upgrade_metadata_db,
)


class ExistingDatabaseMutationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name)
        with sqlite3.connect(self.database / "metadata.db") as connection:
            connection.executescript("""
                CREATE TABLE document_metadata (
                    id INTEGER PRIMARY KEY, file_name TEXT, hash TEXT,
                    file_path TEXT, page_content TEXT
                );
                CREATE TABLE hash_chunk_ids (
                    tiledb_id TEXT PRIMARY KEY, hash TEXT, document_id INTEGER
                );
                INSERT INTO document_metadata VALUES
                    (1, 'alpha.txt', 'shared', '/alpha.txt', 'alpha'),
                    (2, 'beta.txt', 'beta', '/beta.txt', 'beta'),
                    (3, 'copy.txt', 'shared', '/copy.txt', 'alpha');
                INSERT INTO hash_chunk_ids VALUES
                    ('10', 'shared', 1), ('20', 'shared', 1),
                    ('30', 'beta', 2), ('40', 'shared', 3);
                PRAGMA user_version = 2;
            """)
        upgrade_metadata_db(self.database)

    def tearDown(self):
        self.temporary.cleanup()

    def state(self):
        with sqlite3.connect(self.database / "metadata.db") as connection:
            return {
                "documents": connection.execute(
                    "SELECT id, file_name FROM document_metadata ORDER BY id"
                ).fetchall(),
                "mappings": connection.execute(
                    "SELECT tiledb_id, document_id FROM hash_chunk_ids ORDER BY tiledb_id"
                ).fetchall(),
                "removed": connection.execute(
                    "SELECT tiledb_id FROM removed_chunks ORDER BY tiledb_id"
                ).fetchall(),
                "changes": connection.execute(
                    "SELECT change, file_name, chunks FROM index_changes ORDER BY id"
                ).fetchall(),
                "version": connection.execute(
                    "SELECT version FROM database_info WHERE id = 1"
                ).fetchone()[0],
            }

    def test_records_removal_without_touching_other_content_copies(self):
        result = database_mutations.remove_document(self.database, 1)

        self.assertEqual(result, {
            "document_id": 1, "removed_documents": 1,
            "removed_chunks": 2, "remaining_vectors": 2,
        })
        state = self.state()
        self.assertEqual(state["documents"], [(2, "beta.txt"), (3, "copy.txt")])
        self.assertEqual(state["mappings"], [("30", 2), ("40", 3)])
        self.assertEqual(state["removed"], [("10",), ("20",)])
        self.assertEqual(state["changes"], [("removed", "alpha.txt", 2)])
        self.assertEqual(state["version"], 2)
        self.assertEqual(removed_chunk_ids(self.database), [10, 20])
        with sqlite3.connect(self.database / "metadata.db") as connection:
            self.assertEqual(pending_changes(connection), {"documents": 1, "chunks": 2})
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], METADATA_SCHEMA_VERSION)

    def test_same_path_twins_are_removed_and_logged_together(self):
        with sqlite3.connect(self.database / "metadata.db") as connection:
            connection.execute("INSERT INTO document_metadata VALUES (4, 'alpha.txt', 'shared', '/alpha.txt', 'alpha')")
            connection.execute("INSERT INTO hash_chunk_ids VALUES ('50', 'shared', 4)")

        result = database_mutations.remove_document(self.database, 1)

        self.assertEqual(result["removed_documents"], 2)
        self.assertEqual(result["removed_chunks"], 3)
        self.assertEqual(self.state()["removed"], [("10",), ("20",), ("50",)])
        self.assertEqual(self.state()["changes"], [
            ("removed", "alpha.txt", 2), ("removed", "alpha.txt", 1),
        ])
        self.assertEqual(self.state()["documents"], [(2, "beta.txt"), (3, "copy.txt")])

    def test_failure_after_delete_rolls_back_rows_log_and_version(self):
        before = self.state()

        def fail_after_delete(connection, document_id):
            delete_document_metadata(connection, document_id)
            raise RuntimeError("simulated failure")

        with patch.object(database_mutations, "delete_document_metadata", side_effect=fail_after_delete):
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                database_mutations.remove_document(self.database, 1)

        self.assertEqual(self.state(), before)

    def test_failure_after_version_update_rolls_back_everything(self):
        before = self.state()

        def fail_after_bump(connection):
            connection.execute("UPDATE database_info SET version = version + 1 WHERE id = 1")
            raise RuntimeError("simulated version failure")

        with patch.object(database_mutations, "bump_version", side_effect=fail_after_bump):
            with self.assertRaisesRegex(RuntimeError, "simulated version failure"):
                database_mutations.remove_document(self.database, 1)

        self.assertEqual(self.state(), before)

    def test_missing_document_does_not_create_a_change(self):
        before = self.state()
        with self.assertRaises(DocumentNotFoundError):
            database_mutations.remove_document(self.database, 999)
        self.assertEqual(self.state(), before)

    def test_missing_metadata_database_does_not_create_one(self):
        missing = self.database / "missing"
        missing.mkdir()
        with self.assertRaises(FileNotFoundError):
            database_mutations.remove_document(missing, 1)
        self.assertFalse((missing / "metadata.db").exists())


if __name__ == "__main__":
    unittest.main()
