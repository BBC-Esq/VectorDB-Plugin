import sqlite3
import unittest

from db.sqlite_operations import (
    DocumentNotFoundError,
    delete_document_metadata,
)


class DocumentMetadataMutationTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.executescript(
            """
            CREATE TABLE document_metadata (
                id INTEGER PRIMARY KEY,
                file_name TEXT,
                hash TEXT,
                file_path TEXT,
                page_content TEXT
            );
            CREATE TABLE hash_chunk_ids (
                tiledb_id TEXT PRIMARY KEY,
                hash TEXT,
                document_id INTEGER
            );
            INSERT INTO document_metadata VALUES
                (1, 'alpha.txt', 'alpha-hash', '/alpha.txt', 'alpha'),
                (2, 'beta.txt', 'beta-hash', '/beta.txt', 'beta');
            INSERT INTO hash_chunk_ids VALUES
                ('20', 'alpha-hash', 1),
                ('10', 'alpha-hash', 1),
                ('30', 'beta-hash', 2);
            """
        )

    def tearDown(self):
        self.connection.close()

    def test_delete_document_metadata_removes_only_selected_document(self):
        self.connection.execute("BEGIN")
        ids = delete_document_metadata(self.connection, 1)
        self.connection.commit()

        self.assertEqual(sorted(ids), ["10", "20"])
        self.assertEqual(
            self.connection.execute(
                "SELECT file_name FROM document_metadata ORDER BY file_name"
            ).fetchall(),
            [("beta.txt",)],
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT tiledb_id FROM hash_chunk_ids ORDER BY tiledb_id"
            ).fetchall(),
            [("30",)],
        )

    def test_missing_document_does_not_change_database(self):
        with self.assertRaises(DocumentNotFoundError):
            delete_document_metadata(self.connection, 999)
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM document_metadata").fetchone()[0],
            2,
        )

    def test_document_without_chunk_mapping_is_rejected(self):
        self.connection.execute(
            "INSERT INTO document_metadata VALUES (?, ?, ?, ?, ?)",
            (3, "orphan.txt", "orphan-hash", "/orphan.txt", "orphan"),
        )
        with self.assertRaisesRegex(RuntimeError, "no vector mappings"):
            delete_document_metadata(self.connection, 3)
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM document_metadata WHERE hash = 'orphan-hash'"
            ).fetchone()[0],
            1,
        )

    def test_same_content_at_different_paths_is_preserved(self):
        self.connection.execute(
            "INSERT INTO document_metadata VALUES (?, ?, ?, ?, ?)",
            (3, "copy.txt", "alpha-hash", "/copy.txt", "alpha"),
        )
        self.connection.execute(
            "INSERT INTO hash_chunk_ids VALUES (?, ?, ?)", ("40", "alpha-hash", 3)
        )
        delete_document_metadata(self.connection, 1)
        self.assertEqual(self.connection.execute("SELECT id FROM document_metadata ORDER BY id").fetchall(), [(2,), (3,)])
        self.assertEqual(self.connection.execute("SELECT tiledb_id FROM hash_chunk_ids WHERE document_id = 3").fetchall(), [("40",)])

    def test_same_content_at_same_path_is_removed_as_one_group(self):
        self.connection.execute(
            "INSERT INTO document_metadata VALUES (?, ?, ?, ?, ?)",
            (3, "alpha.txt", "alpha-hash", "/alpha.txt", "alpha"),
        )
        self.connection.execute(
            "INSERT INTO hash_chunk_ids VALUES (?, ?, ?)", ("40", "alpha-hash", 3)
        )
        removed = delete_document_metadata(self.connection, 3)
        self.assertEqual(sorted(removed), ["10", "20", "40"])
        self.assertEqual(self.connection.execute("SELECT id FROM document_metadata").fetchall(), [(2,)])


if __name__ == "__main__":
    unittest.main()
