import os

import pytest
import yaml

from core.utilities import format_citations
from db.document_processor import Document
from db.sqlite_operations import create_metadata_db
from gui.tabs_databases import manage_data, query_data, query_search


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(manage_data, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(query_data, "PROJECT_ROOT", tmp_path)
    (tmp_path / "Vector_DB").mkdir()
    (tmp_path / "Vector_DB_Backup").mkdir()
    return tmp_path


def write_config(root, data):
    (root / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def read_config(root):
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


def make_db(root, name):
    folder = root / "Vector_DB" / name
    create_metadata_db(folder, [Document(page_content="t", metadata={"file_name": "a.txt", "file_path": "C:/a.txt", "hash": "h"})], [("1", "h")])
    return folder


def test_parse_citations_reads_the_real_formatter():
    metadata = [
        {"file_path": r"C:\docs\contract.pdf", "file_type": ".pdf", "similarity_score": 0.61, "page_number": 3},
        {"file_path": r"C:\docs\contract.pdf", "file_type": ".pdf", "similarity_score": 0.74, "page_number": 4},
        {"file_path": r"C:\docs\notes & memos.txt", "file_type": ".txt", "similarity_score": 0.88},
        {"file_path": r"C:\docs\scan.pdf", "file_type": ".pdf", "similarity_score": 0.52},
    ]
    citations = query_data.parse_citations(format_citations(metadata))
    assert [c["name"] for c in citations] == ["notes & memos.txt", "contract.pdf", "scan.pdf"]
    assert citations[0] == {"name": "notes & memos.txt", "path": r"C:\docs\notes & memos.txt", "low": 0.88, "high": 0.88, "pages": ""}
    assert citations[1]["low"] == 0.61 and citations[1]["high"] == 0.74 and citations[1]["pages"] == "3-4"
    assert citations[2]["pages"] == ""


def test_parse_citations_tolerates_nothing():
    assert query_data.parse_citations("") == []
    assert query_data.parse_citations(None) == []
    assert query_data.parse_citations("<ol></ol>") == []
    assert query_data.parse_citations("No citations found.") == []


def test_parse_token_counts_matches_the_local_model_format():
    html = ("<span style='color:#2ECC40;'>available tokens (8192)</span>"
            "<span style='color:#FF4136;'> - rag instruction (120) - query (15) - contexts (3200) - response (450)</span>"
            "<span style='color:white;'> = 4407 remaining tokens.</span>")
    assert query_data.parse_token_counts(html) == {"available": 8192, "instruction": 120, "question": 15, "contexts": 3200,
                                                   "response": 450, "remaining": 4407}
    over = html.replace("= 4407", "= -12")
    assert query_data.parse_token_counts(over)["remaining"] == -12
    assert query_data.parse_token_counts("garbage") is None


def test_turn_text_for_answers_and_chunks():
    answer = {"kind": "answer", "answer": "Thirty days.\n", "citations": [
        {"name": "contract.pdf", "path": "C:/contract.pdf", "low": 0.6, "high": 0.7, "pages": "3-4"},
        {"name": "notes.txt", "path": "C:/notes.txt", "low": 0.5, "high": 0.5, "pages": ""}]}
    assert query_data.turn_text(answer) == "Thirty days.\n\nCitations:\n1. contract.pdf [0.6000-0.7000] p.3-4\n2. notes.txt [0.5000]"
    chunks = {"kind": "chunks", "chunks": [{"name": "a.pdf", "page": 2, "text": "First."}, {"name": "b.txt", "page": None, "text": "Second."}]}
    assert query_data.turn_text(chunks) == "CONTEXT 1 | a.pdf (page 2)\nFirst.\n\nCONTEXT 2 | b.txt\nSecond."
    assert query_data.turn_text({"kind": "answer", "answer": "", "citations": []}) == ""


def test_query_settings_summary():
    cfg = {"database": {"contexts": 6, "similarity": 0.7, "search_term": " lease ", "document_types": "image"},
           "Compute_Device": {"database_query": "cuda"}}
    assert query_data.query_settings(cfg) == {"contexts": 6, "similarity": 0.7, "search_term": " lease ", "document_types": "Images", "device": "cuda"}
    assert query_data.query_settings({})["document_types"] == "All files"
    assert query_data.query_settings({})["device"] == "cpu"


def test_readiness_messages():
    models = [{"name": "Small", "needs_token": False}, {"name": "Gated", "needs_token": True}]
    assert query_data.readiness({}, "ChatGPT", None, models)["action"] == "settings"
    assert query_data.readiness({"openai": {"api_key": "k"}}, "ChatGPT", None, models) is None
    minimax = query_data.readiness({}, "MiniMax-M2.7", None, models)
    assert minimax["action"] == "minimax_key" and minimax["label"] == "Enter the MiniMax API key"
    assert query_data.readiness({"minimax": {"api_key": "k"}}, "MiniMax-M3", None, models) is None
    assert query_data.readiness({"server": {"connection_str": ""}}, "LM Studio", None, models)["action"] == "settings"
    assert query_data.readiness({"server": {"connection_str": "http://localhost:1234/v1"}}, "LM Studio", None, models) is None
    assert query_data.readiness({}, "Kobold", None, models) is None
    assert query_data.readiness({}, "Local Model", "Small", models) is None
    gated = query_data.readiness({}, "Local Model", "Gated", models)
    assert gated["action"] == "hf_token" and "access token" in gated["message"]
    assert query_data.readiness({}, "Local Model", None, models) == {"message": "Choose a local model.", "action": None, "label": ""}
    assert query_data.readiness({}, "Local Model", None, [])["message"] == "No local chat model runs on this computer."


def test_queryable_databases_and_memory(root):
    make_db(root, "ready_db")
    make_db(root, "user_manual")
    (root / "Vector_DB" / "leftover").mkdir()
    write_config(root, {"created_databases": {"ready_db": {"model": str(root / "missing-model"), "chunk_size": 1, "chunk_overlap": 0},
                                              "missing_db": {"model": "x", "chunk_size": 1, "chunk_overlap": 0},
                                              "user_manual": {"model": "x", "chunk_size": 1, "chunk_overlap": 0}},
                        "database": {"database_to_search": "gone"}})
    cfg = query_data.load_config()
    rows = query_data.queryable_databases(cfg)
    assert [r["name"] for r in rows] == ["ready_db"] and rows[0]["model_missing"] is True
    assert query_data.remembered_database(cfg, ["ready_db"]) == "ready_db"
    assert query_data.remember_database("ready_db") is True
    assert read_config(root)["database"]["database_to_search"] == "ready_db"
    before = (root / "config.yaml").stat().st_mtime_ns
    assert query_data.remember_database("ready_db") is False
    assert (root / "config.yaml").stat().st_mtime_ns == before
    assert query_data.remembered_database(query_data.load_config(), ["other", "ready_db"]) == "ready_db"


def test_remember_database_never_writes_a_broken_config(root):
    (root / "config.yaml").write_text("database: [broken\n", encoding="utf-8")
    assert query_data.remember_database("anything") is False
    assert (root / "config.yaml").read_text(encoding="utf-8") == "database: [broken\n"
    assert query_data.load_config() == {}


def test_local_models_flags(root, monkeypatch):
    monkeypatch.setattr(query_data, "CHAT_MODELS", {
        "Open": {"cache_dir": "org--open", "vram": 2048, "gated": False},
        "Gated": {"cache_dir": "org--gated", "vram": 4096, "gated": True},
    })
    monkeypatch.setattr(query_data, "runs_on_this_hardware", lambda info: True)
    monkeypatch.setattr(query_data, "cuda_usable", lambda: True)
    (root / "Models" / "chat" / "org--open").mkdir(parents=True)
    (root / "Models" / "chat" / "org--open" / "config.json").write_text("{}", encoding="utf-8")
    rows = query_data.local_models({})
    assert rows == [{"name": "Open", "memory": 2.0, "downloaded": True, "needs_token": False},
                    {"name": "Gated", "memory": 4.0, "downloaded": False, "needs_token": True}]
    assert query_data.local_models({"hf_access_token": "hf_x"})[1]["needs_token"] is False
    monkeypatch.setattr(query_data, "cuda_usable", lambda: False)
    assert query_data.local_models({})[0]["memory"] is None


def test_tts_label():
    assert query_data.tts_label({"tts": {"model": "kokoro"}}) == "Kokoro"
    assert query_data.tts_label({"tts": {"model": "googletts"}}) == "Google TTS"
    assert query_data.tts_label({"tts": {"model": "mystery"}}) == "mystery"
    assert query_data.tts_label({}) == ""


def test_chunk_rows_clean_and_label():
    rows = query_search.chunk_rows(
        ["First line\n  \n\n\nSecond", "  plain  "],
        [{"file_path": r"C:\docs\a.pdf", "file_name": "a.pdf", "similarity_score": 0.8, "page_number": 2},
         {"file_path": r"C:\docs\b.txt", "similarity_score": 0.6}])
    assert rows == [{"text": "First line\n\nSecond", "name": "a.pdf", "path": r"C:\docs\a.pdf", "score": 0.8, "page": 2},
                    {"text": "plain", "name": "b.txt", "path": r"C:\docs\b.txt", "score": 0.6, "page": None}]
