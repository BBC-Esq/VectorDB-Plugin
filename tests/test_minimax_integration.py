"""
Integration tests for the MiniMax provider in VectorDB-Plugin.

These tests verify the end-to-end flow against the real MiniMax API
(requires MINIMAX_API_KEY to be set).  They are skipped automatically when
the key is absent so the test suite stays green in CI.
"""
import os
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Lightweight stubs (same approach as unit tests)
# ---------------------------------------------------------------------------
pyside6 = types.ModuleType("PySide6")
pyside6_core = types.ModuleType("PySide6.QtCore")
pyside6_core.QThread = object
pyside6_core.Signal = lambda *a, **kw: None
pyside6.QtCore = pyside6_core
sys.modules.setdefault("PySide6", pyside6)
sys.modules.setdefault("PySide6.QtCore", pyside6_core)

chat_base = types.ModuleType("chat.base")
chat_base.load_chat_config = lambda: {}
chat_base.save_metadata = lambda x: None
chat_base.build_augmented_query = lambda ctxs, q: "\n\n---\n\n".join(ctxs) + f"\n\n-----\n\n{q}"
chat_base.write_chat_history = lambda x: None
chat_base.cleanup_gpu = lambda: None
sys.modules.setdefault("chat.base", chat_base)

db_mod = types.ModuleType("db.database_interactions")
from unittest.mock import MagicMock
db_mod.QueryVectorDB = MagicMock()
sys.modules.setdefault("db", types.ModuleType("db"))
sys.modules.setdefault("db.database_interactions", db_mod)

core_utils = types.ModuleType("core.utilities")
core_utils.format_citations = lambda x: ""
core_utils.my_cprint = lambda *a, **kw: None
sys.modules.setdefault("core", types.ModuleType("core"))
sys.modules.setdefault("core.utilities", core_utils)

core_const = types.ModuleType("core.constants")
core_const.system_message = "You are a helpful assistant."
core_const.PROJECT_ROOT = "/tmp"
sys.modules.setdefault("core.constants", core_const)

from chat.minimax import MiniMaxChat, MINIMAX_BASE_URL, MINIMAX_MODELS  # noqa: E402

API_KEY = os.getenv("MINIMAX_API_KEY")
SKIP_REASON = "MINIMAX_API_KEY not set – skipping live integration tests"


@unittest.skipUnless(API_KEY, SKIP_REASON)
class TestMiniMaxIntegrationLive(unittest.TestCase):
    """Live calls to the MiniMax API – skipped when API key is missing."""

    def _chat(self, model: str) -> MiniMaxChat:
        chat = MiniMaxChat(override_model=model)
        chat.config["minimax"]["api_key"] = API_KEY
        return chat

    def test_m2_7_returns_text(self):
        chat = self._chat("MiniMax-M2.7")
        chunks = list(chat.connect_to_minimax("Say 'hello' and nothing else."))
        full = "".join(chunks)
        self.assertTrue(len(full) > 0, "Expected non-empty response from MiniMax-M2.7")

    def test_m2_7_highspeed_returns_text(self):
        chat = self._chat("MiniMax-M2.7-highspeed")
        chunks = list(chat.connect_to_minimax("Say 'hi' and nothing else."))
        full = "".join(chunks)
        self.assertTrue(len(full) > 0, "Expected non-empty response from MiniMax-M2.7-highspeed")

    def test_streaming_yields_multiple_chunks(self):
        chat = self._chat("MiniMax-M2.7")
        chunks = list(chat.connect_to_minimax("Count from 1 to 5."))
        # Streaming should produce more than one chunk for a short answer
        self.assertGreater(len(chunks), 0)

    def test_invalid_api_key_raises(self):
        chat = MiniMaxChat(override_model="MiniMax-M2.7")
        chat.config["minimax"] = {"api_key": "invalid-key", "model": "MiniMax-M2.7"}
        with self.assertRaises(Exception):
            list(chat.connect_to_minimax("Hello"))


if __name__ == "__main__":
    unittest.main()
