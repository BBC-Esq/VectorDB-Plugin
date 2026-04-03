"""
Unit tests for the MiniMax chat provider (chat/minimax.py).

These tests run without requiring Qt, a real MiniMax API key, or a real
VectorDB – everything external is mocked out.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without Qt / PySide6
# ---------------------------------------------------------------------------

# PySide6 stubs
pyside6 = types.ModuleType("PySide6")
pyside6_core = types.ModuleType("PySide6.QtCore")
pyside6_core.QThread = object
pyside6_core.Signal = lambda *a, **kw: None
pyside6.QtCore = pyside6_core
sys.modules.setdefault("PySide6", pyside6)
sys.modules.setdefault("PySide6.QtCore", pyside6_core)

# chat.base stubs
chat_base = types.ModuleType("chat.base")
chat_base.load_chat_config = lambda: {}
chat_base.save_metadata = lambda x: None
chat_base.build_augmented_query = lambda ctxs, q: f"context\n\n---\n\n{q}"
chat_base.write_chat_history = lambda x: None
chat_base.cleanup_gpu = lambda: None
sys.modules.setdefault("chat.base", chat_base)

# db.database_interactions stub
db_mod = types.ModuleType("db.database_interactions")
db_mod.QueryVectorDB = MagicMock()
sys.modules.setdefault("db", types.ModuleType("db"))
sys.modules.setdefault("db.database_interactions", db_mod)

# core.utilities stub
core_utils = types.ModuleType("core.utilities")
core_utils.format_citations = lambda x: "cite"
core_utils.my_cprint = lambda *a, **kw: None
sys.modules.setdefault("core", types.ModuleType("core"))
sys.modules.setdefault("core.utilities", core_utils)

# core.constants stub
core_const = types.ModuleType("core.constants")
core_const.system_message = "You are a helpful assistant."
core_const.PROJECT_ROOT = "/tmp"
sys.modules.setdefault("core.constants", core_const)

# Now import the module under test
from chat.minimax import (  # noqa: E402
    MiniMaxChat,
    MINIMAX_BASE_URL,
    MINIMAX_MODELS,
    _MINIMAX_MIN_TEMP,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestMiniMaxConstants(unittest.TestCase):
    def test_base_url(self):
        self.assertEqual(MINIMAX_BASE_URL, "https://api.minimax.io/v1")

    def test_models_list(self):
        self.assertIn("MiniMax-M2.7", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.7-highspeed", MINIMAX_MODELS)

    def test_min_temperature_positive(self):
        self.assertGreater(_MINIMAX_MIN_TEMP, 0.0)
        self.assertLessEqual(_MINIMAX_MIN_TEMP, 1.0)


class TestMiniMaxChatInit(unittest.TestCase):
    def test_default_init(self):
        chat = MiniMaxChat()
        self.assertIsNone(chat.query_vector_db)

    def test_override_model_stored(self):
        chat = MiniMaxChat(override_model="MiniMax-M2.7-highspeed")
        self.assertEqual(chat.config["minimax"]["model"], "MiniMax-M2.7-highspeed")

    def test_default_callbacks_callable(self):
        chat = MiniMaxChat()
        # Should not raise
        chat.response_callback("hello")
        chat.error_callback("err")
        chat.finished_callback()
        chat.citations_callback("cite")


class TestMiniMaxConnectToMinimax(unittest.TestCase):
    """connect_to_minimax is a generator that yields streamed text chunks."""

    def _make_chunk(self, content):
        chunk = MagicMock()
        chunk.choices[0].delta.content = content
        return chunk

    def _make_none_chunk(self):
        chunk = MagicMock()
        chunk.choices[0].delta.content = None
        return chunk

    @patch("chat.minimax.OpenAI")
    def test_yields_text_chunks(self, MockOpenAI):
        stream = [self._make_chunk("Hello"), self._make_chunk(" world")]
        MockOpenAI.return_value.chat.completions.create.return_value = iter(stream)

        chat = MiniMaxChat()
        chat.config = {"minimax": {"api_key": "test-key", "model": "MiniMax-M2.7"}}

        result = list(chat.connect_to_minimax("test query"))
        self.assertEqual(result, ["Hello", " world"])

    @patch("chat.minimax.OpenAI")
    def test_skips_none_content(self, MockOpenAI):
        stream = [self._make_chunk("A"), self._make_none_chunk(), self._make_chunk("B")]
        MockOpenAI.return_value.chat.completions.create.return_value = iter(stream)

        chat = MiniMaxChat()
        chat.config = {"minimax": {"api_key": "test-key", "model": "MiniMax-M2.7"}}

        result = list(chat.connect_to_minimax("query"))
        self.assertEqual(result, ["A", "B"])

    @patch("chat.minimax.OpenAI")
    def test_raises_if_no_api_key(self, MockOpenAI):
        chat = MiniMaxChat()
        chat.config = {"minimax": {"model": "MiniMax-M2.7"}}

        with self.assertRaises(ValueError) as ctx:
            list(chat.connect_to_minimax("query"))
        self.assertIn("MiniMax API key", str(ctx.exception))

    @patch("chat.minimax.OpenAI")
    def test_client_created_with_correct_base_url(self, MockOpenAI):
        stream = []
        MockOpenAI.return_value.chat.completions.create.return_value = iter(stream)

        chat = MiniMaxChat()
        chat.config = {"minimax": {"api_key": "sk-abc", "model": "MiniMax-M2.7"}}
        list(chat.connect_to_minimax("q"))

        MockOpenAI.assert_called_once_with(api_key="sk-abc", base_url=MINIMAX_BASE_URL)

    @patch("chat.minimax.OpenAI")
    def test_temperature_clamped(self, MockOpenAI):
        stream = []
        MockOpenAI.return_value.chat.completions.create.return_value = iter(stream)

        chat = MiniMaxChat()
        chat.config = {"minimax": {"api_key": "key", "model": "MiniMax-M2.7"}}
        list(chat.connect_to_minimax("q"))

        _, kwargs = MockOpenAI.return_value.chat.completions.create.call_args
        self.assertGreater(kwargs["temperature"], 0.0)
        self.assertLessEqual(kwargs["temperature"], 1.0)

    @patch("chat.minimax.OpenAI")
    def test_stream_enabled(self, MockOpenAI):
        MockOpenAI.return_value.chat.completions.create.return_value = iter([])

        chat = MiniMaxChat()
        chat.config = {"minimax": {"api_key": "key", "model": "MiniMax-M2.7"}}
        list(chat.connect_to_minimax("q"))

        _, kwargs = MockOpenAI.return_value.chat.completions.create.call_args
        self.assertTrue(kwargs["stream"])

    @patch("chat.minimax.OpenAI")
    def test_uses_override_model(self, MockOpenAI):
        MockOpenAI.return_value.chat.completions.create.return_value = iter([])

        chat = MiniMaxChat(override_model="MiniMax-M2.7-highspeed")
        chat.config["minimax"]["api_key"] = "key"
        list(chat.connect_to_minimax("q"))

        _, kwargs = MockOpenAI.return_value.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "MiniMax-M2.7-highspeed")


class TestMiniMaxAskMinimax(unittest.TestCase):
    """ask_minimax orchestrates the full RAG + chat flow."""

    def _make_stream_chunk(self, content):
        chunk = MagicMock()
        chunk.choices[0].delta.content = content
        return chunk

    @patch("chat.minimax.QueryVectorDB")
    @patch("chat.minimax.OpenAI")
    def test_error_callback_when_no_contexts(self, MockOpenAI, MockVDB):
        instance = MockVDB.get_instance.return_value
        instance.search.return_value = ([], [])
        instance.selected_database = "db"

        chat = MiniMaxChat()
        chat.config = {"minimax": {"api_key": "key", "model": "MiniMax-M2.7"}}
        errors = []
        finished = []
        chat.error_callback = errors.append
        chat.finished_callback = lambda: finished.append(True)

        chat.ask_minimax("q", "db")

        self.assertEqual(len(errors), 1)
        self.assertIn("No relevant contexts", errors[0])
        self.assertEqual(len(finished), 1)

    @patch("chat.minimax.save_metadata")
    @patch("chat.minimax.write_chat_history")
    @patch("chat.minimax.cleanup_gpu")
    @patch("chat.minimax.QueryVectorDB")
    @patch("chat.minimax.OpenAI")
    def test_response_assembled_correctly(
        self, MockOpenAI, MockVDB, mock_gpu, mock_write, mock_save
    ):
        instance = MockVDB.get_instance.return_value
        instance.search.return_value = (["ctx1"], [{"file": "doc.pdf"}])
        instance.selected_database = "db"

        stream = [self._make_stream_chunk("Hi"), self._make_stream_chunk(" there")]
        MockOpenAI.return_value.chat.completions.create.return_value = iter(stream)

        chat = MiniMaxChat()
        chat.config = {"minimax": {"api_key": "key", "model": "MiniMax-M2.7"}}
        responses = []
        citations = []
        finished = []
        chat.response_callback = responses.append
        chat.citations_callback = citations.append
        chat.finished_callback = lambda: finished.append(True)

        chat.ask_minimax("hello", "db")

        self.assertIn("Hi", responses)
        self.assertIn(" there", responses)
        self.assertEqual(len(finished), 1)
        mock_write.assert_called_once_with("Hi there")

    @patch("chat.minimax.save_metadata")
    @patch("chat.minimax.write_chat_history")
    @patch("chat.minimax.cleanup_gpu")
    @patch("chat.minimax.QueryVectorDB")
    @patch("chat.minimax.OpenAI")
    def test_reuses_existing_query_vector_db(
        self, MockOpenAI, MockVDB, mock_gpu, mock_write, mock_save
    ):
        instance = MockVDB.get_instance.return_value
        instance.search.return_value = (["ctx"], [])
        instance.selected_database = "mydb"
        MockOpenAI.return_value.chat.completions.create.return_value = iter([])

        chat = MiniMaxChat()
        chat.config = {"minimax": {"api_key": "k", "model": "MiniMax-M2.7"}}
        chat.query_vector_db = instance  # pre-assign same db

        chat.ask_minimax("q", "mydb")

        # get_instance should NOT be called again
        MockVDB.get_instance.assert_not_called()


class TestMiniMaxCredentialManager(unittest.TestCase):
    """Test that MiniMaxCredentialManager reads/writes config correctly."""

    def _make_manager(self, config_data):
        """Return a MiniMaxCredentialManager wired to a fake config."""
        # Defer GUI import until test runs; patch the filesystem bits
        sys.modules.setdefault("PySide6.QtWidgets", MagicMock())
        from gui.credentials import MiniMaxCredentialManager

        mgr = object.__new__(MiniMaxCredentialManager)
        mgr.parent_widget = None
        mgr.config_file_path = MagicMock()
        mgr.config = config_data
        return mgr

    def test_get_current_credential_present(self):
        mgr = self._make_manager({"minimax": {"api_key": "secret"}})
        self.assertEqual(mgr.get_current_credential(), "secret")

    def test_get_current_credential_missing(self):
        mgr = self._make_manager({})
        self.assertIsNone(mgr.get_current_credential())

    def test_update_credential_creates_section(self):
        mgr = self._make_manager({})
        mgr.update_credential("new-key")
        self.assertEqual(mgr.config["minimax"]["api_key"], "new-key")

    def test_update_credential_clears(self):
        mgr = self._make_manager({"minimax": {"api_key": "old"}})
        mgr.update_credential(None)
        self.assertIsNone(mgr.config["minimax"]["api_key"])

    def test_credential_name(self):
        mgr = self._make_manager({})
        self.assertEqual(mgr.credential_name, "MiniMax API key")


class TestQueryTabStrategyIntegration(unittest.TestCase):
    """Verify MiniMax model names map to MiniMaxStrategy in the query tab."""

    def test_strategy_map_contains_minimax_models(self):
        """
        _strategy_for_source must return a MiniMaxStrategy for both
        MiniMax model strings without raising.
        """
        # Avoid importing Qt-dependent modules; check the mapping table directly.
        import importlib, ast, textwrap

        src = open(
            "gui/tabs_databases/query.py", encoding="utf-8"
        ).read()

        self.assertIn('"MiniMax-M2.7": MiniMaxStrategy(self)', src)
        self.assertIn('"MiniMax-M2.7-highspeed": MiniMaxStrategy(self)', src)

    def test_combo_box_items_contain_minimax(self):
        src = open(
            "gui/tabs_databases/query.py", encoding="utf-8"
        ).read()

        self.assertIn("MiniMax-M2.7", src)
        self.assertIn("MiniMax-M2.7-highspeed", src)


if __name__ == "__main__":
    unittest.main()
