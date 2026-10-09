import re

import pytest


DB_NAME_PATTERN = re.compile(r'^[a-z0-9_-]*$')


class TestDatabaseNameValidation:

    def test_valid_lowercase(self):
        assert DB_NAME_PATTERN.match("mydb")

    def test_valid_with_numbers(self):
        assert DB_NAME_PATTERN.match("test123")

    def test_valid_with_hyphens(self):
        assert DB_NAME_PATTERN.match("my-db")

    def test_valid_with_underscores(self):
        assert DB_NAME_PATTERN.match("my_db")

    def test_valid_mixed(self):
        assert DB_NAME_PATTERN.match("my-db_123")

    def test_invalid_uppercase(self):
        assert not DB_NAME_PATTERN.match("MyDB")

    def test_invalid_spaces(self):
        assert not DB_NAME_PATTERN.match("my db")

    def test_invalid_special_chars(self):
        assert not DB_NAME_PATTERN.match("db!name")

    def test_invalid_slash(self):
        assert not DB_NAME_PATTERN.match("db/name")

    def test_empty_string_matches(self):
        assert DB_NAME_PATTERN.match("")


PORT_PATTERN = re.compile(r':(\d{1,5})(?=/)')


class TestPortExtraction:

    def test_standard_port(self):
        match = PORT_PATTERN.search("http://localhost:1234/v1")
        assert match is not None
        assert match.group(1) == "1234"

    def test_five_digit_port(self):
        match = PORT_PATTERN.search("http://localhost:65535/v1")
        assert match is not None
        assert match.group(1) == "65535"

    def test_no_port(self):
        match = PORT_PATTERN.search("http://localhost/v1")
        assert match is None

    def test_single_digit_port(self):
        match = PORT_PATTERN.search("http://localhost:8/v1")
        assert match is not None
        assert match.group(1) == "8"

    def test_port_without_path(self):
        match = PORT_PATTERN.search("http://localhost:1234")
        assert match is None


FILE_TYPE_MAP = {
    "All Files": "",
    "Images Only": "image",
    "Documents Only": "document",
    "Audio Only": "audio",
}


class TestFileTypeFilterMapping:

    def test_all_files(self):
        assert FILE_TYPE_MAP["All Files"] == ""

    def test_images(self):
        assert FILE_TYPE_MAP["Images Only"] == "image"

    def test_documents(self):
        assert FILE_TYPE_MAP["Documents Only"] == "document"

    def test_audio(self):
        assert FILE_TYPE_MAP["Audio Only"] == "audio"

    def test_all_keys_present(self):
        expected_keys = {"All Files", "Images Only", "Documents Only", "Audio Only"}
        assert set(FILE_TYPE_MAP.keys()) == expected_keys


class TestDbNamePreconditions:

    def _check_name(self, name):
        if not name or len(name) < 3 or name.lower() in ["null", "none"]:
            return False
        return True

    def test_valid_name(self):
        assert self._check_name("my-database") is True

    def test_too_short(self):
        assert self._check_name("ab") is False

    def test_empty(self):
        assert self._check_name("") is False

    def test_none_value(self):
        assert self._check_name(None) is False

    def test_null_string(self):
        assert self._check_name("null") is False

    def test_none_string(self):
        assert self._check_name("none") is False

    def test_null_case_insensitive(self):
        assert self._check_name("NULL") is False

    def test_exactly_three_chars(self):
        assert self._check_name("abc") is True
