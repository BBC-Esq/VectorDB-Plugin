import os
import platform
import sys
from unittest.mock import patch

import pytest

from core.utilities import (
    normalize_chat_text,
    normalize_text,
    format_citations,
    prepare_long_path,
    is_package_available,
    verify_installation,
    get_platform_info,
    get_python_version,
    get_embedding_batch_size,
    get_embedding_dtype_and_batch,
)


class TestNormalizeChatText:

    def test_section_symbol(self):
        assert 'section' in normalize_chat_text('See \u00a7 3.1')

    def test_doctor_abbreviation(self):
        result = normalize_chat_text('Dr. Smith said hello')
        assert 'Doctor' in result

    def test_mister_abbreviation(self):
        result = normalize_chat_text('Mr. Jones arrived')
        assert 'Mister' in result

    def test_ms_abbreviation(self):
        result = normalize_chat_text('Ms. Clark spoke')
        assert 'Miss' in result

    def test_mrs_abbreviation(self):
        result = normalize_chat_text('Mrs. Davis left')
        assert 'Mrs' in result

    def test_time_oclock(self):
        result = normalize_chat_text("It's 5:00 now")
        assert "5 o'clock" in result

    def test_time_oh_minutes(self):
        result = normalize_chat_text("At 3:05 PM")
        assert '3 oh 5' in result

    def test_time_normal_minutes(self):
        result = normalize_chat_text("At 3:15 PM")
        assert '3 15' in result

    def test_money_dollar_singular(self):
        result = normalize_chat_text("Costs $1 each")
        assert '1 dollar' in result

    def test_money_dollar_plural(self):
        result = normalize_chat_text("Costs $5 each")
        assert '5 dollars' in result

    def test_money_dollar_with_cents(self):
        result = normalize_chat_text("Price is $3.50")
        assert '3 dollars and 50 cents' in result

    def test_money_pound(self):
        result = normalize_chat_text("Costs \u00a35 each")
        assert '5 pounds' in result

    def test_year_hundreds(self):
        result = normalize_chat_text("In 1900 they")
        assert '19 hundred' in result

    def test_year_with_s(self):
        result = normalize_chat_text("The 1990s were")
        assert '19 90s' in result

    def test_decimal_point(self):
        result = normalize_chat_text("Value is 3.14 units")
        assert '3 point 1 4' in result

    def test_smart_quotes_replaced(self):
        text = '\u201cHello\u201d and \u2018world\u2019'
        result = normalize_chat_text(text)
        assert '\u201c' not in result
        assert '\u201d' not in result
        assert '\u2018' not in result
        assert '\u2019' not in result

    def test_whitespace_collapsed(self):
        result = normalize_chat_text("Hello\t  \t world")
        assert '\t' not in result
        assert '  ' not in result

    def test_strips_leading_nonalpha(self):
        result = normalize_chat_text("123. Hello world")
        assert result.startswith('Hello')

    def test_empty_string(self):
        result = normalize_chat_text("")
        assert result == ""


class TestFormatCitations:

    def test_returns_html_ol(self, sample_metadata_list):
        result = format_citations(sample_metadata_list)
        assert result.startswith('<ol>')
        assert result.endswith('</ol>')

    def test_single_file(self):
        metadata = [{
            'file_path': '/docs/test.txt',
            'file_type': '.txt',
            'similarity_score': 0.85,
        }]
        result = format_citations(metadata)
        assert '<li>' in result
        assert 'test.txt' in result

    def test_multiple_files_produce_multiple_items(self, sample_metadata_list):
        result = format_citations(sample_metadata_list)
        assert result.count('<li>') == 2

    def test_pdf_pages_shown(self, sample_metadata_list):
        result = format_citations(sample_metadata_list)
        assert 'p.' in result

    def test_non_pdf_no_pages(self):
        metadata = [{
            'file_path': '/docs/test.txt',
            'file_type': '.txt',
            'similarity_score': 0.85,
        }]
        result = format_citations(metadata)
        assert 'p.' not in result

    def test_score_range_shown(self, sample_metadata_list):
        result = format_citations(sample_metadata_list)
        assert '-' in result or '0.' in result

    def test_single_score_no_range(self):
        metadata = [{
            'file_path': '/docs/test.txt',
            'file_type': '.txt',
            'similarity_score': 0.85,
        }]
        result = format_citations(metadata)
        assert '0.8500' in result


class TestPrepareLongPath:

    def test_short_path_unchanged(self):
        result = prepare_long_path("C:\\Users\\test", "file.txt")
        assert result == os.path.normpath("C:\\Users\\test\\file.txt")

    def test_long_path_gets_prefix_on_windows(self):
        if os.name != 'nt':
            pytest.skip("Windows-only test")
        long_dir = "C:\\" + "a" * 250
        result = prepare_long_path(long_dir, "file.txt")
        assert result.startswith("\\\\?\\")

    def test_path_normalization(self):
        result = prepare_long_path("C:/Users/test", "file.txt")
        assert "/" not in result or os.name != 'nt'


class TestIsPackageAvailable:

    def test_known_package(self):
        available, ver = is_package_available("pip")
        assert available is True
        assert ver != "N/A"

    def test_unknown_package(self):
        available, ver = is_package_available("nonexistent_package_xyz_12345")
        assert available is False
        assert ver == "N/A"


class TestVerifyInstallation:

    def test_correct_version(self):
        import importlib.metadata
        pip_version = importlib.metadata.version("pip")
        assert verify_installation("pip", pip_version) is True

    def test_wrong_version(self):
        assert verify_installation("pip", "0.0.0") is False

    def test_missing_package(self):
        assert verify_installation("nonexistent_pkg_xyz", "1.0.0") is False


class TestPlatformInfo:

    def test_keys(self):
        info = get_platform_info()
        assert 'system' in info
        assert 'platform' in info
        assert 'architecture' in info

    def test_system_value(self):
        info = get_platform_info()
        assert info['system'] == platform.system()


class TestPythonVersion:

    def test_keys(self):
        info = get_python_version()
        assert 'major' in info
        assert 'minor' in info
        assert 'version_string' in info

    def test_values(self):
        info = get_python_version()
        assert info['major'] == sys.version_info.major
        assert info['minor'] == sys.version_info.minor


class TestNormalizeText:

    def test_basic(self):
        result = normalize_text("Hello World")
        assert result == "Hello World"

    def test_none_input(self):
        assert normalize_text(None) is None

    def test_list_input(self):
        result = normalize_text(["Hello", "World"])
        assert result == "Hello World"

    def test_unicode_normalization(self):
        result = normalize_text("\ufb01")
        assert result == "fi"

    def test_invisible_chars_removed(self):
        result = normalize_text("Hello\u200bWorld")
        assert result == "HelloWorld"

    def test_control_chars_removed(self):
        result = normalize_text("Hello\x01World")
        assert result == "HelloWorld"

    def test_empty_returns_none(self):
        assert normalize_text("") is None

    def test_whitespace_only_returns_none(self):
        assert normalize_text("   ") is None

    def test_preserve_whitespace(self):
        result = normalize_text("Line 1\nLine 2", preserve_whitespace=True)
        assert "\n" in result

    def test_collapse_whitespace(self):
        result = normalize_text("Hello   World")
        assert result == "Hello World"


class TestGetEmbeddingBatchSize:

    def test_cpu_always_2(self):
        assert get_embedding_batch_size("any-model", "cpu") == 2

    def test_known_model(self):
        result = get_embedding_batch_size("bge-small-en-v1.5", "cuda")
        assert result == 12

    def test_unknown_model_default(self):
        result = get_embedding_batch_size("unknown-model", "cuda")
        assert result == 8
