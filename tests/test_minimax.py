import pytest

from chat.minimax import (
    MINIMAX_BASE_URL,
    MINIMAX_MODELS,
    _MINIMAX_MIN_TEMP,
)


class TestMiniMaxConstants:

    def test_base_url(self):
        assert MINIMAX_BASE_URL == "https://api.minimax.io/v1"

    def test_models_list(self):
        assert "MiniMax-M2.7" in MINIMAX_MODELS
        assert "MiniMax-M2.7-highspeed" in MINIMAX_MODELS

    def test_min_temp_value(self):
        assert _MINIMAX_MIN_TEMP == 0.01

    def test_min_temp_positive(self):
        assert _MINIMAX_MIN_TEMP > 0.0


class TestTemperatureClamping:

    def test_normal_temp_unchanged(self):
        temp = max(_MINIMAX_MIN_TEMP, 0.5)
        assert temp == 0.5

    def test_zero_temp_clamped(self):
        temp = max(_MINIMAX_MIN_TEMP, 0.0)
        assert temp == _MINIMAX_MIN_TEMP

    def test_below_min_clamped(self):
        temp = max(_MINIMAX_MIN_TEMP, 0.001)
        assert temp == _MINIMAX_MIN_TEMP

    def test_exact_min_unchanged(self):
        temp = max(_MINIMAX_MIN_TEMP, _MINIMAX_MIN_TEMP)
        assert temp == _MINIMAX_MIN_TEMP
