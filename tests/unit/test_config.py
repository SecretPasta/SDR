"""Tests for AppSettings — env-var loading, SecretStr safety, nested prefixes, lru_cache."""
from __future__ import annotations

import pytest

from app.config import AlignmentSettings, AppSettings, ChunkingSettings, get_settings


def _set_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the minimum env vars needed to construct AppSettings."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("PINECONE_API_KEY", "pinecone-test-key")


class TestSettingsLoadFromEnv:
    def test_anthropic_model_overridden_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-8")
        s = AppSettings()
        assert s.anthropic.model == "claude-opus-4-8"

    def test_gemini_embed_dimensions_overridden_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required(monkeypatch)
        monkeypatch.setenv("GEMINI_EMBED_DIMENSIONS", "512")
        s = AppSettings()
        assert s.gemini.embed_dimensions == 512

    def test_alignment_threshold_overridden_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALIGN_THRESHOLD", "0.75")
        s = AlignmentSettings()
        assert s.threshold == 0.75

    def test_chunking_max_tokens_overridden_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "1200")
        s = ChunkingSettings()
        assert s.max_tokens == 1200


class TestSecretStrSafety:
    def test_anthropic_key_not_leaked_via_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required(monkeypatch)
        s = AppSettings()
        assert "sk-ant-test" not in repr(s.anthropic)
        assert "sk-ant-test" not in str(s.anthropic)

    def test_gemini_key_not_leaked_via_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required(monkeypatch)
        s = AppSettings()
        assert "gemini-test-key" not in repr(s.gemini)

    def test_secret_value_accessible_via_get_secret_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required(monkeypatch)
        s = AppSettings()
        assert s.anthropic.api_key.get_secret_value() == "sk-ant-test"


class TestLruCache:
    def test_get_settings_returns_same_instance_on_repeated_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_required(monkeypatch)
        get_settings.cache_clear()
        try:
            s1 = get_settings()
            s2 = get_settings()
            assert s1 is s2
        finally:
            get_settings.cache_clear()


class TestAlignmentSettingsDefaults:
    def test_default_weights_sum_to_one(self) -> None:
        s = AlignmentSettings()
        total = s.w_heading_num + s.w_heading_embed + s.w_levenshtein
        assert abs(total - 1.0) < 1e-9

    def test_default_threshold_is_0_55(self) -> None:
        assert AlignmentSettings().threshold == 0.55
