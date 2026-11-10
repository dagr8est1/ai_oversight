from pathlib import Path

from ai_oversight.config import Settings


def test_defaults_require_no_env_vars(monkeypatch):
    monkeypatch.delenv("AI_OVERSIGHT_VISION_ENABLED", raising=False)
    monkeypatch.delenv("AI_OVERSIGHT_CACHE_DIR", raising=False)
    monkeypatch.delenv("AI_OVERSIGHT_TEXT_MODEL", raising=False)
    monkeypatch.delenv("AI_OVERSIGHT_VISION_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.vision_enabled is False
    assert settings.text_model == "Qwen/Qwen3-4B-Instruct-2507"
    assert settings.vision_model == "ibm-granite/granite-vision-4.1-4b"


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("AI_OVERSIGHT_VISION_ENABLED", "true")
    monkeypatch.setenv("AI_OVERSIGHT_CACHE_DIR", "/tmp/ai-oversight-cache")
    monkeypatch.setenv("AI_OVERSIGHT_TEXT_MODEL", "some/other-model")

    settings = Settings.from_env()

    assert settings.vision_enabled is True
    assert settings.cache_dir == Path("/tmp/ai-oversight-cache")
    assert settings.text_model == "some/other-model"
