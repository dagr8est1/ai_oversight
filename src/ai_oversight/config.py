import os
from pathlib import Path

from pydantic import BaseModel

DEFAULT_TEXT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_VISION_MODEL = "ibm-granite/granite-vision-4.1-4b"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ai_oversight"


class Settings(BaseModel):
    cache_dir: Path = DEFAULT_CACHE_DIR
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    vision_enabled: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        kwargs: dict = {}
        if value := os.environ.get("AI_OVERSIGHT_CACHE_DIR"):
            kwargs["cache_dir"] = Path(value)
        if value := os.environ.get("AI_OVERSIGHT_TEXT_MODEL"):
            kwargs["text_model"] = value
        if value := os.environ.get("AI_OVERSIGHT_VISION_MODEL"):
            kwargs["vision_model"] = value
        if value := os.environ.get("AI_OVERSIGHT_VISION_ENABLED"):
            kwargs["vision_enabled"] = value.strip().lower() in ("1", "true", "yes")
        return cls(**kwargs)
