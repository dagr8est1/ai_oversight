from pathlib import Path

from ai_oversight.schemas import ExtractionResult


def write(result: ExtractionResult, path: Path) -> None:
    path.write_text(result.model_dump_json(indent=2))
