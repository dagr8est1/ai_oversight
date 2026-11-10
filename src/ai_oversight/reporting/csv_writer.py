import csv
from pathlib import Path

from ai_oversight.schemas import ExtractionResult

FIELDNAMES = ["file_name", "field", "status", "value", "confidence"]


def render_rows(results: list[ExtractionResult]) -> list[dict]:
    rows = []
    for result in results:
        for field_name, field in result.fields.items():
            rows.append(
                {
                    "file_name": result.source_file,
                    "field": field_name,
                    "status": field.status,
                    "value": "" if field.value is None else str(field.value),
                    "confidence": field.confidence,
                }
            )
    return rows


def write(results: list[ExtractionResult], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(render_rows(results))
