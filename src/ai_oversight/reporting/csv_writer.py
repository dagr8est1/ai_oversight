import csv
from pathlib import Path

from ai_oversight.schemas import ExtractionResult

FIELDNAMES = [
    "file_name",
    "field",
    "status",
    "value",
    "confidence",
    "vision_rating",
    "vision_success_rate",
    "vision_data_loss",
]


def render_rows(results: list[ExtractionResult]) -> list[dict]:
    rows = []
    for result in results:
        vision_integrity = result.vision_integrity
        vision_data_loss = "; ".join(vision_integrity.loss_notes)
        for field_name, field in result.fields.items():
            rows.append(
                {
                    "file_name": result.source_file,
                    "field": field_name,
                    "status": field.status,
                    "value": "" if field.value is None else str(field.value),
                    "confidence": field.confidence,
                    "vision_rating": vision_integrity.rating,
                    "vision_success_rate": vision_integrity.success_rate,
                    "vision_data_loss": vision_data_loss,
                }
            )
    return rows


def write(results: list[ExtractionResult], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(render_rows(results))
