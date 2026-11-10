from pathlib import Path

from ai_oversight.schemas import ExtractionResult


def render(result: ExtractionResult) -> str:
    lines = [
        f"# Review: {result.source_file}",
        "",
        f"**Document type:** {result.document_type}",
        f"**Extracted at:** {result.extracted_at.isoformat()}",
        f"**Schema version:** {result.schema_version}",
        "",
        "## Fields",
        "",
        "| Field | Status | Value | Confidence |",
        "|---|---|---|---|",
    ]
    for name, field in result.fields.items():
        value = "" if field.value is None else str(field.value)
        lines.append(f"| {name} | {field.status} | {value} | {field.confidence} |")

    if result.metrics:
        lines += ["", "## Metrics"]
        for metric in result.metrics:
            lines.append(f"- {metric.name} ({metric.period}): {metric.value}")

    if result.incidents:
        lines += ["", "## Incidents"]
        for incident in result.incidents:
            lines.append(f"- {incident.get('date')}: {incident.get('event')} — {incident.get('impact')}")

    lines += ["", "## Review flags (rule-generated suggestions, not findings)"]
    if result.review_flags:
        for flag in result.review_flags:
            lines.append(f"- [{flag.code}] {flag.field}: {flag.message}")
    else:
        lines.append("- none")

    if result.vision_observations:
        lines += ["", "## Vision observations"]
        for obs in result.vision_observations:
            lines.append(
                f"- [{obs.task}] {obs.source_element_id} (page/slide {obs.page_or_slide}, "
                f"status={obs.status}): {obs.parsed if obs.parsed is not None else obs.note}"
            )

    lines += ["", "## Extraction warnings"]
    if result.extraction_warnings:
        for warning in result.extraction_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def write(result: ExtractionResult, path: Path) -> None:
    path.write_text(render(result))
