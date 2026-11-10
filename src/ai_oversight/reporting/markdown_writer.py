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
        "| Field | Status | Value |",
        "|---|---|---|",
    ]
    for name, field in result.fields.items():
        value = "" if field.value is None else str(field.value)
        lines.append(f"| {name} | {field.status} | {value} |")

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

    lines += ["", "## Extraction warnings"]
    if result.extraction_warnings:
        for warning in result.extraction_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def write(result: ExtractionResult, path: Path) -> None:
    path.write_text(render(result))
