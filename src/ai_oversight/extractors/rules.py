import re
from datetime import UTC, datetime
from typing import Any

from ai_oversight.parsers.document_ir import Element, ParsedDocument
from ai_oversight.processing.normalize import (
    normalize_date,
    normalize_metric_list,
    normalize_risk_rating,
)
from ai_oversight.schemas import (
    CANONICAL_FIELDS,
    FIELD_ID_TO_NAME,
    Evidence,
    ExtractionResult,
    FieldValue,
    MetricObservation,
    ReviewFlag,
)

# Fields whose raw text gets a structured normalization pass. Everything else
# is passed through verbatim -- PLAN.md's "never fabricate" rule means we only
# transform a value when we have a reliable, reversible parse for it.
_NORMALIZERS = {
    "risk_rating": normalize_risk_rating,
    "model_performance": normalize_metric_list,
    "first_deployment_date": normalize_date,
    "next_review_date": normalize_date,
}

# Fuzzy-matched against a chart series name to decide which metric it
# corresponds to in a narrative field's parsed metric dict. PLAN.md's own
# example ("narrative says 94% while chart says 92%", a 2-point gap) must
# trigger, so the tolerance has to sit below 0.02.
_CONFLICT_TOLERANCE = 0.015

# A 3-cell table row is treated as an incident record only when its first
# cell looks like a short date (e.g. "Sep 07", "Oct 02") -- distinguishes a
# real incident log row from an unrelated 3-column row elsewhere in the deck.
_INCIDENT_DATE_PATTERN = re.compile(r"^[A-Za-z]{3,9}\.?\s+\d{1,2}$")


def extract(doc: ParsedDocument) -> ExtractionResult:
    fields = {name: FieldValue(status="missing") for name in CANONICAL_FIELDS}

    for element in doc.elements:
        if element.type != "key_value":
            continue
        content = element.content or {}
        field_id = content.get("field_id")
        name = FIELD_ID_TO_NAME.get(field_id) if field_id else None
        if name is None:
            continue
        fields[name] = _build_field_value(name, content["value"], element)

    metrics = _collect_metrics(doc.elements)
    review_flags = _detect_conflicts(fields, metrics)
    incidents = _collect_incidents(doc.elements)

    return ExtractionResult(
        document_id=doc.document_id,
        source_file=doc.filename,
        document_type=_infer_document_type(doc.filename),
        extracted_at=datetime.now(UTC),
        fields=fields,
        metrics=metrics,
        incidents=incidents,
        review_flags=review_flags,
        extraction_warnings=list(doc.warnings),
    )


def _build_field_value(name: str, raw_value: str, element: Element) -> FieldValue:
    normalizer = _NORMALIZERS.get(name)
    value: Any = raw_value
    notes: str | None = None
    if normalizer is not None:
        value, notes = normalizer(raw_value)

    return FieldValue(
        value=value,
        status="extracted",
        evidence=[
            Evidence(
                page_or_slide=element.page_or_slide,
                element_id=element.element_id,
                quote=raw_value,
                method=element.method,
            )
        ],
        normalization_notes=notes,
    )


def _infer_document_type(filename: str) -> str:
    lowered = filename.lower()
    if "internal" in lowered:
        return "internal"
    if "external" in lowered:
        return "external"
    return "unknown"


def _collect_metrics(elements: list[Element]) -> list[MetricObservation]:
    metrics: list[MetricObservation] = []
    for element in elements:
        if element.type != "chart":
            continue
        content = element.content or {}
        categories = content.get("categories", [])
        for series in content.get("series", []):
            name = _normalize_metric_key(series.get("name", ""))
            for period, value in zip(categories, series.get("values", []), strict=False):
                metrics.append(
                    MetricObservation(
                        field="model_performance",
                        name=name,
                        period=str(period),
                        value=float(value),
                        source_element_id=element.element_id,
                        page_or_slide=element.page_or_slide,
                    )
                )
    return metrics


def _normalize_metric_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return key.strip("_")


def _detect_conflicts(
    fields: dict[str, FieldValue], metrics: list[MetricObservation]
) -> list[ReviewFlag]:
    performance = fields.get("model_performance")
    if performance is None or not isinstance(performance.value, dict) or not metrics:
        return []

    latest_by_name: dict[str, float] = {}
    for metric in metrics:
        latest_by_name[metric.name] = metric.value  # categories are chronological; last write wins

    flags: list[ReviewFlag] = []
    for narrative_key, narrative_value in performance.value.items():
        matched_name, chart_value = _match_metric(narrative_key, latest_by_name)
        if matched_name is None:
            continue
        if abs(narrative_value - chart_value) > _CONFLICT_TOLERANCE:
            flags.append(
                ReviewFlag(
                    code="performance_conflict",
                    field="model_performance",
                    message=(
                        f"narrative value for {narrative_key!r} is {narrative_value}, "
                        f"but the chart's latest value for {matched_name!r} is {chart_value}"
                    ),
                )
            )
            performance.status = "conflicting"
    return flags


def _match_metric(narrative_key: str, latest_by_name: dict[str, float]) -> tuple[str | None, float]:
    for chart_name, value in latest_by_name.items():
        if narrative_key in chart_name or chart_name in narrative_key:
            return chart_name, value
    return None, 0.0


def _collect_incidents(elements: list[Element]) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    for element in elements:
        if element.type != "table":
            continue
        cells = (element.content or {}).get("cells", [])
        if len(cells) != 3 or not _INCIDENT_DATE_PATTERN.match(cells[0]):
            continue
        incidents.append(
            {
                "date": cells[0],
                "event": cells[1],
                "impact": cells[2],
                "source_element_id": element.element_id,
                "page_or_slide": element.page_or_slide,
            }
        )
    return incidents
