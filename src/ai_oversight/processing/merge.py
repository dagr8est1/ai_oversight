import re
from typing import Any

from ai_oversight.parsers.document_ir import Element
from ai_oversight.processing.normalize import (
    normalize_date,
    normalize_metric_dict,
    normalize_metric_list,
    normalize_risk_rating,
)
from ai_oversight.schemas import Evidence, FieldValue, MetricObservation, ReviewFlag

# Shared by both extractor backends so neither one normalizes a raw string
# into a structured value differently from the other.
_NORMALIZERS = {
    "risk_rating": normalize_risk_rating,
    "model_performance": normalize_metric_list,
    "first_deployment_date": normalize_date,
    "next_review_date": normalize_date,
}


def normalize_value(name: str, raw_value: Any) -> tuple[Any, str, str | None]:
    """Normalize a field's raw value if it has a registered normalizer.

    Returns (value, status, notes). status is "extracted" when the value
    parsed cleanly, "needs_review" when the normalizer couldn't reliably
    parse it (the raw text is still returned, never discarded).
    """
    if name == "model_performance" and isinstance(raw_value, dict):
        # the model can reply with a dict directly (e.g. {"accuracy": "92%"})
        # instead of a flat string -- found via a real run where this sailed
        # through unconverted and later crashed arithmetic that assumes
        # every dict value is already numeric
        value, notes, ok = normalize_metric_dict(raw_value)
        return value, ("extracted" if ok else "needs_review"), notes
    if not isinstance(raw_value, str):
        return raw_value, "extracted", None
    normalizer = _NORMALIZERS.get(name)
    if normalizer is None:
        return raw_value, "extracted", None
    value, notes, ok = normalizer(raw_value)
    return value, ("extracted" if ok else "needs_review"), notes

# A 3-cell table row is treated as an incident record only when its first
# cell looks like a short date (e.g. "Sep 07", "Oct 02") -- distinguishes a
# real incident log row from an unrelated 3-column row elsewhere in a deck.
_INCIDENT_DATE_PATTERN = re.compile(r"^[A-Za-z]{3,9}\.?\s+\d{1,2}$")

# A narrative metric vs. the chart's latest value must disagree by more than
# this to be flagged. PLAN.md's own example ("94% vs 92%") must trigger.
_CONFLICT_TOLERANCE = 0.015

# A chart value this large cannot be a 0-1 fraction -- it was stored as a raw
# 0-100 percent number. Scaling it before comparing avoids a false conflict
# that is really just a units mismatch, not a disagreement in the data.
_LIKELY_PERCENT_THRESHOLD = 1.5


def infer_document_type(filename: str) -> str:
    lowered = filename.lower()
    if "internal" in lowered:
        return "internal"
    if "external" in lowered:
        return "external"
    return "unknown"


def normalize_metric_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return key.strip("_")


def collect_metrics(elements: list[Element]) -> tuple[list[MetricObservation], list[str]]:
    """Build the metrics[] time series from native chart IR elements.

    Returns (metrics, warnings) -- a blank/non-numeric data point (python-pptx
    returns None for an empty cell) is skipped with a warning, never crashing
    the whole extraction over one missing point.
    """
    metrics: list[MetricObservation] = []
    warnings: list[str] = []
    for element in elements:
        if element.type != "chart":
            continue
        content = element.content or {}
        categories = content.get("categories", [])
        for series in content.get("series", []):
            name = normalize_metric_key(series.get("name", ""))
            for period, raw_value in zip(categories, series.get("values", []), strict=False):
                if raw_value is None:
                    warnings.append(
                        f"slide/page {element.page_or_slide}: blank data point for "
                        f"{name!r} ({period}) skipped"
                    )
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    warnings.append(
                        f"slide/page {element.page_or_slide}: non-numeric data point "
                        f"for {name!r} ({period}) skipped: {raw_value!r}"
                    )
                    continue
                metrics.append(
                    MetricObservation(
                        field="model_performance",
                        name=name,
                        period=str(period),
                        value=value,
                        source_element_id=element.element_id,
                        page_or_slide=element.page_or_slide,
                    )
                )
    return metrics, warnings


def collect_incidents(elements: list[Element]) -> list[dict]:
    """Structured incident records from table-type IR elements.

    Native PPTX tables (`shape.has_table`) store `content["rows"]` (a list of
    rows); the aligned-textbox-row heuristic stores a single row directly as
    `content["cells"]`. Both are normalized to a list of rows here so neither
    shape silently drops incident data.
    """
    incidents: list[dict] = []
    for element in elements:
        if element.type != "table":
            continue
        content = element.content or {}
        rows = content.get("rows")
        if rows is None and "cells" in content:
            rows = [content["cells"]]
        for cells in rows or []:
            if len(cells) == 3 and _INCIDENT_DATE_PATTERN.match(str(cells[0]).strip()):
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


def match_metric(narrative_key: str, candidate_names: list[str]) -> str | None:
    """Match a narrative field's metric key to a chart series name.

    Prefers an exact match. Otherwise, matches only when exactly one
    candidate's token set is a superset/subset of the narrative key's tokens
    -- if more than one candidate qualifies (e.g. "accuracy" vs. both
    "balanced_accuracy" and "classification_accuracy"), refuses to guess and
    returns None rather than silently picking the wrong metric.
    """
    if narrative_key in candidate_names:
        return narrative_key

    narrative_tokens = set(narrative_key.split("_"))
    matches = [
        name
        for name in candidate_names
        if narrative_tokens <= set(name.split("_")) or set(name.split("_")) <= narrative_tokens
    ]
    return matches[0] if len(matches) == 1 else None


def detect_performance_conflicts(
    fields: dict[str, FieldValue], metrics: list[MetricObservation]
) -> list[ReviewFlag]:
    performance = fields.get("model_performance")
    if performance is None or not isinstance(performance.value, dict) or not metrics:
        return []

    latest_by_name: dict[str, MetricObservation] = {}
    for metric in metrics:
        latest_by_name[metric.name] = metric  # categories are chronological; last write wins

    flags: list[ReviewFlag] = []
    for narrative_key, narrative_value in performance.value.items():
        if not isinstance(narrative_value, int | float):
            # a `conflicting` field's value never passed through
            # normalize_value/validate_field_shape (see _resolve_field_candidates'
            # early-return on disagreement), so it can still hold a raw,
            # un-normalized entry like a string percentage -- never trusted
            # as comparable to a chart's numeric value, but never a crash either
            continue
        matched_name = match_metric(narrative_key, list(latest_by_name))
        if matched_name is None:
            continue
        chart_obs = latest_by_name[matched_name]
        chart_value = chart_obs.value
        if chart_value > _LIKELY_PERCENT_THRESHOLD:
            chart_value = chart_value / 100.0

        if abs(narrative_value - chart_value) > _CONFLICT_TOLERANCE:
            performance.status = "conflicting"
            performance.evidence.append(
                Evidence(
                    page_or_slide=chart_obs.page_or_slide,
                    element_id=chart_obs.source_element_id,
                    quote=f"{chart_obs.name}={chart_obs.value} ({chart_obs.period})",
                    method="chart_series",
                )
            )
            flags.append(
                ReviewFlag(
                    code="performance_conflict",
                    field="model_performance",
                    message=(
                        f"narrative value for {narrative_key!r} is {narrative_value}, "
                        f"but the chart's latest value for {matched_name!r} is {chart_obs.value}"
                    ),
                )
            )
    return flags
