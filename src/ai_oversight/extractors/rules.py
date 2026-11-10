from collections import defaultdict
from datetime import UTC, datetime

from ai_oversight.parsers.document_ir import Element, ParsedDocument
from ai_oversight.processing.merge import (
    collect_incidents,
    collect_metrics,
    detect_performance_conflicts,
    infer_document_type,
    normalize_value,
)
from ai_oversight.processing.validate import validate_field_shape
from ai_oversight.schemas import (
    CANONICAL_FIELDS,
    FIELD_ID_TO_NAME,
    Evidence,
    ExtractionResult,
    FieldValue,
    ReviewFlag,
)

# A loose cross-check that a field_id's label actually says what that
# position is supposed to mean -- field_id alone is trusted only when this
# also agrees, since a document whose numbering doesn't match PLAN.md's
# canonical order would otherwise silently swap two fields' values.
_LABEL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "submission_id": ("submission id",),
    "reporting_entity": ("legal entity", "reporting organization", "reporting entity"),
    "application_name": ("application name", "ai application"),
    "business_use_case": ("business use case",),
    "business_owner": ("business owner",),
    "technical_owner": ("technical owner",),
    "model_type": ("model type",),
    "model_provider": ("ai provider", "model provider"),
    "model_version": ("model version", "version"),
    "lifecycle_status": ("lifecycle status", "deployment status"),
    "first_deployment_date": ("first deployment", "first deployed"),
    "input_data": ("input data", "data inputs"),
    "data_sensitivity": ("data sensitivity",),
    "financial_criticality": ("financial criticality",),
    "human_oversight": ("human oversight",),
    "validation_status": ("validation status", "validation"),
    "model_performance": ("performance",),
    "risk_rating": ("risk rating",),
    "incident_history": ("incident history",),
    "monitoring_frequency": ("monitoring frequency",),
    "fallback_process": ("fallback process",),
    "next_review_date": ("next periodic review", "next review date", "next review"),
}


def extract(doc: ParsedDocument) -> ExtractionResult:
    candidates: dict[str, list[tuple[str, Element]]] = defaultdict(list)

    for element in doc.elements:
        if element.type != "key_value":
            continue
        content = element.content or {}
        field_id = content.get("field_id")
        name = FIELD_ID_TO_NAME.get(field_id) if field_id else None
        if name is None:
            continue
        candidates[name].append((content["value"], element))

    fields: dict[str, FieldValue] = {}
    resolution_warnings: list[str] = []
    resolution_flags: list[ReviewFlag] = []
    for name in CANONICAL_FIELDS:
        field, warning, flag = _resolve_field(name, candidates.get(name, []))
        fields[name] = field
        if warning:
            resolution_warnings.append(warning)
        if flag:
            resolution_flags.append(flag)

    metrics, metric_warnings = collect_metrics(doc.elements)
    conflict_flags = detect_performance_conflicts(fields, metrics)
    incidents = collect_incidents(doc.elements)

    return ExtractionResult(
        document_id=doc.document_id,
        source_file=doc.filename,
        document_type=infer_document_type(doc.filename),
        extracted_at=datetime.now(UTC),
        fields=fields,
        metrics=metrics,
        incidents=incidents,
        review_flags=[*resolution_flags, *conflict_flags],
        extraction_warnings=[*doc.warnings, *metric_warnings, *resolution_warnings],
    )


def _resolve_field(
    name: str, entries: list[tuple[str, Element]]
) -> tuple[FieldValue, str | None, ReviewFlag | None]:
    if not entries:
        return FieldValue(status="missing"), None, None

    distinct_values = {value for value, _ in entries}
    evidence = [
        Evidence(
            page_or_slide=element.page_or_slide,
            element_id=element.element_id,
            quote=value,
            method=element.method,
        )
        for value, element in entries
    ]

    if len(distinct_values) > 1:
        flag = ReviewFlag(
            code="duplicate_field_conflict",
            field=name,
            message=f"{len(distinct_values)} differing values found for field_id-mapped field {name!r}",
        )
        return FieldValue(value=entries[0][0], status="conflicting", evidence=evidence), None, flag

    raw_value, element = entries[0]
    label = (element.content or {}).get("label", "")
    if not _label_matches(name, label):
        warning = (
            f"{name}: label {label!r} on {element.element_id} doesn't match the expected "
            f"wording for field {name!r}; value kept but marked ambiguous rather than trusted"
        )
        return FieldValue(value=raw_value, status="ambiguous", evidence=evidence), warning, None

    value, status, notes = normalize_value(name, raw_value)
    field = FieldValue(value=value, status=status, evidence=evidence, normalization_notes=notes)
    return validate_field_shape(name, field), None, None


def _label_matches(name: str, label: str) -> bool:
    synonyms = _LABEL_SYNONYMS.get(name)
    if synonyms is None:
        return True
    lowered = label.strip().lower()
    return any(synonym in lowered for synonym in synonyms)
