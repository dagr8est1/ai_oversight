from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

FieldStatus = Literal["extracted", "missing", "ambiguous", "conflicting", "needs_review"]

# Non-overlapping confidence bands per status -- `needs_review` can never
# out-rank `extracted` no matter how much evidence backs it, since their
# ranges never touch. `missing` has no band: there's no value at all, so
# its confidence is always 0.0, handled separately below.
_CONFIDENCE_RANGE: dict[str, tuple[float, float]] = {
    "conflicting": (0.15, 0.29),
    "ambiguous": (0.30, 0.44),
    "needs_review": (0.45, 0.70),
    "extracted": (0.85, 1.00),
}
# how many pieces of evidence it takes to reach a band's ceiling
_CORROBORATION_SATURATION = 3
# for `conflicting` only: how many disagreeing sources it takes to fall
# all the way to the band's floor -- more disagreement is a worse sign,
# the opposite of every other status
_CONFLICT_CONFUSION_SATURATION = 4

# Exactly PLAN.md Section 5's 22 canonical fields, in their numbered order:
# index 0 is field "01", index 21 is field "22".
CANONICAL_FIELDS: list[str] = [
    "submission_id",
    "reporting_entity",
    "application_name",
    "business_use_case",
    "business_owner",
    "technical_owner",
    "model_type",
    "model_provider",
    "model_version",
    "lifecycle_status",
    "first_deployment_date",
    "input_data",
    "data_sensitivity",
    "financial_criticality",
    "human_oversight",
    "validation_status",
    "model_performance",
    "risk_rating",
    "incident_history",
    "monitoring_frequency",
    "fallback_process",
    "next_review_date",
]

FIELD_ID_TO_NAME: dict[str, str] = {
    f"{index + 1:02d}": name for index, name in enumerate(CANONICAL_FIELDS)
}

# PLAN.md Section 5's own description for each field. A bare snake_case name
# gives a semantic mapper (qwen_local) nothing to match an unusually-worded
# source label against -- e.g. a label "RESIDUAL RISK" has no lexical
# overlap with "risk_rating" alone, so the model echoed the label as a new
# JSON key instead of picking the real field. The description closes that
# gap: "Explicitly reported inherent and residual ratings" does overlap.
CANONICAL_FIELD_DESCRIPTIONS: dict[str, str] = {
    "submission_id": "Document submission identifier",
    "reporting_entity": "Legal entity / reporting organization",
    "application_name": "AI application and abbreviation",
    "business_use_case": "Purpose and business workflow",
    "business_owner": "Responsible business person/team",
    "technical_owner": "Responsible technical person/team",
    "model_type": "E.g., XGBoost, RAG LLM + classifier",
    "model_provider": "Internal team or external service",
    "model_version": "All named application/model versions",
    "lifecycle_status": "Pilot, controlled rollout, production, retired, etc.",
    "first_deployment_date": "ISO YYYY-MM-DD date the model first went live",
    "input_data": "Data sources and categories",
    "data_sensitivity": "Classification and data-handling notes",
    "financial_criticality": "Reported criticality and explanation",
    "human_oversight": "Approval/override/decision rights",
    "validation_status": "Validation date, party, scope and result",
    "model_performance": "Metric dictionary with units, periods and samples",
    "risk_rating": "Explicitly reported inherent and residual ratings",
    "incident_history": "Count, dates, statuses and consequences",
    "monitoring_frequency": "Cadence and what is monitored",
    "fallback_process": "Manual/automated failure-mode backup",
    "next_review_date": "ISO YYYY-MM-DD date of the next scheduled review",
}


class Evidence(BaseModel):
    page_or_slide: int
    element_id: str
    quote: str
    method: str


class FieldValue(BaseModel):
    value: Any | None = None
    status: FieldStatus
    evidence: list[Evidence] = Field(default_factory=list)
    normalization_notes: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """A triage-level signal of how much a reviewer should trust this
        field -- not a literal probability. Computed from `status` and
        `evidence` (never a separate stored field, so it can't drift out
        of sync at one of the many FieldValue(...) construction sites
        across rules.py/qwen_local.py/merge.py/validate.py): each status
        has a fixed, non-overlapping band, and more independent evidence
        corroborating the accepted value moves it toward that band's
        ceiling. `conflicting` is the exception -- more *disagreeing*
        sources is a worse sign, so more evidence there moves it toward
        the floor instead.
        """
        if self.status == "missing":
            return 0.0
        floor, ceiling = _CONFIDENCE_RANGE[self.status]
        count = len(self.evidence)
        if self.status == "conflicting":
            span = _CONFLICT_CONFUSION_SATURATION - 2
            confusion = max(_CONFLICT_CONFUSION_SATURATION - count, 0) / span
            corroboration = min(confusion, 1.0)
        else:
            span = _CORROBORATION_SATURATION - 1
            corroboration = max(0.0, min(count - 1, span) / span)
        return round(floor + (ceiling - floor) * corroboration, 2)


class MetricObservation(BaseModel):
    field: str
    name: str
    period: str
    value: float
    unit: str | None = None
    source_element_id: str
    page_or_slide: int


class ReviewFlag(BaseModel):
    code: str
    message: str
    field: str | None = None
    # Explicit per PLAN.md Section 5: a review flag is a rule-generated
    # suggestion, never a compliance finding -- stated in the JSON itself,
    # not only in the Markdown report's heading.
    kind: Literal["rule_generated_suggestion"] = "rule_generated_suggestion"


class VisionObservation(BaseModel):
    task: Literal["chart2csv", "tables_json"]
    source_element_id: str | None = None
    page_or_slide: int
    raw_response: str
    parsed: Any | None = None
    status: Literal["ok", "needs_review"]
    note: str | None = None


class VisionIntegrity(BaseModel):
    """Document-level summary of how reliably Granite Vision's output was
    recovered, so a reviewer can tell at a glance whether this document's
    vision-derived content is trustworthy without reading every
    VisionObservation by hand."""

    tasks_total: int
    tasks_clean: int
    tasks_partial: int
    tasks_failed: int
    success_rate: float
    rating: Literal["pass", "review", "fail"]
    loss_notes: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    document_id: str
    source_file: str
    document_type: Literal["internal", "external", "unknown"] = "unknown"
    extracted_at: datetime
    schema_version: str = "1.0"
    fields: dict[str, FieldValue]
    metrics: list[MetricObservation] = Field(default_factory=list)
    incidents: list[dict[str, Any]] = Field(default_factory=list)
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
    vision_observations: list[VisionObservation] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def vision_integrity(self) -> VisionIntegrity:
        """Derived purely from vision_observations, never a separately
        stored field, so it can't drift out of sync with the observations
        a caller attaches after the fact (cli.py sets vision_observations
        post-extraction). A task is "clean" only with no caveat at all;
        "partial" parsed but with a noted loss (e.g. a truncated response
        repaired by keeping only the complete rows before the cutoff);
        "failed" recovered nothing. Any failure drives the rating to
        "fail" regardless of how many other tasks were clean -- a single
        lost chart/table is a real gap, not something unrelated successes
        elsewhere in the same document should average away.
        """
        observations = self.vision_observations
        total = len(observations)
        clean = sum(1 for o in observations if o.status == "ok" and o.note is None)
        partial = sum(1 for o in observations if o.status == "ok" and o.note is not None)
        failed = total - clean - partial
        success_rate = 1.0 if total == 0 else round(clean / total, 2)

        if total == 0 or clean == total:
            rating: Literal["pass", "review", "fail"] = "pass"
        elif failed > 0:
            rating = "fail"
        else:
            rating = "review"

        loss_notes = [
            f"page/slide {o.page_or_slide} ({o.task}): {o.note}"
            for o in observations
            if o.status != "ok" or o.note is not None
        ]

        return VisionIntegrity(
            tasks_total=total,
            tasks_clean=clean,
            tasks_partial=partial,
            tasks_failed=failed,
            success_rate=success_rate,
            rating=rating,
            loss_notes=loss_notes,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def field_match_rate(self) -> float:
        """Fraction of the 22 canonical fields resolved as a confident
        `extracted` match for this document. `needs_review`, `ambiguous`,
        and `conflicting` all count the same as `missing` here -- each
        means a human still has to confirm the value, same as if nothing
        were found at all. Derived purely from `fields`, never a stored
        field, so it can't drift out of sync as statuses change.
        """
        if not self.fields:
            return 0.0
        extracted = sum(1 for field in self.fields.values() if field.status == "extracted")
        return round(extracted / len(self.fields), 2)
