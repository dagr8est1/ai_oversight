from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

FieldStatus = Literal["extracted", "missing", "ambiguous", "conflicting", "needs_review"]

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
