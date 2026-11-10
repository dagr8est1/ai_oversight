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
