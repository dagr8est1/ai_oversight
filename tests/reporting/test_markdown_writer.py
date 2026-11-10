from datetime import UTC, datetime

from ai_oversight.reporting.markdown_writer import render
from ai_oversight.schemas import (
    CANONICAL_FIELDS,
    Evidence,
    ExtractionResult,
    FieldValue,
    ReviewFlag,
    VisionObservation,
)


def _result(**overrides) -> ExtractionResult:
    fields = {name: FieldValue(status="missing") for name in CANONICAL_FIELDS}
    fields["submission_id"] = FieldValue(
        value="AI-INT-2026-041",
        status="extracted",
        evidence=[Evidence(page_or_slide=2, element_id="page2.kv0", quote="AI-INT-2026-041", method="m")],
    )
    defaults = {
        "document_id": "sha256:abc",
        "source_file": "Sample_Internal_AI_Oversight_Report.pdf",
        "document_type": "internal",
        "extracted_at": datetime.now(UTC),
        "fields": fields,
    }
    defaults.update(overrides)
    return ExtractionResult(**defaults)


def test_render_includes_source_file_and_document_type():
    md = render(_result())
    assert "Sample_Internal_AI_Oversight_Report.pdf" in md
    assert "internal" in md


def test_render_lists_every_field_status_and_value():
    md = render(_result())
    assert "submission_id" in md
    assert "AI-INT-2026-041" in md
    assert "extracted" in md
    assert "missing" in md  # at least one field stays missing in the fixture


def test_render_lists_each_field_s_confidence():
    md = render(_result())
    assert "Confidence" in md
    missing_field = CANONICAL_FIELDS[2]  # application_name, status "missing" in the fixture
    assert "| submission_id | extracted | AI-INT-2026-041 | 0.85 |" in md
    assert f"| {missing_field} | missing |  | 0.0 |" in md


def test_render_lists_review_flags_as_suggestions_not_findings():
    result = _result(
        review_flags=[ReviewFlag(code="performance_conflict", field="model_performance", message="94 vs 92")]
    )
    md = render(result)
    assert "rule-generated suggestion" in md.lower()
    assert "94 vs 92" in md


def test_render_lists_extraction_warnings():
    result = _result(extraction_warnings=["page 3: vector drawing detected (chart-like)"])
    md = render(result)
    assert "vector drawing detected" in md


def test_render_lists_vision_observations_with_their_parsed_content():
    result = _result(
        vision_observations=[
            VisionObservation(
                task="tables_json",
                source_element_id="page1.vision_scan",
                page_or_slide=1,
                raw_response='[{"Field": "Submission ID", "Value": "AI-1"}]',
                parsed=[{"Field": "Submission ID", "Value": "AI-1"}],
                status="ok",
            )
        ]
    )
    md = render(result)
    assert "tables_json" in md
    assert "page1.vision_scan" in md
    assert "Submission ID" in md
