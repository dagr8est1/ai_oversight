import pytest
from pydantic import ValidationError

from ai_oversight.schemas import (
    CANONICAL_FIELD_DESCRIPTIONS,
    CANONICAL_FIELDS,
    FIELD_ID_TO_NAME,
    Evidence,
    ExtractionResult,
    FieldValue,
)


def test_canonical_fields_has_22_entries_in_plan_order():
    assert len(CANONICAL_FIELDS) == 22
    assert CANONICAL_FIELDS[0] == "submission_id"
    assert CANONICAL_FIELDS[21] == "next_review_date"


def test_field_id_to_name_maps_two_digit_ids():
    assert FIELD_ID_TO_NAME["01"] == "submission_id"
    assert FIELD_ID_TO_NAME["22"] == "next_review_date"


def test_every_canonical_field_has_a_description():
    assert set(CANONICAL_FIELD_DESCRIPTIONS) == set(CANONICAL_FIELDS)
    assert all(isinstance(d, str) and d for d in CANONICAL_FIELD_DESCRIPTIONS.values())


def test_field_value_rejects_unknown_status():
    with pytest.raises(ValidationError):
        FieldValue(status="not_a_real_status")


def test_field_value_defaults_to_missing_with_no_evidence():
    fv = FieldValue(status="missing")
    assert fv.value is None
    assert fv.evidence == []


def test_extraction_result_requires_all_22_fields_present():
    fields = {name: FieldValue(status="missing") for name in CANONICAL_FIELDS}
    result = ExtractionResult(
        document_id="sha256:abc",
        source_file="x.pdf",
        document_type="internal",
        extracted_at="2026-01-01T00:00:00Z",
        fields=fields,
    )
    assert set(result.fields) == set(CANONICAL_FIELDS)
    assert result.metrics == []
    assert result.review_flags == []


def test_evidence_requires_page_element_quote_and_method():
    ev = Evidence(page_or_slide=2, element_id="page2.kv0", quote="AI-INT-2026-041", method="pymupdf_text_block_pair")
    assert ev.page_or_slide == 2
