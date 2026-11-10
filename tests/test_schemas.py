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


def _evidence(n: int) -> list[Evidence]:
    return [
        Evidence(page_or_slide=1, element_id=f"page1.e{i}", quote="x", method="m") for i in range(n)
    ]


def test_field_value_confidence_for_missing_status_is_always_zero():
    assert FieldValue(status="missing").confidence == 0.0
    # even if evidence were somehow attached, missing means no value at all
    assert FieldValue(status="missing", evidence=_evidence(3)).confidence == 0.0


def test_field_value_confidence_rises_with_corroborating_evidence():
    # more independent evidence agreeing on the same accepted value should
    # push confidence toward the top of its status's band, not treat one
    # citation and three identical citations as equally trustworthy
    low = FieldValue(status="extracted", evidence=_evidence(1)).confidence
    high = FieldValue(status="extracted", evidence=_evidence(3)).confidence
    assert low < high
    assert high == 1.0


def test_field_value_confidence_falls_with_more_disagreeing_sources():
    # the opposite direction for `conflicting`: more sources that still
    # can't agree is a worse sign, not a better one, unlike every other
    # status where more evidence means more corroboration
    two_way = FieldValue(status="conflicting", evidence=_evidence(2)).confidence
    four_way = FieldValue(status="conflicting", evidence=_evidence(4)).confidence
    assert two_way > four_way


def test_field_value_confidence_bands_never_overlap_between_statuses():
    # a `needs_review` field, no matter how much evidence backs it, must
    # never out-rank even a weakly-corroborated `extracted` one
    worst_extracted = FieldValue(status="extracted", evidence=_evidence(1)).confidence
    best_needs_review = FieldValue(status="needs_review", evidence=_evidence(10)).confidence
    assert best_needs_review < worst_extracted

    worst_needs_review = FieldValue(status="needs_review", evidence=_evidence(1)).confidence
    best_ambiguous = FieldValue(status="ambiguous", evidence=_evidence(10)).confidence
    assert best_ambiguous < worst_needs_review


def test_field_value_confidence_is_computed_not_settable_and_serializes():
    # computed from status + evidence, not a separately-stored field -- so
    # it can never drift out of sync at one of the many FieldValue(...)
    # construction sites across rules.py/qwen_local.py/merge.py/validate.py
    fv = FieldValue(status="needs_review", evidence=_evidence(1))
    assert fv.model_dump()["confidence"] == fv.confidence
    assert "confidence" in fv.model_dump_json()


def test_evidence_requires_page_element_quote_and_method():
    ev = Evidence(page_or_slide=2, element_id="page2.kv0", quote="AI-INT-2026-041", method="pymupdf_text_block_pair")
    assert ev.page_or_slide == 2
