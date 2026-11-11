import pytest
from pydantic import ValidationError

from ai_oversight.schemas import (
    CANONICAL_FIELD_DESCRIPTIONS,
    CANONICAL_FIELDS,
    FIELD_ID_TO_NAME,
    Evidence,
    ExtractionResult,
    FieldValue,
    VisionObservation,
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


def _result_with_observations(observations: list[VisionObservation]) -> ExtractionResult:
    fields = {name: FieldValue(status="missing") for name in CANONICAL_FIELDS}
    return ExtractionResult(
        document_id="sha256:abc",
        source_file="x.pdf",
        document_type="internal",
        extracted_at="2026-01-01T00:00:00Z",
        fields=fields,
        vision_observations=observations,
    )


def test_vision_integrity_passes_trivially_with_no_vision_observations():
    # no vision stage run at all -- nothing to lose, so this isn't the same
    # as "every task happened to succeed"
    result = _result_with_observations([])
    vi = result.vision_integrity
    assert vi.tasks_total == 0
    assert vi.rating == "pass"
    assert vi.success_rate == 1.0
    assert vi.loss_notes == []


def test_vision_integrity_passes_when_every_task_is_clean():
    obs = [
        VisionObservation(task="tables_json", page_or_slide=1, raw_response="x", parsed=[{"a": 1}], status="ok"),
        VisionObservation(task="chart2csv", page_or_slide=2, raw_response="y", parsed=[{"b": 2}], status="ok"),
    ]
    vi = _result_with_observations(obs).vision_integrity
    assert vi.rating == "pass"
    assert vi.tasks_total == 2
    assert vi.tasks_clean == 2
    assert vi.success_rate == 1.0
    assert vi.loss_notes == []


def test_vision_integrity_flags_review_for_a_truncation_repaired_task():
    # status "ok" but with a note means something real was recovered, with
    # a caveat -- worse than clean, but not a total loss
    obs = [
        VisionObservation(task="tables_json", page_or_slide=1, raw_response="x", parsed=[{"a": 1}], status="ok"),
        VisionObservation(
            task="tables_json",
            page_or_slide=2,
            raw_response="y",
            parsed=[{"b": 2}],
            status="ok",
            note="response was truncated mid-generation; recovered the complete rows before the cutoff, dropped the incomplete one",
        ),
    ]
    vi = _result_with_observations(obs).vision_integrity
    assert vi.rating == "review"
    assert vi.tasks_clean == 1
    assert vi.tasks_partial == 1
    assert vi.tasks_failed == 0
    assert 0.0 < vi.success_rate < 1.0
    assert len(vi.loss_notes) == 1
    assert "truncated" in vi.loss_notes[0]


def test_vision_integrity_fails_when_any_task_recovered_nothing():
    obs = [
        VisionObservation(task="tables_json", page_or_slide=1, raw_response="x", parsed=[{"a": 1}], status="ok"),
        VisionObservation(
            task="tables_json",
            page_or_slide=2,
            raw_response="bad",
            parsed=None,
            status="needs_review",
            note="JSON parse error: boom",
        ),
    ]
    vi = _result_with_observations(obs).vision_integrity
    assert vi.rating == "fail"
    assert vi.tasks_failed == 1
    assert len(vi.loss_notes) == 1
    assert "JSON parse error: boom" in vi.loss_notes[0]


def test_vision_integrity_fail_outranks_review_regardless_of_other_successes():
    # one outright-lost chart/table must not be averaged away by many
    # unrelated clean/partial tasks elsewhere in the same document
    obs = [
        VisionObservation(task="tables_json", page_or_slide=1, raw_response="x", parsed=[{"a": 1}], status="ok"),
        VisionObservation(
            task="tables_json", page_or_slide=2, raw_response="x", parsed=[{"a": 1}], status="ok", note="partial"
        ),
        VisionObservation(
            task="tables_json",
            page_or_slide=3,
            raw_response="bad",
            parsed=None,
            status="needs_review",
            note="parse failed",
        ),
    ]
    vi = _result_with_observations(obs).vision_integrity
    assert vi.rating == "fail"


def test_vision_integrity_is_computed_not_settable_and_serializes():
    result = _result_with_observations([])
    assert result.model_dump()["vision_integrity"]["rating"] == "pass"
    assert "vision_integrity" in result.model_dump_json()
