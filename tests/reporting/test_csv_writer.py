from datetime import UTC, datetime

from ai_oversight.reporting.csv_writer import write
from ai_oversight.schemas import (
    CANONICAL_FIELDS,
    Evidence,
    ExtractionResult,
    FieldValue,
    VisionObservation,
)

_HEADER = "file_name,field,status,value,confidence,vision_rating,vision_success_rate,vision_data_loss"


def _result(source_file: str, **field_overrides) -> ExtractionResult:
    fields = {name: FieldValue(status="missing") for name in CANONICAL_FIELDS}
    fields.update(field_overrides)
    return ExtractionResult(
        document_id=f"sha256:{source_file}",
        source_file=source_file,
        document_type="internal",
        extracted_at=datetime.now(UTC),
        fields=fields,
    )


def test_write_produces_one_row_per_field_per_file(tmp_path):
    results = [_result("a.pdf"), _result("b.pptx")]
    out = tmp_path / "report.csv"

    write(results, out)

    rows = out.read_text().splitlines()
    assert rows[0] == _HEADER
    # header + 22 fields * 2 files
    assert len(rows) == 1 + 22 * 2


def test_write_includes_value_status_and_confidence_for_each_field(tmp_path):
    result = _result(
        "a.pdf",
        submission_id=FieldValue(
            value="AI-1",
            status="extracted",
            evidence=[Evidence(page_or_slide=1, element_id="page1.kv0", quote="AI-1", method="m")],
        ),
    )
    out = tmp_path / "report.csv"

    write([result], out)

    content = out.read_text()
    assert "a.pdf,submission_id,extracted,AI-1,0.85" in content


def test_write_handles_no_results_without_crashing(tmp_path):
    out = tmp_path / "report.csv"

    write([], out)

    rows = out.read_text().splitlines()
    assert rows == [_HEADER]


def test_write_reports_pass_rating_and_full_success_rate_when_vision_was_not_used(tmp_path):
    result = _result("a.pdf")
    out = tmp_path / "report.csv"

    write([result], out)

    rows = out.read_text().splitlines()
    assert all(row.endswith(",pass,1.0,") for row in rows[1:])


def test_write_reports_fail_rating_and_loss_notes_when_a_vision_task_recovered_nothing(tmp_path):
    result = _result("a.pdf")
    result.vision_observations = [
        VisionObservation(
            task="tables_json",
            page_or_slide=3,
            raw_response="bad",
            parsed=None,
            status="needs_review",
            note="JSON parse error: boom",
        )
    ]
    out = tmp_path / "report.csv"

    write([result], out)

    content = out.read_text()
    assert ",fail,0.0," in content
    assert "JSON parse error: boom" in content
