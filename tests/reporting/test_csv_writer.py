from datetime import UTC, datetime

from ai_oversight.reporting.csv_writer import write
from ai_oversight.schemas import CANONICAL_FIELDS, Evidence, ExtractionResult, FieldValue


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
    assert rows[0] == "file_name,field,status,value,confidence"
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
    assert rows == ["file_name,field,status,value,confidence"]
