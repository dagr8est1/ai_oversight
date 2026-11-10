from datetime import UTC, datetime

from ai_oversight.reporting.html_report_writer import render, write
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


def test_render_is_a_complete_html_document():
    html = render([_result("a.pdf")])
    assert html.strip().startswith("<!doctype html>")
    assert "<html" in html and "</html>" in html


def test_render_groups_fields_under_each_file_name():
    results = [_result("a.pdf"), _result("b.pptx")]

    html = render(results)

    assert "a.pdf" in html
    assert "b.pptx" in html
    # each file's 22 fields appear, not merged into one shared table
    assert html.count("submission_id") == 2


def test_render_includes_status_value_and_confidence():
    result = _result(
        "a.pdf",
        submission_id=FieldValue(
            value="AI-1",
            status="extracted",
            evidence=[Evidence(page_or_slide=1, element_id="page1.kv0", quote="AI-1", method="m")],
        ),
    )

    html = render([result])

    assert "AI-1" in html
    assert "extracted" in html
    assert "0.85" in html  # confidence for a single-evidence extracted field


def test_render_escapes_html_special_characters_in_values():
    result = _result(
        "a.pdf",
        business_use_case=FieldValue(value="<script>alert(1)</script>", status="extracted"),
    )

    html = render([result])

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_write_saves_the_rendered_html_to_disk(tmp_path):
    out = tmp_path / "report.html"

    write([_result("a.pdf")], out)

    assert out.exists()
    assert out.read_text().strip().startswith("<!doctype html>")
