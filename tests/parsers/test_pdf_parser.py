from pathlib import Path

from ai_oversight.parsers.pdf_parser import parse

SAMPLE = Path(__file__).parents[2] / "data" / "samples" / "Sample_Internal_AI_Oversight_Report.pdf"


def _key_values_on(doc, page: int) -> dict[str, str]:
    return {
        el.content["label"]: el.content["value"]
        for el in doc.elements
        if el.type == "key_value" and el.page_or_slide == page
    }


def test_parses_without_a_model_and_tags_document_identity():
    doc = parse(SAMPLE)

    assert doc.format == "pdf"
    assert doc.filename == "Sample_Internal_AI_Oversight_Report.pdf"
    assert doc.document_id.startswith("sha256:")


def test_page_2_inventory_fields_are_paired_label_to_value():
    doc = parse(SAMPLE)

    fields = _key_values_on(doc, page=2)

    assert fields["01 / Submission ID"] == "AI-INT-2026-041"
    assert fields["03 / Application name"] == "Settlement Failure Early Warning (SFEW)"
    assert fields["18 / Risk rating"] == "Inherent HIGH; residual MEDIUM"
    assert len(fields) == 22


def test_vector_chart_page_is_captured_with_a_warning_not_a_crash():
    doc = parse(SAMPLE)

    assert any("page 3" in w and "chart" in w.lower() for w in doc.warnings)
    # the chart's own text labels (axis ticks, month names) must still be
    # captured as plain text elements, since they ARE present in the text layer
    page_3_text = " ".join(
        el.raw_text for el in doc.elements if el.page_or_slide == 3 and el.raw_text
    )
    assert "Jul" in page_3_text
    assert "0.91" in page_3_text


def test_decorative_vector_graphics_on_other_pages_do_not_trigger_a_chart_warning():
    # pages 1, 2 and 4 use plain background rectangles/borders for styling, not
    # a chart; only page 3's line chart (repeated gridlines) should warn.
    doc = parse(SAMPLE)

    warned_pages = {w for w in doc.warnings}
    assert not any("page 1" in w for w in warned_pages)
    assert not any("page 2" in w for w in warned_pages)
    assert not any("page 4" in w for w in warned_pages)
