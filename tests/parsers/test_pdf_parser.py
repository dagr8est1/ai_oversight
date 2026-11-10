import io
from pathlib import Path

import pymupdf
from PIL import Image

from ai_oversight.parsers.pdf_parser import chart_region_on_page, parse

SAMPLE = Path(__file__).parents[2] / "data" / "samples" / "Sample_Internal_AI_Oversight_Report.pdf"


def _key_values_on(doc, page: int) -> dict[str, dict]:
    return {
        el.content["field_id"]: el.content
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

    assert fields["01"]["label"] == "Submission ID"
    assert fields["01"]["value"] == "AI-INT-2026-041"
    assert fields["03"]["label"] == "Application name"
    assert fields["03"]["value"] == "Settlement Failure Early Warning (SFEW)"
    assert fields["18"]["value"] == "Inherent HIGH; residual MEDIUM"
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


def test_label_far_from_any_text_is_not_fabricated_into_a_pair(tmp_path):
    pdf_path = tmp_path / "missing_value.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((39, 150), "02 / Legal entity")
    # a large gap -- this is unrelated narrative text further down the page,
    # not this field's value
    page.insert_text((39, 400), "RISK NARRATIVE: model drift observed")
    doc.save(pdf_path)
    doc.close()

    parsed = parse(pdf_path)

    assert not any(el.type == "key_value" for el in parsed.elements)
    texts = {el.raw_text for el in parsed.elements if el.raw_text}
    assert "02 / Legal entity" in texts
    assert "RISK NARRATIVE: model drift observed" in texts


def test_page_with_no_text_layer_warns_instead_of_crashing(tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(10, 10, 100, 100))  # drawing only, no text
    doc.save(pdf_path)
    doc.close()

    parsed = parse(pdf_path)

    assert parsed.elements == []
    assert any("no text layer" in w.lower() or "scanned" in w.lower() for w in parsed.warnings)


def test_chart_region_on_page_locates_the_gridline_cluster_for_the_chart_page():
    with pymupdf.open(SAMPLE) as doc:
        page = doc[2]  # page 3, the one with the line chart
        region = chart_region_on_page(page)

        assert region is not None
        # the chart sits in the body of the page, not spanning the full page
        assert region.width < page.rect.width
        assert region.height < page.rect.height
        assert region.width > 50
        assert region.height > 50
        # the real gridlines span y=198 to y=382 (6 evenly-spaced axis lines);
        # a bbox built from zero-height Rects via `|=` silently collapses to
        # just the first gridline, since PyMuPDF treats a zero-area Rect as
        # "empty" and a union with an empty Rect is a no-op -- this assertion
        # catches that regression directly, not just via a width/height floor
        assert region.y0 <= 198
        assert region.y1 >= 382


def test_chart_region_on_page_is_none_for_decorative_pages():
    with pymupdf.open(SAMPLE) as doc:
        for page_index in (0, 1, 3):  # pages 1, 2, 4
            assert chart_region_on_page(doc[page_index]) is None


def test_page_with_raster_image_is_flagged_not_silently_dropped(tmp_path):
    pdf_path = tmp_path / "with_image.pdf"
    png_bytes = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(png_bytes, format="PNG")

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((39, 150), "unrelated caption")
    page.insert_image(pymupdf.Rect(39, 200, 139, 300), stream=png_bytes.getvalue())
    doc.save(pdf_path)
    doc.close()

    parsed = parse(pdf_path)

    assert any("image" in w.lower() for w in parsed.warnings)
