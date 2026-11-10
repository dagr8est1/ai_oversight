import io
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Inches

from ai_oversight.parsers.pptx_parser import parse

SAMPLE = Path(__file__).parents[2] / "data" / "samples" / "Sample_External_AI_Usage_Disclosure.pptx"


def _key_values_on(doc, slide: int) -> dict[str, dict]:
    return {
        el.content["field_id"]: el.content
        for el in doc.elements
        if el.type == "key_value" and el.page_or_slide == slide
    }


def test_parses_without_a_model_and_tags_document_identity():
    doc = parse(SAMPLE)

    assert doc.format == "pptx"
    assert doc.filename == "Sample_External_AI_Usage_Disclosure.pptx"
    assert doc.document_id.startswith("sha256:")


def test_slide_5_native_chart_has_categories_and_two_named_series():
    doc = parse(SAMPLE)

    charts = [el for el in doc.elements if el.type == "chart" and el.page_or_slide == 5]
    assert len(charts) == 1

    content = charts[0].content
    assert content["categories"] == ["Aug 2026", "Sep 2026", "Oct 2026"]
    series_by_name = {s["name"]: s["values"] for s in content["series"]}
    assert series_by_name["Classification accuracy"] == [0.87, 0.9, 0.92]
    assert series_by_name["Grounded response rate"] == [0.94, 0.96, 0.97]


def test_slides_3_and_4_aligned_textbox_rows_are_paired_label_to_value():
    doc = parse(SAMPLE)

    slide_3 = _key_values_on(doc, slide=3)
    slide_4 = _key_values_on(doc, slide=4)

    assert slide_3["01"]["label"] == "Submission ID"
    assert slide_3["01"]["value"] == "EXT-2026-118"
    assert slide_3["03"]["label"] == "AI application"
    assert slide_3["03"]["value"] == "Collateral Exception Triage Copilot (CETC)"
    assert slide_4["18"]["value"] == "Inherent HIGH; residual MEDIUM"
    assert len(slide_3) == 12
    assert len(slide_4) == 10


def test_unlabeled_multi_shape_rows_keep_their_cells_grouped_not_scattered():
    # slide 7's incident table has no numeric field-ID column, just aligned
    # Date / Event / Impact textboxes sharing a row -- losing that grouping
    # would make the table unrecoverable from separate, unlinked text elements.
    doc = parse(SAMPLE)

    rows = [
        el.content["cells"]
        for el in doc.elements
        if el.type == "table" and el.page_or_slide == 7
    ]

    assert ["Date", "Event", "Impact / remediation"] in rows
    assert [
        "Sep 07",
        "Incorrect routing labels in 3 tickets",
        "Analyst rejected; prompt template patched",
    ] in rows


def test_key_value_row_with_extra_trailing_shapes_does_not_swallow_them_into_value(
    tmp_path,
):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    top = Inches(1)
    for left, text in [
        (Inches(0.5), "07"),
        (Inches(1.0), "Model type"),
        (Inches(2.5), "XGBoost"),
        (Inches(4.0), "unrelated extra column"),
    ]:
        box = slide.shapes.add_textbox(left, top, Inches(1.4), Inches(0.4))
        box.text_frame.text = text

    pptx_path = tmp_path / "extra_column.pptx"
    prs.save(pptx_path)

    doc = parse(pptx_path)

    kv = [el for el in doc.elements if el.type == "key_value"]
    assert len(kv) == 1
    assert kv[0].content == {"field_id": "07", "label": "Model type", "value": "XGBoost"}
    # the 4th shape's text must still be recoverable somewhere, not discarded
    all_text = " ".join(el.raw_text or "" for el in doc.elements) + " ".join(
        str(el.content) for el in doc.elements
    )
    assert "unrelated extra column" in all_text


def test_picture_shapes_are_flagged_not_silently_dropped(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    png_bytes = io.BytesIO()
    Image.new("RGB", (4, 4), color="blue").save(png_bytes, format="PNG")
    png_bytes.seek(0)
    slide.shapes.add_picture(png_bytes, Inches(1), Inches(1), Inches(2), Inches(2))

    pptx_path = tmp_path / "with_picture.pptx"
    prs.save(pptx_path)

    doc = parse(pptx_path)

    assert any("picture" in w.lower() or "image" in w.lower() for w in doc.warnings)


def test_text_inside_a_group_shape_is_still_extracted(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    label = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(0.5))
    label.text_frame.text = "Grouped label"
    value = slide.shapes.add_textbox(Inches(1), Inches(1.5), Inches(2), Inches(0.5))
    value.text_frame.text = "Grouped value"
    slide.shapes.add_group_shape([label, value])

    pptx_path = tmp_path / "grouped.pptx"
    prs.save(pptx_path)

    doc = parse(pptx_path)

    texts = {el.raw_text for el in doc.elements if el.raw_text}
    assert "Grouped label" in texts
    assert "Grouped value" in texts
