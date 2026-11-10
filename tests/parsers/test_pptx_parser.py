from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

from ai_oversight.parsers.pptx_parser import parse

SAMPLE = Path(__file__).parents[2] / "data" / "samples" / "Sample_External_AI_Usage_Disclosure.pptx"


def _key_values_on(doc, slide: int) -> dict[str, str]:
    return {
        el.content["label"]: el.content["value"]
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

    assert slide_3["Submission ID"] == "EXT-2026-118"
    assert slide_3["AI application"] == "Collateral Exception Triage Copilot (CETC)"
    assert slide_4["Risk rating"] == "Inherent HIGH; residual MEDIUM"
    assert len(slide_3) == 12
    assert len(slide_4) == 10


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
