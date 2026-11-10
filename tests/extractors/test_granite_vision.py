from pathlib import Path

import pytest
from pptx import Presentation
from pptx.util import Inches

from ai_oversight.extractors import granite_vision
from ai_oversight.parsers.document_ir import ParsedDocument

SAMPLE_PDF = Path(__file__).parents[2] / "data" / "samples" / "Sample_Internal_AI_Oversight_Report.pdf"


def test_extract_chart_parses_clean_csv():
    def fake_vision(image_bytes: bytes, task: str) -> str:
        assert task == "<chart2csv>"
        return "Month,Precision,Recall\nJul,0.89,0.85\nAug,0.90,0.86\n"

    obs = granite_vision.extract_chart(
        b"fake-png-bytes", fake_vision, source_element_id="page3.crop0", page_or_slide=3
    )

    assert obs.status == "ok"
    assert obs.task == "chart2csv"
    assert obs.parsed[0] == {"Month": "Jul", "Precision": "0.89", "Recall": "0.85"}


def test_extract_chart_strips_a_markdown_code_fence():
    def fake_vision(image_bytes: bytes, task: str) -> str:
        return "```csv\nMonth,Precision\nJul,0.89\n```"

    obs = granite_vision.extract_chart(b"x", fake_vision, source_element_id="e", page_or_slide=1)

    assert obs.status == "ok"
    assert obs.parsed[0]["Precision"] == "0.89"


def test_extract_chart_flags_unparseable_output_as_needs_review_not_fabricated():
    def fake_vision(image_bytes: bytes, task: str) -> str:
        return "I cannot read this chart clearly, the axis labels are blurry."

    obs = granite_vision.extract_chart(b"x", fake_vision, source_element_id="e", page_or_slide=1)

    assert obs.status == "needs_review"
    assert obs.parsed is None
    assert obs.note is not None
    assert "I cannot read this chart" in obs.raw_response


def test_extract_table_parses_json_array_of_rows():
    def fake_vision(image_bytes: bytes, task: str) -> str:
        assert task == "<tables_json>"
        return '[{"Measure": "Precision", "Observed": "0.91"}]'

    obs = granite_vision.extract_table(b"x", fake_vision, source_element_id="e", page_or_slide=3)

    assert obs.status == "ok"
    assert obs.parsed == [{"Measure": "Precision", "Observed": "0.91"}]


def test_extract_table_flags_malformed_json_as_needs_review():
    def fake_vision(image_bytes: bytes, task: str) -> str:
        return "{not valid json"

    obs = granite_vision.extract_table(b"x", fake_vision, source_element_id="e", page_or_slide=3)

    assert obs.status == "needs_review"
    assert obs.parsed is None


def test_default_vision_fn_raises_friendly_error_when_vision_extra_missing():
    with pytest.raises(granite_vision.VisionModelNotAvailableError, match="vision"):
        granite_vision.build_default_vision_fn(None)


def test_run_vision_pass_targets_only_the_pdf_chart_page():
    calls = []

    def fake_vision(image_bytes: bytes, task: str) -> str:
        calls.append(task)
        return "Month,Precision,Recall\nJul,0.89,0.85\nAug,0.90,0.86\n"

    doc = ParsedDocument(document_id="sha256:x", filename=SAMPLE_PDF.name, format="pdf", elements=[])
    observations = granite_vision.run_vision_pass(SAMPLE_PDF, doc, fake_vision)

    assert len(observations) == 1
    assert observations[0].page_or_slide == 3
    assert observations[0].task == "chart2csv"
    assert calls == ["<chart2csv>"]


def test_apply_to_result_adds_vision_metrics_and_flags_a_conflict(tmp_path):
    from ai_oversight.schemas import ExtractionResult, FieldValue

    def fake_vision(image_bytes: bytes, task: str) -> str:
        return "Month,Precision\nJul,0.89\nAug,0.90\nSep,0.92\nOct,0.80\n"

    doc = ParsedDocument(document_id="sha256:x", filename=SAMPLE_PDF.name, format="pdf", elements=[])
    result = ExtractionResult(
        document_id="sha256:x",
        source_file=SAMPLE_PDF.name,
        extracted_at="2026-01-01T00:00:00Z",
        fields={"model_performance": FieldValue(value={"precision": 0.91}, status="extracted", evidence=[])},
    )

    updated = granite_vision.apply_to_result(SAMPLE_PDF, doc, result, vision_fn=fake_vision)

    assert len(updated.metrics) == 4
    assert updated.fields["model_performance"].status == "conflicting"
    assert any(flag.field == "model_performance" for flag in updated.review_flags)


def test_apply_to_result_degrades_to_a_warning_on_any_model_failure_not_just_missing_extra():
    # once the extra IS installed, loading or running the real model can
    # still fail for other reasons (offline, missing weights, OOM) --
    # apply_to_result must never let that escape as a crash either
    from ai_oversight.schemas import ExtractionResult

    def broken_vision_fn(image_bytes: bytes, task: str) -> str:
        raise RuntimeError("out of memory")

    doc = ParsedDocument(document_id="sha256:x", filename=SAMPLE_PDF.name, format="pdf", elements=[])
    result = ExtractionResult(
        document_id="sha256:x",
        source_file=SAMPLE_PDF.name,
        extracted_at="2026-01-01T00:00:00Z",
        fields={},
    )

    updated = granite_vision.apply_to_result(SAMPLE_PDF, doc, result, vision_fn=broken_vision_fn)

    assert updated.metrics == []
    assert any("vision" in w.lower() for w in updated.extraction_warnings)


def test_apply_to_result_adds_a_warning_instead_of_crashing_when_model_unavailable():
    from ai_oversight.schemas import ExtractionResult

    doc = ParsedDocument(document_id="sha256:x", filename=SAMPLE_PDF.name, format="pdf", elements=[])
    result = ExtractionResult(
        document_id="sha256:x",
        source_file=SAMPLE_PDF.name,
        extracted_at="2026-01-01T00:00:00Z",
        fields={},
    )

    updated = granite_vision.apply_to_result(SAMPLE_PDF, doc, result)

    assert updated.metrics == []
    assert any("vision" in w.lower() for w in updated.extraction_warnings)


def test_run_vision_pass_targets_a_fully_scanned_pdf_page_with_no_chart(tmp_path):
    # a page with no text layer and no vector-drawn chart (a scanned page,
    # e.g. the whole page rendered as one raster image) must still be routed
    # to vision -- it just can't be routed via the gridline-chart heuristic,
    # since there's no vector content to detect at all
    import io

    import pymupdf
    from PIL import Image

    pdf_path = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    png_bytes = io.BytesIO()
    Image.new("RGB", (100, 100), color="red").save(png_bytes, format="PNG")
    page.insert_image(pymupdf.Rect(0, 0, 100, 100), stream=png_bytes.getvalue())
    doc.save(pdf_path)
    doc.close()

    def fake_vision(image_bytes: bytes, task: str) -> str:
        assert task == "<tables_json>"
        return '[{"Field": "Submission ID", "Value": "AI-1"}]'

    parsed_doc = ParsedDocument(document_id="sha256:x", filename="scanned.pdf", format="pdf", elements=[])
    observations = granite_vision.run_vision_pass(pdf_path, parsed_doc, fake_vision)

    assert len(observations) == 1
    assert observations[0].task == "tables_json"
    assert observations[0].page_or_slide == 1
    assert observations[0].parsed == [{"Field": "Submission ID", "Value": "AI-1"}]


def test_run_vision_pass_does_not_double_route_a_page_that_has_both_a_chart_and_an_image():
    # a page with a detected vector chart is routed via chart2csv already;
    # it must not ALSO be routed again as a "scanned page" just because it
    # happens to have an unrelated image (e.g. a logo) elsewhere on it
    calls = []

    def fake_vision(image_bytes: bytes, task: str) -> str:
        calls.append(task)
        return "Month,Precision\nJul,0.89\n"

    doc = ParsedDocument(document_id="sha256:x", filename=SAMPLE_PDF.name, format="pdf", elements=[])
    observations = granite_vision.run_vision_pass(SAMPLE_PDF, doc, fake_vision)

    # page 3 (the chart page) also has no images, so this just re-confirms
    # exactly one observation and no accidental double-routing
    assert len(observations) == 1
    assert calls == ["<chart2csv>"]


def test_run_vision_pass_targets_pptx_picture_shapes(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    import io

    from PIL import Image

    png_bytes = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(png_bytes, format="PNG")
    png_bytes.seek(0)
    slide.shapes.add_picture(png_bytes, Inches(1), Inches(1), Inches(2), Inches(2))
    pptx_path = tmp_path / "with_picture.pptx"
    prs.save(pptx_path)

    def fake_vision(image_bytes: bytes, task: str) -> str:
        return '[{"a": "1"}]'

    doc = ParsedDocument(document_id="sha256:x", filename="with_picture.pptx", format="pptx", elements=[])
    observations = granite_vision.run_vision_pass(pptx_path, doc, fake_vision)

    assert len(observations) == 1
    assert observations[0].task == "tables_json"
    assert observations[0].page_or_slide == 1
