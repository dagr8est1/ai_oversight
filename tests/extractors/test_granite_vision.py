from pathlib import Path

import pytest
from pptx import Presentation
from pptx.util import Inches

from ai_oversight.extractors import granite_vision
from ai_oversight.parsers.document_ir import ParsedDocument

SAMPLE_PDF = Path(__file__).parents[2] / "data" / "samples" / "Sample_Internal_AI_Oversight_Report.pdf"


def _make_scanned_pdf(tmp_path) -> Path:
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
    return pdf_path


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


def test_extract_table_recovers_complete_rows_from_a_response_truncated_mid_object():
    # found via a real end-to-end run: the model emitted a stop token mid-
    # generation while writing a 16-row table's JSON, discarding all 15
    # fully-formed rows that came before the cutoff along with the one
    # incomplete row. Reproduces the real shape: a list containing one
    # table dict whose "cells" array is cut off partway through an object.
    truncated = (
        '[{"dimensions": {"rows": 0, "columns": 2}, "cells": ['
        '{"row": 1, "col": 1, "type": "header", "content": "MODEL INVENTORY ID"}, '
        '{"row": 1, "col": 2, "type": "data", "content": "INT-SET-042"}, '
        '{"row": 2, "col": 1, "type": "header", "content": "APPLICATION"}, '
        '{"row": 2, "col": 2, "type": "data", "content": "Settlement Failure Early Warning"}, '
        '{"row": 3, "col": 1, "type": "header", "content": "VALIDATION OUTCOME"'
        # cut off mid-object: no closing brace for this cell, no closing
        # brackets for "cells" or the outer object/array at all
    )

    def fake_vision(image_bytes: bytes, task: str) -> str:
        return truncated

    obs = granite_vision.extract_table(b"x", fake_vision, source_element_id="e", page_or_slide=1)

    assert obs.status == "ok"
    assert obs.note is not None
    assert "truncat" in obs.note.lower()
    cells = obs.parsed[0]["cells"]
    assert {"row": 1, "col": 2, "type": "data", "content": "INT-SET-042"} in cells
    assert {"row": 2, "col": 2, "type": "data", "content": "Settlement Failure Early Warning"} in cells
    # the incomplete row-3 cell is correctly dropped, not fabricated
    assert all(c.get("row") != 3 for c in cells)


def test_default_vision_fn_raises_friendly_error_when_vision_extra_missing(monkeypatch):
    # simulate absence via sys.modules rather than relying on the real
    # environment not having mlx_vlm/transformers installed -- this test
    # must stay deterministic whether or not the `vision` extra happens to
    # be installed (installing it, as this project's own README documents,
    # should never make this test start a real multi-GB download instead
    # of exercising the "missing dependency" error path)
    import sys

    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

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


def test_observations_to_elements_converts_chart2csv_into_a_chart_element():
    from ai_oversight.schemas import VisionObservation

    obs = VisionObservation(
        task="chart2csv",
        source_element_id="page3.vision_crop",
        page_or_slide=3,
        raw_response="Month,Precision\nJul,0.89\nAug,0.90\n",
        parsed=[{"Month": "Jul", "Precision": "0.89"}, {"Month": "Aug", "Precision": "0.90"}],
        status="ok",
    )

    elements = granite_vision.observations_to_elements([obs])

    assert len(elements) == 1
    assert elements[0].type == "chart"
    assert elements[0].page_or_slide == 3
    assert elements[0].content["categories"] == ["Jul", "Aug"]
    assert elements[0].content["series"] == [{"name": "Precision", "values": [0.89, 0.90]}]


def test_observations_to_elements_converts_a_two_column_table_into_key_value_elements():
    from ai_oversight.schemas import VisionObservation

    obs = VisionObservation(
        task="tables_json",
        source_element_id="slide2.3",
        page_or_slide=2,
        raw_response="...",
        parsed=[
            {
                "cells": [
                    {"row": 1, "col": 1, "type": "header", "content": "APPLICATION ID"},
                    {"row": 1, "col": 2, "type": "data", "content": "EXT-MRG-2026-031"},
                    {"row": 2, "col": 1, "type": "header", "content": "BUSINESS OWNER"},
                    {"row": 2, "col": 2, "type": "data", "content": "Elena Ortiz"},
                ]
            }
        ],
        status="ok",
    )

    elements = granite_vision.observations_to_elements([obs])

    assert len(elements) == 2
    assert all(el.type == "key_value" for el in elements)
    assert all(el.page_or_slide == 2 for el in elements)
    contents = {el.content["label"]: el.content["value"] for el in elements}
    assert contents == {"APPLICATION ID": "EXT-MRG-2026-031", "BUSINESS OWNER": "Elena Ortiz"}


def test_observations_to_elements_converts_a_wide_table_into_a_single_table_element():
    from ai_oversight.schemas import VisionObservation

    obs = VisionObservation(
        task="tables_json",
        source_element_id="page4.vision_scan",
        page_or_slide=4,
        raw_response="...",
        parsed=[
            {
                "cells": [
                    {"row": 1, "col": 1, "type": "header", "content": "CONTROL ID"},
                    {"row": 1, "col": 2, "type": "header", "content": "OWNER"},
                    {"row": 1, "col": 3, "type": "header", "content": "STATE"},
                    {"row": 2, "col": 1, "type": "data", "content": "C-01"},
                    {"row": 2, "col": 2, "type": "data", "content": "Ops"},
                    {"row": 2, "col": 3, "type": "data", "content": "ACTIVE"},
                    {"row": 3, "col": 1, "type": "data", "content": "C-02"},
                    {"row": 3, "col": 2, "type": "data", "content": "InfoSec"},
                    {"row": 3, "col": 3, "type": "data", "content": "ACTIVE"},
                ]
            }
        ],
        status="ok",
    )

    elements = granite_vision.observations_to_elements([obs])

    # multiple real instance rows -- still one generic, cautioned table
    # element (see test_wide_table_with_exactly_one_data_row_* for the
    # single-instance case, which this is deliberately not)
    assert len(elements) == 1
    assert elements[0].type == "table"
    assert elements[0].content["rows"] == [
        ["CONTROL ID", "OWNER", "STATE"],
        ["C-01", "Ops", "ACTIVE"],
        ["C-02", "InfoSec", "ACTIVE"],
    ]


def test_observations_to_elements_handles_a_bare_dict_not_wrapped_in_a_list():
    from ai_oversight.schemas import VisionObservation

    obs = VisionObservation(
        task="tables_json",
        source_element_id="slide1.3",
        page_or_slide=1,
        raw_response="...",
        parsed={
            "cells": [
                {"row": 1, "col": 1, "type": "header", "content": "STATUS"},
                {"row": 1, "col": 2, "type": "data", "content": "PILOT"},
            ]
        },
        status="ok",
    )

    elements = granite_vision.observations_to_elements([obs])

    assert len(elements) == 1
    assert elements[0].content == {"label": "STATUS", "value": "PILOT"}


def test_observations_to_elements_skips_needs_review_observations():
    from ai_oversight.schemas import VisionObservation

    obs = VisionObservation(
        task="tables_json",
        source_element_id="page1.vision_scan",
        page_or_slide=1,
        raw_response="garbage",
        parsed=None,
        status="needs_review",
        note="JSON parse error",
    )

    assert granite_vision.observations_to_elements([obs]) == []


def test_enrich_document_merges_vision_elements_and_retains_observations(tmp_path):
    pdf_path = _make_scanned_pdf(tmp_path)

    def fake_vision(image_bytes: bytes, task: str) -> str:
        return '[{"cells": [{"row": 1, "col": 1, "type": "header", "content": "STATUS"}, {"row": 1, "col": 2, "type": "data", "content": "PILOT"}]}]'

    doc = ParsedDocument(document_id="sha256:x", filename=pdf_path.name, format="pdf", elements=[])

    enriched, observations, warning = granite_vision.enrich_document(pdf_path, doc, vision_fn=fake_vision)

    assert warning is None
    assert len(observations) == 1
    assert observations[0].status == "ok"
    kv_elements = [el for el in enriched.elements if el.type == "key_value"]
    assert len(kv_elements) == 1
    assert kv_elements[0].content == {"label": "STATUS", "value": "PILOT"}
    # the original document's own elements are preserved, not replaced
    assert enriched.document_id == doc.document_id


def test_enrich_document_degrades_to_a_warning_on_any_model_failure_not_just_missing_extra():
    def broken_vision_fn(image_bytes: bytes, task: str) -> str:
        raise RuntimeError("out of memory")

    doc = ParsedDocument(document_id="sha256:x", filename=SAMPLE_PDF.name, format="pdf", elements=[])

    enriched, observations, warning = granite_vision.enrich_document(
        SAMPLE_PDF, doc, vision_fn=broken_vision_fn
    )

    assert enriched is doc
    assert observations == []
    assert warning is not None
    assert "vision" in warning.lower()


def test_enriched_document_lets_qwen_local_map_a_vision_recovered_label_to_a_canonical_field(
    tmp_path,
):
    # the actual bridge: a vision-recovered label/value pair becomes a
    # normal key_value element, and qwen_local maps it to a canonical field
    # using its existing chunk-and-map logic -- no vision-specific code in
    # qwen_local itself
    from ai_oversight.extractors import qwen_local

    pdf_path = _make_scanned_pdf(tmp_path)

    def fake_vision(image_bytes: bytes, task: str) -> str:
        return (
            '[{"cells": ['
            '{"row": 1, "col": 1, "type": "header", "content": "APPLICATION ID"}, '
            '{"row": 1, "col": 2, "type": "data", "content": "EXT-MRG-2026-031"}'
            "]}]"
        )

    doc = ParsedDocument(document_id="sha256:x", filename=pdf_path.name, format="pdf", elements=[])
    enriched, _observations, _warning = granite_vision.enrich_document(pdf_path, doc, vision_fn=fake_vision)

    kv_element = next(el for el in enriched.elements if el.type == "key_value")

    def fake_generate(prompt: str) -> str:
        assert "APPLICATION ID" in prompt
        assert kv_element.element_id in prompt
        return (
            f'{{"submission_id": {{"value": "EXT-MRG-2026-031", '
            f'"element_id": "{kv_element.element_id}"}}}}'
        )

    result = qwen_local.extract(enriched, generate_fn=fake_generate)

    field = result.fields["submission_id"]
    assert field.value == "EXT-MRG-2026-031"
    assert field.status == "extracted"
    assert field.evidence[0].element_id == kv_element.element_id


def test_enrich_document_degrades_to_a_warning_when_model_unavailable(monkeypatch):
    # simulate absence via sys.modules so this stays deterministic
    # regardless of what's actually installed, rather than relying on a
    # real missing package
    import sys

    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    doc = ParsedDocument(document_id="sha256:x", filename=SAMPLE_PDF.name, format="pdf", elements=[])

    enriched, observations, warning = granite_vision.enrich_document(SAMPLE_PDF, doc)

    assert enriched is doc
    assert observations == []
    assert warning is not None
    assert "vision" in warning.lower()


def test_run_vision_pass_targets_a_fully_scanned_pdf_page_with_no_chart(tmp_path):
    # a page with no text layer and no vector-drawn chart (a scanned page,
    # e.g. the whole page rendered as one raster image) must still be routed
    # to vision -- it just can't be routed via the gridline-chart heuristic,
    # since there's no vector content to detect at all
    pdf_path = _make_scanned_pdf(tmp_path)

    def fake_vision(image_bytes: bytes, task: str) -> str:
        assert task == "<tables_json>"
        return '[{"Field": "Submission ID", "Value": "AI-1"}]'

    parsed_doc = ParsedDocument(document_id="sha256:x", filename="scanned.pdf", format="pdf", elements=[])
    observations = granite_vision.run_vision_pass(pdf_path, parsed_doc, fake_vision)

    assert len(observations) == 1
    assert observations[0].task == "tables_json"
    assert observations[0].page_or_slide == 1
    assert observations[0].parsed == [{"Field": "Submission ID", "Value": "AI-1"}]


def test_run_vision_pass_splits_a_scanned_page_after_a_truncated_whole_page_response(tmp_path):
    # the whole-page response is missing the second row's closing brace and
    # the outer closing bracket -- the exact shape _repair_truncated_json
    # recovers by keeping only the first (complete) row and dropping the
    # second, which is the "partial" condition that should trigger a split
    # retry into top/bottom halves instead of accepting the dropped row
    pdf_path = _make_scanned_pdf(tmp_path)
    calls = []

    def fake_vision(image_bytes: bytes, task: str) -> str:
        calls.append(task)
        if len(calls) == 1:
            return '[{"Field": "A", "Value": "1"}, {"Field": "B", "Value": "2"'
        if len(calls) == 2:
            return '[{"Field": "A", "Value": "1"}]'
        return '[{"Field": "B", "Value": "2"}]'

    parsed_doc = ParsedDocument(document_id="sha256:x", filename="scanned.pdf", format="pdf", elements=[])

    observations = granite_vision.run_vision_pass(pdf_path, parsed_doc, fake_vision)

    assert len(calls) == 3  # one whole-page attempt, then two half-page retries
    assert len(observations) == 2
    assert all(obs.status == "ok" and obs.note is None for obs in observations)
    assert {obs.source_element_id for obs in observations} == {
        "page1.vision_scan.top",
        "page1.vision_scan.bottom",
    }


def test_run_vision_pass_accepts_a_still_lossy_half_without_recursing_further(tmp_path):
    # the top half is deliberately still truncated after the first retry --
    # this is a single-level retry, not recursive splitting, so the lossy
    # half's loss should surface as-is rather than triggering a third call
    pdf_path = _make_scanned_pdf(tmp_path)
    calls = []

    def fake_vision(image_bytes: bytes, task: str) -> str:
        calls.append(task)
        if len(calls) == 1:
            return '[{"Field": "A", "Value": "1"}, {"Field": "B", "Value": "2"'
        if len(calls) == 2:
            return '[{"Field": "A", "Value": "1"}, {"Field": "C", "Value": "3"'
        return '[{"Field": "B", "Value": "2"}]'

    parsed_doc = ParsedDocument(document_id="sha256:x", filename="scanned.pdf", format="pdf", elements=[])

    observations = granite_vision.run_vision_pass(pdf_path, parsed_doc, fake_vision)

    assert len(calls) == 3  # no further recursive splitting attempted
    by_id = {obs.source_element_id: obs for obs in observations}
    assert by_id["page1.vision_scan.top"].note is not None
    assert by_id["page1.vision_scan.bottom"].status == "ok"
    assert by_id["page1.vision_scan.bottom"].note is None


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


def test_save_and_load_observations_round_trips(tmp_path):
    from ai_oversight.schemas import VisionObservation

    observations = [
        VisionObservation(
            task="tables_json",
            source_element_id="page2.vision_scan",
            page_or_slide=2,
            raw_response="...",
            parsed=[{"cells": [{"row": 1, "col": 1, "content": "X"}]}],
            status="ok",
        ),
        VisionObservation(
            task="chart2csv",
            source_element_id="page3.vision_crop",
            page_or_slide=3,
            raw_response="garbage",
            parsed=None,
            status="needs_review",
            note="no CSV rows parsed from model output",
        ),
    ]
    path = tmp_path / "vision_observations.json"

    granite_vision.save_observations(observations, path)
    loaded = granite_vision.load_observations(path)

    assert loaded == observations


def test_enrich_document_from_observations_merges_elements_without_running_vision():
    from ai_oversight.schemas import VisionObservation

    doc = ParsedDocument(document_id="sha256:x", filename="x.pdf", format="pdf", elements=[])
    observations = [
        VisionObservation(
            task="tables_json",
            source_element_id="slide2.3",
            page_or_slide=2,
            raw_response="...",
            parsed=[
                {
                    "cells": [
                        {"row": 1, "col": 1, "type": "header", "content": "STATUS"},
                        {"row": 1, "col": 2, "type": "data", "content": "PILOT"},
                    ]
                }
            ],
            status="ok",
        )
    ]

    enriched = granite_vision.enrich_document_from_observations(doc, observations)

    kv_elements = [el for el in enriched.elements if el.type == "key_value"]
    assert len(kv_elements) == 1
    assert kv_elements[0].content == {"label": "STATUS", "value": "PILOT"}


def _cell(row, col, content, colspan=1):
    return {"row": row, "col": col, "colspan": colspan, "content": content}


def test_wide_table_with_exactly_one_data_row_becomes_key_value_elements_per_column():
    from ai_oversight.schemas import VisionObservation

    obs = VisionObservation(
        task="tables_json",
        source_element_id="page3.vision_scan",
        page_or_slide=3,
        raw_response="...",
        parsed=[
            {
                "cells": [
                    _cell(1, 1, "IMPACT AREA"),
                    _cell(1, 2, "EXPOSURE"),
                    _cell(1, 3, "RESIDUAL RISK"),
                    _cell(2, 1, "Settlement timeliness"),
                    _cell(2, 2, "HIGH"),
                    _cell(2, 3, "MEDIUM"),
                ]
            }
        ],
        status="ok",
    )

    elements = granite_vision.observations_to_elements([obs])

    assert all(el.type == "key_value" for el in elements)
    contents = {el.content["label"]: el.content["value"] for el in elements}
    assert contents == {
        "IMPACT AREA": "Settlement timeliness",
        "EXPOSURE": "HIGH",
        "RESIDUAL RISK": "MEDIUM",
    }


def test_wide_table_extracts_a_narrower_summary_row_separately_from_repeated_rows():
    from ai_oversight.schemas import VisionObservation

    obs = VisionObservation(
        task="tables_json",
        source_element_id="page2.vision_scan",
        page_or_slide=2,
        raw_response="...",
        parsed=[
            {
                "cells": [
                    _cell(1, 1, "PREDICTED FAILURE"),
                    _cell(1, 2, "Yes"),
                    _cell(1, 3, "No"),
                    _cell(2, 1, "Actual failure"),
                    _cell(2, 2, "174"),
                    _cell(2, 3, "26"),
                    _cell(3, 1, "Actual success"),
                    _cell(3, 2, "17"),
                    _cell(3, 3, "783"),
                    _cell(4, 1, "n=1,000 records | Precision 91.1% | Recall 87.0%", colspan=3),
                ]
            }
        ],
        status="ok",
    )

    elements = granite_vision.observations_to_elements([obs])

    text_elements = [el for el in elements if el.type == "text"]
    table_elements = [el for el in elements if el.type == "table"]
    assert len(text_elements) == 1
    assert "Precision 91.1%" in text_elements[0].raw_text
    assert len(table_elements) == 1
    assert table_elements[0].content["rows"] == [
        ["PREDICTED FAILURE", "Yes", "No"],
        ["Actual failure", "174", "26"],
        ["Actual success", "17", "783"],
    ]


def test_wide_table_with_multiple_full_rows_and_no_summary_row_is_unchanged():
    from ai_oversight.schemas import VisionObservation

    obs = VisionObservation(
        task="tables_json",
        source_element_id="page4.vision_scan",
        page_or_slide=4,
        raw_response="...",
        parsed=[
            {
                "cells": [
                    _cell(1, 1, "CONTROLID"),
                    _cell(1, 2, "MITIGATION"),
                    _cell(1, 3, "OWNER"),
                    _cell(2, 1, "C-01"),
                    _cell(2, 2, "Human authorization"),
                    _cell(2, 3, "Operations Lead"),
                    _cell(3, 1, "C-02"),
                    _cell(3, 2, "Input data validation"),
                    _cell(3, 3, "Data Engineering"),
                ]
            }
        ],
        status="ok",
    )

    elements = granite_vision.observations_to_elements([obs])

    assert len(elements) == 1
    assert elements[0].type == "table"
    assert elements[0].content["rows"] == [
        ["CONTROLID", "MITIGATION", "OWNER"],
        ["C-01", "Human authorization", "Operations Lead"],
        ["C-02", "Input data validation", "Data Engineering"],
    ]


def test_build_mlx_vision_fn_overrides_the_default_100_token_cap():
    # found via a real end-to-end run: mlx_vlm.generate's own default is
    # max_tokens=100 (confirmed from its docstring), which truncated a
    # genuine 16-row inventory table mid-JSON on page 1 of the real PDF --
    # the whole response was discarded since it never became valid JSON,
    # even though most of the real data was right there in the raw text.
    import io

    from PIL import Image

    from ai_oversight.config import Settings

    calls = []

    class _FakeUtils:
        @staticmethod
        def load_config(repo):
            return {"fake": "config"}

    class _FakeMlxVlm:
        utils = _FakeUtils

        @staticmethod
        def load(repo):
            return "fake-model", "fake-processor"

        @staticmethod
        def apply_chat_template(processor, config, task, num_images):
            return "fake-prompt"

        @staticmethod
        def generate(model, processor, prompt, images, verbose=False, **kwargs):
            calls.append(kwargs)

            class _Result:
                text = "ok"

            return _Result()

    vision_fn = granite_vision._build_mlx_vision_fn(Settings(), _FakeMlxVlm)

    png_bytes = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(png_bytes, format="PNG")
    vision_fn(png_bytes.getvalue(), "<tables_json>")

    assert len(calls) == 1
    assert calls[0].get("max_tokens", 100) >= 2048
