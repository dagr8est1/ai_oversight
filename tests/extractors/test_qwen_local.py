import pytest

from ai_oversight.extractors import qwen_local
from ai_oversight.parsers.document_ir import Element, ParsedDocument


def _doc_with_one_page(text: str, element_id: str = "page1.text0") -> ParsedDocument:
    return ParsedDocument(
        document_id="sha256:x",
        filename="Sample_Internal_AI_Oversight_Report.pdf",
        format="pdf",
        elements=[
            Element(
                element_id=element_id,
                type="text",
                page_or_slide=1,
                method="pymupdf_text_block",
                raw_text=text,
            )
        ],
    )


def test_extract_uses_valid_first_response_without_retrying():
    calls = []

    def fake_generate(prompt: str) -> str:
        if "ONE specific structured field" in prompt:
            return '{"value": null}'  # second-pass call for some other missing field
        calls.append(prompt)
        return '{"submission_id": {"value": "AI-INT-2026-041", "element_id": "page1.text0"}}'

    doc = _doc_with_one_page("01 / Submission ID\nAI-INT-2026-041")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert len(calls) == 1
    field = result.fields["submission_id"]
    assert field.value == "AI-INT-2026-041"
    assert field.status == "extracted"
    assert field.evidence[0].method == "qwen_local"
    assert field.evidence[0].element_id == "page1.text0"


def test_extract_retries_once_on_invalid_json_then_succeeds():
    responses = iter(
        [
            "not valid json at all",
            '{"submission_id": {"value": "AI-INT-2026-041", "element_id": "page1.text0"}}',
        ]
    )

    def fake_generate(prompt: str) -> str:
        if "ONE specific structured field" in prompt:
            return '{"value": null}'  # second-pass call for some other missing field
        return next(responses)

    doc = _doc_with_one_page("01 / Submission ID\nAI-INT-2026-041")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.fields["submission_id"].value == "AI-INT-2026-041"
    assert result.fields["submission_id"].status == "extracted"


def test_a_cited_element_id_outside_the_chunk_falls_back_to_needs_review():
    # a made-up element_id means the model's citation can't be verified at
    # all -- PLAN.md Section 9 requires every extracted field to point to a
    # specific source element, so this can never be a confident "extracted",
    # independent of whether the value itself happens to be right
    def fake_generate(prompt: str) -> str:
        return '{"submission_id": {"value": "AI-INT-2026-041", "element_id": "slide99.made_up"}}'

    doc = _doc_with_one_page("01 / Submission ID\nAI-INT-2026-041", element_id="page1.text0")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    field = result.fields["submission_id"]
    assert field.value == "AI-INT-2026-041"
    assert field.status == "needs_review"
    # falls back to a chunk-level citation rather than trusting a made-up id
    assert field.evidence[0].element_id == "page1.text0"


def test_extract_falls_back_to_needs_review_and_keeps_raw_response_as_evidence():
    def fake_generate(prompt: str) -> str:
        return "the model rambles and never produces JSON"

    doc = _doc_with_one_page("01 / Submission ID\nAI-INT-2026-041")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    field = result.fields["submission_id"]
    assert field.status == "needs_review"
    assert field.value is None
    assert len(field.evidence) == 1
    assert field.evidence[0].method == "qwen_local_unvalidated"
    assert "never produces JSON" in field.evidence[0].quote
    assert any("needs_review" in w for w in result.extraction_warnings)


def test_a_value_not_actually_present_in_its_cited_evidence_is_needs_review():
    # the model cites a real element, but the value it reports isn't
    # actually in that element's text -- accepting it as "extracted" would
    # make an unsubstantiated claim look verified
    def fake_generate(prompt: str) -> str:
        return '{"submission_id": {"value": "TOTALLY-MADE-UP", "element_id": "page1.text0"}}'

    doc = _doc_with_one_page("01 / Submission ID: AI-INT-2026-041", element_id="page1.text0")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    field = result.fields["submission_id"]
    assert field.status == "needs_review"
    assert field.value == "TOTALLY-MADE-UP"
    assert field.evidence[0].element_id == "page1.text0"


def test_a_value_actually_present_in_its_cited_evidence_is_extracted():
    def fake_generate(prompt: str) -> str:
        return '{"submission_id": {"value": "AI-INT-2026-041", "element_id": "page1.text0"}}'

    doc = _doc_with_one_page("01 / Submission ID: AI-INT-2026-041", element_id="page1.text0")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.fields["submission_id"].status == "extracted"


def test_an_explicit_null_value_is_ambiguous_not_missing():
    def fake_generate(prompt: str) -> str:
        return '{"business_owner": {"value": null, "element_id": "page1.text0"}}'

    doc = _doc_with_one_page("ownership unclear from this excerpt")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.fields["business_owner"].status == "ambiguous"
    assert result.fields["business_owner"].value is None


def test_extract_flags_conflicting_values_across_chunks():
    responses = iter(
        [
            '{"model_type": {"value": "XGBoost", "element_id": "page1.text0"}}',
            '{"model_type": {"value": "Random Forest", "element_id": "page2.text0"}}',
        ]
    )

    def fake_generate(prompt: str) -> str:
        if "ONE specific structured field" in prompt:
            return '{"value": null}'  # second-pass call for some other missing field
        return next(responses)

    doc = ParsedDocument(
        document_id="sha256:x",
        filename="x.pdf",
        format="pdf",
        elements=[
            Element(element_id="page1.text0", type="text", page_or_slide=1, method="m", raw_text="model info A"),
            Element(element_id="page2.text0", type="text", page_or_slide=2, method="m", raw_text="model info B"),
        ],
    )

    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.fields["model_type"].status == "conflicting"
    assert len(result.fields["model_type"].evidence) == 2
    assert any(flag.field == "model_type" for flag in result.review_flags)


def test_model_performance_value_is_normalized_and_shape_validated():
    def fake_generate(prompt: str) -> str:
        return '{"model_performance": {"value": "Accuracy 92%", "element_id": "page1.text0"}}'

    doc = _doc_with_one_page("Accuracy 92%")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    field = result.fields["model_performance"]
    assert field.value == {"accuracy": 0.92}
    assert field.status == "extracted"


def test_prompt_includes_each_canonical_field_s_description_not_just_its_name():
    # found via a real end-to-end run: given a key_value element labeled
    # "RESIDUAL RISK", Qwen invented a JSON key "residual_risk" (echoing
    # the source label) instead of picking "risk_rating" from the
    # canonical list -- a bare snake_case name gave it nothing to match
    # "residual risk" against. Each field's PLAN.md description should
    # give the model a real chance at that association.
    captured_prompts = []

    def fake_generate(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "{}"

    doc = _doc_with_one_page("some excerpt")
    qwen_local.extract(doc, generate_fn=fake_generate)

    prompt = captured_prompts[0]
    assert "risk_rating" in prompt
    assert "inherent and residual" in prompt.lower()


def test_prompt_warns_that_a_wide_table_is_a_record_list_not_a_field_value():
    # found via a real end-to-end run: Qwen pulled individual MITIGATION
    # cells out of a 5-row control-matrix table and confidently assigned
    # each one to a different, unrelated canonical field (e.g. one
    # control's mitigation text became "human_oversight"). The prompt must
    # warn against treating any one row/cell of a multi-record table as a
    # document-level fact.
    captured_prompts = []

    def fake_generate(prompt: str) -> str:
        if "ONE specific structured field" in prompt:
            return '{"value": null}'  # second-pass calls for the other missing fields
        captured_prompts.append(prompt)
        return "{}"

    doc = ParsedDocument(
        document_id="sha256:x",
        filename="x.pdf",
        format="pdf",
        elements=[
            Element(
                element_id="page4.table0",
                type="table",
                page_or_slide=4,
                method="granite_vision_tables_json",
                content={
                    "rows": [
                        ["CONTROLID", "MITIGATION", "OWNER", "STATE"],
                        ["C-01", "Human authorization", "Operations Lead", "ACTIVE"],
                    ]
                },
            )
        ],
    )

    qwen_local.extract(doc, generate_fn=fake_generate)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "record" in prompt.lower()
    assert "Human authorization" in prompt  # the content itself is still included


def test_the_table_caution_instruction_does_not_break_plain_field_extraction():
    # the instruction is a general, always-true rule (harmless on a chunk
    # with no table), not conditional on one being present -- what matters
    # is that normal single-fact extraction still works unaffected by it
    def fake_generate(prompt: str) -> str:
        return '{"submission_id": {"value": "AI-1", "element_id": "page1.text0"}}'

    doc = _doc_with_one_page("01 / Submission ID\nAI-1")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.fields["submission_id"].value == "AI-1"
    assert result.fields["submission_id"].status == "extracted"


def test_a_value_cited_from_a_table_type_element_is_never_confidently_extracted():
    # found via a real end-to-end run: despite the prompt's caution
    # instruction, the model still cited a `table`-type element (a
    # 5-row control matrix) for "fallback_process", picking out one
    # control's mitigation text ("Manual fallback drill") because the
    # field's own description shares words with it. A `table`-type
    # element is, by construction, only ever created when there's more
    # than one real repeated-instance row -- there's no legitimate case
    # where citing one is a confident, unambiguous document-level fact.
    # This must be enforced structurally, not left to the model actually
    # following an instruction it can choose to ignore.
    def fake_generate(prompt: str) -> str:
        return (
            '{"fallback_process": {"value": "Manual fallback drill", '
            '"element_id": "page4.table0"}}'
        )

    doc = ParsedDocument(
        document_id="sha256:x",
        filename="x.pdf",
        format="pdf",
        elements=[
            Element(
                element_id="page4.table0",
                type="table",
                page_or_slide=4,
                method="granite_vision_tables_json",
                content={
                    "rows": [
                        ["CONTROLID", "MITIGATION"],
                        ["C-01", "Human authorization"],
                        ["C-02", "Manual fallback drill"],
                    ]
                },
            )
        ],
    )

    result = qwen_local.extract(doc, generate_fn=fake_generate)

    field = result.fields["fallback_process"]
    assert field.value == "Manual fallback drill"
    assert field.status == "needs_review"


def test_a_page_with_many_candidates_is_split_into_smaller_chunks():
    # found via a real end-to-end run: a page with ~15 recovered key_value
    # candidates asked the model to match all of them against all 22
    # canonical fields in a single generation, and it missed some
    # otherwise-clear matches (e.g. "MODEL INVENTORY ID" never got mapped
    # to submission_id at all). Splitting one page into several smaller
    # chunks keeps each matching decision more tractable.
    captured_prompts = []

    def fake_generate(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "{}"

    elements = [
        Element(
            element_id=f"page1.kv{i}",
            type="key_value",
            page_or_slide=1,
            method="m",
            content={"label": f"LABEL {i}", "value": f"VALUE {i}"},
        )
        for i in range(12)
    ]
    doc = ParsedDocument(document_id="sha256:x", filename="x.pdf", format="pdf", elements=elements)

    qwen_local.extract(doc, generate_fn=fake_generate)

    assert len(captured_prompts) > 1
    # each individual chunk's prompt has only a subset of the 12 labels,
    # not every one of them crammed into a single call
    assert not all(f"LABEL {i}" in captured_prompts[0] for i in range(12))


def test_second_pass_recovers_a_field_the_main_pass_missed():
    # found via a real end-to-end run: `submission_id` stayed `missing`
    # even after the main pass was chunked down to 5 elements per call --
    # it was still competing for output budget against every other field
    # mentioned in its chunk. A second, single-field-focused pass over the
    # whole page should be able to recover it.
    calls = []

    def fake_generate(prompt: str) -> str:
        calls.append(prompt)
        if "ONE specific structured field" in prompt:
            if "field is: submission_id" in prompt:
                return '{"value": "AI-INT-2026-041", "element_id": "page1.text0"}'
            return '{"value": null}'
        return "{}"  # the main pass misses every field on this chunk

    doc = _doc_with_one_page("01 / Submission ID\nAI-INT-2026-041")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    field = result.fields["submission_id"]
    assert field.value == "AI-INT-2026-041"
    assert field.status == "extracted"
    assert len(calls) > 1


def test_second_pass_skips_fields_the_main_pass_already_extracted():
    captured_second_pass_prompts = []

    def fake_generate(prompt: str) -> str:
        if "ONE specific structured field" in prompt:
            captured_second_pass_prompts.append(prompt)
            return '{"value": null}'
        return '{"submission_id": {"value": "AI-1", "element_id": "page1.text0"}}'

    doc = _doc_with_one_page("01 / Submission ID\nAI-1")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.fields["submission_id"].status == "extracted"
    assert not any("field is: submission_id" in p for p in captured_second_pass_prompts)


def test_second_pass_leaves_a_field_missing_when_no_page_states_it():
    def fake_generate(prompt: str) -> str:
        if "ONE specific structured field" in prompt:
            return '{"value": null}'
        return "{}"

    doc = _doc_with_one_page("nothing relevant here")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.fields["submission_id"].status == "missing"


def test_second_pass_citation_from_a_table_type_element_is_not_confidently_extracted():
    # the structural table-citation check must apply equally in the
    # second pass, not just the main one
    def fake_generate(prompt: str) -> str:
        if "ONE specific structured field" in prompt:
            if "field is: fallback_process" in prompt:
                return '{"value": "Manual fallback drill", "element_id": "page4.table0"}'
            return '{"value": null}'
        return "{}"

    doc = ParsedDocument(
        document_id="sha256:x",
        filename="x.pdf",
        format="pdf",
        elements=[
            Element(
                element_id="page4.table0",
                type="table",
                page_or_slide=4,
                method="granite_vision_tables_json",
                content={
                    "rows": [
                        ["CONTROLID", "MITIGATION"],
                        ["C-01", "Human authorization"],
                        ["C-02", "Manual fallback drill"],
                    ]
                },
            )
        ],
    )

    result = qwen_local.extract(doc, generate_fn=fake_generate)

    field = result.fields["fallback_process"]
    assert field.value == "Manual fallback drill"
    assert field.status == "needs_review"


def test_second_pass_prompt_includes_the_whole_page_not_a_5_element_slice():
    captured_second_pass_prompts = []

    def fake_generate(prompt: str) -> str:
        if "ONE specific structured field" in prompt:
            captured_second_pass_prompts.append(prompt)
            return '{"value": null}'
        return "{}"

    elements = [
        Element(
            element_id=f"page1.kv{i}",
            type="key_value",
            page_or_slide=1,
            method="m",
            content={"label": f"LABEL {i}", "value": f"VALUE {i}"},
        )
        for i in range(12)
    ]
    doc = ParsedDocument(document_id="sha256:x", filename="x.pdf", format="pdf", elements=elements)

    qwen_local.extract(doc, generate_fn=fake_generate)

    assert captured_second_pass_prompts
    assert any(all(f"LABEL {i}" in p for i in range(12)) for p in captured_second_pass_prompts)


def test_metrics_incidents_and_document_type_come_from_the_ir_not_the_model():
    def fake_generate(prompt: str) -> str:
        return "{}"

    doc = ParsedDocument(
        document_id="sha256:x",
        filename="Sample_External_AI_Usage_Disclosure.pptx",
        format="pptx",
        elements=[
            Element(
                element_id="slide5.chart0",
                type="chart",
                page_or_slide=5,
                method="pptx_native_chart",
                content={"categories": ["Aug 2026"], "series": [{"name": "Accuracy", "values": [0.9]}]},
            ),
            Element(
                element_id="slide7.table0",
                type="table",
                page_or_slide=7,
                method="pptx_aligned_textbox_row_cells",
                content={"cells": ["Sep 07", "Incident", "Resolved"]},
            ),
        ],
    )

    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.document_type == "external"
    assert len(result.metrics) == 1
    assert len(result.incidents) == 1


def test_a_truncated_outer_object_does_not_fall_back_to_its_own_inner_object():
    # a response cut off mid-stream (plausible at a fixed max_new_tokens) is
    # invalid JSON overall; the scanner must not then accept the nested
    # {"value": ..., "element_id": ...} fragment as if IT were the whole
    # answer -- none of its keys are canonical fields, so silently treating
    # it as "ok" would report every field "missing" with no warning at all.
    def fake_generate(prompt: str) -> str:
        return (
            '{"submission_id": {"value": "AI-1", "element_id": "page1.text0"}, '
            '"risk_rating": {"value": "Inherent HIGH residual MEDIUM", "element_id": "page1.text0"},'
        )

    doc = _doc_with_one_page("01 / Submission ID\nAI-1")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.fields["submission_id"].status == "needs_review"
    assert any("needs_review" in w for w in result.extraction_warnings)


def test_an_echoed_example_object_is_not_mistaken_for_the_real_answer():
    def fake_generate(prompt: str) -> str:
        return (
            'Example: {"not_a_real_field": {"value": "ignore me"}}\n'
            'Answer: {"submission_id": {"value": "AI-INT-2026-041", "element_id": "page1.text0"}}'
        )

    doc = _doc_with_one_page("01 / Submission ID\nAI-INT-2026-041")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.fields["submission_id"].value == "AI-INT-2026-041"


def test_default_generate_fn_raises_friendly_error_when_llm_extra_missing(monkeypatch):
    # simulate absence via sys.modules so this stays deterministic whether
    # or not the `llm` extra happens to be installed in this environment --
    # otherwise, once installed, this would try to load the real multi-GB
    # model instead of exercising the friendly-error path
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    doc = _doc_with_one_page("some text")

    with pytest.raises(qwen_local.ModelNotAvailableError, match="llm"):
        qwen_local.extract(doc)
