import pytest

from ai_oversight.extractors import qwen_local
from ai_oversight.parsers.document_ir import Element, ParsedDocument


def _doc_with_one_page(text: str) -> ParsedDocument:
    return ParsedDocument(
        document_id="sha256:x",
        filename="x.pdf",
        format="pdf",
        elements=[
            Element(
                element_id="page1.text0",
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
        calls.append(prompt)
        return '{"submission_id": "AI-INT-2026-041", "application_name": null}'

    doc = _doc_with_one_page("01 / Submission ID\nAI-INT-2026-041")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert len(calls) == 1
    assert result.fields["submission_id"].value == "AI-INT-2026-041"
    assert result.fields["submission_id"].status == "extracted"
    assert result.fields["submission_id"].evidence[0].method == "qwen_local"
    # a field the model reported as null stays missing, not fabricated
    assert result.fields["application_name"].status == "missing"


def test_extract_retries_once_on_invalid_json_then_succeeds():
    responses = iter(["not valid json at all", '{"submission_id": "AI-INT-2026-041"}'])

    def fake_generate(prompt: str) -> str:
        return next(responses)

    doc = _doc_with_one_page("01 / Submission ID\nAI-INT-2026-041")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.fields["submission_id"].value == "AI-INT-2026-041"
    assert result.fields["submission_id"].status == "extracted"


def test_extract_falls_back_to_needs_review_without_fabricating(monkeypatch):
    monkeypatch.setattr(qwen_local, "MAX_ATTEMPTS", 2)

    def fake_generate(prompt: str) -> str:
        return "the model rambles and never produces JSON"

    doc = _doc_with_one_page("01 / Submission ID\nAI-INT-2026-041")
    result = qwen_local.extract(doc, generate_fn=fake_generate)

    assert result.fields["submission_id"].status == "missing"
    assert result.fields["submission_id"].value is None
    assert any("needs_review" in w for w in result.extraction_warnings)


def test_extract_flags_conflicting_values_across_chunks():
    responses = iter(['{"model_type": "XGBoost"}', '{"model_type": "Random Forest"}'])

    def fake_generate(prompt: str) -> str:
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


def test_default_generate_fn_raises_friendly_error_when_llm_extra_missing():
    doc = _doc_with_one_page("some text")

    with pytest.raises(qwen_local.ModelNotAvailableError, match="llm"):
        qwen_local.extract(doc)
