import pytest
from pydantic import ValidationError

from ai_oversight.parsers.document_ir import Element, ParsedDocument


def test_parsed_document_round_trips_through_json():
    doc = ParsedDocument(
        document_id="sha256:abc",
        filename="example.pdf",
        format="pdf",
        elements=[
            Element(
                element_id="page2.kv1",
                type="key_value",
                page_or_slide=2,
                bbox=[39.0, 147.6, 113.3, 154.7],
                method="pymupdf_text_block_pair",
                content={"label": "Submission ID", "value": "AI-INT-2026-041"},
                raw_text="AI-INT-2026-041",
            )
        ],
        warnings=[],
    )

    restored = ParsedDocument.model_validate_json(doc.model_dump_json())

    assert restored == doc
    assert restored.elements[0].content["label"] == "Submission ID"


def test_element_type_is_restricted_to_known_kinds():
    with pytest.raises(ValidationError):
        Element(
            element_id="x",
            type="not_a_real_type",
            page_or_slide=1,
            method="m",
        )
