from pathlib import Path

from ai_oversight.extractors import rules
from ai_oversight.parsers import pdf_parser, pptx_parser
from ai_oversight.parsers.document_ir import Element, ParsedDocument
from ai_oversight.schemas import CANONICAL_FIELDS

SAMPLES = Path(__file__).parents[2] / "data" / "samples"
PDF_SAMPLE = SAMPLES / "Sample_Internal_AI_Oversight_Report.pdf"
PPTX_SAMPLE = SAMPLES / "Sample_External_AI_Usage_Disclosure.pptx"


def test_every_canonical_field_present_for_the_internal_pdf():
    doc = pdf_parser.parse(PDF_SAMPLE)
    result = rules.extract(doc)

    assert set(result.fields) == set(CANONICAL_FIELDS)
    assert result.document_type == "internal"


def test_representative_pdf_fields_match_section_2_of_the_plan():
    doc = pdf_parser.parse(PDF_SAMPLE)
    result = rules.extract(doc)
    fields = result.fields

    assert fields["submission_id"].value == "AI-INT-2026-041"
    assert fields["submission_id"].status == "extracted"
    assert fields["submission_id"].evidence[0].page_or_slide == 2
    assert fields["application_name"].value == "Settlement Failure Early Warning (SFEW)"
    assert fields["model_type"].value == "Gradient-boosted trees (XGBoost)"
    assert fields["risk_rating"].value == {"inherent": "HIGH", "residual": "MEDIUM"}
    assert fields["monitoring_frequency"].value == "Daily metrics; monthly drift review"
    assert fields["human_oversight"].value == "Analyst approves every escalation"
    assert fields["model_performance"].value == {
        "precision": 0.91,
        "recall": 0.86,
        "roc_auc": 0.95,
    }
    # the PDF's chart is a vector drawing, not a native chart element -- no
    # structured time series is recoverable in Phase 1/2, so metrics stays empty
    assert result.metrics == []


def test_representative_pptx_fields_match_section_2_of_the_plan():
    doc = pptx_parser.parse(PPTX_SAMPLE)
    result = rules.extract(doc)
    fields = result.fields

    assert result.document_type == "external"
    assert fields["submission_id"].value == "EXT-2026-118"
    assert fields["application_name"].value == "Collateral Exception Triage Copilot (CETC)"
    assert fields["model_type"].value == "Retrieval-augmented LLM + classifier"
    assert fields["risk_rating"].value == {"inherent": "HIGH", "residual": "MEDIUM"}
    assert fields["monitoring_frequency"].value == "Daily prompt logs; weekly QA sampling"
    assert fields["human_oversight"].value == "Analyst must approve every classification & action"
    assert fields["model_performance"].value == {
        "accuracy": 0.92,
        "grounded_response_rate": 0.97,
    }
    # consistent with Section 2's note: time periods and metric names must be
    # preserved from the native chart, not collapsed to a single number
    assert len(result.metrics) == 6
    periods = {m.period for m in result.metrics if m.name == "classification_accuracy"}
    assert periods == {"Aug 2026", "Sep 2026", "Oct 2026"}


def test_pptx_incident_table_rows_become_structured_incident_records():
    doc = pptx_parser.parse(PPTX_SAMPLE)
    result = rules.extract(doc)

    assert len(result.incidents) == 2
    dates = {incident["date"] for incident in result.incidents}
    assert dates == {"Sep 07", "Oct 02"}


def test_missing_field_is_null_with_missing_status_not_fabricated():
    doc = ParsedDocument(document_id="sha256:x", filename="x.pdf", format="pdf", elements=[])
    result = rules.extract(doc)

    assert result.fields["submission_id"].status == "missing"
    assert result.fields["submission_id"].value is None
    assert result.fields["submission_id"].evidence == []


def test_narrative_and_chart_disagreement_is_flagged_as_conflicting_not_resolved():
    doc = ParsedDocument(
        document_id="sha256:conflict",
        filename="conflict.pptx",
        format="pptx",
        elements=[
            Element(
                element_id="slide1.kv0",
                type="key_value",
                page_or_slide=1,
                method="pptx_aligned_textbox_row",
                content={"field_id": "17", "label": "Performance", "value": "Accuracy 94%"},
            ),
            Element(
                element_id="slide1.chart0",
                type="chart",
                page_or_slide=1,
                method="pptx_native_chart",
                content={
                    "categories": ["Jan 2026", "Feb 2026"],
                    "series": [{"name": "Accuracy", "values": [0.90, 0.92]}],
                },
            ),
        ],
    )

    result = rules.extract(doc)

    assert result.fields["model_performance"].status == "conflicting"
    assert any(flag.field == "model_performance" for flag in result.review_flags)
    flag = next(flag for flag in result.review_flags if flag.field == "model_performance")
    assert "0.94" in flag.message
    assert "0.92" in flag.message
