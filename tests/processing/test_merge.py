from ai_oversight.parsers.document_ir import Element
from ai_oversight.processing.merge import (
    collect_incidents,
    collect_metrics,
    detect_performance_conflicts,
    infer_document_type,
    match_metric,
    normalize_value,
)
from ai_oversight.schemas import FieldValue, MetricObservation


def _chart_element(categories, series, element_id="slide1.chart0", page_or_slide=1):
    return Element(
        element_id=element_id,
        type="chart",
        page_or_slide=page_or_slide,
        method="pptx_native_chart",
        content={"categories": categories, "series": series},
    )


def test_collect_metrics_preserves_time_periods_and_names():
    el = _chart_element(
        ["Aug 2026", "Sep 2026", "Oct 2026"],
        [{"name": "Classification accuracy", "values": [0.87, 0.90, 0.92]}],
    )
    metrics, warnings = collect_metrics([el])

    assert warnings == []
    periods = {m.period for m in metrics}
    assert periods == {"Aug 2026", "Sep 2026", "Oct 2026"}
    assert all(m.name == "classification_accuracy" for m in metrics)


def test_collect_metrics_skips_a_blank_data_point_without_crashing():
    el = _chart_element(["Jan", "Feb"], [{"name": "Accuracy", "values": [0.9, None]}])

    metrics, warnings = collect_metrics([el])

    assert len(metrics) == 1
    assert metrics[0].period == "Jan"
    assert any("blank" in w.lower() or "skip" in w.lower() for w in warnings)


def test_collect_incidents_handles_native_table_rows_key():
    el = Element(
        element_id="slide7.table0",
        type="table",
        page_or_slide=7,
        method="pptx_native_table",
        content={
            "rows": [
                ["Date", "Event", "Impact"],
                ["Sep 07", "Incorrect routing", "Analyst rejected"],
            ]
        },
    )

    incidents = collect_incidents([el])

    assert len(incidents) == 1
    assert incidents[0]["date"] == "Sep 07"


def test_collect_incidents_handles_aligned_textbox_cells_key():
    el = Element(
        element_id="slide7.table0",
        type="table",
        page_or_slide=7,
        method="pptx_aligned_textbox_row_cells",
        content={"cells": ["Oct 02", "Provider latency", "Manual queue used"]},
    )

    incidents = collect_incidents([el])

    assert len(incidents) == 1
    assert incidents[0]["date"] == "Oct 02"


def test_infer_document_type_from_filename():
    assert infer_document_type("Sample_Internal_AI_Oversight_Report.pdf") == "internal"
    assert infer_document_type("Sample_External_AI_Usage_Disclosure.pptx") == "external"
    assert infer_document_type("whatever.pdf") == "unknown"


def test_match_metric_prefers_exact_match():
    assert match_metric("grounded_response_rate", ["grounded_response_rate", "accuracy"]) == "grounded_response_rate"


def test_match_metric_matches_a_unique_token_subset():
    assert match_metric("accuracy", ["classification_accuracy", "grounded_response_rate"]) == "classification_accuracy"


def test_match_metric_refuses_to_guess_when_ambiguous():
    # "accuracy" is a token-subset of BOTH candidates -- picking either would
    # silently guess the wrong one, so neither should be returned.
    assert match_metric("accuracy", ["balanced_accuracy", "classification_accuracy"]) is None


def test_match_metric_returns_none_when_nothing_relates():
    assert match_metric("grounded_responses", ["precision", "recall"]) is None


def test_detect_performance_conflicts_flags_disagreement_and_attaches_chart_evidence():
    fields = {
        "model_performance": FieldValue(
            value={"accuracy": 0.94}, status="extracted", evidence=[]
        )
    }
    metrics = [
        MetricObservation(
            field="model_performance",
            name="accuracy",
            period="Feb 2026",
            value=0.92,
            source_element_id="slide1.chart0",
            page_or_slide=1,
        )
    ]

    flags = detect_performance_conflicts(fields, metrics)

    assert fields["model_performance"].status == "conflicting"
    assert len(flags) == 1
    assert "0.94" in flags[0].message
    assert "0.92" in flags[0].message
    # the chart's own evidence must be attached, not just the narrative's
    assert any(ev.element_id == "slide1.chart0" for ev in fields["model_performance"].evidence)


def test_detect_performance_conflicts_scales_a_chart_value_stored_as_0_to_100():
    fields = {
        "model_performance": FieldValue(value={"accuracy": 0.92}, status="extracted", evidence=[])
    }
    metrics = [
        MetricObservation(
            field="model_performance",
            name="accuracy",
            period="Feb 2026",
            value=92.0,  # stored as a raw percent number, not a fraction
            source_element_id="slide1.chart0",
            page_or_slide=1,
        )
    ]

    flags = detect_performance_conflicts(fields, metrics)

    assert flags == []
    assert fields["model_performance"].status == "extracted"


def test_detect_performance_conflicts_does_not_flag_agreement():
    fields = {
        "model_performance": FieldValue(value={"accuracy": 0.92}, status="extracted", evidence=[])
    }
    metrics = [
        MetricObservation(
            field="model_performance",
            name="classification_accuracy",
            period="Oct 2026",
            value=0.92,
            source_element_id="slide5.chart0",
            page_or_slide=5,
        )
    ]

    flags = detect_performance_conflicts(fields, metrics)

    assert flags == []
    assert fields["model_performance"].status == "extracted"


def test_detect_performance_conflicts_skips_non_numeric_entries_without_crashing():
    # found via a real run: a `conflicting`-status model_performance field
    # never goes through normalize_value/validate_field_shape at all (the
    # early-return conflict path in qwen_local's merge keeps the model's
    # raw, un-normalized value) -- so a dict with string percentage values
    # like {"accuracy": "92%"} can reach here directly and must not crash
    # arithmetic that assumes every entry is already numeric.
    fields = {
        "model_performance": FieldValue(
            value={"accuracy": "92%", "grounded_response_rate": "97%"},
            status="conflicting",
            evidence=[],
        )
    }
    metrics = [
        MetricObservation(
            field="model_performance",
            name="accuracy",
            period="Oct 2026",
            value=0.90,
            source_element_id="slide5.chart0",
            page_or_slide=5,
        )
    ]

    flags = detect_performance_conflicts(fields, metrics)

    assert flags == []
    assert fields["model_performance"].status == "conflicting"


def test_normalize_value_converts_a_dict_shaped_model_performance_value():
    # found via a real run: qwen_local's response for model_performance was
    # already a dict ({"accuracy": "92%", ...}) rather than a flat string --
    # normalize_value previously only normalized string raw values, so this
    # sailed through with un-normalized string percentages
    value, status, notes = normalize_value("model_performance", {"accuracy": "92%"})
    assert value == {"accuracy": 0.92}
    assert status == "extracted"
    assert notes is not None


def test_normalize_value_leaves_other_dict_shaped_fields_untouched():
    # risk_rating's dict shape ({"inherent": "HIGH", "residual": "MEDIUM"})
    # is already its correct final form -- must not be routed through the
    # numeric metric-dict normalizer
    value, status, notes = normalize_value("risk_rating", {"inherent": "HIGH", "residual": "MEDIUM"})
    assert value == {"inherent": "HIGH", "residual": "MEDIUM"}
    assert status == "extracted"
    assert notes is None
