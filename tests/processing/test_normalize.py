from ai_oversight.processing.normalize import (
    normalize_date,
    normalize_metric_list,
    normalize_risk_rating,
)


def test_normalize_date_passes_through_valid_iso_date():
    value, notes = normalize_date("2026-07-15")
    assert value == "2026-07-15"
    assert notes is None


def test_normalize_date_flags_unparseable_text_without_fabricating():
    value, notes = normalize_date("sometime next quarter")
    assert value == "sometime next quarter"
    assert notes is not None
    assert "could not" in notes.lower()


def test_normalize_risk_rating_splits_inherent_and_residual():
    value, notes = normalize_risk_rating("Inherent HIGH; residual MEDIUM")
    assert value == {"inherent": "HIGH", "residual": "MEDIUM"}
    assert notes is not None


def test_normalize_risk_rating_keeps_raw_text_if_pattern_does_not_match():
    value, notes = normalize_risk_rating("Rating pending committee review")
    assert value == "Rating pending committee review"
    assert notes is not None
    assert "could not" in notes.lower()


def test_normalize_metric_list_parses_fraction_values():
    value, _notes = normalize_metric_list("Precision 0.91; recall 0.86; ROC-AUC 0.95")
    assert value == {"precision": 0.91, "recall": 0.86, "roc_auc": 0.95}


def test_normalize_metric_list_converts_percent_to_fraction():
    value, notes = normalize_metric_list("Accuracy 92%; grounded response rate 97%")
    assert value == {"accuracy": 0.92, "grounded_response_rate": 0.97}
    assert notes is not None
    assert "percent" in notes.lower()
