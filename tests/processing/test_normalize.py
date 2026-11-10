from ai_oversight.processing.normalize import (
    normalize_date,
    normalize_metric_list,
    normalize_risk_rating,
)


def test_normalize_date_passes_through_valid_iso_date():
    value, notes, ok = normalize_date("2026-07-15")
    assert value == "2026-07-15"
    assert notes is None
    assert ok is True


def test_normalize_date_flags_unparseable_text_without_fabricating():
    value, notes, ok = normalize_date("sometime next quarter")
    assert value == "sometime next quarter"
    assert notes is not None
    assert "could not" in notes.lower()
    assert ok is False


def test_normalize_risk_rating_splits_inherent_and_residual():
    value, notes, ok = normalize_risk_rating("Inherent HIGH; residual MEDIUM")
    assert value == {"inherent": "HIGH", "residual": "MEDIUM"}
    assert notes is not None
    assert ok is True


def test_normalize_risk_rating_keeps_raw_text_if_pattern_does_not_match():
    value, notes, ok = normalize_risk_rating("Rating pending committee review")
    assert value == "Rating pending committee review"
    assert notes is not None
    assert "could not" in notes.lower()
    assert ok is False


def test_normalize_risk_rating_does_not_capture_arbitrary_words_as_a_rating():
    # adversarial: extra words between "inherent"/"residual" and the actual
    # rating word must not get captured as if they were the rating itself
    value, _notes, ok = normalize_risk_rating("Inherent rating: HIGH; residual rating: MEDIUM")
    assert value == {"inherent": "HIGH", "residual": "MEDIUM"}
    assert ok is True

    value, _notes, ok = normalize_risk_rating("Inherent risk is HIGH, residual not yet assessed")
    assert ok is False
    assert value == "Inherent risk is HIGH, residual not yet assessed"


def test_normalize_metric_list_parses_fraction_values():
    value, _notes, ok = normalize_metric_list("Precision 0.91; recall 0.86; ROC-AUC 0.95")
    assert value == {"precision": 0.91, "recall": 0.86, "roc_auc": 0.95}
    assert ok is True


def test_normalize_metric_list_converts_percent_to_fraction():
    value, notes, ok = normalize_metric_list("Accuracy 92%; grounded response rate 97%")
    assert value == {"accuracy": 0.92, "grounded_response_rate": 0.97}
    assert notes is not None
    assert "percent" in notes.lower()
    assert ok is True


def test_normalize_metric_list_handles_comma_separated_segments():
    value, _notes, ok = normalize_metric_list("Precision 0.91, recall 0.86")
    assert value == {"precision": 0.91, "recall": 0.86}
    assert ok is True


def test_normalize_metric_list_ignores_trailing_parenthetical_notes():
    value, _notes, ok = normalize_metric_list("Precision 0.91; recall 0.86 (Oct 2026); ROC-AUC 0.95")
    assert value == {"precision": 0.91, "recall": 0.86, "roc_auc": 0.95}
    assert ok is True


def test_normalize_metric_list_does_not_crash_on_a_malformed_number():
    value, notes, ok = normalize_metric_list("Accuracy 0.9.2")
    assert value == "Accuracy 0.9.2"
    assert ok is False
    assert notes is not None


def test_normalize_metric_list_reports_failure_when_nothing_parses():
    value, notes, ok = normalize_metric_list("Not yet measured")
    assert ok is False
    assert value == "Not yet measured"
    assert notes is not None
