from ai_oversight.processing.validate import validate_field_shape
from ai_oversight.schemas import FieldValue


def test_extracted_risk_rating_with_correct_shape_passes_through():
    field = FieldValue(value={"inherent": "HIGH", "residual": "MEDIUM"}, status="extracted")
    result = validate_field_shape("risk_rating", field)
    assert result.status == "extracted"


def test_extracted_risk_rating_with_wrong_shape_is_downgraded_to_needs_review():
    field = FieldValue(value="not a dict", status="extracted")
    result = validate_field_shape("risk_rating", field)
    assert result.status == "needs_review"
    assert "shape" in (result.normalization_notes or "").lower()


def test_extracted_model_performance_must_be_a_nonempty_numeric_dict():
    field = FieldValue(value={}, status="extracted")
    result = validate_field_shape("model_performance", field)
    assert result.status == "needs_review"


def test_fields_without_a_shape_rule_pass_through_unchanged():
    field = FieldValue(value="Maya Chen, Director", status="extracted")
    result = validate_field_shape("business_owner", field)
    assert result.status == "extracted"


def test_non_extracted_fields_are_never_touched():
    field = FieldValue(value=None, status="missing")
    result = validate_field_shape("risk_rating", field)
    assert result.status == "missing"
