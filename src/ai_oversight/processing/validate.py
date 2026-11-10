from typing import Any

from ai_oversight.schemas import FieldValue

_EXPECTED_SHAPE = {
    "risk_rating": lambda v: isinstance(v, dict) and set(v) == {"inherent", "residual"},
    "model_performance": lambda v: (
        isinstance(v, dict) and bool(v) and all(isinstance(x, int | float) for x in v.values())
    ),
}


def validate_field_shape(name: str, field: FieldValue) -> FieldValue:
    """Downgrade an `extracted` field to `needs_review` if its value doesn't
    match the structural shape that field's normalizer promises to produce.
    This is the Pydantic-validation gate PLAN.md Section 6 (Phase 2) asks
    for: a value can pass the normalizer's own regex yet still come out
    malformed, and that should never surface silently as `extracted`.
    """
    if field.status != "extracted":
        return field

    checker = _EXPECTED_SHAPE.get(name)
    if checker is None:
        return field

    value: Any = field.value
    if checker(value):
        return field

    note = field.normalization_notes
    note = f"{note}; value failed shape validation" if note else "value failed shape validation"
    return field.model_copy(update={"status": "needs_review", "normalization_notes": note})
