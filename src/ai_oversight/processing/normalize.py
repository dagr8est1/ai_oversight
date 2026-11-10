import re
from datetime import date

RISK_RATING_PATTERN = re.compile(
    r"inherent\s+(\w+).*?residual\s+(\w+)", re.IGNORECASE | re.DOTALL
)

# "Precision 0.91", "Accuracy 92%", "ROC-AUC 0.95" -- a name (letters, spaces,
# hyphens) followed by a number, optionally with a trailing percent sign.
METRIC_SEGMENT_PATTERN = re.compile(r"^(.+?)\s+([\d.]+)\s*(%)?$")


def normalize_date(raw: str) -> tuple[str, str | None]:
    """Pass an ISO YYYY-MM-DD date through unchanged; never fabricate a parse."""
    try:
        date.fromisoformat(raw.strip())
        return raw.strip(), None
    except ValueError:
        return raw, "could not parse as an ISO YYYY-MM-DD date; kept raw text"


def normalize_risk_rating(raw: str) -> tuple[dict[str, str] | str, str | None]:
    match = RISK_RATING_PATTERN.search(raw)
    if not match:
        return raw, "could not parse inherent/residual rating; kept raw text"
    inherent, residual = match.group(1).upper(), match.group(2).upper()
    return {"inherent": inherent, "residual": residual}, "parsed inherent/residual from free text"


def _normalize_metric_key(name: str) -> str:
    key = name.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def normalize_metric_list(raw: str) -> tuple[dict[str, float], str | None]:
    """Parse a "Name value[%]; Name2 value2[%]" string into {key: fraction}."""
    metrics: dict[str, float] = {}
    saw_percent = False
    for segment in raw.split(";"):
        segment = segment.strip().rstrip(",")
        if not segment:
            continue
        match = METRIC_SEGMENT_PATTERN.match(segment)
        if not match:
            continue
        name, number, percent_sign = match.groups()
        value = float(number)
        if percent_sign:
            value /= 100.0
            saw_percent = True
        metrics[_normalize_metric_key(name)] = value

    notes = "converted percent values to fractions" if saw_percent else None
    return metrics, notes
