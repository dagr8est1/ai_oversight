import re
from datetime import date

# Only a whitelisted rating vocabulary is captured -- never an arbitrary word
# that happens to sit between "inherent"/"residual" and the next clause. A
# loose `\w+` here previously captured "RATING" from "Inherent rating: HIGH"
# or "IS" from "...residual is MEDIUM", silently reporting a wrong value as
# `extracted`.
_RATING_WORD = r"(HIGH|MEDIUM|LOW|CRITICAL|MODERATE|SEVERE|MINIMAL)"
RISK_RATING_PATTERN = re.compile(
    rf"inherent\D*?\b{_RATING_WORD}\b.*?residual\D*?\b{_RATING_WORD}\b",
    re.IGNORECASE | re.DOTALL,
)

# A name (letters/spaces/hyphens, no digits) followed by a number and an
# optional percent sign, end-anchored. Parenthetical trailing notes like
# " (Oct 2026)" are stripped before matching (see normalize_metric_list), so
# the end anchor is safe to keep -- without it, a segment with no separator
# before a second metric (e.g. "Precision 0.91 recall 0.86") or trailing
# prose (e.g. "Recall dropped 0.05 to 0.81") would match only its first
# number and silently discard or misread the rest.
METRIC_SEGMENT_PATTERN = re.compile(r"^([A-Za-z][A-Za-z\s\-]*?)\s+([\d.]+)\s*(%)?\s*$")

_PARENTHETICAL = re.compile(r"\([^)]*\)")


def normalize_date(raw: str) -> tuple[str, str | None, bool]:
    """Pass an ISO YYYY-MM-DD date through unchanged; never fabricate a parse."""
    try:
        date.fromisoformat(raw.strip())
        return raw.strip(), None, True
    except ValueError:
        return raw, "could not parse as an ISO YYYY-MM-DD date; kept raw text", False


def normalize_risk_rating(raw: str) -> tuple[dict[str, str] | str, str | None, bool]:
    match = RISK_RATING_PATTERN.search(raw)
    if not match:
        return raw, "could not parse inherent/residual rating; kept raw text", False
    inherent, residual = match.group(1).upper(), match.group(2).upper()
    return {"inherent": inherent, "residual": residual}, "parsed inherent/residual from free text", True


def _normalize_metric_key(name: str) -> str:
    key = name.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def normalize_metric_list(raw: str) -> tuple[dict[str, float] | str, str | None, bool]:
    """Parse a "Name value[%]; Name2 value2[%], ..." string into {key: fraction}.

    Never raises on a malformed number and never silently drops a segment
    that looked like a metric but couldn't be parsed -- both are reported in
    `notes` instead, and `ok=False` tells the caller not to trust the result
    as a clean extraction.
    """
    metrics: dict[str, float] = {}
    saw_percent = False
    unparsed: list[str] = []

    for segment in re.split(r"[;,]", raw):
        segment = _PARENTHETICAL.sub("", segment).strip()
        if not segment:
            continue
        match = METRIC_SEGMENT_PATTERN.match(segment)
        if not match:
            unparsed.append(segment)
            continue
        name, number, percent_sign = match.groups()
        try:
            value = float(number)
        except ValueError:
            unparsed.append(segment)
            continue
        if percent_sign:
            value /= 100.0
            saw_percent = True
        metrics[_normalize_metric_key(name)] = value

    if not metrics:
        return raw, "could not parse any metric from free text; kept raw text", False

    notes_parts = []
    if saw_percent:
        notes_parts.append("converted percent values to fractions")
    if unparsed:
        notes_parts.append(f"could not parse segment(s): {'; '.join(unparsed)}")
    notes = "; ".join(notes_parts) if notes_parts else None
    return metrics, notes, not unparsed


_SINGLE_METRIC_VALUE = re.compile(r"^\s*([\d.]+)\s*(%)?\s*$")


def normalize_metric_dict(raw: dict) -> tuple[dict, str | None, bool]:
    """Like `normalize_metric_list`, but for when the model already replied
    with a dict of {name: value} instead of a flat string -- the model can
    do this, so a dict-shaped value must be normalized the same way a
    string would be, not passed through unconverted (found via a real run:
    qwen_local replied with {"accuracy": "92%", ...}, which survived as
    string percentages and later crashed arithmetic that assumes numeric
    dict values). An already-numeric entry passes through unchanged; an
    unparseable entry is dropped from the result (never left in as a string
    that could crash a numeric consumer downstream) and reported in notes,
    with ok=False so the caller knows not to trust this as a clean parse.
    """
    metrics: dict[str, float] = {}
    saw_percent = False
    unparsed: list[str] = []

    for key, value in raw.items():
        if isinstance(value, int | float):
            metrics[key] = float(value)
            continue
        match = _SINGLE_METRIC_VALUE.match(str(value))
        if not match:
            unparsed.append(f"{key}={value!r}")
            continue
        number, percent_sign = match.groups()
        parsed = float(number)
        if percent_sign:
            parsed /= 100.0
            saw_percent = True
        metrics[key] = parsed

    if not metrics:
        return raw, "could not parse any metric value from dict; kept raw", False

    notes_parts = []
    if saw_percent:
        notes_parts.append("converted percent values to fractions")
    if unparsed:
        notes_parts.append(f"could not parse value(s): {'; '.join(unparsed)}")
    notes = "; ".join(notes_parts) if notes_parts else None
    return metrics, notes, not unparsed
