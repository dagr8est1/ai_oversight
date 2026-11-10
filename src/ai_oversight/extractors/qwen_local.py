import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ai_oversight.config import Settings
from ai_oversight.parsers.document_ir import Element, ParsedDocument
from ai_oversight.processing.merge import (
    collect_incidents,
    collect_metrics,
    detect_performance_conflicts,
    infer_document_type,
    normalize_value,
)
from ai_oversight.processing.validate import validate_field_shape
from ai_oversight.schemas import (
    CANONICAL_FIELD_DESCRIPTIONS,
    CANONICAL_FIELDS,
    Evidence,
    ExtractionResult,
    FieldValue,
    ReviewFlag,
)

# Qwen3-4B has no documented grammar-constrained decoding -- only prompt-level
# JSON hints -- so a raw response is never trusted on the first try. This caps
# how many times a single chunk is re-prompted before falling back to
# needs_review rather than fabricating a value from unparseable output.
MAX_ATTEMPTS = 2

GenerateFn = Callable[[str], str]


class ModelNotAvailableError(RuntimeError):
    """The qwen_local backend's model dependencies (transformers/torch) are missing."""


@dataclass
class _Chunk:
    page_or_slide: int
    element_texts: dict[str, str]  # element_id -> its own source text


@dataclass
class _Candidate:
    value: Any
    evidence: Evidence
    substantiated: bool


def extract(
    doc: ParsedDocument,
    *,
    generate_fn: GenerateFn | None = None,
    settings: Settings | None = None,
) -> ExtractionResult:
    settings = settings or Settings.from_env()
    generate_fn = generate_fn or _build_default_generate_fn(settings)

    candidates: dict[str, list[_Candidate]] = defaultdict(list)
    ambiguous_signaled: set[str] = set()
    failed_chunks: list[tuple[_Chunk, str]] = []  # (chunk, raw_response)
    extraction_warnings = list(doc.warnings)

    for chunk in _chunk_by_page(doc):
        parsed, raw_response, ok = _extract_chunk(chunk, generate_fn)
        if not ok:
            failed_chunks.append((chunk, raw_response))
            extraction_warnings.append(
                f"page/slide {chunk.page_or_slide}: model response did not validate "
                f"after {MAX_ATTEMPTS} attempt(s); needs_review (raw response kept, "
                "no value fabricated)"
            )
            continue

        for field_name, entry in parsed.items():
            if field_name not in CANONICAL_FIELDS:
                continue
            value, element_id = _unpack_field_entry(entry)
            if value is None:
                # the model explicitly flagged this field as present-but-unclear
                # for this chunk, rather than not mentioning it at all
                ambiguous_signaled.add(field_name)
                continue
            evidence, substantiated = _build_evidence(chunk, element_id, raw_response, value)
            candidates[field_name].append(
                _Candidate(value=value, evidence=evidence, substantiated=substantiated)
            )

    fields, review_flags = _merge_candidates(candidates, ambiguous_signaled, failed_chunks)

    metrics, metric_warnings = collect_metrics(doc.elements)
    conflict_flags = detect_performance_conflicts(fields, metrics)
    incidents = collect_incidents(doc.elements)

    return ExtractionResult(
        document_id=doc.document_id,
        source_file=doc.filename,
        document_type=infer_document_type(doc.filename),
        extracted_at=datetime.now(UTC),
        fields=fields,
        metrics=metrics,
        incidents=incidents,
        review_flags=[*review_flags, *conflict_flags],
        extraction_warnings=[*extraction_warnings, *metric_warnings],
    )


def _unpack_field_entry(entry: Any) -> tuple[Any, str | None]:
    """A field entry is {"value": ..., "element_id": ...}; tolerate a bare
    scalar too, in case the model skips the wrapper object."""
    if isinstance(entry, dict) and "value" in entry:
        return entry.get("value"), entry.get("element_id")
    return entry, None


def _normalize_for_match(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _build_evidence(
    chunk: _Chunk, cited_element_id: str | None, raw_response: str, value: Any
) -> tuple[Evidence, bool]:
    """Returns (evidence, substantiated). `substantiated` is True only when
    the model cited a real element in this chunk AND (for a string value)
    that value actually appears in that element's own text -- PLAN.md
    Section 9 requires every extracted field to point to a specific source
    element, so neither a made-up citation nor a citation whose value isn't
    actually there can count as a confident `extracted`.
    """
    if cited_element_id and cited_element_id in chunk.element_texts:
        source_text = chunk.element_texts[cited_element_id]
        evidence = Evidence(
            page_or_slide=chunk.page_or_slide,
            element_id=cited_element_id,
            quote=source_text,
            method="qwen_local",
        )
        substantiated = (
            not isinstance(value, str) or _normalize_for_match(value) in _normalize_for_match(source_text)
        )
        return evidence, substantiated

    # the model didn't cite a real element in this chunk -- degrade to a
    # chunk-level citation rather than rejecting an otherwise-usable value,
    # but an unverifiable citation can never be "extracted" either
    evidence = Evidence(
        page_or_slide=chunk.page_or_slide,
        element_id=",".join(chunk.element_texts),
        quote=raw_response,
        method="qwen_local",
    )
    return evidence, False


def _chunk_by_page(doc: ParsedDocument) -> list[_Chunk]:
    by_page: dict[int, list[Element]] = defaultdict(list)
    for element in doc.elements:
        by_page[element.page_or_slide].append(element)

    chunks = []
    for page in sorted(by_page):
        element_texts = {el.element_id: text for el in by_page[page] if (text := _element_text(el))}
        chunks.append(_Chunk(page_or_slide=page, element_texts=element_texts))
    return chunks


# Prefixes a table-shaped element's text so the prompt's own instructions
# (see _build_prompt) can warn against treating one row/cell of a repeated
# record (an incident, a control, a risk) as if it were a single
# document-level field value -- found via a real run where Qwen pulled four
# different controls' mitigation text out of one table and confidently
# assigned each to an unrelated canonical field.
_TABLE_MARKER = "[TABLE of repeated records, not a single document-level fact] "


def _element_text(element: Element) -> str:
    if element.raw_text:
        return element.raw_text
    content = element.content or {}
    if "label" in content and "value" in content:
        return f"{content['label']}: {content['value']}"
    if "cells" in content:
        return _TABLE_MARKER + " | ".join(str(c) for c in content["cells"])
    if "rows" in content:
        rows_text = "\n".join(" | ".join(str(c) for c in row) for row in content["rows"])
        return _TABLE_MARKER + rows_text
    if "categories" in content and "series" in content:
        series_text = "; ".join(f"{s['name']}={s['values']}" for s in content["series"])
        return f"chart categories={content['categories']} {series_text}"
    return ""


def _build_prompt(chunk: _Chunk, previous_error: str | None = None, previous_response: str | None = None) -> str:
    field_descriptions = "\n".join(
        f"- {name}: {CANONICAL_FIELD_DESCRIPTIONS[name]}" for name in CANONICAL_FIELDS
    )
    excerpt = "\n".join(f"[{element_id}] {text}" for element_id, text in chunk.element_texts.items())
    base = (
        "You extract structured fields from an AI-governance report excerpt. "
        f'Respond with ONLY a JSON object: {{"field_name": {{"value": ..., "element_id": "..."}}}}. '
        "field_name must be exactly one of these field names (never a label or word taken "
        f"from the excerpt itself, even if it seems to match):\n{field_descriptions}\n\n"
        "element_id must be one of the bracketed IDs "
        "below, the one that actually states the value. Only include a key if this excerpt "
        'states that field. If a field is mentioned but unclear, use {"value": null, '
        '"element_id": "..."}. Never invent a value not stated in the excerpt. '
        "Content marked [TABLE of repeated records, not a single document-level fact] lists "
        "multiple instances of the same kind of thing (e.g. several different controls, "
        "incidents, or risks) -- one row or cell from it describes only that one instance, "
        "never the whole document. Do not use a value from inside such a table for any field "
        "unless the excerpt's own wording states it applies document-wide, not just to one row.\n\n"
        f"Excerpt (page/slide {chunk.page_or_slide}):\n{excerpt}\n"
    )
    if previous_error is not None:
        base += (
            f"\nYour previous response was invalid: {previous_error}\n"
            f"Previous response was:\n{previous_response}\n"
            "Respond again with ONLY a valid JSON object.\n"
        )
    return base


def _extract_chunk(chunk: _Chunk, generate_fn: GenerateFn) -> tuple[dict[str, Any] | None, str, bool]:
    prompt = _build_prompt(chunk)
    response = ""
    for attempt in range(MAX_ATTEMPTS):
        response = generate_fn(prompt)
        parsed, error = _parse_and_validate(response)
        if parsed is not None:
            return parsed, response, True
        if attempt + 1 < MAX_ATTEMPTS:
            prompt = _build_prompt(chunk, previous_error=error, previous_response=response)
    return None, response, False


def _is_plausible_response(candidate: Any) -> bool:
    """A real answer's keys are canonical field names, and each value is
    either a scalar/null or a {"value": ..., ...} object. This is what
    stops a truncated outer object from falling back to its own inner
    {"value": ..., "element_id": ...} fragment -- that fragment decodes
    fine on its own, but none of its keys are canonical fields, so it must
    never be accepted as if it were the whole response."""
    if not isinstance(candidate, dict):
        return False
    for key, value in candidate.items():
        if key not in CANONICAL_FIELDS:
            return False
        if isinstance(value, dict) and "value" not in value:
            return False
        if not isinstance(value, dict | str | int | float | bool) and value is not None:
            return False
    return True


def _parse_and_validate(response: str) -> tuple[dict[str, Any] | None, str | None]:
    """Find the last top-level JSON object that both decodes AND looks like
    a real field-mapping response (see `_is_plausible_response`), scanning
    from each '{' rather than trusting the first-to-last-brace span -- that
    span breaks on trailing prose containing its own brace, or an echoed
    example object followed by the real answer. A '{' nested inside the
    current winning object's span is skipped, so only sibling top-level
    objects compete (the last one wins, e.g. the real answer after an echoed
    example), never the already-matched object's own inner keys/values."""
    decoder = json.JSONDecoder()
    last_valid: dict[str, Any] | None = None
    last_valid_end = 0
    for index, char in enumerate(response):
        if char != "{" or index < last_valid_end:
            continue
        try:
            candidate, end = decoder.raw_decode(response, index)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and _is_plausible_response(candidate):
            last_valid = candidate
            last_valid_end = end
    if last_valid is None:
        return None, "no JSON object with canonical field keys found in response"
    return last_valid, None


def _merge_candidates(
    candidates: dict[str, list[_Candidate]],
    ambiguous_signaled: set[str],
    failed_chunks: list[tuple[_Chunk, str]],
) -> tuple[dict[str, FieldValue], list[ReviewFlag]]:
    fields: dict[str, FieldValue] = {}
    review_flags: list[ReviewFlag] = []

    for name in CANONICAL_FIELDS:
        entries = candidates.get(name, [])
        if not entries:
            fields[name] = _field_with_no_candidates(name, ambiguous_signaled, failed_chunks)
            continue

        distinct_values = {_stable_key(c.value) for c in entries}
        evidence = [c.evidence for c in entries]
        if len(distinct_values) > 1:
            fields[name] = FieldValue(value=entries[0].value, status="conflicting", evidence=evidence)
            review_flags.append(
                ReviewFlag(
                    code="multiple_candidate_values",
                    field=name,
                    message=f"{len(distinct_values)} differing candidate values found across chunks",
                )
            )
            continue

        if not any(c.substantiated for c in entries):
            fields[name] = FieldValue(
                value=entries[0].value,
                status="needs_review",
                evidence=evidence,
                normalization_notes="model's citation could not be verified against source text",
            )
            continue

        value, status, notes = normalize_value(name, entries[0].value)
        field = FieldValue(value=value, status=status, evidence=evidence, normalization_notes=notes)
        fields[name] = validate_field_shape(name, field)

    return fields, review_flags


def _field_with_no_candidates(
    name: str, ambiguous_signaled: set[str], failed_chunks: list[tuple[_Chunk, str]]
) -> FieldValue:
    if name in ambiguous_signaled:
        return FieldValue(status="ambiguous", normalization_notes="model flagged as present but unclear")
    if failed_chunks:
        evidence = [
            Evidence(
                page_or_slide=chunk.page_or_slide,
                element_id=",".join(chunk.element_texts),
                quote=raw_response,
                method="qwen_local_unvalidated",
            )
            for chunk, raw_response in failed_chunks
        ]
        return FieldValue(status="needs_review", evidence=evidence)
    return FieldValue(status="missing")


def _stable_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True) if isinstance(value, dict | list) else str(value)


def _build_default_generate_fn(settings: Settings) -> GenerateFn:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ModelNotAvailableError(
            "qwen_local backend requires the 'llm' extra (transformers + torch). "
            "Install with `uv sync --extra llm`, then re-run. No model weights "
            "are downloaded unless this backend is actually invoked."
        ) from exc

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(settings.text_model, cache_dir=str(settings.cache_dir))
    model = AutoModelForCausalLM.from_pretrained(
        settings.text_model, cache_dir=str(settings.cache_dir), torch_dtype="auto"
    ).to(device)

    few_shot = [
        {
            "role": "user",
            "content": (
                'Excerpt (page/slide 1):\n[page1.text0] 03 / Application name: Example App\n\n'
                'Respond with ONLY a JSON object: {"field_name": {"value": ..., "element_id": "..."}}.'
            ),
        },
        {
            "role": "assistant",
            "content": '{"application_name": {"value": "Example App", "element_id": "page1.text0"}}',
        },
    ]

    def generate(prompt: str) -> str:
        messages = [*few_shot, {"role": "user", "content": prompt}]
        # return_dict=True: apply_chat_template with return_tensors="pt" gives
        # a BatchEncoding (input_ids + attention_mask), not a bare tensor, on
        # this transformers version -- model.generate needs **inputs, and the
        # prompt length for slicing the response comes from inputs["input_ids"].
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)
        prompt_length = inputs["input_ids"].shape[1]
        return tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)

    return generate
