import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ai_oversight.config import Settings
from ai_oversight.parsers.document_ir import Element, ParsedDocument
from ai_oversight.schemas import (
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
    element_ids: list[str]
    text: str


def extract(
    doc: ParsedDocument,
    *,
    generate_fn: GenerateFn | None = None,
    settings: Settings | None = None,
) -> ExtractionResult:
    settings = settings or Settings.from_env()
    generate_fn = generate_fn or _build_default_generate_fn(settings)

    candidates: dict[str, list[tuple[Any, Evidence]]] = defaultdict(list)
    extraction_warnings = list(doc.warnings)

    for chunk in _chunk_by_page(doc):
        parsed, raw_response, ok = _extract_chunk(chunk, generate_fn)
        if not ok:
            extraction_warnings.append(
                f"page/slide {chunk.page_or_slide}: model response did not validate "
                f"after {MAX_ATTEMPTS} attempt(s); needs_review (raw response kept, "
                "no value fabricated)"
            )
            continue

        evidence = Evidence(
            page_or_slide=chunk.page_or_slide,
            element_id=",".join(chunk.element_ids),
            quote=raw_response,
            method="qwen_local",
        )
        for field_name, value in parsed.items():
            if field_name not in CANONICAL_FIELDS or value in (None, ""):
                continue
            candidates[field_name].append((value, evidence))

    fields, review_flags = _merge_candidates(candidates)

    return ExtractionResult(
        document_id=doc.document_id,
        source_file=doc.filename,
        document_type="unknown",
        extracted_at=datetime.now(UTC),
        fields=fields,
        review_flags=review_flags,
        extraction_warnings=extraction_warnings,
    )


def _chunk_by_page(doc: ParsedDocument) -> list[_Chunk]:
    by_page: dict[int, list[Element]] = defaultdict(list)
    for element in doc.elements:
        by_page[element.page_or_slide].append(element)

    chunks = []
    for page in sorted(by_page):
        elements = by_page[page]
        text = "\n".join(_element_text(el) for el in elements if _element_text(el))
        chunks.append(_Chunk(page_or_slide=page, element_ids=[el.element_id for el in elements], text=text))
    return chunks


def _element_text(element: Element) -> str:
    if element.raw_text:
        return element.raw_text
    content = element.content or {}
    if "label" in content and "value" in content:
        return f"{content['label']}: {content['value']}"
    if "cells" in content:
        return " | ".join(content["cells"])
    if "categories" in content and "series" in content:
        series_text = "; ".join(f"{s['name']}={s['values']}" for s in content["series"])
        return f"chart categories={content['categories']} {series_text}"
    return ""


def _build_prompt(chunk: _Chunk, previous_error: str | None = None, previous_response: str | None = None) -> str:
    field_list = ", ".join(CANONICAL_FIELDS)
    base = (
        "You extract structured fields from an AI-governance report excerpt. "
        f"Respond with ONLY a JSON object whose keys are a subset of: {field_list}. "
        "Only include a key if this excerpt states that field; use null if present "
        "but unclear. Never invent a value not stated in the excerpt.\n\n"
        f"Excerpt (page/slide {chunk.page_or_slide}):\n{chunk.text}\n"
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


def _parse_and_validate(response: str) -> tuple[dict[str, Any] | None, str | None]:
    start, end = response.find("{"), response.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None, "no JSON object found in response"
    try:
        parsed = json.loads(response[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}"
    if not isinstance(parsed, dict):
        return None, "response was valid JSON but not an object"
    return parsed, None


def _merge_candidates(
    candidates: dict[str, list[tuple[Any, Evidence]]]
) -> tuple[dict[str, FieldValue], list[ReviewFlag]]:
    fields: dict[str, FieldValue] = {}
    review_flags: list[ReviewFlag] = []

    for name in CANONICAL_FIELDS:
        entries = candidates.get(name, [])
        if not entries:
            fields[name] = FieldValue(status="missing")
            continue

        distinct_values = {_stable_key(value) for value, _ in entries}
        evidence = [ev for _, ev in entries]
        if len(distinct_values) == 1:
            fields[name] = FieldValue(value=entries[0][0], status="extracted", evidence=evidence)
        else:
            fields[name] = FieldValue(value=entries[0][0], status="conflicting", evidence=evidence)
            review_flags.append(
                ReviewFlag(
                    code="multiple_candidate_values",
                    field=name,
                    message=f"{len(distinct_values)} differing candidate values found across chunks",
                )
            )
    return fields, review_flags


def _stable_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)


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

    tokenizer = AutoTokenizer.from_pretrained(settings.text_model, cache_dir=str(settings.cache_dir))
    model = AutoModelForCausalLM.from_pretrained(settings.text_model, cache_dir=str(settings.cache_dir))

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512)
        return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return generate
