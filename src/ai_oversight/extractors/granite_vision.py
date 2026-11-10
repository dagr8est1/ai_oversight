import csv
import io
import json
import re
from collections.abc import Callable
from pathlib import Path

from ai_oversight.config import Settings
from ai_oversight.parsers import pdf_parser
from ai_oversight.parsers.document_ir import Element, ParsedDocument
from ai_oversight.schemas import VisionObservation

# (image_bytes, task_tag) -> raw text response, e.g. task_tag="<chart2csv>"
VisionFn = Callable[[bytes, str], str]

_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n```$")

# A multi-row table (e.g. a 16-row inventory) can need well over a thousand
# tokens of JSON; mlx_vlm.generate's own default (max_tokens=100) silently
# truncates mid-JSON, which _parse_table_json then rejects outright --
# discarding real, mostly-complete data along with the broken tail.
VISION_MAX_TOKENS = 2048


class VisionModelNotAvailableError(RuntimeError):
    """The vision backend's model dependencies (mlx-vlm, or transformers+torch) are missing."""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = _CODE_FENCE.sub("", text).strip()
    return text


def extract_chart(
    image_bytes: bytes, vision_fn: VisionFn, *, source_element_id: str | None, page_or_slide: int
) -> VisionObservation:
    raw = vision_fn(image_bytes, "<chart2csv>")
    parsed, note = _parse_csv(raw)
    return VisionObservation(
        task="chart2csv",
        source_element_id=source_element_id,
        page_or_slide=page_or_slide,
        raw_response=raw,
        parsed=parsed,
        status="ok" if parsed is not None else "needs_review",
        note=note,
    )


def extract_table(
    image_bytes: bytes, vision_fn: VisionFn, *, source_element_id: str | None, page_or_slide: int
) -> VisionObservation:
    raw = vision_fn(image_bytes, "<tables_json>")
    parsed, note = _parse_table_json(raw)
    return VisionObservation(
        task="tables_json",
        source_element_id=source_element_id,
        page_or_slide=page_or_slide,
        raw_response=raw,
        parsed=parsed,
        status="ok" if parsed is not None else "needs_review",
        note=note,
    )


def _parse_csv(raw: str) -> tuple[list[dict[str, str]] | None, str | None]:
    text = _strip_code_fence(raw)
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = [row for row in reader if any((v or "").strip() for v in row.values())]
    except csv.Error as exc:
        return None, f"CSV parse error: {exc}"
    if not rows or not reader.fieldnames:
        return None, "no CSV rows parsed from model output"
    return rows, None


def _parse_table_json(raw: str) -> tuple[list | dict | None, str | None]:
    text = _strip_code_fence(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        repaired = _repair_truncated_json(text)
        if repaired is None:
            return None, f"JSON parse error: {exc}"
        return repaired, (
            f"response was truncated mid-generation (original error: {exc}); "
            "recovered the complete rows before the cutoff, dropped the incomplete one"
        )
    if not isinstance(parsed, list | dict) or not parsed:
        return None, "model output was valid JSON but not a non-empty table"
    return parsed, None


def _repair_truncated_json(text: str) -> list | dict | None:
    """A response cut off by a premature stop token (not a token-budget
    issue -- confirmed by a real run reproducing byte-for-byte identical
    output after raising max_tokens) leaves early rows fully formed and
    only the last one incomplete. Finds the longest prefix of `text` that,
    closed with the brackets still open at that point, is itself valid
    JSON -- i.e. the state right after the last complete array element --
    so those real rows are recovered instead of discarding the lot because
    one trailing row never finished.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    cut_points: list[tuple[int, list[str]]] = []

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()
            cut_points.append((index, list(stack)))

    for index, remaining in reversed(cut_points):
        candidate = text[: index + 1].rstrip().rstrip(",")
        candidate += "".join("]" if opener == "[" else "}" for opener in reversed(remaining))
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list | dict) and parsed:
            return parsed
    return None


def run_vision_pass(document_path: Path, doc: ParsedDocument, vision_fn: VisionFn) -> list[VisionObservation]:
    """Route ambiguous chart/table content to the vision model, narrowly
    cropped. Native chart/table IR elements never reach here -- Phase 1
    already extracted those exactly; this only covers what Phase 1 could
    only warn about (a vector-drawn chart, a raster image/picture)."""
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        return _run_pdf_vision(document_path, vision_fn)
    if suffix == ".pptx":
        return _run_pptx_vision(document_path, vision_fn)
    return []


def _run_pdf_vision(document_path: Path, vision_fn: VisionFn) -> list[VisionObservation]:
    import pymupdf

    observations = []
    with pymupdf.open(document_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            region = pdf_parser.chart_region_on_page(page)
            if region is not None:
                pixmap = page.get_pixmap(clip=region, matrix=pymupdf.Matrix(2, 2))
                observations.append(
                    extract_chart(
                        pixmap.tobytes("png"),
                        vision_fn,
                        source_element_id=f"page{page_number}.vision_crop",
                        page_or_slide=page_number,
                    )
                )
                continue

            # no vector-drawn chart detected -- if the page also has no text
            # layer (a scanned page, e.g. the whole page is one raster
            # image), there's no native content at all to fall back on, so
            # route the rendered page itself, as an ambiguous raster table
            # per Section 3's "rasterized tables" case (no way to tell chart
            # from table from a bare image, so <tables_json> is the default,
            # same choice already made for a PPTX picture shape)
            if not page.get_text("text").strip() and page.get_images():
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                observations.append(
                    extract_table(
                        pixmap.tobytes("png"),
                        vision_fn,
                        source_element_id=f"page{page_number}.vision_scan",
                        page_or_slide=page_number,
                    )
                )
    return observations


def _run_pptx_vision(document_path: Path, vision_fn: VisionFn) -> list[VisionObservation]:
    from pptx import Presentation

    from ai_oversight.parsers.pptx_parser import MSO_SHAPE_TYPE, _flatten_shapes

    observations = []
    prs = Presentation(document_path)
    for slide_index, slide in enumerate(prs.slides):
        slide_number = slide_index + 1
        for shape in _flatten_shapes(slide.shapes):
            if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                continue
            observations.append(
                extract_table(
                    shape.image.blob,
                    vision_fn,
                    source_element_id=f"slide{slide_number}.{shape.shape_id}",
                    page_or_slide=slide_number,
                )
            )
    return observations


def enrich_document(
    document_path: Path,
    doc: ParsedDocument,
    *,
    vision_fn: VisionFn | None = None,
    settings: Settings | None = None,
) -> tuple[ParsedDocument, list[VisionObservation], str | None]:
    """Run the vision pass and merge its recovered content into the IR as
    ordinary elements (chart/key_value/table), so whichever text backend
    runs next (rules or qwen_local) sees vision-recovered content the same
    way it already sees native content -- no backend-specific vision code
    needed; a recovered chart feeds the same collect_metrics/conflict-check
    path a native chart does, and a recovered label/value pair becomes a
    normal key_value element qwen_local can map semantically.

    Returns (document_for_backend, observations, warning). Never raises: on
    any failure (missing extra, offline, OOM, a generate error), returns the
    original document unchanged, no observations, and a warning message for
    the caller to attach to its result -- so a constrained Mac still gets a
    usable, vision-less result instead of a crash.
    """
    try:
        vision_fn = vision_fn or build_default_vision_fn(settings)
        observations = run_vision_pass(document_path, doc, vision_fn)
    except Exception as exc:  # noqa: BLE001 -- deliberately blind, see docstring
        return doc, [], f"vision pass requested but unavailable: {exc}"

    return enrich_document_from_observations(doc, observations), observations, None


def enrich_document_from_observations(
    doc: ParsedDocument, observations: list[VisionObservation]
) -> ParsedDocument:
    """Merge already-recovered vision observations into the IR, without
    running the vision model at all. Pairs with `save_observations`/
    `load_observations`: the vision pass (slow, a real model) runs once via
    the `vision` CLI command, and iterating on how its output gets mapped
    to canonical fields (prompt changes, element-conversion changes) can
    then re-run the fast text backend alone, as many times as needed,
    against the same cached observations.
    """
    vision_elements = observations_to_elements(observations)
    return doc.model_copy(update={"elements": [*doc.elements, *vision_elements]})


def save_observations(observations: list[VisionObservation], path: Path) -> None:
    path.write_text(json.dumps([obs.model_dump(mode="json") for obs in observations], indent=2))


def load_observations(path: Path) -> list[VisionObservation]:
    return [VisionObservation.model_validate(item) for item in json.loads(path.read_text())]


def observations_to_elements(observations: list[VisionObservation]) -> list[Element]:
    """Convert successful vision observations into IR elements.

    A chart2csv result becomes one `chart` element (categories/series),
    feeding the same metrics/conflict-detection path a native chart does. A
    tables_json result becomes one `key_value` element per row for a 2-column
    table (a label/value pair qwen_local can map to a canonical field), or
    one `table` element for a wider table (a record list like an incident or
    control table, not a single field value).
    """
    elements: list[Element] = []
    for observation in observations:
        if observation.status != "ok":
            continue
        if observation.task == "chart2csv":
            element = _chart_element(observation)
            if element is not None:
                elements.append(element)
        elif observation.task == "tables_json":
            elements.extend(_table_elements(observation))
    return elements


def _chart_element(observation: VisionObservation) -> Element | None:
    if not isinstance(observation.parsed, list) or not observation.parsed:
        return None

    period_key: str | None = None
    categories: list[str] = []
    series_values: dict[str, list[float | None]] = {}
    for row in observation.parsed:
        if not isinstance(row, dict) or not row:
            continue
        if period_key is None:
            period_key = next(iter(row))
        categories.append(row.get(period_key))
        for column, raw_value in row.items():
            if column == period_key:
                continue
            try:
                value: float | None = float(raw_value)
            except (TypeError, ValueError):
                value = None
            series_values.setdefault(column, []).append(value)

    if not categories:
        return None

    series = [{"name": name, "values": values} for name, values in series_values.items()]
    return Element(
        element_id=f"{observation.source_element_id}.chart",
        type="chart",
        page_or_slide=observation.page_or_slide,
        method=f"granite_vision_{observation.task}",
        content={"categories": categories, "series": series},
    )


def _table_elements(observation: VisionObservation) -> list[Element]:
    tables = observation.parsed if isinstance(observation.parsed, list) else [observation.parsed]
    elements: list[Element] = []
    for table_index, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        cells = table.get("cells")
        if not cells:
            continue
        by_row = _cells_by_row(cells)
        if not by_row:
            continue
        max_col = max(col for row in by_row.values() for col in row)
        if max_col <= 2:
            elements.extend(_key_value_elements(observation, by_row, table_index))
        else:
            elements.extend(_wide_table_elements(observation, by_row, table_index))
    return elements


def _cells_by_row(cells: list[dict]) -> dict[int, dict[int, str]]:
    by_row: dict[int, dict[int, str]] = {}
    for cell in cells:
        row, col = cell.get("row"), cell.get("col")
        if row is None or col is None:
            continue
        by_row.setdefault(row, {})[col] = cell.get("content")
    return by_row


def _key_value_elements(
    observation: VisionObservation, by_row: dict[int, dict[int, str]], table_index: int
) -> list[Element]:
    elements: list[Element] = []
    for row_index in sorted(by_row):
        label, value = by_row[row_index].get(1), by_row[row_index].get(2)
        if not label or not value:
            continue
        elements.append(
            Element(
                element_id=f"{observation.source_element_id}.table{table_index}.row{row_index}",
                type="key_value",
                page_or_slide=observation.page_or_slide,
                method=f"granite_vision_{observation.task}",
                content={"label": str(label), "value": str(value)},
                raw_text=f"{label}: {value}",
            )
        )
    return elements


def _wide_table_elements(
    observation: VisionObservation, by_row: dict[int, dict[int, str]], table_index: int
) -> list[Element]:
    """A wide (>2-column) table's rows aren't automatically a 'repeated
    records' list -- two structural signals, both read off Granite's own
    cell data (not guessed), say otherwise:

    - A row with fewer cells than the table's full width is a merged/
      colspan cell -- a summary or footer fact (e.g. "n=1,000 records |
      Precision 91.1% | Recall 87.0%"), not one more instance. It's
      extracted as its own plain-text fact, safe regardless of how many
      real instance rows the table has.
    - If, after removing such summary rows, there's exactly one real body
      row left, there's no "which instance" ambiguity at all -- each
      header/value column pair is as safe as a 2-column table's row.

    Only when more than one full-width body row remains does the content
    stay a generic `table` element, cautioned in qwen_local's prompt as a
    list of repeated records.
    """
    sorted_rows = sorted(by_row)
    if len(sorted_rows) < 2:
        return []

    full_width = max(len(cols) for cols in by_row.values())
    header_row_index = sorted_rows[0]
    body_row_indices = sorted_rows[1:]
    full_body_rows = [i for i in body_row_indices if len(by_row[i]) == full_width]
    summary_row_indices = [i for i in body_row_indices if len(by_row[i]) < full_width]

    elements: list[Element] = []
    for row_index in summary_row_indices:
        text = " | ".join(str(value) for value in by_row[row_index].values() if value)
        if text:
            elements.append(
                Element(
                    element_id=f"{observation.source_element_id}.table{table_index}.summary{row_index}",
                    type="text",
                    page_or_slide=observation.page_or_slide,
                    method=f"granite_vision_{observation.task}",
                    raw_text=text,
                )
            )

    if len(full_body_rows) == 1:
        header_cells = by_row[header_row_index]
        value_cells = by_row[full_body_rows[0]]
        for col in sorted(header_cells):
            label, value = header_cells.get(col), value_cells.get(col)
            if label and value:
                elements.append(
                    Element(
                        element_id=f"{observation.source_element_id}.table{table_index}.col{col}",
                        type="key_value",
                        page_or_slide=observation.page_or_slide,
                        method=f"granite_vision_{observation.task}",
                        content={"label": str(label), "value": str(value)},
                        raw_text=f"{label}: {value}",
                    )
                )
    elif full_body_rows:
        rows = [
            [str(by_row[i].get(col) or "") for col in range(1, full_width + 1)]
            for i in [header_row_index, *full_body_rows]
        ]
        elements.append(
            Element(
                element_id=f"{observation.source_element_id}.table{table_index}",
                type="table",
                page_or_slide=observation.page_or_slide,
                method=f"granite_vision_{observation.task}",
                content={"rows": rows},
            )
        )

    return elements


def build_default_vision_fn(settings: Settings | None) -> VisionFn:
    settings = settings or Settings.from_env()
    try:
        import mlx_vlm
    except ImportError:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise VisionModelNotAvailableError(
                "vision extraction requires the 'vision' extra: mlx-vlm on Apple "
                "Silicon, or transformers + torch elsewhere. Install with "
                "`uv sync --extra vision`, then re-run. No model weights are "
                "downloaded unless this backend is actually invoked."
            ) from exc
        return _build_transformers_vision_fn(settings, torch, AutoModelForImageTextToText, AutoProcessor)
    return _build_mlx_vision_fn(settings, mlx_vlm)


def _build_mlx_vision_fn(settings: Settings, mlx_vlm_module) -> VisionFn:
    model, processor = mlx_vlm_module.load(settings.vision_model)
    config = mlx_vlm_module.utils.load_config(settings.vision_model)

    def vision_fn(image_bytes: bytes, task: str) -> str:
        image = _bytes_to_temp_image(image_bytes)
        prompt = mlx_vlm_module.apply_chat_template(processor, config, task, num_images=1)
        # mlx_vlm.generate's own default is max_tokens=100 (per its
        # docstring) -- found via a real run truncating a genuine 16-row
        # table mid-JSON, discarding the whole response even though most
        # of the real data was in the raw text before the cutoff.
        result = mlx_vlm_module.generate(
            model, processor, prompt, [image], verbose=False, max_tokens=VISION_MAX_TOKENS
        )
        return result.text

    return vision_fn


def _build_transformers_vision_fn(settings: Settings, torch, model_cls, processor_cls) -> VisionFn:
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    processor = processor_cls.from_pretrained(settings.vision_model, cache_dir=str(settings.cache_dir))
    model = model_cls.from_pretrained(
        settings.vision_model, cache_dir=str(settings.cache_dir), torch_dtype="auto"
    ).to(device)

    def vision_fn(image_bytes: bytes, task: str) -> str:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=VISION_MAX_TOKENS)
        return processor.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return vision_fn


def _bytes_to_temp_image(image_bytes: bytes):
    import tempfile

    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    image.save(path)
    return path
