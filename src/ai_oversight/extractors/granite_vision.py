import csv
import io
import json
import re
from collections.abc import Callable
from pathlib import Path

from ai_oversight.config import Settings
from ai_oversight.parsers import pdf_parser
from ai_oversight.parsers.document_ir import ParsedDocument
from ai_oversight.processing.merge import detect_performance_conflicts, normalize_metric_key
from ai_oversight.schemas import ExtractionResult, MetricObservation, VisionObservation

# (image_bytes, task_tag) -> raw text response, e.g. task_tag="<chart2csv>"
VisionFn = Callable[[bytes, str], str]

_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\n|\n```$")


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
        return None, f"JSON parse error: {exc}"
    if not isinstance(parsed, list | dict) or not parsed:
        return None, "model output was valid JSON but not a non-empty table"
    return parsed, None


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


def apply_to_result(
    document_path: Path,
    doc: ParsedDocument,
    result: ExtractionResult,
    *,
    vision_fn: VisionFn | None = None,
    settings: Settings | None = None,
) -> ExtractionResult:
    """Opt-in vision pass: recovers chart/table data Phase 1 could only warn
    about, cross-checks it against the already-extracted narrative, and
    flags (never silently resolves) a disagreement. Never raises -- an
    unavailable model degrades to a `needs_review`-style warning so a
    constrained Mac still gets a usable (if vision-less) result.
    """
    try:
        vision_fn = vision_fn or build_default_vision_fn(settings)
        observations = run_vision_pass(document_path, doc, vision_fn)
    except Exception as exc:  # noqa: BLE001 -- deliberately blind, see comment below
        # anything here -- missing extra, offline/missing weights, OOM, a
        # generate error -- must degrade to a warning, never break the
        # baseline result the text backend already produced successfully.
        result.extraction_warnings.append(f"vision pass requested but unavailable: {exc}")
        return result

    vision_metrics: list[MetricObservation] = []
    for observation in observations:
        if observation.status != "ok":
            result.extraction_warnings.append(
                f"page/slide {observation.page_or_slide}: vision {observation.task} needs_review: "
                f"{observation.note}"
            )
            continue
        if observation.task == "chart2csv":
            vision_metrics.extend(_metrics_from_chart_observation(observation))

    if not vision_metrics:
        return result

    result.metrics.extend(vision_metrics)
    # favor native data: this only ever flags a disagreement against the
    # narrative value already extracted, never overwrites it with the
    # vision-derived number.
    result.review_flags.extend(detect_performance_conflicts(result.fields, vision_metrics))
    return result


def _metrics_from_chart_observation(observation: VisionObservation) -> list[MetricObservation]:
    metrics: list[MetricObservation] = []
    if not isinstance(observation.parsed, list):
        return metrics
    for row in observation.parsed:
        if not isinstance(row, dict) or not row:
            continue
        period_key, period = next(iter(row.items()))
        for column, raw_value in row.items():
            if column == period_key:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            metrics.append(
                MetricObservation(
                    field="model_performance",
                    name=normalize_metric_key(column),
                    period=str(period),
                    value=value,
                    source_element_id=f"vision:{observation.source_element_id}",
                    page_or_slide=observation.page_or_slide,
                )
            )
    return metrics


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
        prompt = mlx_vlm_module.prompt_utils.apply_chat_template(processor, config, task, num_images=1)
        return mlx_vlm_module.generate(model, processor, prompt, [image], verbose=False)

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
            output_ids = model.generate(**inputs, max_new_tokens=1024)
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
