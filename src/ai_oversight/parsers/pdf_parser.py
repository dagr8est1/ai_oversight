import hashlib
import re
from pathlib import Path

import pymupdf

from ai_oversight.parsers.document_ir import Element, ParsedDocument

# Matches the sample reports' "NN / Label" inventory-field headings, e.g. "01 / Submission ID".
LABEL_PATTERN = re.compile(r"^(\d{2})\s*/\s*(.+)$")

# A candidate value block pairs with a pending label only if it starts within
# this many points of the label's bottom edge. The real sample's label->value
# gap is ~9pt; anything much larger is unrelated text lower on the page, and
# pairing it would fabricate a value the document never gave for that field.
MAX_LABEL_VALUE_GAP = 20.0

# A page is treated as containing a vector-drawn chart when it has at least this
# many near-zero-height horizontal strokes wider than a stray rule/border --
# i.e. repeated gridlines. Plain decorative backgrounds/borders draw at most a
# top and bottom rule (2 strokes); a line/bar chart's axis gridlines do not.
MIN_GRIDLINES_FOR_CHART = 4
GRIDLINE_MIN_WIDTH = 100.0
GRIDLINE_MAX_HEIGHT = 1.0

# Margin (points) added around the gridline cluster's bounding box so a crop
# still includes nearby axis labels/legends, not just the bare plot area.
CHART_REGION_MARGIN = 40.0


def parse(path: Path | str) -> ParsedDocument:
    path = Path(path)
    document_id = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

    elements: list[Element] = []
    warnings: list[str] = []

    with pymupdf.open(path) as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            page_elements = _parse_page_text(page, page_number)
            elements.extend(page_elements)

            if not page_elements:
                warnings.append(
                    f"page {page_number}: no text layer extracted (likely scanned); "
                    "OCR was not performed in Phase 1"
                )

            image_count = len(page.get_images())
            if image_count:
                warnings.append(
                    f"page {page_number}: {image_count} raster image(s) present; "
                    "image content is not extracted in Phase 1 (would need OCR/vision)"
                )

            warnings.extend(_chart_warnings(page, page_number))

    return ParsedDocument(
        document_id=document_id,
        filename=path.name,
        format="pdf",
        elements=elements,
        warnings=warnings,
    )


def _parse_page_text(page, page_number: int) -> list[Element]:
    # Group blocks into left-to-right columns first, then read each column
    # top-to-bottom independently. A page-wide y-then-x sort interleaves
    # side-by-side columns (e.g. fields 01-11 at x0=39 next to fields 12-22 at
    # x0=312, same row of y-coordinates) and pairs each label with the wrong
    # column's value -- PLAN.md Section 9 calls this out explicitly.
    elements: list[Element] = []
    for column in _columns_left_to_right(page.get_text("blocks")):
        elements.extend(_pair_labels_and_values(column, page_number, len(elements)))
    return elements


def _columns_left_to_right(blocks: list[tuple], tolerance: float = 20.0) -> list[list[tuple]]:
    anchors: list[float] = []
    for x0 in sorted({round(b[0], 1) for b in blocks}):
        if not anchors or x0 - anchors[-1] > tolerance:
            anchors.append(x0)

    columns: dict[float, list[tuple]] = {anchor: [] for anchor in anchors}
    for block in blocks:
        nearest = min(anchors, key=lambda anchor: abs(anchor - block[0]))
        columns[nearest].append(block)

    return [sorted(columns[anchor], key=lambda b: round(b[1], 1)) for anchor in anchors]


def _pair_labels_and_values(
    blocks: list[tuple], page_number: int, id_offset: int
) -> list[Element]:
    elements: list[Element] = []
    pending: tuple[str, str, list[float]] | None = None  # (field_id, label, bbox)

    def flush_unpaired_label() -> None:
        if pending is None:
            return
        field_id, label, label_bbox = pending
        elements.append(
            Element(
                element_id=f"page{page_number}.text{id_offset + len(elements)}",
                type="text",
                page_or_slide=page_number,
                bbox=label_bbox,
                method="pymupdf_text_block",
                raw_text=f"{field_id} / {label}",
            )
        )

    for block in blocks:
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        text = text.strip()
        if not text:
            continue
        bbox = [x0, y0, x1, y1]

        label_match = LABEL_PATTERN.match(text)
        if label_match:
            flush_unpaired_label()
            pending = (label_match.group(1), label_match.group(2), bbox)
            continue

        if pending is not None:
            field_id, label, label_bbox = pending
            gap = y0 - label_bbox[3]
            if 0 <= gap <= MAX_LABEL_VALUE_GAP:
                elements.append(
                    Element(
                        element_id=f"page{page_number}.kv{id_offset + len(elements)}",
                        type="key_value",
                        page_or_slide=page_number,
                        bbox=[label_bbox[0], label_bbox[1], bbox[2], bbox[3]],
                        method="pymupdf_text_block_pair",
                        content={"field_id": field_id, "label": label, "value": text},
                        raw_text=text,
                    )
                )
                pending = None
                continue
            # too far below the label to be its value -- flush the label
            # unpaired rather than fabricate a pairing, then fall through to
            # emit this block as its own plain text element.
            flush_unpaired_label()
            pending = None

        elements.append(
            Element(
                element_id=f"page{page_number}.text{id_offset + len(elements)}",
                type="text",
                page_or_slide=page_number,
                bbox=bbox,
                method="pymupdf_text_block",
                raw_text=text,
            )
        )

    flush_unpaired_label()
    return elements


def _largest_gridline_group(page) -> list:
    """Candidate gridline-type strokes, grouped by their (x0, x1) span and
    reduced to the largest group. A page-wide header/footer rule also matches
    the raw stroke filter but only ever appears once or twice at its own
    distinct span, so grouping by span (not just counting every stroke)
    keeps a decorative rule from being unioned into the chart's bounding box
    or from padding the warning's stroke count."""
    candidates = [
        drawing
        for drawing in page.get_drawings()
        if drawing["type"] == "s"
        and drawing["rect"].height < GRIDLINE_MAX_HEIGHT
        and drawing["rect"].width > GRIDLINE_MIN_WIDTH
    ]
    groups: dict[tuple[float, float], list] = {}
    for drawing in candidates:
        rect = drawing["rect"]
        key = (round(rect.x0, 1), round(rect.x1, 1))
        groups.setdefault(key, []).append(drawing)
    return max(groups.values(), key=len, default=[])


def chart_region_on_page(page) -> "pymupdf.Rect | None":
    """The gridline cluster's bounding box (with a margin), or None if this
    page has no chart-like vector drawing. Shared by Phase 1's warning and
    Phase 3's vision-crop routing so both agree on what counts as a chart."""
    gridlines = _largest_gridline_group(page)
    if len(gridlines) < MIN_GRIDLINES_FOR_CHART:
        return None

    region = gridlines[0]["rect"]
    for drawing in gridlines[1:]:
        region |= drawing["rect"]

    margin = CHART_REGION_MARGIN
    expanded = pymupdf.Rect(
        region.x0 - margin, region.y0 - margin, region.x1 + margin, region.y1 + margin
    )
    return expanded & page.rect


def _chart_warnings(page, page_number: int) -> list[str]:
    if chart_region_on_page(page) is not None:
        return [
            (
                f"page {page_number}: vector drawing detected (chart-like); structured "
                "numeric series extraction not attempted in Phase 1 (nearby text labels "
                "were still captured as text elements)"
            )
        ]
    return []
