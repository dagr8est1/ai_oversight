import hashlib
import re
from pathlib import Path

import pymupdf

from ai_oversight.parsers.document_ir import Element, ParsedDocument

# Matches the sample reports' "NN / Label" inventory-field headings, e.g. "01 / Submission ID".
LABEL_PATTERN = re.compile(r"^\d{2}\s*/\s*.+$")

# A page is treated as containing a vector-drawn chart when it has at least this
# many near-zero-height horizontal strokes wider than a stray rule/border --
# i.e. repeated gridlines. Plain decorative backgrounds/borders draw at most a
# top and bottom rule (2 strokes); a line/bar chart's axis gridlines do not.
MIN_GRIDLINES_FOR_CHART = 4
GRIDLINE_MIN_WIDTH = 100.0
GRIDLINE_MAX_HEIGHT = 1.0


def parse(path: Path | str) -> ParsedDocument:
    path = Path(path)
    document_id = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

    elements: list[Element] = []
    warnings: list[str] = []

    with pymupdf.open(path) as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            elements.extend(_parse_page_text(page, page_number))
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
    pending_label: tuple[str, list[float]] | None = None

    def flush_unpaired_label() -> None:
        if pending_label is None:
            return
        label_text, label_bbox = pending_label
        elements.append(
            Element(
                element_id=f"page{page_number}.text{id_offset + len(elements)}",
                type="text",
                page_or_slide=page_number,
                bbox=label_bbox,
                method="pymupdf_text_block",
                raw_text=label_text,
            )
        )

    for block in blocks:
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        text = text.strip()
        if not text:
            continue
        bbox = [x0, y0, x1, y1]

        if LABEL_PATTERN.match(text):
            flush_unpaired_label()
            pending_label = (text, bbox)
            continue

        if pending_label is not None:
            label_text, label_bbox = pending_label
            elements.append(
                Element(
                    element_id=f"page{page_number}.kv{id_offset + len(elements)}",
                    type="key_value",
                    page_or_slide=page_number,
                    bbox=[label_bbox[0], label_bbox[1], bbox[2], bbox[3]],
                    method="pymupdf_text_block_pair",
                    content={"label": label_text, "value": text},
                    raw_text=text,
                )
            )
            pending_label = None
            continue

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


def _chart_warnings(page, page_number: int) -> list[str]:
    gridlines = [
        drawing
        for drawing in page.get_drawings()
        if drawing["type"] == "s"
        and drawing["rect"].height < GRIDLINE_MAX_HEIGHT
        and drawing["rect"].width > GRIDLINE_MIN_WIDTH
    ]
    if len(gridlines) >= MIN_GRIDLINES_FOR_CHART and not page.get_images():
        return [
            (
                f"page {page_number}: vector drawing detected (chart-like); structured "
                "numeric series extraction not attempted in Phase 1 (nearby text labels "
                "were still captured as text elements)"
            )
        ]
    return []
