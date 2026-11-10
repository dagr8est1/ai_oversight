import hashlib
import re
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from ai_oversight.parsers.document_ir import Element, ParsedDocument

# A bare 1-2 digit field-ID shape, e.g. "01" or "12", as used in the sample
# disclosure's aligned-textbox inventory rows (slides 3-4).
ROW_ID_PATTERN = re.compile(r"^\d{1,2}$")

# Shapes are grouped into one row when their *top* edges fall within this many
# EMU of each other. Same-row shapes in the sample deck share an identical top
# even when their heights differ (an "ID" cell is shorter than its label/value
# neighbors), so top -- not vertical center -- is the stable row key.
ROW_TOLERANCE_EMU = 10_000


def parse(path: Path | str) -> ParsedDocument:
    path = Path(path)
    document_id = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

    prs = Presentation(path)
    elements: list[Element] = []
    warnings: list[str] = []
    for slide_index, slide in enumerate(prs.slides):
        slide_number = slide_index + 1
        shapes = list(_flatten_shapes(slide.shapes))

        text_shapes = []
        for shape in shapes:
            if getattr(shape, "has_table", False):
                elements.append(_table_element(shape, slide_number, len(elements)))
            elif getattr(shape, "has_chart", False):
                elements.append(_chart_element(shape, slide_number, len(elements)))
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                warnings.append(
                    f"slide {slide_number}: picture shape present; image content "
                    "is not extracted in Phase 1 (would need OCR/vision)"
                )
            elif shape.has_text_frame and shape.text_frame.text.strip():
                text_shapes.append(shape)

        elements.extend(_pair_row_shapes(text_shapes, slide_number, len(elements)))

    return ParsedDocument(
        document_id=document_id,
        filename=path.name,
        format="pptx",
        elements=elements,
        warnings=warnings,
    )


def _flatten_shapes(shapes):
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _flatten_shapes(shape.shapes)
        else:
            yield shape


def _table_element(shape, slide_number: int, id_offset: int) -> Element:
    table = shape.table
    rows = [[cell.text for cell in row.cells] for row in table.rows]
    return Element(
        element_id=f"slide{slide_number}.table{id_offset}",
        type="table",
        page_or_slide=slide_number,
        bbox=_bbox(shape),
        method="pptx_native_table",
        content={"rows": rows},
    )


def _chart_element(shape, slide_number: int, id_offset: int) -> Element:
    chart = shape.chart
    plot = chart.plots[0]
    series = [{"name": s.name, "values": list(s.values)} for s in plot.series]
    title = None
    if chart.has_title:
        title = chart.chart_title.text_frame.text
    return Element(
        element_id=f"slide{slide_number}.chart{id_offset}",
        type="chart",
        page_or_slide=slide_number,
        bbox=_bbox(shape),
        method="pptx_native_chart",
        title=title,
        content={"categories": list(plot.categories), "series": series},
    )


def _pair_row_shapes(shapes, slide_number: int, id_offset: int) -> list[Element]:
    rows = _cluster_rows(shapes)
    elements: list[Element] = []

    for row in rows:
        row.sort(key=lambda s: s.left)
        texts = [s.text_frame.text.strip() for s in row]

        if len(texts) >= 3 and ROW_ID_PATTERN.match(texts[0]):
            field_id, label, value = texts[0], texts[1], texts[2]
            elements.append(
                Element(
                    element_id=f"slide{slide_number}.kv{id_offset + len(elements)}",
                    type="key_value",
                    page_or_slide=slide_number,
                    bbox=_row_bbox(row),
                    method="pptx_aligned_textbox_row",
                    content={"field_id": field_id, "label": label, "value": value},
                    raw_text=value,
                )
            )
            if len(texts) > 3:
                # extra shapes beyond [id, label, value] belong to a different
                # column than "value" -- keep them, grouped, instead of either
                # discarding them or folding them into the value string.
                elements.append(
                    Element(
                        element_id=f"slide{slide_number}.table{id_offset + len(elements)}",
                        type="table",
                        page_or_slide=slide_number,
                        bbox=_row_bbox(row),
                        method="pptx_aligned_textbox_row_cells",
                        content={"cells": texts},
                    )
                )
            continue

        if len(texts) >= 2:
            # no numeric field-ID: still an aligned multi-column row (e.g. a
            # plain data table's header/data rows) -- keep the cells grouped
            # rather than scattering them into unlinked text elements.
            elements.append(
                Element(
                    element_id=f"slide{slide_number}.table{id_offset + len(elements)}",
                    type="table",
                    page_or_slide=slide_number,
                    bbox=_row_bbox(row),
                    method="pptx_aligned_textbox_row_cells",
                    content={"cells": texts},
                )
            )
            continue

        for shape, text in zip(row, texts):
            elements.append(
                Element(
                    element_id=f"slide{slide_number}.text{id_offset + len(elements)}",
                    type="text",
                    page_or_slide=slide_number,
                    bbox=_bbox(shape),
                    method="pptx_text_shape",
                    raw_text=text,
                )
            )

    return elements


def _cluster_rows(shapes) -> list[list]:
    rows: list[list] = []
    for shape in sorted(shapes, key=lambda s: s.top):
        for row in rows:
            if abs(row[0].top - shape.top) <= ROW_TOLERANCE_EMU:
                row.append(shape)
                break
        else:
            rows.append([shape])
    return rows


def _bbox(shape) -> list[float]:
    return [float(shape.left), float(shape.top), float(shape.left + shape.width), float(shape.top + shape.height)]


def _row_bbox(row: list) -> list[float]:
    left = min(s.left for s in row)
    top = min(s.top for s in row)
    right = max(s.left + s.width for s in row)
    bottom = max(s.top + s.height for s in row)
    return [float(left), float(top), float(right), float(bottom)]
