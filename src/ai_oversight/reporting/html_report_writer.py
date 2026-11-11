import html
from pathlib import Path

from ai_oversight.schemas import ExtractionResult

_STYLE = """
body { font-family: -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; }
h1 { font-size: 1.4rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin-top: 0.5rem; }
th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; }
th { background: #f4f4f4; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.03em; }
.status-extracted { color: #1a7f37; }
.status-needs_review { color: #9a6700; }
.status-missing { color: #6e7781; }
.status-conflicting { color: #cf222e; }
.status-ambiguous { color: #8250df; }
.vision-integrity { margin: 0.4rem 0 0.8rem; padding: 0.5rem 0.7rem; border-radius: 6px; font-size: 0.85rem; }
.vision-integrity ul { margin: 0.3rem 0 0; padding-left: 1.2rem; }
.vision-pass { background: #dafbe1; color: #1a7f37; }
.vision-review { background: #fff8c5; color: #9a6700; }
.vision-fail { background: #ffebe9; color: #cf222e; }
.field-match-rate { margin: 0.4rem 0; font-size: 0.85rem; color: #57606a; }
"""


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _vision_integrity_block(result: ExtractionResult) -> str:
    vi = result.vision_integrity
    if vi.tasks_total == 0:
        return ""
    summary = (
        f"<strong>Vision parsing: {_escape(vi.rating.upper())}</strong> — "
        f"success rate {_escape(vi.success_rate)} "
        f"({vi.tasks_clean}/{vi.tasks_total} clean)"
    )
    parts = [f'<div class="vision-integrity vision-{vi.rating}">{summary}']
    if vi.loss_notes:
        items = "".join(f"<li>{_escape(note)}</li>" for note in vi.loss_notes)
        parts.append(f"<ul>{items}</ul>")
    parts.append("</div>")
    return "".join(parts)


def _field_match_rate_block(result: ExtractionResult) -> str:
    rate = result.field_match_rate
    matched = sum(1 for field in result.fields.values() if field.value is not None)
    total = len(result.fields)
    return (
        f'<div class="field-match-rate">'
        f"<strong>Field match rate:</strong> {_escape(rate)} ({matched}/{total} fields matched)"
        f"</div>"
    )


def render(results: list[ExtractionResult]) -> str:
    body_parts = ["<h1>Extraction Report</h1>"]
    for result in results:
        body_parts.append(f"<h2>{_escape(result.source_file)}</h2>")
        body_parts.append(_field_match_rate_block(result))
        body_parts.append(_vision_integrity_block(result))
        body_parts.append("<table>")
        body_parts.append("<tr><th>Field</th><th>Status</th><th>Value</th><th>Confidence</th></tr>")
        for name, field in result.fields.items():
            status_class = f"status-{field.status}"
            body_parts.append(
                f"<tr><td>{_escape(name)}</td>"
                f'<td class="{status_class}">{_escape(field.status)}</td>'
                f"<td>{_escape(field.value)}</td>"
                f"<td>{field.confidence}</td></tr>"
            )
        body_parts.append("</table>")

    body = "\n".join(body_parts)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>Extraction Report</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def write(results: list[ExtractionResult], path: Path) -> None:
    path.write_text(render(results))
