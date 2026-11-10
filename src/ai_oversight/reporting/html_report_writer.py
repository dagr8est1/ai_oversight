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
"""


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def render(results: list[ExtractionResult]) -> str:
    body_parts = ["<h1>Extraction Report</h1>"]
    for result in results:
        body_parts.append(f"<h2>{_escape(result.source_file)}</h2>")
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
