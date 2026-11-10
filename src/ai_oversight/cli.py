from pathlib import Path

import typer

import ai_oversight
from ai_oversight.extractors import qwen_local, rules
from ai_oversight.parsers import pdf_parser, pptx_parser
from ai_oversight.reporting import json_writer, markdown_writer

app = typer.Typer(help="Local AI-usage-report extraction CLI.")

_PARSERS = {
    ".pdf": pdf_parser.parse,
    ".pptx": pptx_parser.parse,
}

_BACKENDS = {
    "rules": rules.extract,
    "qwen_local": qwen_local.extract,
}


def _print_version_and_exit(show_version: bool) -> None:
    # A --version eager option, not an @app.command(), because Typer collapses
    # a Typer() app with exactly one registered command into that command's own
    # callback (dropping the command name) -- an extra subcommand like "parse"
    # would silently change this app's top-level invocation shape.
    if show_version:
        typer.echo(ai_oversight.__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version_and_exit,
        is_eager=True,
        help="Show the installed package version and exit.",
    ),
) -> None:
    pass


def _parse_document(document: Path):
    parser = _PARSERS.get(document.suffix.lower())
    if parser is None:
        supported = ", ".join(sorted(_PARSERS))
        typer.echo(f"unsupported file type {document.suffix!r}; supported: {supported}")
        raise typer.Exit(code=1)

    try:
        return parser(document)
    except Exception as exc:  # pragma: no cover - exact exception type is parser-dependent
        typer.echo(f"failed to parse {document}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def parse(
    document: Path = typer.Argument(..., exists=True, help="PDF or PPTX report to parse."),
    out: Path = typer.Option(..., "--out", help="Where to write the parsed IR JSON."),
) -> None:
    """Parse a report into the intermediate representation (IR). No model is used."""
    parsed = _parse_document(document)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(parsed.model_dump_json(indent=2))
    typer.echo(f"wrote {out} ({len(parsed.elements)} elements, {len(parsed.warnings)} warnings)")


@app.command()
def extract(
    document: Path = typer.Argument(..., exists=True, help="PDF or PPTX report to extract from."),
    backend: str = typer.Option("rules", "--backend", help="rules (default, no model) or qwen_local."),
    out: Path = typer.Option(..., "--out", help="Where to write the extraction result JSON."),
) -> None:
    """Extract the 22 canonical fields, writing <out> and a <out>.review.md summary."""
    backend_fn = _BACKENDS.get(backend)
    if backend_fn is None:
        supported = ", ".join(sorted(_BACKENDS))
        typer.echo(f"unsupported backend {backend!r}; supported: {supported}", err=True)
        raise typer.Exit(code=1)

    parsed = _parse_document(document)

    try:
        result = backend_fn(parsed)
    except Exception as exc:
        typer.echo(f"extraction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    json_writer.write(result, out)
    review_path = out.with_suffix(".review.md")
    markdown_writer.write(result, review_path)
    typer.echo(f"wrote {out} and {review_path}")
