from pathlib import Path

import typer

import ai_oversight
from ai_oversight.parsers import pdf_parser, pptx_parser

app = typer.Typer(help="Local AI-usage-report extraction CLI.")

_PARSERS = {
    ".pdf": pdf_parser.parse,
    ".pptx": pptx_parser.parse,
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


@app.command()
def parse(
    document: Path = typer.Argument(..., exists=True, help="PDF or PPTX report to parse."),
    out: Path = typer.Option(..., "--out", help="Where to write the parsed IR JSON."),
) -> None:
    """Parse a report into the intermediate representation (IR). No model is used."""
    parser = _PARSERS.get(document.suffix.lower())
    if parser is None:
        supported = ", ".join(sorted(_PARSERS))
        typer.echo(f"unsupported file type {document.suffix!r}; supported: {supported}")
        raise typer.Exit(code=1)

    try:
        parsed = parser(document)
    except Exception as exc:  # pragma: no cover - exact exception type is parser-dependent
        typer.echo(f"failed to parse {document}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(parsed.model_dump_json(indent=2))
    typer.echo(f"wrote {out} ({len(parsed.elements)} elements, {len(parsed.warnings)} warnings)")
