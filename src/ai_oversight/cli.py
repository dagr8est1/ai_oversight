from pathlib import Path

import typer

import ai_oversight
from ai_oversight.config import Settings
from ai_oversight.extractors import granite_vision, qwen_local, rules
from ai_oversight.parsers import pdf_parser, pptx_parser
from ai_oversight.reporting import csv_writer, json_writer, markdown_writer

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


def _try_parse_document(document: Path):
    """Like `_parse_document`, but for a batch of files: a single bad file
    (missing, unsupported type, unparseable) is reported and skipped rather
    than aborting every other file in the batch."""
    parser = _PARSERS.get(document.suffix.lower())
    if parser is None:
        supported = ", ".join(sorted(_PARSERS))
        typer.echo(f"skipping {document}: unsupported file type {document.suffix!r}; supported: {supported}", err=True)
        return None
    try:
        return parser(document)
    except Exception as exc:  # noqa: BLE001 - one bad file must not abort the whole batch
        typer.echo(f"skipping {document}: failed to parse ({exc})", err=True)
        return None


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
def vision(
    document: Path = typer.Argument(..., exists=True, help="PDF or PPTX report to run the vision pass on."),
    out: Path = typer.Option(..., "--out", help="Where to write the raw vision observations JSON."),
) -> None:
    """Run only the Granite Vision pass and save its raw observations (no
    text backend). Pairs with `extract --vision-cache`: the slow model load
    + inference happens once here, and iterating on how that output maps to
    fields can re-run the fast text backend alone, as many times as needed,
    against the same saved observations.
    """
    parsed = _parse_document(document)
    try:
        vision_fn = granite_vision.build_default_vision_fn(None)
        observations = granite_vision.run_vision_pass(document, parsed, vision_fn)
    except Exception as exc:
        detail = str(exc)
        message = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
        typer.echo(f"vision pass failed: {message}", err=True)
        raise typer.Exit(code=1) from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    granite_vision.save_observations(observations, out)
    typer.echo(f"wrote {out} ({len(observations)} observation(s))")


@app.command()
def extract(
    document: Path = typer.Argument(..., exists=True, help="PDF or PPTX report to extract from."),
    backend: str = typer.Option("rules", "--backend", help="rules (default, no model) or qwen_local."),
    vision: bool = typer.Option(
        None,  # sentinel: resolved to AI_OVERSIGHT_VISION_ENABLED below if --vision isn't passed
        "--vision",
        help="Also recover chart/table data Phase 1 could only flag, via Granite Vision. "
        "Defaults to the AI_OVERSIGHT_VISION_ENABLED env var.",
    ),
    vision_cache: Path = typer.Option(
        None,
        "--vision-cache",
        help="Reuse vision observations saved by the `vision` command instead of "
        "re-running the model. Implies --vision.",
    ),
    out: Path = typer.Option(..., "--out", help="Where to write the extraction result JSON."),
) -> None:
    """Extract the 22 canonical fields, writing <out> and a <out>.review.md summary."""
    if vision_cache is not None:
        vision = True
    elif vision is None:
        vision = Settings.from_env().vision_enabled

    backend_fn = _BACKENDS.get(backend)
    if backend_fn is None:
        supported = ", ".join(sorted(_BACKENDS))
        typer.echo(f"unsupported backend {backend!r}; supported: {supported}", err=True)
        raise typer.Exit(code=1)

    parsed = _parse_document(document)

    vision_observations = []
    vision_warning = None
    backend_input = parsed
    if vision_cache is not None:
        vision_observations = granite_vision.load_observations(vision_cache)
        backend_input = granite_vision.enrich_document_from_observations(parsed, vision_observations)
    elif vision:
        backend_input, vision_observations, vision_warning = granite_vision.enrich_document(document, parsed)

    try:
        result = backend_fn(backend_input)
    except Exception as exc:
        # str(exc) is empty for an argumentless exception (e.g. a bare
        # `raise AttributeError`) -- always include the type so the message
        # is never just "extraction failed: "
        detail = str(exc)
        message = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
        typer.echo(f"extraction failed: {message}", err=True)
        raise typer.Exit(code=1) from exc

    if vision:
        result.vision_observations = vision_observations
        for observation in vision_observations:
            if observation.status != "ok":
                result.extraction_warnings.append(
                    f"page/slide {observation.page_or_slide}: vision {observation.task} "
                    f"needs_review: {observation.note}"
                )
        if vision_warning:
            result.extraction_warnings.append(vision_warning)

    out.parent.mkdir(parents=True, exist_ok=True)
    json_writer.write(result, out)
    review_path = out.with_suffix(".review.md")
    markdown_writer.write(result, review_path)
    typer.echo(f"wrote {out} and {review_path}")


@app.command()
def batch(
    documents: list[Path] = typer.Argument(..., help="PDF/PPTX reports to extract from."),
    backend: str = typer.Option("rules", "--backend", help="rules (default, no model) or qwen_local."),
    vision: bool = typer.Option(
        None,
        "--vision",
        help="Also recover chart/table data Phase 1 could only flag, via Granite Vision. "
        "Defaults to the AI_OVERSIGHT_VISION_ENABLED env var.",
    ),
    out: Path = typer.Option(..., "--out", help="Where to write the combined CSV report."),
) -> None:
    """Extract the 22 canonical fields from each of several documents and
    write one combined CSV: one row per (file, field), with that field's
    status, value, and confidence. A file that fails to parse or extract is
    reported and skipped rather than aborting the whole batch."""
    resolved_vision = Settings.from_env().vision_enabled if vision is None else vision

    backend_fn = _BACKENDS.get(backend)
    if backend_fn is None:
        supported = ", ".join(sorted(_BACKENDS))
        typer.echo(f"unsupported backend {backend!r}; supported: {supported}", err=True)
        raise typer.Exit(code=1)

    results = []
    for document in documents:
        parsed = _try_parse_document(document)
        if parsed is None:
            continue

        backend_input = parsed
        if resolved_vision:
            backend_input, _vision_observations, vision_warning = granite_vision.enrich_document(
                document, parsed
            )
            if vision_warning:
                typer.echo(f"{document}: {vision_warning}", err=True)

        try:
            results.append(backend_fn(backend_input))
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the whole batch
            detail = str(exc)
            message = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
            typer.echo(f"skipping {document}: extraction failed ({message})", err=True)

    if not results:
        typer.echo("no file extracted successfully; nothing written", err=True)
        raise typer.Exit(code=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    csv_writer.write(results, out)
    typer.echo(f"wrote {out} ({len(results)} file(s), {len(results) * 22} row(s))")
