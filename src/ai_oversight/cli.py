import typer

import ai_oversight

app = typer.Typer(help="Local AI-usage-report extraction CLI.")


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
