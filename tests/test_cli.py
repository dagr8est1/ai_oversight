from typer.testing import CliRunner

from ai_oversight.cli import app

runner = CliRunner()


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_version_flag_prints_package_version():
    import ai_oversight

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert ai_oversight.__version__ in result.stdout
