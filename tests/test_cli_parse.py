import json
from pathlib import Path

from typer.testing import CliRunner

from ai_oversight.cli import app

runner = CliRunner()

SAMPLE_PDF = Path(__file__).parent.parent / "data" / "samples" / "Sample_Internal_AI_Oversight_Report.pdf"
SAMPLE_PPTX = Path(__file__).parent.parent / "data" / "samples" / "Sample_External_AI_Usage_Disclosure.pptx"


def test_parse_pdf_writes_ir_json_without_downloading_a_model(tmp_path):
    out = tmp_path / "internal.ir.json"

    result = runner.invoke(app, ["parse", str(SAMPLE_PDF), "--out", str(out)])

    assert result.exit_code == 0, result.stdout
    data = json.loads(out.read_text())
    assert data["format"] == "pdf"
    assert any(el["type"] == "key_value" for el in data["elements"])


def test_parse_pptx_writes_ir_json(tmp_path):
    out = tmp_path / "external.ir.json"

    result = runner.invoke(app, ["parse", str(SAMPLE_PPTX), "--out", str(out)])

    assert result.exit_code == 0, result.stdout
    data = json.loads(out.read_text())
    assert data["format"] == "pptx"


def test_parse_rejects_unsupported_extension_with_a_friendly_error(tmp_path):
    bogus = tmp_path / "report.txt"
    bogus.write_text("not a real report")

    result = runner.invoke(app, ["parse", str(bogus), "--out", str(tmp_path / "out.json")])

    assert result.exit_code != 0
    assert "unsupported" in result.stdout.lower()
