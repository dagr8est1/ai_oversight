import json
from pathlib import Path

from typer.testing import CliRunner

from ai_oversight.cli import app

runner = CliRunner()

SAMPLE_PDF = Path(__file__).parent.parent / "data" / "samples" / "Sample_Internal_AI_Oversight_Report.pdf"


def test_extract_rules_backend_writes_json_and_review_markdown(tmp_path):
    out = tmp_path / "internal.json"

    result = runner.invoke(app, ["extract", str(SAMPLE_PDF), "--backend", "rules", "--out", str(out)])

    assert result.exit_code == 0, result.stdout
    data = json.loads(out.read_text())
    assert len(data["fields"]) == 22
    assert data["fields"]["submission_id"]["value"] == "AI-INT-2026-041"

    review_path = out.with_suffix(".review.md")
    assert review_path.exists()
    assert "submission_id" in review_path.read_text()


def test_extract_defaults_to_rules_backend(tmp_path):
    out = tmp_path / "internal.json"

    result = runner.invoke(app, ["extract", str(SAMPLE_PDF), "--out", str(out)])

    assert result.exit_code == 0, result.stdout


def test_extract_rejects_unknown_backend_with_a_friendly_error(tmp_path):
    out = tmp_path / "internal.json"

    result = runner.invoke(app, ["extract", str(SAMPLE_PDF), "--backend", "made_up", "--out", str(out)])

    assert result.exit_code != 0
    assert "made_up" in result.output
    assert "Traceback" not in result.output


def test_extract_qwen_local_without_llm_extra_gives_a_friendly_error(tmp_path):
    out = tmp_path / "internal.json"

    result = runner.invoke(app, ["extract", str(SAMPLE_PDF), "--backend", "qwen_local", "--out", str(out)])

    assert result.exit_code != 0
    assert "llm" in result.output.lower()
    assert "Traceback" not in result.output
