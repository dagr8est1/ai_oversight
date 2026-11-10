import json
import sys
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


def test_extract_qwen_local_without_llm_extra_gives_a_friendly_error(tmp_path, monkeypatch):
    # simulate absence via sys.modules so this test stays deterministic
    # whether or not the `llm` extra happens to be installed in this
    # environment -- otherwise, once installed, this would try to load the
    # real multi-GB model instead of exercising the friendly-error path
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)

    out = tmp_path / "internal.json"

    result = runner.invoke(app, ["extract", str(SAMPLE_PDF), "--backend", "qwen_local", "--out", str(out)])

    assert result.exit_code != 0
    assert "llm" in result.output.lower()
    assert "Traceback" not in result.output


def test_extract_honors_ai_oversight_vision_enabled_env_var(tmp_path, monkeypatch):
    # block the real vision deps so this exercises the "unavailable" path
    # deterministically rather than the real model, whether or not the
    # `vision` extra happens to be installed in this environment
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setenv("AI_OVERSIGHT_VISION_ENABLED", "true")
    out = tmp_path / "internal.json"

    # no --vision flag at all -- the env var alone should turn it on
    result = runner.invoke(app, ["extract", str(SAMPLE_PDF), "--out", str(out)])

    assert result.exit_code == 0, result.stdout
    data = json.loads(out.read_text())
    assert any("vision" in w.lower() for w in data["extraction_warnings"])


def test_extract_vision_without_vision_extra_degrades_to_a_warning_not_a_crash(tmp_path, monkeypatch):
    # --vision must still produce a usable (if vision-less) result when the
    # model is unavailable, per PLAN.md Phase 3's "provide a no-VLM result
    # for constrained Macs with needs_review warnings" -- blocked here via
    # sys.modules so the test is deterministic regardless of whether the
    # `vision` extra happens to be installed in this environment
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    out = tmp_path / "internal.json"

    result = runner.invoke(app, ["extract", str(SAMPLE_PDF), "--vision", "--out", str(out)])

    assert result.exit_code == 0, result.stdout
    data = json.loads(out.read_text())
    assert data["fields"]["submission_id"]["value"] == "AI-INT-2026-041"
    assert any("vision" in w.lower() for w in data["extraction_warnings"])
