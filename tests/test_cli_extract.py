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


def test_extract_failure_message_is_never_blank_even_for_an_argumentless_exception(tmp_path, monkeypatch):
    # a bare `raise SomeError` (no message) str()s to "" -- f"...: {exc}"
    # alone then prints a useless blank message. Reproduced for real: a
    # transformers version mismatch raised a bare AttributeError here and
    # the CLI showed only "extraction failed: ", no clue what broke.
    from ai_oversight import cli

    def broken_backend(parsed):
        raise AttributeError

    monkeypatch.setitem(cli._BACKENDS, "rules", broken_backend)
    out = tmp_path / "internal.json"

    result = runner.invoke(app, ["extract", str(SAMPLE_PDF), "--backend", "rules", "--out", str(out)])

    assert result.exit_code != 0
    assert "AttributeError" in result.output
    assert result.output.strip() != "extraction failed:"


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


def test_vision_command_saves_observations_without_running_a_text_backend(tmp_path, monkeypatch):
    from ai_oversight import cli

    def fake_vision_fn(image_bytes: bytes, task: str) -> str:
        return "Month,Precision\nJul,0.89\n"

    monkeypatch.setattr(cli.granite_vision, "build_default_vision_fn", lambda settings: fake_vision_fn)
    out = tmp_path / "vision_observations.json"

    result = runner.invoke(app, ["vision", str(SAMPLE_PDF), "--out", str(out)])

    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["task"] == "chart2csv"
    assert data[0]["status"] == "ok"


def test_vision_command_friendly_error_when_model_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    out = tmp_path / "vision_observations.json"

    result = runner.invoke(app, ["vision", str(SAMPLE_PDF), "--out", str(out)])

    assert result.exit_code != 0
    assert "vision" in result.output.lower()
    assert "Traceback" not in result.output
    assert not out.exists()


def test_extract_with_vision_cache_never_runs_the_vision_model(tmp_path, monkeypatch):
    from ai_oversight import cli
    from ai_oversight.extractors import granite_vision
    from ai_oversight.schemas import VisionObservation

    def exploding_build_default_vision_fn(settings):
        raise AssertionError("--vision-cache must never load/run the real vision model")

    monkeypatch.setattr(cli.granite_vision, "build_default_vision_fn", exploding_build_default_vision_fn)

    cache_path = tmp_path / "vision_observations.json"
    granite_vision.save_observations(
        [
            VisionObservation(
                task="tables_json",
                source_element_id="page1.vision_scan",
                page_or_slide=1,
                raw_response="...",
                parsed=[
                    {
                        "cells": [
                            {"row": 1, "col": 1, "type": "header", "content": "STATUS"},
                            {"row": 1, "col": 2, "type": "data", "content": "PILOT"},
                        ]
                    }
                ],
                status="ok",
            )
        ],
        cache_path,
    )
    out = tmp_path / "internal.json"

    result = runner.invoke(
        app,
        [
            "extract",
            str(SAMPLE_PDF),
            "--backend",
            "rules",
            "--vision",
            "--vision-cache",
            str(cache_path),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert len(data["vision_observations"]) == 1
