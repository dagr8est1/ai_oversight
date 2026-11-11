from pathlib import Path

from typer.testing import CliRunner

from ai_oversight.cli import app

runner = CliRunner()

SAMPLES = Path(__file__).parent.parent / "data" / "samples"
SAMPLE_PDF = SAMPLES / "Sample_Internal_AI_Oversight_Report.pdf"
SAMPLE_PPTX = SAMPLES / "Sample_External_AI_Usage_Disclosure.pptx"

_HEADER = "file_name,field,status,value,confidence,vision_rating,vision_success_rate,vision_data_loss"


def test_batch_writes_one_combined_csv_row_per_field_per_file(tmp_path):
    out = tmp_path / "report.csv"

    result = runner.invoke(
        app, ["batch", str(SAMPLE_PDF), str(SAMPLE_PPTX), "--backend", "rules", "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    rows = out.read_text().splitlines()
    assert rows[0] == _HEADER
    assert len(rows) == 1 + 22 * 2
    assert any(row.startswith("Sample_Internal_AI_Oversight_Report.pdf,submission_id,") for row in rows)
    assert any(row.startswith("Sample_External_AI_Usage_Disclosure.pptx,") for row in rows)


def test_batch_defaults_to_rules_backend(tmp_path):
    out = tmp_path / "report.csv"

    result = runner.invoke(app, ["batch", str(SAMPLE_PDF), "--out", str(out)])

    assert result.exit_code == 0, result.output


def test_batch_skips_a_file_that_fails_to_parse_and_continues(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    out = tmp_path / "report.csv"

    result = runner.invoke(
        app, ["batch", str(missing), str(SAMPLE_PDF), "--backend", "rules", "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert "does_not_exist.pdf" in result.output
    rows = out.read_text().splitlines()
    # only the surviving file's 22 rows -- the missing one contributes none
    assert len(rows) == 1 + 22
    assert all("does_not_exist" not in row for row in rows)


def test_batch_exits_with_an_error_if_every_file_fails(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    out = tmp_path / "report.csv"

    result = runner.invoke(app, ["batch", str(missing), "--backend", "rules", "--out", str(out)])

    assert result.exit_code != 0
    assert not out.exists()


def test_batch_writes_an_html_report_when_out_ends_in_html(tmp_path):
    out = tmp_path / "report.html"

    result = runner.invoke(
        app, ["batch", str(SAMPLE_PDF), str(SAMPLE_PPTX), "--backend", "rules", "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    content = out.read_text()
    assert content.strip().startswith("<!doctype html>")
    assert "Sample_Internal_AI_Oversight_Report.pdf" in content
    assert "Sample_External_AI_Usage_Disclosure.pptx" in content


def test_batch_rejects_unknown_backend_with_a_friendly_error(tmp_path):
    out = tmp_path / "report.csv"

    result = runner.invoke(app, ["batch", str(SAMPLE_PDF), "--backend", "made_up", "--out", str(out)])

    assert result.exit_code != 0
    assert "made_up" in result.output
    assert "Traceback" not in result.output


def test_batch_attaches_vision_observations_and_surfaces_a_failed_task_in_the_csv(tmp_path, monkeypatch):
    from ai_oversight import cli
    from ai_oversight.schemas import VisionObservation

    def fake_enrich_document(document_path, doc, vision_fn=None, settings=None):
        observations = [
            VisionObservation(
                task="tables_json",
                page_or_slide=1,
                raw_response="bad",
                parsed=None,
                status="needs_review",
                note="JSON parse error: boom",
            )
        ]
        return doc, observations, None

    monkeypatch.setattr(cli.granite_vision, "enrich_document", fake_enrich_document)
    out = tmp_path / "report.csv"

    result = runner.invoke(
        app, ["batch", str(SAMPLE_PDF), "--backend", "rules", "--vision", "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    content = out.read_text()
    assert ",fail,0.0," in content
    assert "JSON parse error: boom" in content
