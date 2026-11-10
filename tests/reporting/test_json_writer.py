import json
from datetime import UTC, datetime

from ai_oversight.reporting.json_writer import write
from ai_oversight.schemas import CANONICAL_FIELDS, ExtractionResult, FieldValue


def test_write_produces_valid_json_with_all_22_fields(tmp_path):
    result = ExtractionResult(
        document_id="sha256:abc",
        source_file="x.pdf",
        document_type="internal",
        extracted_at=datetime.now(UTC),
        fields={name: FieldValue(status="missing") for name in CANONICAL_FIELDS},
    )
    out = tmp_path / "out.json"

    write(result, out)

    data = json.loads(out.read_text())
    assert set(data["fields"]) == set(CANONICAL_FIELDS)
    assert data["document_id"] == "sha256:abc"
