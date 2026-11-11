# AI Usage Report Extraction — Implementation Plan for Claude in VS Code

**Project name:** `ai-oversight-extractor`  
**Status:** Implementation plan / coding-agent handoff  
**Scope:** Local, open-source prototype using fictional AI usage reports; **not** tied to any real organization's workflow or reporting standard.  
**Primary goal:** Parse PDF and PowerPoint reports containing text, visually formatted tables, and charts; extract a standardized set of 22 AI-governance fields with traceable source evidence; identify missing information and inconsistencies.

> **Instructions for Claude Code / Claude in VS Code:** Implement this project incrementally, starting with the deterministic MVP. Inspect the repository and sample files before changing anything. Prefer working code, small modules, tests, and a README over speculative infrastructure. Do not assume the user's Mac has a GPU or a particular memory size. All inference must be local by default, with no paid API or uploaded confidential data. Build an end-to-end CLI workflow first; add the optional visual model only after the baseline works. If a proposed library API has changed, consult its current official documentation and adapt. Do not manufacture extracted values or report an unsupported compliance conclusion. Report completed steps, test results, and remaining limitations at the end.

## 1. Product requirements and boundaries

### Required outcomes

1. Accept `.pdf` and `.pptx` files.
2. Extract ordinary text, paragraphs, page/slide metadata, native tables, native chart series, and table-like aligned text boxes.
3. Recover values from visual PDF charts or embedded-image charts through an **optional** vision-language model (VLM) pass.
4. Preserve the provenance of each element: source filename, page/slide number, element ID, bounding box if available, extraction method, and evidence text.
5. Normalize extracted information into **22 canonical fields**, plus separately structured performance metrics, incident records, and review flags.
6. Allow fields to be absent (`null` / `missing`); never fill gaps from speculation.
7. Detect conflicts (e.g., narrative says 94% while chart says 92%) without silently resolving them.
8. Export machine-readable JSON and a human-readable Markdown review summary.
9. Run offline once local dependencies and model weights have been installed; do not require an OpenAI, Anthropic, or paid Hugging Face API.
10. Provide automated tests and a small hand-checked evaluation dataset for the two samples.

### Non-goals for Version 1

- No integration with any real organization's systems, SEC or FINRA compliance certification, or actual external regulatory submissions.
- No enterprise authentication, cloud hosting, vector database, RAG search service, or complex dashboard.
- No automatic regulatory decision-making or unreviewed risk approval.
- No promise that plotted PDF lines can always be reconstructed numerically from pixels.

## 2. Input fixtures

Put the two existing synthetic documents in `data/samples/`:

- `Sample_Internal_AI_Oversight_Report.pdf` — **4 pages**. Internally developed XGBoost settlement-failure warning model. PDF includes selectable text, field inventory, validation table, charts drawn with vector graphics, and incidents/risk narrative.
- `Sample_External_AI_Usage_Disclosure.pptx` — **7 slides**. External firm's fictional collateral-exception triage copilot. Includes text, field inventory presented as **aligned text shapes (not necessarily native PowerPoint tables)**, and a **native PowerPoint chart on slide 5**.

Both reports contain synthetic values and a 22-field inventory. Do not hard-code their values into extraction logic. The hand-curated expected answers belong in `tests/fixtures/ground_truth.json` only.

### Representative acceptance checks from the documents

| Item | Internal PDF | External PPTX |
|---|---|---|
| Submission ID | `AI-INT-2026-041` | `EXT-2026-118` |
| Application name | `Settlement Failure Early Warning (SFEW)` | `Collateral Exception Triage Copilot (CETC)` |
| Model type | `Gradient-boosted trees (XGBoost)` | `Retrieval-augmented LLM + classifier` |
| Inherent / residual risk | `HIGH` / `MEDIUM` | `HIGH` / `MEDIUM` |
| Monitoring | Daily metrics; monthly drift review | Daily prompt logs; weekly QA sampling |
| Core metrics | Precision 0.91; recall 0.86; ROC-AUC 0.95 | Accuracy 0.92; grounded response rate 0.97 |
| Human oversight | Analyst approves every escalation | Analyst approves every classification and action |

Note: The PDF's page 3 visual time series has July–October precision/recall observations. The PPTX slide 5 native chart has August–October classification accuracy and grounded-response rates. The implementation must preserve time periods and metric names, not just select the last number without context.

## 3. Technology recommendations

### Baseline (implement first)

- **Language:** Python 3.11+.
- **Environment:** `uv` or `venv` + pip, with a reproducible `pyproject.toml`.
- **PDF:** `PyMuPDF` (`fitz`) for text spans, geometry, page renders, embedded images and chart crops.
- **PowerPoint:** `python-pptx` for text, grouped shapes, native tables, charts/series, and slide metadata. Be aware that visible tables may be made of shapes/text boxes instead of `has_table` objects.
- **Document layout fallback:** `Docling` for challenging PDFs and structured page tables; optional in basic installation if it adds heavy dependencies.
- **Schema + validation:** `pydantic`.
- **CLI:** `typer` (or `argparse` if simpler).
- **Tests:** `pytest`.
- **Text field extractor:** `Qwen/Qwen3-4B-Instruct-2507` from Hugging Face, loaded locally through a swappable inference backend. Also implement `--backend rules` for deterministic smoke tests when no model is installed.
- **Structured-output reliability:** Qwen3-4B has no documented grammar-constrained decoding, only prompt-level JSON hints. Do not trust raw model JSON on the first try. Validate every response against the Pydantic schema and retry (with the validation error fed back into the prompt) up to a small fixed number of attempts; fall back to `needs_review` with the raw text preserved as evidence if validation keeps failing. Consider a constrained-decoding library (e.g. `outlines`) if the retry loop proves unreliable in practice.
- **Optional UI:** None for MVP. JSON and Markdown are sufficient.

### Vision enhancement (implement after MVP)

- **Model:** `ibm-granite/granite-vision-4.1-4b` from Hugging Face.
- **Use only when warranted:** scanned pages, inaccessible chart data, rasterized tables, or failed native extraction.
- **Model tasks:** `<chart2csv>` for chart data, `<tables_json>` (or `<tables_html>`) for complex tables. For chart extraction, retain raw output and parse it with normal CSV validation.
- **Routing priority:** native chart data > native text/table objects > PyMuPDF/Docling text/layout > targeted VLM crop > human review.
- **Execution:** choose compatible Transformers/CUDA, Apple Silicon MLX-VLM, or another *verified* local runtime based on hardware. Vision inference is opt-in (`--vision`), not a universal dependency.
- **Memory discipline:** never hold the text LLM and the vision LLM in memory at the same time on constrained hardware (e.g. 16 GB unified memory). Load, use, and release one model before loading the other; prefer quantized/MLX builds (~2.4–3 GB each) over full BF16 (~8 GB each) when both are needed in the same run.

**Design principle:** Do not feed entire PDFs as raw screenshots to the same LLM when native text/underlying chart data can be recovered exactly. Use the LLM for semantic interpretation and normalization, not as a replacement for reliable file parsing.

Useful documentation:

- PyMuPDF: https://pymupdf.readthedocs.io/
- python-pptx: https://python-pptx.readthedocs.io/
- Docling: https://github.com/docling-project/docling
- Qwen3 4B Instruct model: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Granite Vision 4.1 4B model: https://huggingface.co/ibm-granite/granite-vision-4.1-4b

## 4. Processing architecture

```text
.pdf / .pptx
     |
     v
[Document type router]
     |------------------------------|
     v                              v
[PDF adapter: PyMuPDF/Docling]   [PPTX adapter: python-pptx]
     |                              |
     | extract text/tables,         | extract shapes/text,
     | images, chart crops          | charts/series, tables
     |------------------------------|
                    |
                    v
         [Normalized Document IR]
           text / table / chart
             + source evidence
                    |
          [Optional visual routing]
          Granite Vision on crops
                    |
                    v
        [Local text LLM: Qwen3 4B]
         canonicalize 22 fields
                    |
                    v
       [Pydantic validation + rules]
         missing / conflicts / flags
                    |
                    v
       output.json + review.md
```

### Intermediate Representation (IR)

Create a consistent document representation independent of original format. Example:

```json
{
  "document_id": "sha256:...",
  "filename": "Sample_External_AI_Usage_Disclosure.pptx",
  "format": "pptx",
  "elements": [
    {
      "element_id": "slide5.chart1",
      "type": "chart",
      "page_or_slide": 5,
      "bbox": null,
      "method": "pptx_native_chart",
      "title": "Monthly quality trend",
      "content": {
        "categories": ["2026-08", "2026-09", "2026-10"],
        "series": [
          {"name": "classification_accuracy", "values": [0.87, 0.90, 0.92]},
          {"name": "grounded_response_rate", "values": [0.94, 0.96, 0.97]}
        ]
      },
      "raw_text": null
    }
  ],
  "warnings": []
}
```

**IR requirements:**

- Use **1-based** page/slide references in outputs.
- Keep tabular structure as arrays of rows/cells, not flattened whitespace.
- Capture original raw strings and normalizations separately.
- Store chart units (`%`, fraction, count), legend labels, axis labels, and dates.
- Distinguish direct/native values from model-inferred values.
- Preserve multiple possible observations if sources disagree.
- Avoid data loss from grouped PowerPoint shapes and side-by-side PDF columns: do not rely solely on naive reading-order joins.

## 5. Canonical 22-field extraction schema

Exactly these top-level identifiers, shared across PDF/PPTX:

| # | JSON field | Meaning / normalization |
|---:|---|---|
| 01 | `submission_id` | Document submission identifier |
| 02 | `reporting_entity` | Legal entity / reporting organization |
| 03 | `application_name` | AI application and abbreviation |
| 04 | `business_use_case` | Purpose and business workflow |
| 05 | `business_owner` | Responsible business person/team |
| 06 | `technical_owner` | Responsible technical person/team |
| 07 | `model_type` | E.g., XGBoost, RAG LLM + classifier |
| 08 | `model_provider` | Internal team or external service |
| 09 | `model_version` | All named application/model versions |
| 10 | `lifecycle_status` | Pilot, controlled rollout, production, retired, etc. |
| 11 | `first_deployment_date` | ISO `YYYY-MM-DD` |
| 12 | `input_data` | Data sources and categories |
| 13 | `data_sensitivity` | Classification and data-handling notes |
| 14 | `financial_criticality` | Reported criticality and explanation |
| 15 | `human_oversight` | Approval/override/decision rights |
| 16 | `validation_status` | Validation date, party, scope and result |
| 17 | `model_performance` | Metric dictionary with units, periods and samples |
| 18 | `risk_rating` | Explicitly reported inherent and residual ratings |
| 19 | `incident_history` | Count, dates, statuses and consequences |
| 20 | `monitoring_frequency` | Cadence and what is monitored |
| 21 | `fallback_process` | Manual/automated failure-mode backup |
| 22 | `next_review_date` | ISO `YYYY-MM-DD` |

### Value and evidence structure

Do not output untraceable scalar-only fields. Each field should have an evidence wrapper:

```json
{
  "application_name": {
    "value": "Collateral Exception Triage Copilot (CETC)",
    "status": "extracted",
    "evidence": [
      {
        "page_or_slide": 3,
        "element_id": "slide3.shape17",
        "quote": "Collateral Exception Triage Copilot (CETC)",
        "method": "pptx_native_text"
      }
    ]
  }
}
```

Valid `status` values: `extracted`, `missing`, `ambiguous`, `conflicting`, `needs_review`. `value` may be `null`; for complex fields it may contain nested arrays/objects. Include a separate `normalization_notes` when converting dates, percentages, or classification labels. Never use unsupported LLM certainty scores as calibrated probabilities.

### Output shape

```json
{
  "document_id": "sha256:...",
  "source_file": "...",
  "document_type": "internal|external|unknown",
  "extracted_at": "ISO-8601 timestamp",
  "schema_version": "1.0",
  "fields": {"submission_id": {}, "reporting_entity": {}, "...": {}},
  "metrics": [],
  "incidents": [],
  "review_flags": [],
  "extraction_warnings": []
}
```

Use a concrete Pydantic model, not free-form unchecked JSON. `review_flags` must be explicitly labeled **rule-generated suggestions**, not final compliance determinations.

In addition to the stored fields above, derive two document-level signals as Pydantic `computed_field`s (never separately stored, so they can't drift out of sync with the data they're computed from):

- `field_match_rate` — the fraction of the 22 canonical fields that resolved to *any* non-null value, not only a confident `extracted` match; a `needs_review`/`ambiguous`/`conflicting` field with a real value still counts as matched, since there's something for a reviewer to look at.
- `vision_integrity` — only meaningful when the vision pass ran: how many Granite Vision tasks were clean, partially recovered (e.g. a truncated response repaired by keeping only the complete rows), or outright failed, a success rate, and a `pass`/`review`/`fail` rating (any outright failure forces `fail` regardless of other successes on the same document).

## 6. Implementation phases (do these in order)

### Phase 0 — Initialize a clean repository

- [ ] Create project, `pyproject.toml`, `.gitignore`, README, test directories, and samples path.
- [ ] Support Python 3.11+ on macOS/Linux.
- [ ] Define a config file or environment variables for models, local cache and vision opt-in; defaults must not require credentials.
- [ ] Add `ruff` or equivalent lint/format tooling and `pytest`.
- [ ] Do not commit downloaded model weights, private customer documents, caches or generated output files.

**Acceptance:** project installs; `python -m ai_oversight --help` (or equivalent console command) works; tests run without model downloads.

### Phase 1 — Deterministic document parsing (MVP foundation)

- [ ] Implement `PDFParser` using PyMuPDF: extract pages, text blocks/spans and geometry; capture drawings/images; render target pages/crops if required.
- [ ] Add Docling as an **optional** PDF fallback for complex tables/layout and preserve provenance.
- [ ] Implement `PPTXParser` using python-pptx: walk slide shapes (including groups), text, native tables, and charts with category labels and series values.
- [ ] For PPTX aligned textbox tables, create a spatial/text-order grouping heuristic (or an explicit row-layout method), and make sure slides 3 and 4 of the sample are parsed into key/value associations.
- [ ] Normalize both formats into the IR, retaining raw document elements and original numbers.
- [ ] Add parse-only CLI command saving IR JSON.

**Acceptance:** both sample documents parse without a model. Native PPT chart slide 5 produces categories and two named series; PDF page 2 and PPT slides 3–4 preserve paired field names and values. Missing OCR or chart extraction must produce warnings, not crashes.

### Phase 2 — Semantic extraction with local text LLM

- [ ] Implement a pluggable `TextExtractor` interface with `rules` and `qwen_local` backends.
- [ ] Use `Qwen/Qwen3-4B-Instruct-2507` for semantic field mapping from IR elements. Use an explicit schema in the prompt, few-shot examples, and JSON-only response constraints where supported.
- [ ] Wrap each model call in a validate-against-Pydantic-then-retry loop (feeding the validation error back into the prompt) up to a small fixed attempt count; on repeated failure, mark the field `needs_review` and keep the raw model output as evidence rather than discarding it.
- [ ] Chunk by logical page/slide section; pass evidence element IDs with content. Avoid a whole-document blob that destroys provenance.
- [ ] Merge candidate values from multiple chunks; retain all cited evidence and raise a conflict when values differ materially.
- [ ] Run Pydantic validations and a deterministic normalization pass: ISO dates, percentage-to-fraction conversion, consistent risk labels, metric periods, counts.
- [ ] Generate `output.json` and `review.md`.
- [ ] Keep an inference option compatible with local hardware and document the commands for at least one verified execution backend. No model weights should be fetched during ordinary unit tests.

**Acceptance:** for both sample files, the 22 canonical fields are present in the JSON *schema*, with `null` where unavailable; extracted fields cite the correct page/slide and evidence; no fabricated facts; tests verify correct values for the representative fields in Section 2.

### Phase 3 — Optional chart/table VLM extraction

- [ ] Add Granite Vision 4.1 4B behind `--vision` and a `VisionExtractor` interface.
- [ ] Identify pages/shapes with ambiguous chart visuals or raster tables. Crop narrowly, include labels/legends, and use an appropriate render resolution.
- [ ] Use `<chart2csv>` for chart data and `<tables_json>` for tables; parse output defensively and retain raw model output, crop path, and provenance.
- [ ] Cross-check VLM chart numbers against nearby captions/narrative and native extracted values; favor native data where available.
- [ ] Flag unreadable labels, suspicious precision, missing units and conflicts for manual review.
- [ ] Provide a no-VLM result for constrained Macs with `needs_review` warnings.
- [ ] Track Granite Vision's own parsing reliability explicitly, rather than assuming a response that merely parses is lossless: a response truncated by a premature stop token (confirmed not a token-budget issue — raising the output-token cap reproduced identical truncated output) is repaired by keeping only the complete rows and dropping the incomplete trailing one; a response that never parses at all contributes nothing. Surface this per document as `vision_integrity` (see Section 5) instead of silently accepting a partial result as if it were complete.
- [ ] Mitigate the common case of that truncation on a dense, fully-scanned PDF page: since vision generation here is deterministic (no sampling), retrying the identical whole-page call does nothing — instead, a lossy whole-page result triggers one retry that splits the page into top/bottom halves and queries each separately, halving the content either call has to output. Single-level retry only, not recursive; a half still lossy after the split is accepted and reported as-is.

**Acceptance:** opt-in visual processing can recover or explicitly flag the PDF performance graph; PPT native chart does **not** require VLM. Unit tests can mock the vision provider; integration tests run only when model weights are available. Verified against the real model: a document that used to come back `review` (one of several per-page vision tasks truncated) comes back `pass` after the split retry, with the same fields ultimately matched either way.

### Phase 4 — Risk review and evaluation

- [ ] Implement rule-based review flags for missing owner, missing validation, absent human oversight, high financial criticality without a fallback, sensitive data with unclear controls, stale review date, incidents without closure status, and contradictory performance values.
- [ ] Keep **reported risk rating** separate from any **derived review priority**. Never present rule flags as a regulatory finding.
- [ ] Create `tests/fixtures/ground_truth.json` from hand inspection of the two samples (not generated by the same extraction pipeline).
- [ ] Add field-level metrics: exact match for IDs/dates/rating, normalized string match for names, tolerance-aware comparisons for numerical performance metrics, source-citation match, missing-field false-positive rate and conflict-detection accuracy.
- [ ] Surface a document-level match rate alongside the per-field detail, not only an aggregate pass/fail: `field_match_rate` (see Section 5) across all 22 fields, and the same rate narrowed to only the fields a ground-truth check confirms are genuinely present in that document — the latter avoids understating a document that simply states fewer of the 22 facts. The `batch` CLI command's combined CSV/HTML report carries both this and `vision_integrity` per file.
- [ ] Create an evaluation CLI that writes a readable summary plus per-field error details.
- [ ] Add a synthetic conflict test where text says 94% and a chart says 92%; output must expose both pieces of evidence and a `conflicting`/review flag.

**Acceptance:** `pytest` passes offline; evaluation can be run against the two sample documents, and failure details are shown. Do not claim perfect accuracy solely from two samples; these are smoke tests, not a statistical benchmark.

## 7. Suggested folder structure

```text
ai-oversight-extractor/
├── PLAN.md
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── data/
│   └── samples/
│       ├── Sample_Internal_AI_Oversight_Report.pdf
│       └── Sample_External_AI_Usage_Disclosure.pptx
├── src/ai_oversight/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── schemas.py
│   ├── parsers/
│   │   ├── pdf_parser.py
│   │   ├── pptx_parser.py
│   │   └── document_ir.py
│   ├── extractors/
│   │   ├── rules.py
│   │   ├── qwen_text.py
│   │   └── granite_vision.py
│   ├── processing/
│   │   ├── normalize.py
│   │   ├── merge.py
│   │   ├── validate.py
│   │   └── risk_flags.py
│   └── reporting/
│       ├── json_writer.py
│       └── markdown_writer.py
├── tests/
│   ├── fixtures/ground_truth.json
│   ├── test_pdf_parser.py
│   ├── test_pptx_parser.py
│   ├── test_schema.py
│   ├── test_conflicts.py
│   └── test_end_to_end.py
└── outputs/                         # gitignored
```

Prefer a smaller directory structure if implementation needs are simpler; maintain clear separations between parsing, inference, normalization, and reporting.

## 8. Target CLI and example usage

Implement these commands or a simpler equivalent with the same behavior:

```bash
# Install; choose uv or pip, document one primary method
uv sync

# Baseline parsing; should not download models
uv run python -m ai_oversight parse \
  data/samples/Sample_Internal_AI_Oversight_Report.pdf \
  --out outputs/internal.ir.json

uv run python -m ai_oversight parse \
  data/samples/Sample_External_AI_Usage_Disclosure.pptx \
  --out outputs/external.ir.json

# Deterministic smoke-test extraction
uv run python -m ai_oversight extract \
  data/samples/Sample_External_AI_Usage_Disclosure.pptx \
  --backend rules --out outputs/external.json

# Local text model extraction after installing/download of model
uv run python -m ai_oversight extract \
  data/samples/Sample_Internal_AI_Oversight_Report.pdf \
  --backend qwen_local --out outputs/internal.json

# Optional targeted visual extraction
uv run python -m ai_oversight extract \
  data/samples/Sample_Internal_AI_Oversight_Report.pdf \
  --backend qwen_local --vision --out outputs/internal_vision.json

# Several documents, combined into one report: one row per (file, field),
# plus each document's field_match_rate and vision_integrity rating. CSV by
# default; --out ending in .html writes a styled HTML report instead.
uv run python -m ai_oversight batch \
  data/samples/Sample_Internal_AI_Oversight_Report.pdf \
  data/samples/Sample_External_AI_Usage_Disclosure.pptx \
  --backend qwen_local --vision --out outputs/report.html

# Test and evaluate
uv run pytest -q
uv run python -m ai_oversight evaluate --ground-truth tests/fixtures/ground_truth.json
```

Do not mark the CLI complete until the displayed commands match implemented arguments. If an optional backend is not installed, return a friendly actionable error rather than silently calling a hosted API.

## 9. Important engineering decisions

### PowerPoint tables are not always actual tables

Slides 3 and 4 of the supplied PPTX visually resemble tables but are composed of multiple text shapes. An implementation checking only `shape.has_table` will miss their row/column relationships. Use positions (`top`, `left`, `width`, `height`) and reading order, or pair adjacent field labels and value shapes using layout heuristics.

### Chart extraction should be source-first

Native PPTX chart series preserve values exactly. PDF charts are often drawings with text labels but without an accessible data table; recover text values from nearby labels and tables first, then use a chart VLM on crops. Treat vision-derived numbers as estimates requiring review when precision or axis interpretation is uncertain.

### Evidence and auditability

Every extracted field must point to specific source elements. Store the original quote, numeric representation, element ID, page/slide number, and processing method. If the same field has values in a paragraph and a chart, record both.

### Risk and privacy

These fixtures are entirely fictional. For future real financial reports, never transmit data to a hosted inference service without explicit authorization; avoid logging raw confidential content; apply data minimization and local access controls. Separate inference/extraction problems from governance-policy judgments.

### Hardware portability

No assumption that a Mac has at least 16 GB unified memory. Model inference must be optional/configurable; parsing, validation and tests should still run without a GPU. Offer a local quantized runtime if confirmed compatible, and document trade-offs rather than assuming one fixed model package works on every Mac. Never require the text LLM and the vision LLM to be resident simultaneously; load one, finish its pass, release it, then load the other.

## 10. Definition of done

- [ ] Clean installation on Python 3.11+.
- [ ] Both sample documents parse successfully without any LLM.
- [ ] PPT native chart is extracted into numeric series, with dates/legend labels.
- [ ] Aligned text-box inventory rows in PPT are associated correctly.
- [ ] An output contains the same **22 named fields** for every document (including missing values).
- [ ] Every populated field has a traceable, valid source location.
- [ ] No fabricated values; numerical units and date formats are consistent.
- [ ] `review.md` summarizes key facts, missing disclosures, conflicts, and review flags.
- [ ] Tests run offline; sample-ground-truth evaluation reports errors clearly.
- [ ] Optional vision feature is isolated and does not break the baseline.
- [ ] Granite Vision's own parsing reliability is tracked per document and surfaced in the `batch` report, not silently assumed to be lossless.
- [ ] README documents setup, model downloads, CLI commands, limitations, and examples.

## 11. First instruction to Claude

**Start now with Phases 0 and 1.** Inspect the two input documents, initialize the Python project, implement deterministic parsing into the intermediate JSON format, and write tests proving that the PDF page-2 inventory, PPT slides 3–4 inventory, and PPT slide-5 native chart are captured. Then proceed to Phase 2 using a local Qwen model through a provider interface. Do not spend time on a UI, cloud architecture, or fine-tuning before the extraction pipeline and evaluation work. If model hardware constraints block local inference, finish and validate the deterministic parser and clearly document the model step that remains to run.
