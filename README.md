# ai_oversight

Local, open-source prototype for extracting a standardized set of fields from
AI-governance reports (PDF/PPTX). Fictional sample data only; not an official
DTCC workflow or reporting standard. See [PLAN.md](PLAN.md) for the full
design and phased roadmap.

## Setup

```bash
just venv
```

## Development

```bash
just test   # uv run pytest
just lint   # uv run ruff check .
```

## CLI

Parse a report into the intermediate representation (IR) JSON. This step is
fully deterministic and never downloads or runs a model:

```bash
uv run python -m ai_oversight parse data/samples/Sample_Internal_AI_Oversight_Report.pdf --out outputs/internal.ir.json
uv run python -m ai_oversight parse data/samples/Sample_External_AI_Usage_Disclosure.pptx --out outputs/external.ir.json
uv run python -m ai_oversight --version
```

Extract the 22 canonical fields (plus metrics/incidents/review flags) from the
IR. The default `rules` backend is deterministic and downloads nothing; it
maps each document's own numbered field inventory directly to the canonical
schema. Writes `<out>` and a `<out>.review.md` human-readable summary:

```bash
uv run python -m ai_oversight extract data/samples/Sample_Internal_AI_Oversight_Report.pdf --out outputs/internal.json
uv run python -m ai_oversight extract data/samples/Sample_External_AI_Usage_Disclosure.pptx --out outputs/external.json
```

The `qwen_local` backend instead prompts `Qwen/Qwen3-4B-Instruct-2507` per
page/slide for semantic field mapping, with a validate-then-retry loop against
the schema (falls back to `needs_review` rather than fabricating a value) and
conflict detection across chunks. It requires the `llm` extra and downloads
the model on first use — nothing is fetched by the test suite:

```bash
uv sync --extra llm
uv run python -m ai_oversight extract data/samples/Sample_Internal_AI_Oversight_Report.pdf --backend qwen_local --out outputs/internal.qwen.json
```

## Configuration

All settings are environment variables; defaults require no credentials and
no network access:

| Variable | Default | Purpose |
|---|---|---|
| `AI_OVERSIGHT_CACHE_DIR` | `~/.cache/ai_oversight` | Local model/cache directory |
| `AI_OVERSIGHT_TEXT_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Phase 2 local text extraction model |
| `AI_OVERSIGHT_VISION_MODEL` | `ibm-granite/granite-vision-4.1-4b` | Phase 3 optional vision model |
| `AI_OVERSIGHT_VISION_ENABLED` | `false` | Opt in to vision extraction (Phase 3) |

## Known limitations (Phase 2 — semantic extraction)

- The `rules` backend maps a canonical field only when the source document
  used the same "NN / Label" numbered-inventory convention as the two
  samples; a differently structured report would need the `qwen_local`
  backend (or a new rules heuristic) instead.
- Conflict detection between a narrative field and a chart compares the
  chart's *last* period's value against the narrative value — it does not
  check every historical period for a mismatch.
- `incidents` is only populated from PPTX aligned-textbox table rows whose
  first cell looks like a short date (e.g. "Sep 07"); PDF incident narratives
  stay as the `incident_history` field's free text (see the PDF-table
  limitation below).
- `qwen_local`'s retry loop re-prompts on invalid JSON but does not yet use
  constrained decoding (e.g. `outlines`); if retries prove unreliable in
  practice, that is the next thing to add, per PLAN.md Section 3.

## Known limitations (Phase 1 — deterministic parsing only)

- No OCR: a scanned/image-only page produces a `no text layer` warning rather
  than fabricated text.
- Raster images (PDF or PPTX) are flagged in `warnings`, not parsed — their
  content (e.g. a chart screenshot) is not extracted in Phase 1.
- PDF vector-drawn charts (e.g. the sample's monthly performance line chart)
  are flagged with a warning; their numeric series are not reconstructed.
  The chart's own text labels (axis ticks, categories) are still captured as
  plain text elements, since PDF charts are source-first: recover what the
  text layer already gives before ever reaching for a vision model.
- PDF tables (e.g. the sample's validation-results table) are captured as
  plain text blocks, not structured rows/cells. Reconstructing them is
  deferred to the optional `docling` extra (`pip install .[docling]`), which
  is declared but not yet wired up — no code depends on it today.
- PPTX aligned-textbox rows without a numeric field ID (e.g. a risk register
  or incident table) are captured as a `table` element's `cells`, keeping the
  row grouping instead of scattering it into disconnected text elements.
