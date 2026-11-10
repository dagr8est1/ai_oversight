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

Add `--vision` to recover chart/table content Phase 1 could only flag (e.g.
the PDF's vector-drawn monthly performance chart, or a fully-scanned page),
using `ibm-granite/granite-vision-4.1-4b` via `<chart2csv>`/`<tables_json>`.
Vision runs *before* the text backend, not after: its recovered content is
converted into ordinary IR elements (a `chart` element for chart2csv, a
`key_value` element per row for a 2-column table, a `table` element for a
wider one) and merged into the document, so `rules` or `qwen_local` picks it
up through their normal processing — a recovered chart feeds the same
conflict check a native chart does, and a recovered label/value pair gets
mapped to a canonical field by whichever backend is semantically capable of
it. `rules` only matches by numbered `field_id`, so it ignores vision-derived
elements (they have none); `--backend qwen_local --vision` is what actually
maps vision-recovered, arbitrarily-worded labels (e.g. "APPLICATION ID",
"SUBMITTING FIRM") to the 22 canonical fields. The raw recovered content is
also retained in full on `vision_observations`, independent of whether
anything maps to a field. Native chart/table data (e.g. the PPTX's slide-5
chart) never needs vision — Phase 1 already extracted it exactly. Requires
the `vision` extra; without it, `--vision` degrades to an extraction warning
rather than failing the whole command, so it's safe to try on a constrained
Mac:

```bash
uv sync --extra vision
uv run python -m ai_oversight extract data/samples/Sample_Internal_AI_Oversight_Report.pdf --vision --out outputs/internal.vision.json

# the combination that maps vision-recovered labels to the 22 fields:
uv sync --extra llm --extra vision
uv run python -m ai_oversight extract data/samples/Vision_Required_Internal_AI_Risk_Report.pdf --backend qwen_local --vision --out outputs/internal.vision_qwen.json
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

## Known limitations (Phase 3 — vision extraction)

- Verified end-to-end with the real `ibm-granite/granite-vision-4.1-4b`
  model (not just an injected fake vision call) against the two
  `Vision_Required_*` sample fixtures: real recovered table/chart content
  from both a fully-scanned PDF and a PPTX with picture-only slides. The
  `qwen_local` + `--vision` combination was also verified end-to-end this
  way, mapping recovered labels like "APPLICATION ID" to `submission_id`.
- `--vision` without `--backend qwen_local` recovers content into
  `vision_observations` and (for chart2csv) `metrics`, but `rules` has no
  way to map an arbitrarily-worded recovered label to a canonical field
  (it only matches by numbered `field_id`) — the 22 `fields` stay as
  whatever `rules` found natively, which is often nothing on a fully
  scanned/picture-only document.
- PDF chart cropping renders the gridline-cluster's own bounding box (plus a
  margin) at 2x zoom, not the whole page; it does not yet try multiple crop
  sizes if the model misreads a cramped legend or axis label.
- A PPTX picture shape is always routed to `<tables_json>`, never
  `<chart2csv>` — there's no signal in a bare picture shape to tell which
  task fits, and neither original sample contains one (only the
  `Vision_Required_*` fixtures and the unit tests do).
- Vision-recovered metrics are cross-checked only against the single
  already-extracted narrative value for `model_performance`, not against the
  chart's own native data labels when present as text (e.g. the PDF page 3's
  axis values, already captured as plain text in Phase 1) — a source-first
  three-way check (native labels, narrative, vision) is not yet implemented.
- The chart2csv-to-chart-element conversion picks the "latest" period by row
  order (via the same `collect_metrics`/`detect_performance_conflicts` both
  backends already use for native charts), not by parsing and sorting the
  period column — a model that returns rows out of chronological order would
  pick the wrong "latest" value for conflict comparison.
- Vision-derived metrics aren't flagged as estimates in their own right
  (PLAN.md Section 9 asks to treat them as such) — they're only flagged
  indirectly, via a conflict, if they disagree with the narrative.
- A wide (>2-column) recovered table becomes one generic `table` element
  (e.g. an incident or control list) that `qwen_local` sees as context but
  doesn't map cell-by-cell to a field; only 2-column label/value tables get
  mapped as individual canonical-field candidates.
- `normalize_risk_rating`'s inherent/residual capture can still cross a
  clause boundary on unusual wording (e.g. "Inherent rating TBD; residual
  rating MEDIUM. Prior residual LOW" misreads which value is inherent vs.
  residual) — the whitelist fix closed the original bug (capturing a
  non-rating word) but not this narrower one; neither real sample's wording
  triggers it.

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
