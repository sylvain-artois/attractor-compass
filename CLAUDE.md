# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`attractor_compass` (Attractor Compass): a small, standalone Python library that scores a **French** political text on Bruno Latour's two attractor axes (Local↔Global, Hors-Sol↔Terrestre, from *Où atterrir ?*). Public API is a single function, `attractor_compass.score(text)`, plus an `attractor-compass` CLI ([cli.py](src/attractor_compass/cli.py)). Distribution name `attractor-compass`, import name `attractor_compass`.

## Commands

```bash
pip install -e ".[dev,baselines]"
python -m spacy download fr_core_news_lg   # required; NOT a pip dependency

pytest                                     # full suite (model tests self-skip if models absent)
pytest tests/test_stats.py::test_name      # single test
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest   # mimic CI: no-model layer only

ruff check . --fix && ruff format .        # CI runs `ruff check .` and `ruff format --check .`

# Regenerate the golden after an intentional scoring change:
rm tests/golden_score.json && pytest tests/test_score.py   # first run mints, second asserts

bash examples/run_baselines.sh             # Wordscores baseline benchmark on the fixture corpus (writes to out/)
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) has three jobs: `lint` (ruff), `test` (Python 3.10–3.12 with HF forced offline, so only the no-model tests — `test_decoupling`, `test_stats`, `test_calibrate`, `test_compare`, `test_seed_selection`, `test_text_ci` — run there), and `models` (x86 + arm64 runners: downloads spaCy FR + both HF checkpoints, writes a golden drift table to the run summary, then runs `test_score` and `test_baselines`). The golden is float-sensitive across machines/torch builds; tune its tolerance in `tests/test_score.py` from the `models` job's drift table, not from a single local run.

## Architecture

Scoring pipeline ([score.py](src/attractor_compass/score.py)):

1. **Segment** — `base._segment_answer` splits text into spaCy docs (≥3 blank-line paragraphs → per paragraph; long OCR-ish text → per single-newline line; else 3-sentence sliding window). Must stay identical to what calibration used, or scores drift from the golden/calibration numbers.
2. **Stance** — `metrics/latour_stance.LatourStanceMetric`: zero-shot NLI per chunk against pro/contra hypotheses per pole (from `config/attractor-compass-stance-hypotheses.yml`); `stance[P] = mean(max pro) − mean(max contra)` ∈ [-1, 1]. Never raises — returns a zero-filled payload with `error` on failure.
3. **Cosine SemAxis + blend** — `metrics/attractor_compass.AttractorCompassMetric`: mean-pooled seed embeddings (from `config/attractor-compass-seeds.yml`) → pole vectors; per-chunk softmax at τ, averaged; then reads the stance result from `ctx.metrics_so_far["latour_stance"]` and applies the additive-γ blend `max(0, cos + γ·stance)` renormalised. (A legacy `stance_blend_alpha` blend also exists; `gamma` takes precedence.)

Key pieces:
- `metrics/thematic_radar.py` is the shared geometry backbone (`_chunk_doc`, `_softmax_aggregate`, `DEFAULT_SOFTMAX_TAU=0.05`) imported by both metrics so they chunk identically.
- `base.py` defines `AbstractMetric` (`options` + `compute(ctx) -> dict`) and `AnalysisContext` (segmented docs, lazy zero-arg `embedder`/`nli` loaders, `project_root` for resolving packaged YAML, `metrics_so_far`).
- `runtime.py` holds `lru_cache` lazy singletons for spaCy, the embedder (`dangvantuan/sentence-camembert-large`, `max_seq_length` capped at 512 to avoid CamemBERT position overflow) and the NLI pipeline (`cmarkea/distilcamembert-base-nli`). `ATTRACTOR_COMPASS_EMBEDDER_DEVICE` / `ATTRACTOR_COMPASS_NLI_DEVICE` opt onto CUDA; default CPU. Models are frozen — the calibration depends on them.
- Calibrated defaults live in `score.py`: seeds v4, γ=1.0, τ=0.03 (overriding the metric's own 0.05 fallback). Seed/hypothesis YAML files carry their own version history and authoring principles in header comments — read them before editing (hypotheses must target generic inversion families, not specific authors). YAML key order (terrestre → global → hors_sol → local) is a convention.
- `stats.py`: pure-numpy bootstrap CIs for accuracy reporting.
- `baselines/` (optional `[baselines]` extra): Wordscores lexical baseline. `calibrate.py` builds a lexicon, `compare.py` scores texts/inversions/hold-out and writes a comparison report; both are `python -m` CLIs driven by `CORPUS_BASE_PATH` (expects a `corpus_attractor_compass/` subdir). `_corpus_loader.py` owns the corpus file format (YAML frontmatter or legacy flat header; `attracteur` labels folded to 4 poles). These never load the embedder/NLI.

## Hard invariants

- **Decoupling** ([tests/test_decoupling.py](tests/test_decoupling.py)): the package must stay a pure library — no Redis/Postgres/Qdrant/server-runtime imports anywhere in `src/`, and `import attractor_compass` must not eagerly import torch/transformers. Keep heavy imports inside functions (as in `runtime.py`).
- Config YAMLs are package data (`[tool.setuptools.package-data]`); resolve them via `ctx.project_root`, not CWD.
- Test fixture corpus under `tests/fixtures/` is synthetic and CC0; don't add license-gated prose. The real calibration corpus is the HF dataset `afk-live/attractor-compass-corpus`.
