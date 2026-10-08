# Attractor Compass

[![CI](https://github.com/sylvain-artois/attractor-compass/actions/workflows/ci.yml/badge.svg)](https://github.com/sylvain-artois/attractor-compass/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Score a French political text on **Bruno Latour's two attractor axes** from
*Où atterrir ?* (2017):

```
        Terrestre
            │
   Local ───┼─── Global
            │
        Hors-Sol
```

Each pole is defined by seed phrases. The metric mean-pools the seed embeddings
into a pole vector (a [SemAxis](https://arxiv.org/abs/1806.05201)-style
projection), scores each sentence-chunk by softmax-normalised cosine similarity,
then **blends in a zero-shot NLI stance layer** so a text that *uses* a pole's
vocabulary to *attack* it (a climate-denier writing dense ecological prose) is
not misread as belonging to that pole.

It is a self-contained, dependency-light library: `pip install`, call `score()`,
and place a single text on the two axes — no server, no database, no pipeline
runtime.

## Install

```bash
pip install -e .
python -m spacy download fr_core_news_lg     # FR pipeline (not a pip dep)
```

The two transformer models load lazily on the first `score()` call and are
cached afterwards:

- embedder — [`dangvantuan/sentence-camembert-large`](https://huggingface.co/dangvantuan/sentence-camembert-large)
- NLI head — [`cmarkea/distilcamembert-base-nli`](https://huggingface.co/cmarkea/distilcamembert-base-nli)

## Use

```python
from attractor_compass import score

result = score("Nous devons habiter la Terre et composer avec le vivant.")
print(result["dominant_pole"])  # 'terrestre'
print(
    result["scores"]
)  # {'terrestre': 0.79, 'global': 0.15, 'hors_sol': 0.06, 'local': 0.0}
```

Or from the command line:

```bash
attractor-compass "Nous croyons en la croissance technologique illimitée."
attractor-compass --file speech.txt --json
```

`score()` returns the `attractor_compass` block: the four-pole blended `scores`, the
pre-blend `cosine_scores`, the NLI `stance_scores` (in `[-1, +1]`), and a
convenience `dominant_pole`. Pass `use_stance=False` for the pure cosine
projection (no NLI load), or `gamma=` to tune the additive stance blend (γ = 1.0
and the softmax temperature τ = 0.03 are the calibrated defaults).

## Method

| Layer | What it does |
|---|---|
| **Cosine SemAxis** | mean-pool each pole's seed phrases → pole vector; per-chunk softmax over poles at temperature τ, averaged across chunks. |
| **NLI stance** | per chunk, zero-shot entailment against pro/contra hypotheses per pole; `stance[P] = mean(pro) − mean(contra)` ∈ `[-1, +1]`. |
| **Additive-γ blend** | `blended[P] = max(0, cosine[P] + γ·stance[P])`, renormalised. Neutral stance passes the cosine through; a strong contra stance suppresses a pole. |

The seed phrases and stance hypotheses live in
[`src/attractor_compass/config/`](src/attractor_compass/config/), with the calibration
(seeds v4, γ = 1.0, τ = 0.03) baked in as the defaults.

## Baselines & reproducible benchmark

The repo also ships the **Wordscores** (Laver–Benoit–Garry 2003)
lexical-scaling baseline and a benchmark that regenerates a method-comparison
table — entirely offline, from a tiny synthetic CC0 fixture corpus (no
license-gated prose):

```bash
pip install -e ".[baselines]"
python -m spacy download fr_core_news_lg
bash examples/run_baselines.sh
```

## Calibration corpus

The labelled calibration corpus is published as the Hugging Face dataset
[`afk-live/attractor-compass-corpus`](https://huggingface.co/datasets/afk-live/attractor-compass-corpus).
It is **non-consumptive**: each record carries the gold pole annotation, the
audit role (SEED / INVERSION-HOLD / TEST / SANITY), provenance and derived
features (doc-level embedding, shuffled lemma bag-of-words, NLI stance vector).
Full prose ships only for openly licensed texts.

```python
from datasets import load_dataset

ds = load_dataset("afk-live/attractor-compass-corpus")
ds["train"], ds["test"]
```

The baseline CLIs above read a markdown corpus laid out like
[`tests/fixtures/corpus/`](tests/fixtures/corpus/) (`CORPUS_BASE_PATH`
containing a `corpus_attractor_compass/` directory, hold-out under `tests/`).

## In use

The Attractor Compass is one of the metrics behind
[*Le poids des mots*](https://afk.live/fr/le-poids-des-mots/) on afk.live, a
series of analyses of French political speech. Its calibration and limits are
documented in the
[methodology page](https://afk.live/fr/le-poids-des-mots/methodologie/).

## Tests

```bash
pip install -e ".[dev,baselines]"
pytest
```

`tests/test_decoupling.py` runs without any model and asserts the package stays
a pure library — no server or database client creeps in, and a bare `import
attractor_compass` never eagerly loads torch. The functional, golden and benchmark
tests skip themselves when the models or the FR spaCy pipeline are not installed.

## Contributing

Bug reports, baselines, and calibration improvements are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the layered test suite, and
the lint/format workflow.

## License

MIT — see [LICENSE](LICENSE). The synthetic fixture corpus under
`tests/fixtures/` is CC0.
