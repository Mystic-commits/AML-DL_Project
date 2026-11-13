# Causal Workbench

Classical machine learning for causal claim reasoning and directed relation extraction, with reviewed graphs and Bayesian intervention queries. Team T094, Section E. Phase 1 baseline with a Phase 2 neural extension in progress.

**[Open the live Streamlit app](https://causal-workbench-t094.streamlit.app/)**

## Phase 2 development

DistilBERT fine-tuning and a frozen-encoder control are complete for seed 42. Their **development** directed edge F1 scores are **0.839** and **0.380**, versus the historical TF-IDF/LR development score of **0.739** on the same split. Neural held-out test evaluation and hosted integration are pending. These development-selected scores do not establish a test-set improvement.

See the [five-day plan, training commands and evaluation contract](docs/phase2.md) and [verified development results](artifacts/phase2/development/day2_20261113/summary.json). The Phase 1 results below remain the existing held-out baselines.

## Run locally

Use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open `http://localhost:8501`. On macOS, `run.command` launches the app after environment setup. Bundled JSON model snapshots and demo inputs allow inference without dataset downloads or API keys.

## What the app does

- **Walkthrough:** guided synthetic example, from candidate review to DAG validation, Bayesian fitting and intervention queries.
- **Reasoning lab:** binary claim entailment using the saved Corr2Cause classifier.
- **Extraction benchmark:** directed classification between supplied entity spans using a separate SemEval model.
- **Text → DAG:** review candidate relations, reject invalid edges, and fit probability tables from matching binary observations.
- **Intervention lab:** compare observation with intervention in the rain/sprinkler teaching system.
- **Evaluation:** inspect baselines, data summaries, shortcut diagnostics and validation results.

See [app usage](docs/usage.md) for input formats and the guided workflow.

## Repository structure

```text
app.py                 Streamlit application entrypoint
.streamlit/            Streamlit theme and server configuration
src/                   Classifiers, extraction, graph validation and Bayesian logic
scripts/               Dataset fetching, training, analysis and result verification
notebooks/             Reproducible experiment walkthrough
tests/                 Unit and Streamlit interaction tests
data/                  Demo examples, validation fixtures and source manifests
artifacts/             Inference snapshots, measured results and evaluation evidence
assets/                App stylesheet, research image and licensed fonts
docs/                  Technical usage, data, experiments, deployment and references
.github/workflows/     Continuous validation
requirements.txt       Runtime and experiment dependencies
requirements-dev.txt   Development tools
pyproject.toml         Test and lint settings
run.command            Optional macOS launcher
```

`app.py` contains the interface. `.streamlit/config.toml` configures that same app; it is not a second application. Keeping both paths stable preserves the existing cloud deployment.

## Verified results

| Task | Model or comparison | Held-out metric |
|---|---|---:|
| Corr2Cause entailment | TF-IDF + balanced Logistic Regression | Macro F1 0.700 |
| Corr2Cause entailment | Majority baseline / Naive Bayes | Macro F1 0.460 / 0.465 |
| Corr2Cause shortcut control | Hypothesis only | Macro F1 0.653 |
| SemEval directed extraction | Context TF-IDF + Logistic Regression | Edge F1 0.820 |
| SemEval directed extraction | Lexical rules / entity-only control | Edge F1 0.742 / 0.503 |

Corr2Cause retains 12,000 train, 1,040 dev and 1,044 test examples. The derived three-class SemEval task uses 6,400 train, 1,600 dev and 2,700 retained test examples, with given entity spans. These tasks measure different capabilities and are not an end-to-end causal discovery benchmark.

The Bayesian teaching system uses 6,000 synthetic observations and a supplied generating DAG. Fitted intervention probabilities agree with the known generator within approximately 0.011 absolute error. Causal interpretation requires justified structure, observed relevant confounders and stable mechanisms. Classifier scores never become probability tables.

## Reproduce and validate

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.verify_extraction_run artifacts/runs/phase1_extraction_verified_20261101 --portable
```

Full training and analysis commands are in [experiment reproduction](docs/experiments.md). Dataset contracts and limitations are in [the data dictionary](docs/data_dictionary.md). Background sources are in [references](docs/literature_review.md).

The [deployment notes](docs/deployment.md) describe hosting from `main` with entrypoint `app.py` on Python 3.12.

## Team

Yug Johri, Vidit Sachan, Aditya Verma and Ishita Thakur.

Special Mention: Kaavya Gala.
