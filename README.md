# Causal Workbench

Phase 1 classical machine learning for Team T094, Section E.

A local research application for Corr2Cause reasoning, reviewed text-to-DAG extraction, and Bayesian intervention queries. No neural models or API keys are needed.

## First run after cloning

The repository contains all application and experiment source, documentation, a notebook with cleared outputs, and compact measured results. Full dataset slices, the trained estimator, prediction exports, and the virtual environment are generated locally rather than tracked.

After recreating the environment below, run these commands from the repository root:

```sh
.venv/bin/python scripts/fetch_data.py
.venv/bin/python -m src.experiment
.venv/bin/python -m scripts.validate_components
.venv/bin/python -m scripts.analyze
```

These commands create the model and prediction files required by the app and the data needed by the notebook and tests. The initial download requires internet access. Run the notebook after completing these commands.

## Open the application

Double-click `run.command`, or run from this folder:

```sh
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. Stop the process with Control-C. This is a local preview, not a deployed website.

## Recreate the environment

Use Python 3.12 and either pip or uv. Create a fresh environment for this checkout. The separate original local working copy already has an installed environment.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

With uv, use `uv venv --python 3.12 .venv` and `uv pip install --python .venv/bin/python -r requirements-lock.txt`.

## Reproduce the experiment

Run commands from this directory:

```sh
.venv/bin/python scripts/fetch_data.py
.venv/bin/python -m src.experiment
.venv/bin/python -m scripts.validate_components
.venv/bin/python -m scripts.analyze
.venv/bin/python -m pytest -q
```

The original local working copy includes the raw slices; a fresh clone must fetch them first. `notebooks/phase1_experiments.ipynb` contains a reproducible walkthrough with outputs cleared for source control. Open it with a Python environment containing the locked packages. Downloads require internet; inference on included examples and graph queries do not.

## Read the results correctly

The selected TF-IDF logistic model obtains 0.700 macro F1 on 1,044 retained test rows. The training set is a bounded prefix of 12,000 rows, not the whole Corr2Cause corpus. Exact premise overlap is removed from later splits. Full metrics and the group-bootstrap interval are in `artifacts/metrics.json`.

Corr2Cause labels reasoning entailment, not free-text causal relations. The lexical extractor is evaluated only on 16 authored regression documents. The Bayesian demonstration uses 6,000 seeded synthetic observations with a known generating DAG. The reviewed text graph remains separate from that probability model. These boundaries are visible throughout the application and report.

## Project map

- `app.py`: six-page dark Streamlit workspace.
- `src/experiment.py`: train-only features, classical baselines, split audit and model selection.
- `src/causal.py`: provenance, aliases, graph checks, synthetic observations and inference.
- `scripts/`: source fetching, diagnostics and component validation.
- `data/raw/`: source manifest and hashes; fetch the dataset slices locally.
- `data/processed/`: retained premise-disjoint splits generated locally.
- `data/extraction_gold.json`: authored extraction regression fixture.
- `artifacts/`: compact measured outputs and figures; generate the estimator and prediction files locally.
- `notebooks/`: reproducible walkthrough with outputs cleared.
- `docs/`: report, literature notes, presentation and viva guide.
- `tests/`: meaningful checks of leakage, causal behavior and the interface.

## Evaluation

Monday, 2 November 2026, 3:20-3:45 PM, A403. Faculty: Sumit Kumar and Indrapal.

Team: Yug Johri (2401010520), Vidit Sachan (2401010502), Aditya Verma (2401010040), Kaavya Gala (2401010204), Ishita Thakur (2401010195).

The submission is limited to Phase 1. Contribution claims should be recorded by the team based on actual work; none have been invented. The dataset remains attributed to its authors; no new license is asserted for third-party data. Full third-party dataset copies are excluded from source control. Use the included downloader and review the source dataset terms before redistributing data.
