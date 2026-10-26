# Causal Workbench

Phase 1 Classical Machine Learning · Team T094, Section E

A clean, white monochrome research application for Corr2Cause reasoning, reviewed text-to-DAG extraction, and Bayesian intervention queries. Strictly classical machine learning and probabilistic graphical models (no neural models or external API keys).

---

## Quickstart & Local Execution

The repository contains all application and experiment source code, documentation, tests, and portable evaluation assets. Full dataset slices, the trained joblib model, and the virtual environment are generated locally.

### 1. Set Up the Environment
Requires Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 2. Run the Application
Launch the Streamlit app:

```sh
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```
Or double-click `run.command` on macOS. Open `http://127.0.0.1:8501`.

The app starts immediately using the bundled JSON inference snapshot and 20 held-out Corr2Cause examples without requiring external downloads.

---

## Reproduce Experiments from Scratch

To reproduce all results from raw data:

```sh
# 1. Download official Corr2Cause splits (train, dev, test)
.venv/bin/python scripts/fetch_data.py

# 2. Train classical baselines on premise-disjoint splits
.venv/bin/python -m src.experiment

# 3. Validate components (extractor, DAG cycle check, Bayesian oracle)
.venv/bin/python -m scripts.validate_components

# 4. Generate EDA visualizations and error analysis
.venv/bin/python -m scripts.analyze

# 5. Run shortcut / spurious cue diagnostic
.venv/bin/python -m scripts.audit_shortcuts

# 6. Export portable model snapshot and demo examples
.venv/bin/python -m scripts.export_app_assets

# 7. Run test suite
.venv/bin/python -m pytest -v
```

`notebooks/phase1_experiments.ipynb` contains the reproducible interactive walkthrough.

---

## Key Experimental Results

- **Selected Model**: TF-IDF (unigrams + bigrams) with balanced Logistic Regression achieves **0.700 test macro F1** on 1,044 retained test rows (positive precision: 0.392, recall: 0.923).
- **Baseline Comparison**: Majority baseline achieves 0.460 macro F1 (0.0 recall); Naive Bayes achieves 0.465; Unigram ablation achieves 0.521.
- **Premise Grouping**: Verbatim premise overlap is strictly purged across splits.
- **Shortcut Diagnostic**: A hypothesis-only control model reaches **0.653 macro F1** without seeing the premise, demonstrating that a large portion of benchmark score stems from template wording artifacts.
- **Intervention Accuracy**: Estimated $P(\text{wet} \mid do(\text{sprinkler}=s))$ matches the ground-truth generating oracle within 0.011 absolute error.

See [`docs/production_audit.md`](docs/production_audit.md) for detailed verification logs and identified methodological limits.

---

## Project Structure

- `app.py`: Six-page white monochrome Streamlit workspace with high-contrast black rules.
- `.streamlit/config.toml`: Clean white theme configuration.
- `src/experiment.py`: Premise-disjoint splitting, classical ML training, bootstrap CI, and model selection.
- `src/runtime.py`: Portable zero-pickle inference snapshot loader (`FrozenReasoner`).
- `src/causal.py`: Lexical relation extraction, graph cycle/self-loop rejection, synthetic SCM, and do-calculus inference.
- `scripts/`:
  - `fetch_data.py`: Downloads pinned Corr2Cause revision from HuggingFace with atomic writes.
  - `validate_components.py`: Evaluates extractor, DAG, and intervention queries against gold fixtures.
  - `analyze.py`: Computes EDA statistics, generates plots, and extracts top feature weights.
  - `audit_shortcuts.py`: Hypothesis-only control experiment measuring spurious cue reliance.
  - `export_app_assets.py`: Generates portable JSON model snapshot and demo data.
- `artifacts/`: Serialized model parameters, metrics, EDA plots, and validation JSONs.
- `docs/`: Phase 1 IEEE report, presentation slides, viva defense guide, literature review, and production audit.
- `tests/`: Automated unit and integration tests for app rendering, causal queries, and split integrity.

---

## Evaluation & Team Details

- **Date**: Monday, 2 November 2026, 3:20–3:45 PM
- **Venue**: Room A403, Newton School of Technology, Rishihood University
- **Evaluators**: Sumit Kumar and Indrapal
- **Team T094, Section E**:
  - Yug Johri (2401010520)
  - Vidit Sachan (2401010502)
  - Aditya Verma (2401010040)
  - Kaavya Gala (2401010204)
  - Ishita Thakur (2401010195)
