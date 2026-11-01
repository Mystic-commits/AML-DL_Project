# Causal Workbench

Phase 1 Classical Machine Learning · Team T094, Section E

A presentation-matched research application in warm beige, deep red, and locally bundled Poppins typography for Corr2Cause reasoning, reviewed text-to-DAG extraction, and Bayesian intervention queries. Strictly classical machine learning and probabilistic graphical models (no neural models or external API keys).

---

## Viva demonstration

Start with **Viva demo** in the sidebar: review candidates → validate DAG → fit the teaching network → compare observation and intervention. See [the click-by-click viva script](docs/streamlit_viva_demo.md). Detailed labs remain available for custom experiments.

## Hosted app

[Open Causal Workbench](https://causal-workbench-t094.streamlit.app/)

Deployed from `main`, entrypoint `app.py`, on Streamlit Community Cloud with Python 3.12. No API keys are required. Changes pushed to `main` automatically update the hosted app. The free host may need a short wake-up before the viva; open the link in advance.

Active team: Yug Johri, Vidit Sachan, Aditya Verma, and Ishita Thakur.

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

## Phase 1 completion update (1 November 2026)

This submission remains **classical ML only**, as confirmed by the team. The new external extraction experiment supplements Corr2Cause reasoning and the synthetic Bayesian example.

| Directed extraction model | Precision | Recall | Edge F1 |
|---|---:|---:|---:|
| Majority (no relation) | 0.000 | 0.000 | 0.000 |
| Explicit lexical rules | 0.847 | 0.661 | 0.742 |
| Entity-only ablation | 0.532 | 0.477 | 0.503 |
| Context TF-IDF + Logistic Regression | 0.820 | 0.820 | 0.820 |

SemEval-2010 Task 8 supplies entity spans. This is a derived three-class task, not the official 19-class benchmark. Sentence-grouped partitions contain 6,400 train, 1,600 dev and 2,700 retained test examples. Dev selects C=10. Test edge-F1 sentence-group bootstrap interval: [0.786, 0.851]. These measurements do not establish scientific/news domain extraction or causal truth.

The **Text → DAG** view now fits approved, provenance-bearing graphs to uploaded complete binary observations (or explicitly labeled synthetic teaching data). It records reviewer assumptions, fits BDeu CPDs and supports observational/interventional queries. Classifier scores never become CPDs.

### Reproduce and verify extraction

```sh
python -m scripts.fetch_extraction_data
python -m scripts.train_extraction --output artifacts/runs/my_extraction_run
python -m scripts.verify_extraction_run artifacts/runs/my_extraction_run
python -m pytest -q
```

Choose a new run directory. The runner refuses to overwrite an existing run. Raw parquet files, full text partitions and pickle models are generated locally. A bundled JSON model snapshot enables offline extraction inference. Audit the bundled historical numerical records without downloading corpus text:

```sh
python -m scripts.verify_extraction_run artifacts/runs/phase1_extraction_verified_20261101 --portable
```

This checks available hashes, split membership, development selection and reported metrics. Full model prediction reproduction needs locally generated text partitions and an estimator; a new run records its own hashes and environment.

### Submission and study files

- [Formal PowerPoint](docs/phase1_presentation.pptx) in the supplied Canva red/beige style, with editable charts and a full speaking script in presenter notes
- [Project report PDF](docs/phase1_report.pdf) and [IEEEtran LaTeX source](docs/ieee-report/main.tex)
- [Detailed viva tutorial, slide-by-slide presentation script and demo guidance](docs/viva_masterclass.md)
- [Gap audit against the reference and rubric](docs/gap_audit.md)
- [Dataset contracts](docs/data_dictionary.md)
- [Actual contribution record to complete](docs/contributions.md)
- [Extraction run results](artifacts/runs/phase1_extraction_verified_20261101/metrics.json)

Remaining boundaries: automatic entity discovery, independently annotated scientific/news documents, real-domain causal identification and individual counterfactuals are not completed. Transformers and a measured neural hybrid remain Phase 2. Team members must fill their actual contribution evidence and understand the code before the individual viva.

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

- `app.py`: Seven-page white monochrome Streamlit workspace with high-contrast black rules.
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
