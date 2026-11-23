# Causal Workbench

Fine-tuned DistilBERT for directed relation extraction, connected to reviewed graphs and Bayesian intervention queries. Classical baselines remain available for fair comparison. Team T094, Section E.

**[Open the live Streamlit app](https://causal-workbench-t094.streamlit.app/)**

## Phase 2 results

The predeclared primary DistilBERT model achieves a **test directed edge F1 score of 0.870**, compared with **0.820** for TF-IDF/Logistic Regression, evaluated on the same 2,700 held-out pairs. This corresponds to an improvement of **5.09 F1 points**, with a paired sentence-group bootstrap 95% confidence interval of **+2.34 to +7.92 points**. Across three weighted random seeds, the model achieves scores of **0.870, 0.875, and 0.873**, yielding a mean F1 of **0.873** and a sample standard deviation of **0.0024**.

The unweighted seed-42 ablation achieves an F1 score of **0.899**, exceeding the weighted primary model. The frozen-encoder control achieves an F1 score of **0.306** under the same computational budget, without independent learning-rate tuning. These ablation results, including the negative findings, are reported transparently and were not used to retrospectively select the primary model following test-set evaluation. The portable hosted 8-bit model achieves an F1 score of **0.866**.

**Walkthrough workflow:** DistilBERT proposes candidate relations → human review → directed acyclic graph (DAG) validation → Bayesian parameter fitting using matching observations → distinction between observational and interventional reasoning.

The graph evaluation is conducted in a controlled synthetic setting and should not be interpreted as evidence of automated causal discovery in real-world environments. The availability of entity spans is assumed as part of the task specification.

[Final predictions and measured results](artifacts/phase2/final_evaluation/) · [Architecture, training and evaluation](docs/phase2.md) · [Published model release](https://github.com/Mystic-commits/AML-DL_Project/releases/tag/phase2-v1.1)

## Run locally

Use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open `http://localhost:8501`. On macOS, `run.command` launches the app after environment setup. Classical JSON snapshots and demo inputs are bundled. The neural model downloads its checksum-verified public release on first use and is then cached; no API key is needed. Training is never run by the app.

## What the app does

- **Walkthrough:** actual DistilBERT extraction followed by review, DAG validation, Bayesian fitting and intervention queries in a controlled synthetic example.
- **Neural lab:** interactive three-class extraction with the deployed ONNX DistilBERT model.
- **Reasoning lab:** binary claim entailment using the saved Corr2Cause classifier.
- **Extraction benchmark:** directed classification between supplied entity spans using a separate SemEval model.
- **Text → DAG:** review candidate relations, reject invalid edges, and fit probability tables from matching binary observations.
- **Intervention lab:** compare observation with intervention in the rain/sprinkler teaching system.
- **Evaluation:** final neural/ML comparison, ablations, learning curves, seed stability and retained classical task audits.

See [app usage](docs/usage.md) for input formats and the guided workflow.

## Repository structure

```text
app.py                 Streamlit application entrypoint
.streamlit/            Streamlit theme and server configuration
src/                   Classifiers, extraction, graph validation and Bayesian logic
scripts/               Dataset fetching, training, analysis and result verification
notebooks/             Executed Phase 1 and Phase 2 experiment walkthroughs
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
| SemEval directed extraction | Primary DistilBERT / deployed 8-bit | Edge F1 0.870 / 0.866 |
| SemEval ablations | Frozen encoder / unweighted fine-tuning | Edge F1 0.306 / 0.899 |
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
.venv/bin/python -m scripts.verify_final_results artifacts/phase2/final_evaluation
.venv/bin/python -m scripts.verify_extraction_run artifacts/runs/phase1_extraction_verified_20261101 --portable
```

The executed [Phase 2 notebook](notebooks/phase2_experiments.ipynb) runs the released classifier, reconstructs final metrics and demonstrates the reviewed DAG/CPD/intervention workflow. Full training and analysis commands are in [experiment reproduction](docs/experiments.md). Dataset contracts and limitations are in [the data dictionary](docs/data_dictionary.md). Background sources are in [references](docs/literature_review.md).

The [deployment notes](docs/deployment.md) describe hosting from `main` with entrypoint `app.py` on Python 3.12.

## Team

Yug Johri, Vidit Sachan, Aditya Verma and Ishita Thakur.

Special Mention: Kaavya Gala.
