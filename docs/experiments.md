# Experiment reproduction

Run commands from the repository root in the Python 3.12 environment described in README.

## Corr2Cause reasoning

```sh
python -m scripts.fetch_data
python -m src.experiment
python -m scripts.validate_components
python -m scripts.analyze
python -m scripts.audit_shortcuts
python -m scripts.export_app_assets
```

The fetcher records the source revision and hashes. The experiment fits TF-IDF only on training data, separates exact premise groups, compares majority prediction, Naive Bayes and Logistic Regression, and reports a unigram ablation. Development macro F1 selects the main model. The hypothesis-only control diagnoses wording shortcuts. Generated plots, metrics and portable inference parameters are stored under artifacts.

The notebook `notebooks/phase1_experiments.ipynb` provides the interactive walkthrough. Full datasets, joblib models and generated prediction/error tables are local outputs excluded from Git.

## Directed extraction

```sh
python -m scripts.fetch_extraction_data
python -m scripts.train_extraction --output artifacts/runs/my_extraction_run
python -m scripts.verify_extraction_run artifacts/runs/my_extraction_run
```

Choose a new output directory; existing runs are not overwritten. The derived SemEval task uses given entity spans and three labels: no causal relation, e1 causes e2, and e2 causes e1. Sentence grouping reduces exact overlap across partitions. No document IDs are available. Development edge F1 selects C from 0.1, 1 and 10.

The runner compares lexical rules with a context TF-IDF/logistic-regression model and an entity-only ablation. Each run saves settings, source/data hashes, split membership, predictions, metrics, environment information and a portable model snapshot. Its code snapshots are retained as experiment provenance, rather than imported as application code.

## Verify the bundled extraction evidence

```sh
python -m scripts.verify_extraction_run artifacts/runs/phase1_extraction_verified_20261101 --portable
python -m pytest -q
```

Portable verification checks available hashes, partition separation, development-only selection, prediction alignment and reported metrics without corpus downloads. Full prediction reproduction needs the locally generated text partitions and fitted estimator. A fresh training run creates its own audit trail.

## Interpretation

Corr2Cause entailment, directed relation classification and synthetic intervention validation are separate evaluations. A good extraction score does not establish the causal correctness of a graph. Hypothesis-only performance limits the reasoning claim. The teaching Bayesian system checks calculations against a known generator under supplied assumptions; it does not validate causal identification in scientific or news documents.
