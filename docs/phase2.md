# Phase 2: neural extraction and Bayesian reasoning

Status: implementation in progress. Existing Phase 1 results remain historical baselines. No deep-learning score is claimed until training and held-out evaluation finish.

## Five-day delivery plan

| Day (IST) | Work | Completion evidence |
|---|---|---|
| 12 November | Freeze the comparison protocol, verify the dataset/splits, add optional DL dependencies, implement transformer input handling and run a real forward/backward/checkpoint smoke test | Small independent commits, split audit, smoke-test receipt, unchanged app tests |
| 13 November | Fine-tune compact DistilBERT on the directed extraction task, select checkpoints by development edge F1, run a frozen-encoder control | Learning curves, exact configuration, model revision, saved checkpoints and development metrics |
| 14 November | Connect neural candidates to the existing reviewed-DAG and Bayesian interface; measure raw versus graph-validated candidate behaviour | Working candidate review, preserved provenance, cycle checks and documented intervention examples |
| 15 November | Complete held-out ML/DL comparison, weighted-loss and frozen-encoder ablations, seed runs and error analysis | Aligned predictions, directed precision/recall/F1, uncertainty estimates, runtime/memory observations and verified evidence |
| 16 November | Package the tested integration, update the notebook and technical docs, prepare separate LaTeX report, presentation and viva material | Hosted smoke check, reproducible commands, final submission files each below 10 MB |

Evaluation supplied by the team: 23 November 2026, 16:20-16:45 IST, A411. Submission deadline: 23 November, 15:20 IST. Form submission is a separate user action; it is not part of this implementation workflow.

The dated plan is an execution schedule, not an automatic background job. Progress and completed checks are recorded at the end of each work session. Keep each code/data/model/integration change in its own focused commit. Reports, slides and study handouts stay outside the repository.

## Architecture and rationale

Fine-tune a pretrained DistilBERT encoder with a three-class classification head: no causal relation, e1 causes e2, e2 causes e1. Contextual representations address limitations of fixed TF-IDF features. The encoder is a language model backbone, not an instruction-following chat model. Automatic named-entity discovery is not claimed.

Use the same entity masking, SemEval source revision, sentence-group partitions and directed-edge scoring contract as the Phase 1 extraction experiment. Given entity spans remain an explicit task limitation. Do not compare the extraction F1 with Corr2Cause entailment F1 as if they measured the same task.

The hybrid connects neural language predictions to human review, canonical entities, provenance-bearing DAG validation, Bayesian probability estimation from matching observations and intervention queries under recorded assumptions. A neural score never becomes a Bayesian CPD, and a prediction is never automatically human-approved. This is a structured neural/symbolic pipeline, rather than an arbitrary classifier ensemble.

## Evaluation contract

- Fit text representations/weights on training data only. Development directed edge F1 selects checkpoints; test scores never select settings.
- Pin the upstream model to a revision and use safetensors with remote code disabled.
- Preserve original split membership and source hashes. Keep full corpus text and checkpoints local, with compact reproducibility records in Git.
- Record entity-marker coverage after tokenisation. Never silently discard a marked entity during truncation.
- Separate engineering smoke runs from completed training/evaluation runs. A smoke run is not evidence of model quality.
- Main comparisons: existing TF-IDF/LR and lexical baselines, fine-tuned DistilBERT, frozen-encoder control, and weighted versus unweighted classification loss. Record seeds and exact sample budgets.
- Graph validity, edge quality and intervention calculations require separate checks. Use the known synthetic generator for intervention correctness and label this scope clearly.
- Report improvements or regressions honestly. Adding a transformer does not guarantee a higher test score or causal identification.
- Keep heavy training dependencies separate from the running Streamlit environment until an evaluated deployment artifact and hosting strategy are ready.

## Primary references

- Sanh et al., DistilBERT (2019): https://arxiv.org/abs/1910.01108
- DistilBERT model card: https://huggingface.co/distilbert/distilbert-base-uncased
- Hugging Face text classification: https://huggingface.co/docs/transformers/tasks/sequence_classification
- PyTorch MPS backend: https://docs.pytorch.org/docs/stable/notes/mps.html
- SemEval Task 8: https://aclanthology.org/S10-1006/
- Existing dataset contracts: data_dictionary.md. Classical reproduction: experiments.md.

## Day 1 commands

Use Python 3.12 and install the optional training environment separately from the host:

```sh
python -m pip install -r requirements-dl.txt
python -m scripts.prepare_neural_data
python -m scripts.smoke_neural --output artifacts/phase2/runs/my_smoke
python -m pytest -q
```

Prepared corpus text lives in ignored `data/processed/phase2_extraction`. A fresh output directory is required for each smoke run. The smoke command downloads the pinned pretrained model, checks entity-marker retention on training inputs, takes two optimizer steps on a small training sample, and verifies safetensors save/reload on CPU. It does not evaluate test examples or claim a trained release.

`configs/phase2_extraction.json` is the intended full experiment configuration, not trained weights. `artifacts/phase2/data_audit.json` records the exact shared data contract. The full fine-tuning runner and hosted neural inference integration belong to the following stages.
