# Phase 2: neural extraction and Bayesian reasoning

Status: completed and held-out evaluated on 23 November 2026. Five full neural training runs, a public CPU-compatible release, hosted integration, ablations, stability checks and numeric verification are available. Phase 1 results remain separately labelled historical baselines.

## Original five-day delivery plan (historical)

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

`configs/phase2_extraction.json` is the intended full experiment configuration, not trained weights. `artifacts/phase2/data_audit.json` records the exact shared data contract. The full fine-tuning runner is documented under Day 2 below. Hosted neural inference integration belongs to Day 3.

## Completed: Day 1 (12 November)

- Pinned the official DistilBERT revision and isolated the optional DL requirements.
- Verified all data hashes and exact equality with the classical split: 6,400 train / 1,600 dev / 2,700 test.
- Added entity masking, token-truncation guards, training-only class weights and neural training primitives.
- Ran the real 66M-parameter pretrained model on Apple MPS for two engineering optimizer steps. The classification weights changed, losses were finite, and all training entity markers survived the 128-token budget (maximum observed training length: 122).
- Saved and reloaded a safetensors checkpoint. Maximum probability difference on CPU: approximately 0.00000003.
- All 22 local tests passed, including the existing Streamlit pages. The historical extraction verifier passed.
- `artifacts/phase2/smoke_report.json` is an engineering receipt, not a trained release or an extraction score. The checkpoint stays local and is not offered in the hosted app.

Historical Day 1 checkpoint: superseded by the completed Day 2 and final runs below. The two-step smoke checkpoint is not deployed.

## Completed: Day 2 (13 November)

Full training now runs through `scripts/train_neural.py`. Each run validates the frozen train/dev file hashes, preserves entity markers, records all four epochs, selects the highest development directed edge F1 (earliest epoch wins ties), and reloads the selected safetensors checkpoint to reproduce its development predictions. The runner never opens the test partition.

| Model / variant | Seed | Selected epoch | Development edge precision | Development edge recall | Development edge F1 |
|---|---:|---:|---:|---:|---:|
| Fine-tuned DistilBERT, weighted loss | 42 | 4 | 0.813 | 0.866 | 0.839 |
| Frozen DistilBERT encoder + trained head, weighted loss | 42 | 3 | 0.373 | 0.388 | 0.380 |
| Historical TF-IDF/LR, C=10 | 42 | n/a | 0.753 | 0.726 | 0.739 |

All scores in this table are **development scores**, selected using development data. They are not the Phase 1 test F1 of 0.820 and do not replace a held-out neural test evaluation. The two neural variants start from the same upstream model revision and seed, use identical input rows and masking, batch size 16, four epochs, learning rate 0.00002, weight decay 0.01, dropout 0.2 and gradient clipping at 1.0. Loss weights are computed from training labels only.

The frozen encoder stays in evaluation mode with its parameters disabled for gradients. Its classification head remains trainable. This removes encoder updates and encoder dropout noise; it is a fixed-budget control, not an optimally tuned frozen-feature classifier. The frozen head may benefit from its own learning-rate search or a different fitting approach. Different historical ML tuning budgets also limit interpretation of the comparison.

The fine-tuned run updated 66,957,315 parameters. The frozen run trained 592,899 head parameters. Training plus per-epoch development evaluation/checkpoint saving took approximately 205 seconds and 54 seconds respectively on this Apple GPU; these are observed local run durations, not deployment latency benchmarks. The final checkpoint-reload maximum probability differences were below 0.000003 for both runs. Fine-tuned development F1 improved across four epochs while cross-entropy reached its minimum earlier; confidence calibration is not established by this experiment.

Evidence: `artifacts/phase2/development/day2_20261113/` contains the comparison, learning curves, epoch histories, aligned development predictions, full run configuration/environment/hash records and verification receipts. Full corpus text and checkpoint weights remain local under ignored directories. This historical Day 2 snapshot contains one seed per variant. The final evaluation below adds repeated weighted seeds, the loss ablation and held-out testing.

### Reproduce Day 2

```sh
python -m scripts.train_neural --variant fine_tuned_weighted --seed 42 --output artifacts/phase2/runs/new_finetuned_seed42
python -m scripts.train_neural --variant frozen_encoder_weighted --seed 42 --output artifacts/phase2/runs/new_frozen_seed42
python -m scripts.verify_neural_run artifacts/phase2/runs/new_finetuned_seed42
python -m scripts.verify_neural_run artifacts/phase2/runs/new_frozen_seed42
python -m scripts.summarize_neural_development --runs artifacts/phase2/runs/new_finetuned_seed42 artifacts/phase2/runs/new_frozen_seed42 --output artifacts/phase2/development/new_comparison
```

To verify the committed records without downloading model weights:

```sh
python -m scripts.verify_neural_run artifacts/phase2/development/day2_20261113/fine_tuned_weighted_seed42 --portable
python -m scripts.verify_neural_run artifacts/phase2/development/day2_20261113/frozen_encoder_weighted_seed42 --portable
```

The final deployment below supersedes this historical Day 2 integration status.


## Completed final evaluation and integration (23 November)

| Held-out model | Directed edge F1 |
|---|---:|
| Lexical | 0.742 |
| Context TF-IDF/LR | 0.820 |
| Weighted fine-tuning, seed 42 — predeclared primary | 0.870 |
| Weighted fine-tuning, seed 17 | 0.875 |
| Weighted fine-tuning, seed 2026 | 0.873 |
| Frozen encoder, weighted head, seed 42 | 0.306 |
| Unweighted fine-tuning, seed 42 | 0.899 |
| Primary model after portable 8-bit deployment export | 0.866 |

The primary checkpoint is selected on development edge F1, with primary variant/seed fixed before held-out evaluation in `configs/phase2_final_evaluation.json`. All models use the identical retained test rows and source groups. No test threshold tuning or test-based seed selection is performed. The primary 95% group-bootstrap interval is [0.842, 0.897]; the paired improvement over ML is [0.0234, 0.0792] in F1 units. The primary difference is 0.0509, or 5.09 F1 points. Three weighted seeds have mean 0.872998 and sample SD 0.002370. Unweighted fine-tuning wins this seed-42 ablation; further unweighted seeds have not been run. Frozen-head hyperparameters were not independently tuned.

`artifacts/phase2/final_evaluation/` includes aligned predictions, hashes, full metrics, training records, replication and deployment comparisons, and the controlled graph query evidence. `artifacts/phase2/development/final_20261123/` includes weighted/frozen/unweighted development learning curves and records. Neither a classifier score nor a graph-validity result establishes real-world causality.

The synthetic graph case starts with four model-proposed edges, three correct under the supplied generator (raw edge P/R/F1 0.75/1.0/0.857). Structure checks keep the three-edge DAG and reject wet_grass → rain. Validated case P/R/F1 is 1/1/1, explicitly a single scripted known-generator case rather than a real-world graph benchmark. BDeu-smoothed CPDs use 6,000 binary observations; maximum intervention error against the known generator is 0.01051. No individual counterfactual claim is made.

### Reproduce final training and evaluation

Install `requirements-dl.txt`, prepare the pinned corpus, and use fresh output paths. The existing seed-42 weighted/frozen checkpoints can be reproduced with the Day 2 commands above.

```sh
python -m scripts.train_neural --variant fine_tuned_weighted --seed 17 --output artifacts/phase2/runs/repro_weighted17
python -m scripts.train_neural --variant fine_tuned_weighted --seed 2026 --output artifacts/phase2/runs/repro_weighted2026
python -m scripts.train_neural --variant fine_tuned_unweighted --seed 42 --output artifacts/phase2/runs/repro_unweighted42
python -m scripts.export_neural_onnx --help
python -m scripts.evaluate_neural_final --help
python -m scripts.verify_final_results artifacts/phase2/final_evaluation
```

For a complete re-evaluation, pass all five freshly reproduced run directories to `evaluate_neural_final --runs`, the portable primary ONNX bundle directory to `--onnx`, and a fresh directory to `--output`. `--help` documents required export arguments. Stored probability arrays/full corpus/checkpoints stay local; compact predictions and numerical evidence are committed. The independent final verifier runs without PyTorch and reconstructs all ten reported models from saved test predictions. It checks available hashes, split alignment, seed statistics, graph acceptance and oracle values.

### Deployment

Release `phase2-v1.1` uses unsigned 8-bit MatMul quantization with floating embeddings. Signed Gather quantization was replaced after a cross-platform prediction failure; the old v1.0 export is superseded. Public release, model byte size and SHA-256 are pinned in `artifacts/neural/release.json`. Model cache keys include the release archive hash. Linux CI checks real forward, reverse and no-relation predictions, while Streamlit tests check review/fit gates. The cloud host needs ONNX Runtime and tokenizers, not PyTorch or Transformers. Warm local CPU latency of 5.23 ms is a 20-probe, two-thread measurement, not a cloud latency guarantee.

Reports, slide decks and study material remain separate submission artifacts outside the code repository, as requested. The original five-day plan above records intended milestones; work was finalised on 23 November, not backdated.
