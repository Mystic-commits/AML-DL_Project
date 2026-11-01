# Data contract and EDA interpretation

## Corr2Cause
`input`: premise plus hypothesis text. `label`: binary entailment (1) / non-entailment (0). `group`: SHA-256 of normalized premise before Hypothesis or Conclusion. Raw files use an immutable dataset revision, with slice hashes. First 12,000 train rows are a bounded prefix, not representative random sampling. Training-only vocabulary fitting prevents TF-IDF leakage. Exact text grouping does not eliminate graph isomorphisms or template similarity.

## SemEval-2010 Task 8 extension
Source: https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8 at d58325907b9bf84a7c81f81fe5372791d911c10c. Source paper: https://aclanthology.org/S10-1006/.

| Field | Meaning |
|---|---|
| sentence | Text containing given `<e1>...</e1>` and `<e2>...</e2>` nominal spans |
| relation | Original 0..18 relation class |
| label | 0: all other relations; 1: original class 0, e1 causes e2; 2: original class 1, e2 causes e1 |
| group | SHA-256 of normalized sentence after removing span tags. Multiple entity pairs from the same sentence remain together |
| pair_key | Normalized marked sentence, used to detect duplicate pairs and conflicting pair labels |

Train and dev derive only from the original 8,000-row training partition. A seeded stratified group fold reserves approximately 20% for dev. Original test remains separate, with normalized-sentence overlap purged. No document identifiers are supplied; sentence separation cannot guarantee independence across source documents. Entity masking substitutes ENTITY1 and ENTITY2 in the context model. The entity-only control measures how well nominal strings alone predict labels.

The downloaded parquet files remain local. Their source license is not clearly specified in the dataset card; fetch from source and retain its attribution rather than redistribute the corpus. Predictions and derived summaries are saved for audit.

## Reviewed graph and observations
Every edge carries cause, effect, document_id, sentence_id, sentence, method and explicit approval. The fitting interface accepts only complete binary observations named exactly after canonical graph nodes. Aliases include rainfall/rain and sprinklers/sprinkler. Unknown names normalize to underscores but require review. CPDs derive from observations, never classifier scores. BDeu equivalent sample size 5 supplies smoothing. Sparse parent-state coverage, confounding and wrong graph directions remain substantive modeling risks.
