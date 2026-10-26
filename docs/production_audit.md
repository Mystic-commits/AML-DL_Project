# Application and Research Audit
**Project 1: Neuro-Symbolic Causal Inference (Phase 1)**  
**Team T094, Section E**

Audit scope: Local and cloud deployment of the Phase 1 educational causal reasoning workbench. Findings are based on code inspection, repeatable offline experiments, and regression tests.

---

## Verified Corrections

| Finding | Correction | Verification |
|---|---|---|
| Fresh-checkout training wrote into missing output folders (`FileNotFoundError`) | Training (`src/experiment.py`) and validation scripts automatically create `data/processed/` and `artifacts/` | Clean checkout pipeline reproduction |
| Downloads could leave partially updated source files if interrupted | Fetch into a temporary `.tmp` staging file, verify rows and hashes, then atomically replace destination file | Interrupted-download simulation |
| Whitespace variations could bypass exact premise overlap deduplication | Normalize internal and boundary whitespace (`" ".join(text.split())`) before SHA-256 grouping | Cross-split verification with differently spaced identical premises |
| Split verification tests crashed on fresh clones without raw data | Added clean `@pytest.mark.skipif` checking presence of raw CSVs | Clean-checkout test execution |
| App re-fitted 6,000 synthetic observations and Bayesian network on every query click | Cached the fitted network using `@st.cache_resource` and passed reference to `query()` | Latency benchmark and repeated query assertions |
| Published interpretation lacked a shortcut/prior diagnostic | Implemented hypothesis-only control model (`scripts/audit_shortcuts.py`) | Control achieves 0.653 Macro F1 vs 0.700 for full input |
| Dark theme and minified code obscured the workflow and code readability | Switched to clean white background with black accent lines; expanded all one-liners into well-commented student code | App rendering and code inspection |
| Brittle regex on uncertain or negated statements | Added conservative filtering rejecting modal verbs (`may`, `might`, `could`) and negation (`not`, `if`) | Regression test suite |
| Self-loops and directed cycles entered causal graphs | Added sequential greedy acyclicity check using NetworkX pathfinding | Fixture validation on cyclical examples |

---

## Remaining Research Boundaries & Methodological Limits

1. **Dataset Prefix**: The training split uses the first 12,000 rows rather than a full randomized sample of the 400k+ Corr2Cause corpus.
2. **Variable Invariance**: Exact premise hashing blocks identical textual premises across splits, but does not identify isomorphic causal structures under variable renaming (e.g. $A \to B$ vs $X \to Y$).
3. **Imbalance & Positive Precision**: The test set is heavily imbalanced (888 negatives vs 156 positives). While macro F1 is 0.700, positive precision is 0.392 with 223 false positives.
4. **Shortcut Signals**: The hypothesis-only diagnostic reaches 0.653 Macro F1 without reading the premise, proving that template surface cues account for a major fraction of the model's predictive ability.
5. **Lexical Extractor Scope**: The rule-based extractor handles simple affirmative patterns only ("X causes Y", "X leads to Y", "X results in Y"). It does not handle passive voice, nested clauses, or general scientific prose.
6. **Synthetic SCM**: The Bayesian network is a hardcoded 3-variable didactic model (Rain, Sprinkler, Wet Grass) with seeded synthetic data. User-reviewed graphs from the text extractor do not automatically acquire these probability distributions.
7. **Intervention vs Counterfactuals**: The system answers population-level intervention queries ($P(Y \mid do(X))$) using mutilated graph factorization; it does not implement individual-level counterfactual inference ($P(Y_{X=x} \mid X=x', Y=y')$).
