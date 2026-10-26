# Literature review

| Source | What it contributes | How this project uses it | Boundary |
|---|---|---|---|
| Pearl, *Causality*, second edition, 2009 | Interventions modify generating mechanisms | Truncated factorization and explicit causal assumptions | A learned association alone does not license do-inference |
| Jin et al., ICLR 2024, [Corr2Cause paper](https://arxiv.org/abs/2306.05836) | Controlled causal reasoning from statistical premises | Binary reasoning labels and explicit benchmark limits | Not an annotated open-text relation extraction corpus |
| [Scikit-learn feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html) | Sparse text representations | Train-only TF-IDF and inspectable feature weights | Surface features can learn shortcuts |
| Ankan and Textor, [pgmpy, JMLR 2024](https://www.jmlr.org/papers/volume25/23-0487/23-0487.pdf) | Probabilistic graphical-model tooling | CPD estimation and variable elimination | Software inference cannot validate the assumed causal graph |

The gap is between proposing a plausible linguistic relation and justifying an intervention. Our implementation exposes that gap: it preserves provenance, asks for review, checks graph structure and makes data-generating assumptions visible. Phase 1 uses classical models only.
