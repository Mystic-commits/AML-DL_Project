# Phase 1 gap audit and completion record

Reference inspected: nishtha-09-gupta/AML_Capstone_Project, main as downloaded 1 November 2026. It is a different classification task. Its clinical data, models and results are not transferable to causal NLP. No reference code or artwork was copied. The useful parallels are rigorous validation, saved experiment records, an interactive demonstration, a report and a presentation.

| Requirement / reference practice | Before | Completion / honest boundary |
|---|---|---|
| Data ingestion and validation | Pinned Corr2Cause downloader and exact-premise separation | Retained and reproduced. Added pinned SemEval ingestion, entity-span/schema checks, label-conflict checks and sentence-group isolation |
| EDA and data dictionary | Corr2Cause EDA present | Added extraction label counts, dataset contract and feature documentation |
| Classical baseline comparison | Majority, NB, LR, unigram ablation present | Reproduced original results; added directed extraction rules, tuned LR, majority and entity-only ablation |
| Hyperparameter tuning | Fixed reasoning LR C | New extraction C grid selected on dev edge F1, never test |
| Independent extraction evidence | 16 authored regression documents | Added external SemEval test set. Given entities only. Scientific/news domain validation still uncompleted |
| Text to DAG to BN | Reviewed graph and synthetic BN separate | Added reviewed/provenance-gated fitting to matching binary observations and queries in app |
| Bayesian theory and validation | Didactic synthetic SCM and oracle | Retained oracle checks; added end-to-end text/review/fit/intervention regression |
| Saved run verification | JSON metrics without complete run reconstruction | Added per-run partitions, predictions, model, hashes, source snapshots, environment and verifier |
| Presentation | HTML slides only | New formal editable PowerPoint with presenter notes |
| Report | Existing Phase 1 PDF and LaTeX | Updated IEEE-format LaTeX source and two-column PDF with new measured results |
| Viva | Existing short guide | Added detailed tutorial, equations, demo script, difficult questions and preparation plan |
| Individual contribution | Names listed without evidence of roles | Five-member contribution template. Each member must fill actual work; no invented ownership |
| DL / transformer comparison | Absent | Explicit Phase 2 scope, as requested by the team |
| Individual counterfactuals | Absent | Remains a research boundary. Interventions are population queries, not individual counterfactuals |

No live deployment is implied. The test set has now been inspected, so future changes must not reuse it for selection and call it newly untouched. The extraction benchmark is a three-class task derived from SemEval, not the official 19-class leaderboard.
