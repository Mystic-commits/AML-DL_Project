# Viva guide

Evaluation: 2 November 2026, 3:20-3:45 PM, A403. Team T094, Section E.

## A practical 25-minute run

Use 3 minutes for the problem and dataset; 5 minutes for ML features, splitting and metrics; 4 minutes for extraction and review; 4 minutes for the Bayesian demonstration; and reserve 9 minutes for individual questions. Every member should be ready to explain the complete pipeline.

## Questions everyone should answer

1. **Why can the majority classifier reach 85% accuracy and still be unhelpful?** Negatives dominate the test set. It misses all 156 positives. Macro F1 gives equal weight to the classes.
2. **What does the logistic model predict?** Whether the hypothesis is entailed by the textual premise under the dataset labels. It does not identify a real-world causal effect.
3. **Where could leakage occur?** Repeated premises, feature fitting before splitting, or model selection on test results. Features fit only on train; exact premise groups are excluded from later splits; selection uses development macro F1. Renamed equivalent graphs may still overlap semantically.
4. **Why not claim bigrams solved the task?** The ablation changes macro F1 from 0.521 to 0.700 on this test set. That supports the local feature choice, while broader causal reasoning requires further validation.
5. **Where do graph edges come from?** Explicit lexical patterns and user approval. Provenance connects every candidate to a sentence. Acyclicity does not establish causal truth.
6. **Why does do(S=1) differ from observing S=1?** Rain affects both sprinkler use and wet grass. Intervening cuts the Rain-to-Sprinkler edge and preserves the population rain distribution.
7. **Where do CPDs come from?** Six thousand synthetic observations and BDeu smoothing with equivalent sample size five. Text frequencies and classifier scores are never treated as CPDs.
8. **Is this a counterfactual system?** It answers population interventions. Individual counterfactuals need structural mechanisms and an abduction/action/prediction procedure that are not provided.
9. **What do extraction metrics establish?** Coverage on a small authored regression fixture. They do not establish generalization to scientific abstracts or news.
10. **What remains limited?** The nonrandom training prefix, few test premise groups, low positive precision, narrow parser, synthetic causal system and missing independent extraction benchmark.
11. **What does the hypothesis-only control tell us about Corr2Cause?** Training TF-IDF on hypotheses alone scores 0.653 Macro F1 (versus 0.700 with the premise included). This shows synthetic templates carry stylistic artifacts and label priors that classical models exploit without performing causal deduction.
12. **Why are the Text-to-DAG graph and Bayesian intervention network decoupled?** Phase 1 establishes rigorous, verifiable baselines for each component independently. Automatically parameterizing arbitrary real-world causal graphs from free text without observational data would violate causal identification assumptions.

## Demo sequence

Open Overview and state the three tasks. In Reasoning lab, evaluate a held-out example and distinguish model score from gold label. In Text to DAG, approve all four default mentions; show that the last edge is rejected for a cycle. In Intervention lab, compare sprinkler states zero and one with the known oracle. Finish with Evaluation, the confusion matrix and the dataset audit.

## Team record

No roles or contributions are assigned retroactively. Before evaluation, keep a truthful dated record of each member’s changes, experiments, reviews and explanations. The team roster is included in the interface and report.
