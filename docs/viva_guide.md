# Causal Workbench: tutorial and viva defense

## Your 60-second explanation

Our project separates language processing from formal causal reasoning. In Phase 1 we use classical machine learning. We first test whether a causal hypothesis follows from a Corr2Cause premise. Separately, we classify directed cause-effect relations in SemEval sentences with supplied entity spans. Candidate edges need human review before entering a DAG. The graph rejects cycles and self-loops. Given matching binary observations, we estimate Bayesian conditional probability tables. Finally, we compare observational queries with intervention queries in a controlled synthetic system. Our experiments show that text classification can exploit wording shortcuts, so we do not claim that a high F1 establishes causal understanding. Neural language models are Phase 2 work.

## 1. What enters and leaves each component?

Corr2Cause reasoning input is a premise and a hypothesis, output is entailed or not entailed. It does not return entity pairs. SemEval input is a sentence with two marked entity spans, output is no causal relation, e1 causes e2, or e2 causes e1. The lexical extraction demo takes plain text and returns candidate triples with sentence provenance. A reviewer approves edges. The graph stage produces an acyclic structure. The parameter stage additionally requires a table of observations. The query stage returns a probability conditional on the graph, fitted tables and assumptions.

Example: 'Rain causes wet grass.' yields rain -> wet_grass with a document ID and sentence ID. This states what a sentence says. To estimate the probability of wet grass, we need observations of rain and wet_grass. The text model's score cannot substitute for a CPD.

## 2. How does TF-IDF work?

A bag of words counts tokens without full syntax. TF-IDF reduces the importance of common words. Smooth IDF(t) = log((1+N)/(1+df(t))) + 1. Sublinear TF uses 1+log(count) for present tokens. L2 normalization scales the vector to length one. Bigrams preserve short phrases such as 'caused by'. Retaining single-letter tokens matters because Corr2Cause uses A, B and C as variables. Fit vocabulary and IDF on training text only. Test text must never influence feature fitting.

Logistic regression learns linear class scores. Binary reasoning uses sigmoid(w.x+b). Directed extraction uses a three-class logistic classifier. C is inverse regularization strength: small C gives stronger regularization. Balanced weights make rare classes matter more during fitting. The extraction experiment selects C from 0.1, 1 and 10 on development edge F1. The test set reports performance after that choice. This search does not guarantee a globally optimal model.

## 3. What is leakage, and what do our splits prevent?

Several Corr2Cause rows share one premise but ask different hypotheses. Random row splitting could put almost identical situations on both sides. We hash normalized premises and purge overlap across train, dev and test. This blocks exact premise reuse. It does not block isomorphic graphs under variable renaming or shared templates.

SemEval has multiple marked pairs in some sentences. We group by the underlying normalized sentence without span tags. All pairs in that sentence remain in one split. We deduplicate marked pairs separately, because two different pairs in one sentence can legitimately have different labels. Document IDs are unavailable, so we cannot honestly claim full document-disjoint evaluation.

## 4. Precision, recall and F1

Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = 2PR/(P+R). For directed extraction, predicting B->A when gold is A->B creates one FP and one FN. A correct direction is essential. Macro F1 averages per-class F1 equally. Majority accuracy can look strong on imbalanced data while causal-relation recall is zero. Always name the task and metric before giving a number. Corr2Cause macro F1 and SemEval directed edge F1 are different measurements.

A bootstrap interval resamples evaluation units to show sampling variability conditional on the frozen fitted model. Corr2Cause resamples premise groups. The extraction run resamples sentence groups. Neither interval includes all variability from retraining, selecting hyperparameters or changing annotation policy.

## 5. Why the shortcut experiment matters

Full Corr2Cause input produces macro F1 about 0.700. A hypothesis-only model obtains about 0.653 without reading the premise. Therefore, premise-free wording cues support much of the observed score. Do not say this proves that every prediction is a shortcut, or interpret the difference as an exact causal contribution of the premise. The diagnostic challenges the strength of the reasoning claim.

The SemEval entity-only ablation asks a different question: can nominal strings predict the relation without surrounding context? The main extraction representation masks actual entity strings while preserving ENTITY1/ENTITY2 markers. This encourages use of surrounding context but does not guarantee genuine causal comprehension.

## 6. What does the graph add?

A directed acyclic graph has directed edges and no directed cycles. For the selected static Bayesian model, acyclicity enables a factorization into P(node | parents). A->B, B->C, C->A fails this contract. Rejecting the last edge ensures valid structure but does not show which edge is actually wrong. The current algorithm is greedy and source-order dependent. Human approval records responsibility and provenance, not ground-truth proof.

Real causal feedback can exist. It requires a time-indexed or other appropriate model rather than silently violating this static DAG representation. A graph can be acyclic and still causally false.

## 7. Where do probabilities come from?

A Bayesian network stores one conditional probability distribution per node given its parents. Our teaching system factors as P(R,S,W)=P(R)P(S|R)P(W|R,S). We generate 6,000 binary observations from a declared synthetic SCM. BDeu smoothing with equivalent sample size five adds prior mass to avoid brittle zero-frequency estimates. The reviewed-network path fits the same kind of tables to uploaded binary observations with matching node names.

Sparse parent combinations weaken estimates. Smoothing prevents zeros but does not create evidence. Causal interpretation needs correct structure, measured confounders, adequate support/positivity and stable mechanisms. An observational BN without defensible causal assumptions only supports probabilistic association.

## 8. Observation versus intervention, with numbers

Observation asks P(W=1 | S=1): among cases with sprinkler on, how often is grass wet? Intervention asks P(W=1 | do(S=1)): if we force the sprinkler on, how often would grass be wet in the population? Rain affects both sprinkler use and wet grass, so observing sprinkler status carries information about rain. An intervention breaks the incoming rain-to-sprinkler mechanism.

The oracle is P(W=1 | do(S=s)) = sum_r P(W=1 | R=r,S=s)P(R=r). For s=1: 0.3*0.99 + 0.7*0.8 = 0.857. For s=0: 0.3*0.9 + 0.7*0.02 = 0.284. The average causal effect is 0.857-0.284=0.573. Fitted estimates differ due to finite sampling and smoothing. Rain->Sprinkler in this model means rain changes sprinkler use, and that influence is negative. A directed edge does not imply a positive effect.

Variable elimination sums nuisance variables using factors. The intervention implementation removes incoming edges, then fixes the intervention variable using evidence. For a fixed state the replacement prior cancels on normalization. This implementation is for the supplied discrete causal DAG under the stated assumptions.

## 9. Why an intervention is not an individual counterfactual

Our result averages over the population under a changed mechanism. An individual counterfactual asks what would have happened to a specific case under a different treatment, conditional on what actually happened. It requires abduction, action and prediction in an SCM with a defensible joint model of exogenous variables. The current code does not implement individual counterfactual inference. Say this directly if asked.

## 10. Five-minute live demonstration

1. Open Overview and name the three separate tasks. Mention Phase 1 scope.
2. Open Reasoning lab. Run a bundled example and distinguish the model decision from its uncalibrated score and dataset label.
3. Open Extraction benchmark. Show measured rule/LR comparison and entity-only ablation. Classify a marked sentence. Explain given-entity evaluation.
4. Open Text -> DAG. Use 'Rain causes sprinkler. Rain causes wet grass. Sprinkler causes wet grass. Wet grass causes rain.' Approve all. Show the rejected cycle. Explain source-order dependence.
5. Keep the three accepted edges. Select synthetic observations, enter your reviewer name and 'Known synthetic generating graph', then explicitly check assumptions. Fit the graph and compare sprinkler interventions.
6. Open Intervention lab. Set sprinkler to 0 and 1. Explain the oracle and the observational difference. End with the limits of the demonstration.

## 11. Questions examiners may ask

**Why is the course title neuro-symbolic if there is no neural model?** The title describes the full two-phase project. This submission implements the classical language-processing and symbolic/probabilistic baseline. A compact transformer comparison is reserved for Phase 2. We do not claim a completed neural hybrid.

**Why add SemEval when Corr2Cause is recommended?** We retain Corr2Cause. Its labels evaluate causal reasoning entailment, so it cannot independently validate extraction. SemEval supplies directed cause-effect relation labels and given entity pairs as a separate benchmark. It does not establish performance on scientific/news documents. The faculty's dataset-flexibility rule permits an additional suitable dataset.

**Why not use random forest like the reference?** Sparse high-dimensional text has a natural linear baseline. The reference's tabular medical preprocessing and tree models solve another problem. We adopt its validation discipline, not an unjustified model checklist.

**Does review make the graph causal?** No. Review filters predictions and records assumptions. We still need domain knowledge, identification assumptions and suitable data.

**Can you estimate CPDs by counting causal mentions?** No. Mention frequencies and classifier scores do not describe joint observations of modeled variables.

**Why not keep tuning until test F1 improves?** That would use test labels for development. New tuning decisions require fresh evaluation or a declared retrospective analysis.

**What if an edge has no observations?** The fitting contract rejects missing variable columns. A text-derived graph with no measured variables remains a structure proposal.

**What does the entity-only ablation prove?** It is diagnostic evidence about lexical information, not a proof of complete robustness. Compare results on the same frozen test rows.

**What is original about the project?** The contribution is the explicit separation and audit of language classification, edge review, graph constraints and Bayesian querying, with a demonstrable bridge using matching observations. Do not claim a new causal-identification theorem or a new state-of-the-art architecture.

## 12. Preparation for all five members

Each member should explain the complete pipeline, reproduce one result, derive one query, explain one failure case and locate the relevant code. Rehearse a 60-second summary, a five-minute demo and five difficult questions without reading slides. Then fill docs/contributions.md with actual completed work and commit/evidence links. Suggested practice areas may be shared, but all five remain individually accountable for all concepts.
