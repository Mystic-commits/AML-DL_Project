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



# Presentation delivery and speaking script

## Timing and suggested speaking roles

The fifteen-slide talk takes approximately 10-12 minutes at a calm pace, including handovers. Allow another 2-3 minutes if the panel requests the live demo. Use the script to rehearse, then speak naturally rather than memorizing every word. Do not read the reference slide aloud.

Suggested speaking blocks: Yug slides 1-3, Vidit slides 4-5, Aditya slides 6-7, Kaavya slides 8-11, Ishita slides 12-15. These are presentation assignments only, not statements of who implemented the code. Change the order if your team prefers another distribution. Every member should understand all components for the individual viva.

Before starting, open the deck in presentation mode, check the screen ratio and test the clicker. Keep the application running with the synthetic teaching example ready. Use eye contact for the main message and point to evidence only when discussing a chart or graph. Pause briefly after each result. Say the task and metric before stating a number.

For a shortened talk, keep slides 1, 2, 4, 6, 7, 8, 10, 11 and 14. Summarize methodology orally and keep the other slides available for questions. For a five-minute limit, skip the live demo and give one sentence per retained slide.

## Slide 1: Neuro-Symbolic Causal Inference

Suggested presenter: Yug Johri. Target: 35 seconds.

Delivery cue: Face the panel. Introduce the team, then pause before naming the problem.

Say:

Good morning. We are Team T094 from Section E. Our project is Neuro-Symbolic Causal Inference through language-based knowledge extraction and Bayesian networks. For Phase 1, we implement the classical machine-learning foundation. The central idea is to separate what a sentence claims from what a causal model can formally answer. We will show our datasets, the language-model comparisons, and how reviewed relations become a Bayesian network.

Transition: Let me begin with the problem this separation solves.

## Slide 2: Problem and Phase 1 scope

Suggested presenter: Yug Johri. Target: 40 seconds.

Delivery cue: Point to the three objectives in order. Explain the distinction, without reading each box.

Say:

A sentence such as rain causes wet grass gives us a candidate relation. It does not by itself provide probabilities or establish that the relation is causally correct. Our first objective is to extract or classify candidate relations. We then retain provenance and require human review before building a directed acyclic graph. Finally, matching observations estimate conditional probabilities. Phase 1 uses classical classifiers. The transformer-based neural component belongs to Phase 2.

Transition: The next slide shows how these components fit together.

## Slide 3: System components

Suggested presenter: Yug Johri. Target: 40 seconds.

Delivery cue: Distinguish the separate language tasks before describing the graph path.

Say:

There are two language evaluations. Corr2Cause asks whether a causal hypothesis follows from a premise. SemEval asks which direction of cause and effect connects two supplied entity spans, or whether there is no causal relation. These are different tasks, so their scores must remain separate. Candidate edges can then enter a reviewed DAG. When matching observations are available, we fit a Bayesian network and compare observational queries with interventions.

Transition: Vidit will now explain the data and the classical models.

## Slide 4: Dataset quality and separation

Suggested presenter: Vidit Sachan. Target: 50 seconds.

Delivery cue: Point to both split sizes. Emphasize the evaluation unit and the limitation.

Say:

For Corr2Cause, we use a bounded prefix of twelve thousand training rows. After removing exact premise overlap, we retain one thousand forty development rows and one thousand forty-four test rows. For SemEval, the partitions contain sixty-four hundred training, sixteen hundred development, and twenty-seven hundred retained test examples. Multiple entity pairs from the same normalized sentence stay together. SemEval supplies entity spans but no document IDs, so we can claim sentence-group separation, not full document independence. All learned text features fit training data only.

Transition: Here is how we represent those texts and train the models.

## Slide 5: Features and classical models

Suggested presenter: Vidit Sachan. Target: 50 seconds.

Delivery cue: Explain TF-IDF in one sentence and C in one sentence. Avoid deriving every equation unless asked.

Say:

TF-IDF represents a sentence using weighted words and short phrases. Frequent corpus-wide terms receive less weight, while bigrams preserve cues such as caused by. Balanced Logistic Regression gives rare classes more influence during training. Corr2Cause compares majority prediction, Naive Bayes and Logistic Regression, with a unigram ablation. For directed extraction, the context model masks actual entity strings while preserving entity-one and entity-two markers. We select regularization C from zero point one, one and ten using development edge F1. The entity-only ablation removes surrounding context.

Transition: Aditya will present the results and explain what they do and do not establish.

## Slide 6: Reasoning results and shortcut diagnostic

Suggested presenter: Aditya Verma. Target: 50 seconds.

Delivery cue: Point first to the full-input score, then the hypothesis-only control. Say macro F1 explicitly.

Say:

On Corr2Cause, the selected full-text model achieves a test macro F1 of zero point seven zero zero. The majority baseline reaches about zero point four six zero. However, a hypothesis-only control reaches zero point six five three without reading the premise. This tells us that wording patterns provide substantial predictive signal. We therefore describe the model as a useful text-classification baseline rather than proof of formal causal reasoning. The full-input model also has positive precision of about zero point three nine two, despite high recall.

Transition: The extraction benchmark evaluates a more direct language-processing task.

## Slide 7: Directed extraction results

Suggested presenter: Aditya Verma. Target: 50 seconds.

Delivery cue: Point to the context model and the two controls. Define a reversed edge before stating the conclusion.

Say:

Here the target is a directed cause-effect relation between two given entities. A reverse-direction prediction counts as both a false positive and a false negative. Context TF-IDF with Logistic Regression achieves directed edge F1 of zero point eight two zero. Lexical rules reach zero point seven four two, and the entity-only ablation reaches zero point five zero three. The comparison supports the value of sentence context. It does not establish causal truth or end-to-end entity discovery. These are results for a derived three-class task, not the official nineteen-class SemEval score.

Transition: Kaavya will now show how candidate relations enter the formal reasoning stage.

## Slide 8: Edge review and DAG validation

Suggested presenter: Kaavya Gala. Target: 40 seconds.

Delivery cue: Trace the accepted DAG, then point to the rejected reverse relation in the caption.

Say:

Each candidate retains the document ID, sentence ID, original sentence and method. A reviewer explicitly approves an edge before graph construction. The graph builder rejects self-loops, duplicate edges and an edge that would close a directed cycle. For example, rain to wet grass and wet grass to rain cannot both remain in our static DAG. Our procedure follows source order. This guarantees acyclicity, but it does not determine which causal explanation is true. Domain review is still necessary.

Transition: After reviewing the structure, we need a separate source for probabilities.

## Slide 9: Reviewed graph to Bayesian network

Suggested presenter: Kaavya Gala. Target: 45 seconds.

Delivery cue: Stress that observations supply CPDs. Do not call classifier scores causal probabilities.

Say:

Approved edges define the network structure. A table of complete binary observations supplies the variable values used to estimate conditional probability distributions, or CPDs. Column names must match the graph nodes. We fit the CPDs using BDeu smoothing with equivalent sample size five. The reviewer records a causal rationale and assumptions about observed confounders and stable mechanisms. Mention counts and classifier scores never become CPDs. The application can now fit a reviewed graph and run queries under these stated assumptions.

Transition: We can now explain why observing a variable differs from intervening on it.

## Slide 10: Observation and intervention

Suggested presenter: Kaavya Gala. Target: 50 seconds.

Delivery cue: Say the two questions in everyday language before reading their formulas.

Say:

Observation asks: among cases where the sprinkler is on, how often is the grass wet? Intervention asks: if we force the sprinkler on, how often would the grass be wet in the population? In our synthetic system, rain affects sprinkler use and also wets the grass. Observing sprinkler status therefore changes our information about rain. Intervening removes the incoming mechanism that normally determines sprinkler status. This difference depends on the supplied causal graph and its assumptions. It is a population intervention, not an individual counterfactual.

Transition: The next slide checks those intervention calculations against known answers.

## Slide 11: Intervention validation

Suggested presenter: Kaavya Gala. Target: 50 seconds.

Delivery cue: Point to observed, fitted and oracle values. Work through the sprinkler-on oracle verbally.

Say:

We generate six thousand observations from a known three-variable system. The exact sprinkler-on intervention probability is zero point three times zero point nine nine, plus zero point seven times zero point eight, which equals zero point eight five seven. With the sprinkler off, the oracle is zero point two eight four. Our fitted values are about zero point eight four six and zero point two seven five. The maximum absolute error is about zero point zero one one. This validates our implementation on the known teaching system, not causal discovery in the real world.

Transition: Ishita will finish with reproducibility, limitations and our main findings.

## Slide 12: Reproducibility and demonstration

Suggested presenter: Ishita Thakur. Target: 40 seconds.

Delivery cue: Describe the verification evidence, then offer the live demonstration only if time permits.

Say:

We pin dataset versions and record input hashes, split memberships and model-selection choices. The extraction run saves predictions, metrics, model parameters and environment details. A separate verifier reconstructs reported metrics and checks split isolation and saved-model predictions. Seventeen automated tests pass, and the notebook executes successfully. The Streamlit application demonstrates classification, edge review, Bayesian fitting and intervention queries. The synthetic teaching data and the real text benchmarks remain clearly labeled.

Transition: These checks make the implementation inspectable, but important research limitations remain.

## Slide 13: Limitations and Phase 2

Suggested presenter: Ishita Thakur. Target: 40 seconds.

Delivery cue: Deliver this calmly as scope control. Do not describe unimplemented work as completed.

Say:

The extraction benchmark supplies entity spans and does not validate scientific or news documents. Sentence grouping cannot guarantee document independence. Acyclicity and reviewer approval do not prove causal correctness. Real-world effects still require defensible structure, measured confounders and adequate observational support. We have not implemented individual counterfactual inference. Phase 2 will compare a compact transformer under the same evaluation contract and assess whether a neural hybrid adds measurable value.

Transition: Within that Phase 1 scope, these are the conclusions supported by our evidence.

## Slide 14: Key findings

Suggested presenter: Ishita Thakur. Target: 35 seconds.

Delivery cue: Make eye contact and summarize the two main contributions without reading a checklist.

Say:

Our first finding is that classical context features improve directed relation classification over lexical rules and entity strings alone. Our second is that language predictions must remain separate from causal identification. The implemented bridge combines reviewed structure, matching observations and Bayesian queries, while retaining provenance and explicit assumptions. The strongest contribution is an auditable workflow that makes its evidence and failure modes visible. Thank you. We are ready for your questions.

Transition: The final slide lists the main references supporting our approach.

## Slide 15: Selected references

Suggested presenter: Ishita Thakur. Target: 15 seconds.

Delivery cue: Keep this slide available during questions. Do not read every reference aloud.

Say:

Our formal foundation comes from Pearl's work on probabilistic and causal reasoning. Corr2Cause provides the reasoning benchmark, SemEval provides directed relation labels, and pgmpy supplies Bayesian-network inference. BERT is part of our Phase 2 reading. We can explain how each reference influenced a specific design choice.

Transition: Invite questions, or return to the relevant result slide when answering.

## Live-demo script and recovery plan

Say: "I will now show the reviewed-graph path using a synthetic system with known assumptions. This validates the calculation, not a real-world causal discovery claim."

In Text to DAG, enter: Rain causes sprinkler. Rain causes wet grass. Sprinkler causes wet grass. Wet grass causes rain. Approve the candidates, show the cycle rejection and keep the three accepted edges. Say: "The final edge closes a cycle. Our static Bayesian network requires a DAG, so the graph builder rejects it. Source order affects this decision."

Select synthetic observations, enter your actual reviewer name and the rationale "Known synthetic generating graph", and explicitly check the stated assumptions. Fit the network. Set sprinkler to one and query wet grass. Say: "The fitted intervention probability is close to the known value 0.857. Its small difference reflects sampled observations and smoothing."

If the internet fails, use the bundled offline inference and synthetic example. If the live app fails, return to slide 11 and explain the saved oracle comparison. Do not invent a result or claim the live demo succeeded. If time runs short, skip the demo and move to the key findings.

## Answering questions professionally

Listen to the complete question. Begin with a direct answer, then connect it to one equation, code module or measured result. If asked whether the model establishes causation, say: "The text model identifies candidate relations. Causal effects follow only under the supplied graph and identification assumptions." If you do not know an answer, state what you do know and identify the additional experiment or assumption needed. Avoid guessing.

Before the viva, every member should rehearse the observation-versus-intervention explanation, derive 0.857 from the oracle, explain why a reversed edge counts as both an FP and FN, and explain why the hypothesis-only control limits our reasoning claim.
