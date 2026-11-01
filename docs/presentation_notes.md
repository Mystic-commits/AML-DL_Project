# Presentation script



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
