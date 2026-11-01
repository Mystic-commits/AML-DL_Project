# Eight-slide presentation script



## Timing and suggested speaking roles

This eight-slide talk takes approximately 6 minutes. Allow another 2 minutes only if the panel requests a live demonstration. Rehearse with the script, then speak naturally.

Suggested speaking blocks: Yug slides 1-2, Vidit slides 3-4, Aditya slide 5, Kaavya slide 6, Ishita slides 7-8. These are suggested speaking assignments, not claims of implementation ownership. Every member should understand the complete workflow for the individual viva.

Before presenting, open the deck in slideshow mode, check the display and have the offline synthetic demonstration ready. State the task and metric before giving each score. Pause after the key result and use the script's transition to hand over to the next speaker.

## Slide 1: Causal Reasoning

Suggested presenter: Yug Johri. Target: 30 seconds.

Delivery cue: Introduce the team and the project in everyday language.

Say:

Good morning. We are Team T094 from Section E. Our project studies how language-based causal claims can be connected to Bayesian networks. We use classical machine learning to evaluate text and predict candidate relations. We then separate those predictions from the assumptions needed for causal queries. Our presentation covers the workflow, datasets, measured results and the observation-versus-intervention example.

Transition: Here is the overall workflow.

## Slide 2: System Components

Suggested presenter: Yug Johri. Target: 40 seconds.

Delivery cue: Point to the language evaluation and formal reasoning branches.

Say:

The language component evaluates two different tasks. Corr2Cause asks whether a hypothesis follows from a premise. SemEval classifies a directed causal relation between two supplied entities. Candidate edges need human approval and sentence provenance before entering a directed acyclic graph. Matching observations then estimate the Bayesian probability tables. This keeps language predictions, graph validity and causal assumptions separate.

Transition: Vidit will explain the data and model choices.

## Slide 3: Datasets and Split Integrity

Suggested presenter: Vidit Sachan. Target: 40 seconds.

Delivery cue: State the split sizes briefly, then focus on leakage prevention.

Say:

Corr2Cause uses twelve thousand training rows, with one thousand forty development and one thousand forty-four retained test rows. The extraction extension uses sixty-four hundred training, sixteen hundred development and twenty-seven hundred retained test pairs. We separate exact premise groups for reasoning and sentence groups for extraction. SemEval supplies the entity spans and lacks document identifiers, which limits our evaluation claim. All learned text features fit training data only.

Transition: The next slide shows how those texts become model inputs.

## Slide 4: Classical Text Models

Suggested presenter: Vidit Sachan. Target: 45 seconds.

Delivery cue: Explain TF-IDF simply and emphasize development-only selection.

Say:

TF-IDF represents texts using weighted words and short phrases. Bigrams preserve cues such as caused by. Balanced Logistic Regression helps account for class imbalance. We compare it with majority prediction, Naive Bayes and simpler feature controls. In extraction, the context model replaces entity strings with entity-one and entity-two markers. We choose regularization using development edge F1, not test labels. The entity-only ablation removes surrounding context.

Transition: Aditya will explain the reasoning results and their main limitation.

## Slide 5: Reasoning Results

Suggested presenter: Aditya Verma. Target: 45 seconds.

Delivery cue: Name macro F1 before pointing to the full-input and hypothesis-only bars.

Say:

The full-text Corr2Cause model reaches a test macro F1 of zero point seven zero zero. The majority baseline reaches about zero point four six zero. However, the hypothesis-only control reaches zero point six five three without reading the premise. This indicates that wording shortcuts provide substantial signal. We therefore present the model as a classical text-classification baseline and do not treat its score as proof of formal causal understanding.

Transition: Kaavya will now explain the separate extraction result.

## Slide 6: Directed Extraction

Suggested presenter: Kaavya Gala. Target: 45 seconds.

Delivery cue: Compare the context model with both controls and explain direction errors.

Say:

For directed extraction, context TF-IDF with Logistic Regression reaches edge F1 of zero point eight two zero. Lexical rules reach zero point seven four two, and the entity-only ablation reaches zero point five zero three. Predicting the reverse direction creates both a false positive and a false negative. These results show the value of sentence context on this given-entity task. They do not establish causal truth or automatic entity discovery. Proposed edges still require review before Bayesian fitting.

Transition: Ishita will explain what the Bayesian stage adds.

## Slide 7: Observation and Intervention

Suggested presenter: Ishita Thakur. Target: 50 seconds.

Delivery cue: Explain both questions in plain language before referring to formulas.

Say:

Observation asks how often the grass is wet among cases where the sprinkler is on. Intervention asks what happens if we force the sprinkler on in the population. Rain affects both sprinkler use and wet grass, so observing sprinkler status also gives information about rain. An intervention breaks the incoming sprinkler mechanism. Our approved graph is a DAG, and matching binary observations estimate its conditional probability tables using BDeu smoothing. Causal interpretation depends on the graph, observed confounders and stable mechanisms.

Transition: The final slide checks those calculations and summarizes our findings.

## Slide 8: Results and Conclusions

Suggested presenter: Ishita Thakur. Target: 50 seconds.

Delivery cue: Point to fitted and oracle values, then end with the project contribution.

Say:

We validate the Bayesian implementation using six thousand synthetic observations from a known system. With the sprinkler forced on, the exact wet-grass probability is zero point eight five seven and the fitted value is about zero point eight four six. With it off, the values are zero point two eight four and zero point two seven five. The maximum absolute error is approximately zero point zero one one. Our conclusion is that context helps relation classification, while formal causal querying additionally needs reviewed structure, suitable observations and explicit assumptions. The notebook, saved-run verifier and seventeen passing tests make the workflow inspectable. Thank you. We are ready for questions.

Transition: Keep this result slide visible for questions. Offer the live demonstration if requested.

## Live-demo script and recovery plan

Say: "I will now show the reviewed-graph path using a synthetic system with known assumptions. This validates the calculation, not a real-world causal discovery claim."

In Text to DAG, enter: Rain causes sprinkler. Rain causes wet grass. Sprinkler causes wet grass. Wet grass causes rain. Approve the candidates, show the cycle rejection and keep the three accepted edges. Say: "The final edge closes a cycle. Our static Bayesian network requires a DAG, so the graph builder rejects it. Source order affects this decision."

Select synthetic observations, enter your actual reviewer name and the rationale "Known synthetic generating graph", and explicitly check the stated assumptions. Fit the network. Set sprinkler to one and query wet grass. Say: "The fitted intervention probability is close to the known value 0.857. Its small difference reflects sampled observations and smoothing."

If the internet fails, use the bundled offline inference and synthetic example. If the live app fails, return to slide 8 and explain the saved oracle comparison. Do not invent a result or claim the live demo succeeded. If time runs short, skip the demo and move to the key findings.

## Answering questions professionally

Listen to the complete question. Begin with a direct answer, then connect it to one equation, code module or measured result. If asked whether the model establishes causation, say: "The text model identifies candidate relations. Causal effects follow only under the supplied graph and identification assumptions." If you do not know an answer, state what you do know and identify the additional experiment or assumption needed. Avoid guessing.

Before the viva, every member should rehearse the observation-versus-intervention explanation, derive 0.857 from the oracle, explain why a reversed edge counts as both an FP and FN, and explain why the hypothesis-only control limits our reasoning claim.
