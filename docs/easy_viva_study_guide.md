# Causal Workbench: Easy Study Guide and Faculty Script

Phase 1 | Team T094, Section E | Presenters: Ishita, Aditya and Yug


## Your viva, made simple

01 / START HERE

# What you need to understand
Our project connects causal language with a formal graph and probability model. Language suggests a relationship. Human review and graph checks decide what enters the network. Observations provide the probabilities. The network then answers questions about interventions under stated assumptions.

> ONE LINE TO REMEMBER: We study causal claims in text, review their structure, and use a Bayesian network to compare observation with intervention.

# Who studies what
Ishita: problem, datasets, majority baseline, TF-IDF and Naive Bayes. Present slides 1-3.
Aditya: Logistic Regression, model comparisons, metrics and extraction results. Present slides 4-6.
Yug: rule-based extraction, DAG checks, Bayesian network and intervention. Present slides 7-8 and operate the live demo.
Vidit: majority baseline as his easy preparation topic. He has no speaking slot because he is absent. Ishita covers the baseline if asked.
These are study and speaking assignments. For questions about who built each part, describe your actual work.

# How to use this guide tonight
First, everyone reads pages 1-2 and 5. Then read your assigned model pages. Practise the exact scripts on pages 8-10 with the eight-slide deck. Run the demo on page 11 twice. Finish by asking each other the questions on page 12.

# Keep the scope clear
This is Phase 1: classical machine learning and Bayesian networks. We have not implemented a transformer or an LLM. The rain/sprinkler demonstration uses synthetic data and a supplied generating structure. Text alone does not prove causation.

---

## The project story

02 / EVERYONE SHOULD KNOW THIS

# The problem statement in easy English
A sentence can sound causal without providing enough evidence for a causal conclusion. Text extraction alone cannot ensure a valid graph or tell us what will happen when we take an action. We therefore separate language predictions from formal graph validation and probabilistic reasoning.

# Three parts, three different jobs
1. Claim reasoning: Corr2Cause gives a premise and a hypothesis. The classifier predicts whether the hypothesis follows from the premise. Label 1 means entailed. Label 0 means not entailed.
2. Directed extraction: SemEval gives a sentence with two marked entities. A separate classifier predicts no causal relation, entity 1 causes entity 2, or the reverse direction.
3. Graph and inference: reviewed relations enter a DAG. Matching observations estimate probability tables. Intervention queries need the graph and its causal assumptions.
The guided demo uses the rule-based extractor for the graph path. The trained text classifiers have their own separate evaluations.

# How the project was developed
Prepare and audit the datasets. Keep overlapping groups out of later splits. Learn text features on training data. Compare baselines on development data. Evaluate the chosen model on test data. Review candidate edges, reject invalid graph structures, fit Bayesian tables and check intervention answers against a known synthetic generator. Streamlit exposes these functions in an interactive interface.

# The data you should remember
Corr2Cause: 12,000 training, 1,040 development and 1,044 retained test examples.
SemEval: 6,400 training, 1,600 development and 2,700 retained test examples. This setup combines non-causal relations into the no-causal-relation class.
Bayesian teaching example: 6,000 seeded binary observations. Each value is 0 or 1.
Exact premise groups stay separate for reasoning. Sentence groups stay separate for extraction. SemEval lacks document IDs, so do not claim a document-level split.

---

## Ishita: the easy models

03 / BASELINE, TF-IDF AND NAIVE BAYES

# Majority baseline
It always predicts the most common training class. It learns no useful text pattern. It gives us a simple comparison: can our trained model do better than always choosing one answer?
In the Corr2Cause retained test set, 888 of 1,044 labels are negative. Always predicting negative gives about 85.1% accuracy but finds none of the positive examples. This is why accuracy alone can mislead us.

# TF-IDF in plain words
A computer needs numerical inputs. TF-IDF converts words and short phrases into numbers. A word gets weight from how often it appears in a text and how unusual it is across training texts. Common words tend to be less informative.
Unigram means one word. Bigram means two adjacent words. A phrase such as 'caused by' can carry direction information. Our reasoning tokenizer keeps single letters such as A and B because they represent variables in Corr2Cause.
TF-IDF is a feature method. The classifier makes the prediction. We fit vocabulary and word-frequency statistics on training data only.

# Multinomial Naive Bayes
It uses Bayes' rule to compare how likely each class is, given the text features. 'Naive' means it assumes features are conditionally independent when the class is known. Real language violates that assumption, but the model is fast and useful as a baseline.
Our model uses TF-IDF unigrams and bigrams with smoothing alpha = 1. Smoothing prevents unseen features from making a class score collapse to zero. It is a practical baseline using non-negative TF-IDF weights, rather than a literal raw-count generative model.
Its Corr2Cause test macro F1 is about 0.465. It misses most positive examples, so we do not choose it as the main classifier.

> SAY THIS: TF-IDF makes numerical text features. Naive Bayes uses those features and Bayes' rule to predict a class. It is a simple comparison model in our project.

---

## Aditya: Logistic Regression

04 / THE MAIN TEXT CLASSIFIERS

# What it does
Logistic Regression learns a weight for each TF-IDF feature. A weighted sum gives a score. For binary classification, a sigmoid converts the score into a number between zero and one. The decision threshold turns that score into a class prediction.
Formula to recognise: score = w1*x1 + w2*x2 + ... + b. Then p = 1 / (1 + exp(-score)).
A positive weight pushes the positive-class score up. A negative weight pushes it down. The model learns weights from labelled training examples. Its name contains 'regression', but we use it for classification.

# We train two separate classifiers
Corr2Cause classifier: two outputs, entailed or not entailed. It reads the premise and hypothesis. Its test macro F1 is 0.700. Its regularization setting is C = 1.
SemEval classifier: three outputs, no causal relation, e1 causes e2, or e2 causes e1. It predicts direction between supplied entity spans. Its test edge F1 is about 0.820. Development search over C = 0.1, 1 and 10 selects C = 10.
Do not say that one model produces both reported results.

# Important settings in easy English
Regularization discourages excessively large weights and helps control overfitting. Smaller C means stronger regularization. Larger C means weaker regularization.
class_weight = balanced gives more weight to examples from rarer classes. It does not create new examples.
The extraction context model replaces entity names with ENTITY1 and ENTITY2. This reduces reliance on memorising names while retaining their positions and surrounding words. An entity-only control removes the surrounding context instead.

# What to say if asked why this model
It is efficient on sparse text features, easy to reproduce and a suitable classical ML baseline. We select models using development performance and report test performance separately. Its score is uncalibrated and does not measure whether a causal statement is true in the real world.

---

## The results and what they mean

05 / EVERYONE SHOULD LEARN THESE NUMBERS

# Corr2Cause: claim reasoning
Test macro F1: majority baseline 0.460, Naive Bayes 0.465, unigram Logistic Regression 0.521, full-text Logistic Regression 0.700.
A separate hypothesis-only control reaches 0.653 without using the premise. This suggests substantial wording shortcuts. The full model's performance alone does not show that it learned formal causal logic.
The selected model's positive precision is 0.392 and positive recall is 0.923. It finds most positive examples but also makes many false-positive predictions. Do not describe it as almost perfectly accurate.

# SemEval: directed extraction
Test directed edge F1: lexical rules 0.742, context TF-IDF plus Logistic Regression 0.820, entity-only control 0.503. The context model's edge precision and recall are both approximately 0.820. Its edge F1 bootstrap interval is approximately 0.786 to 0.851, conditional on the frozen model and available sentence groups.
These comparisons support the value of context on this task. They do not establish graph correctness or causal truth. SemEval provides the entity spans, so we are not claiming automatic entity discovery.

# Metrics in easy words
Precision: of the positive predictions, how many were correct? TP / (TP + FP).
Recall: of the real positives, how many did we find? TP / (TP + FN).
F1: the harmonic mean of precision and recall. 2PR / (P + R).
Macro F1: calculate F1 for each class, then average equally.
For directed edges, predicting B causes A when A causes B is correct creates a false positive for the wrong edge and a false negative for the missing correct edge.

# A fair experiment
Training learns features and weights. Development selects settings. Test estimates performance on held-out examples. An ablation removes information to see how performance changes. Leakage happens when information from evaluation data influences training or model selection.

---

## Yug: text to a Bayesian network

06 / THE STRUCTURE AND PROBABILITY MODEL

# Rule-based extraction
The teaching extractor recognises simple affirmative patterns: 'X causes Y', 'X leads to Y' and 'X results in Y'. It normalises aliases, such as 'wet grass' to wet_grass, and keeps document ID, sentence ID and source sentence for each candidate.
It skips clauses with uncertainty or negation cues such as may, might or not. It cannot understand unrestricted prose. The separate trained SemEval classifier also proposes candidates, which require human review before graph construction.

# DAG in easy English
DAG means directed acyclic graph. Directed means edges have a direction. Acyclic means you cannot follow arrows and come back to the starting node.
The builder rejects self-loops, duplicate edges and edges that create cycles. It processes edges in source order. A different order can produce a different accepted graph, so cycle rejection is a structural safeguard rather than proof that the rejected claim is causally false.
Teaching structure: rain -> sprinkler, rain -> wet_grass and sprinkler -> wet_grass.

# Bayesian network
A Bayesian network combines a DAG with conditional probability tables, called CPDs or CPTs. The DAG identifies each node's parents. A table describes the node's probability for each parent setting.
For our system: P(R,S,W) = P(R) * P(S given R) * P(W given R,S).
The graph alone does not give numerical probabilities. We estimate those from matching observations. Uploaded teaching data must have matching column names, complete values and binary states 0 or 1.

# Bayesian estimation and inference
We fit with a BDeu prior, equivalent sample size 5. This adds prior smoothing, especially useful for rare parent configurations. It is not five extra observed rows. The fitted tables must form a valid probability model.
Variable elimination computes a requested marginal probability by summing over variables we do not query. For causal interpretation, we additionally assume a correct causal graph, observed relevant confounders and stable mechanisms.

---

## Observation versus intervention

07 / THE EXAMPLE EVERYONE MUST UNDERSTAND

# The everyday explanation
Observation: look at cases where the sprinkler happened to be on.
Intervention: force the sprinkler on, regardless of why it would normally turn on.
Rain affects sprinkler use and wet grass. Rain suppresses sprinkler use in this generator. So observing sprinkler status also tells us something about rain. When we force a state, we remove that selection mechanism.
An arrow records causal dependence. It does not necessarily mean a positive effect. This is why rain -> sprinkler is valid even though rain reduces sprinkler use.

# What do() changes
For do(sprinkler = 1), remove the incoming rain -> sprinkler edge and fix the sprinkler state. Keep rain -> wet_grass and sprinkler -> wet_grass. Rain still follows its original population distribution.
The code uses graph surgery and probability inference. It does not merely condition on the original, unmodified network.

# One calculation you can explain
P(rain = 1) = 0.3, so P(rain = 0) = 0.7.
When the sprinkler is on, P(wet = 1 given rain = 1) = 0.99 and P(wet = 1 given rain = 0) = 0.80.
Therefore P(wet = 1 given do(sprinkler = 1)) = 0.3*0.99 + 0.7*0.80 = 0.857.
When it is off, the exact answer is 0.3*0.90 + 0.7*0.02 = 0.284.

# What the app displays
Sprinkler OFF: observed 0.452, fitted intervention 0.275, known generator 0.284.
Sprinkler ON: observed 0.801, fitted intervention 0.846, known generator 0.857.
The fitted results differ slightly from exact answers because we estimate tables from finite sampled data and use smoothing. 'Oracle' here means the known generating-process answer, not another trained model.
These are population intervention queries. We do not claim individual counterfactual answers.

---

## Ishita: your speaking script

08 / READ ALOUD AND REHEARSE

# Slide 1: Causal Reasoning | about 35 seconds
Good morning. We are Team T094 from Section E. Our project is about causal reasoning from text. A sentence may describe a cause and an effect, but recognising those words is not enough to answer what happens when we take an action. We study the language with classical machine learning, then use reviewed graphs and Bayesian networks for formal probability queries.

# Slide 2: System Components | about 40 seconds
Our system has separate language and reasoning parts. One classifier checks whether a claim follows from a premise. Another predicts a causal direction between two given entities. The graph stage accepts reviewed candidate edges and checks that they form a directed acyclic graph. Matching observations provide the probability tables. This separation helps us distinguish a language prediction from a causal conclusion.

# Slide 3: Datasets and Split Integrity | about 40 seconds
We use Corr2Cause for claim reasoning and SemEval for directed relation classification. Our retained training sets contain twelve thousand and sixty-four hundred examples respectively. We keep repeated premise or sentence groups separate across splits. We learn text features only from training data, use development data to select settings, and use test data for evaluation. The Bayesian demonstration uses six thousand synthetic observations with a known generating process. Aditya will now explain the models and results.

# How to practise
Read each paragraph aloud once. Then look away and explain the same idea in your own words. Remember the first sentence and the main number. Pause before handing over. Speak slowly enough that the faculty can follow.

---

## Aditya: your speaking script

09 / READ ALOUD AND REHEARSE

# Slide 4: Classical Text Models | about 45 seconds
We first convert text into numerical features using TF-IDF. It gives weights to words and short phrases. Our comparisons include majority prediction, Naive Bayes and Logistic Regression. Majority prediction always chooses the most common class. Naive Bayes is a simple probabilistic baseline. Our selected reasoning classifier is Logistic Regression. We train a separate three-class Logistic Regression model for extraction. That model masks entity names and uses surrounding context to predict the relation direction.

# Slide 5: Reasoning Results | about 40 seconds
On Corr2Cause, our full-text Logistic Regression model achieves test macro F1 of zero point seven zero zero. The majority baseline achieves about zero point four six zero. However, the hypothesis-only control reaches zero point six five three without reading the premise. This shows that wording shortcuts provide useful signals. We therefore describe this as a text-classification baseline, rather than claiming that it has learned complete causal logic.

# Slide 6: Directed Extraction | about 40 seconds
For directed relation extraction, the context model achieves edge F1 of about zero point eight two zero. Lexical rules achieve zero point seven four two, while the entity-only control achieves zero point five zero three. This supports the value of sentence context. Direction matters: reversing an edge gives both a false positive and a false negative. The task supplies entity spans, and predicted edges still require review. Yug will now explain the Bayesian stage.

# How to practise
Read each paragraph aloud once. Then look away and explain the same idea in your own words. Remember the first sentence and the main number. Pause before handing over. Speak slowly enough that the faculty can follow.

---

## Yug: your speaking script

10 / READ ALOUD AND REHEARSE

# Slide 7: Observation and Intervention | about 50 seconds
Observation asks what happens among cases where the sprinkler is on. Intervention asks what happens if we force it on. In our teaching system, rain affects both sprinkler use and wet grass. Observing the sprinkler gives information about rain. An intervention removes the incoming rain-to-sprinkler edge and fixes the sprinkler state. We fit the network's conditional probability tables from observations using Bayesian smoothing. Causal interpretation depends on the supplied structure and assumptions about confounding and stable mechanisms.

# Slide 8: Results and Conclusions | about 45 seconds
We check the Bayesian calculation against the known synthetic generator. When we force the sprinkler on, the exact wet-grass probability is zero point eight five seven and the fitted result is about zero point eight four six. Our main finding is that context helps classify causal language, while intervention queries also require reviewed structure, observations and explicit assumptions. The text benchmarks and the synthetic validation measure different parts of the project. Thank you. We can now show the live demonstration or take questions.

# How to practise
Read each paragraph aloud once. Then look away and explain the same idea in your own words. Remember the first sentence and the main number. Pause before handing over. Speak slowly enough that the faculty can follow.

---

## The live Streamlit demonstration

11 / YUG OPERATES; EVERYONE UNDERSTANDS

# Open the correct page
Open https://causal-workbench-t094.streamlit.app/ and choose Viva demo in the sidebar. Wake the app before the presentation. Keep the deck open in another tab.
Say: "I will demonstrate the graph pipeline using a synthetic teaching system with known assumptions. The trained text classifiers are evaluated separately."

# Review the source and graph
Point to the four sentences. Tick 'I reviewed these four teaching candidates; check them for graph validity'.
Say: "The extractor proposes four relations and retains their source sentences. Three edges enter the DAG. Wet grass to rain would create a cycle, so the builder rejects it in source order. Acyclicity is a structural check, not proof of causal truth."

# Fit the network
Tick the teaching-assumptions checkbox. Press Fit teaching network.
Say: "We use six thousand synthetic observations to estimate conditional probability tables. The supplied generator justifies the teaching graph and its assumptions."
The button stays disabled until the assumptions are acknowledged. This button fits Bayesian tables, not the text classifiers. Those classifiers load saved models trained offline.

# Compare OFF and ON
Start at OFF. Point to 0.452 observed and 0.275 intervened.
Say: "When the sprinkler happens to be off, rain is relatively common. Forcing it off is a different question."
Click ON. Point to 0.801 observed and 0.846 intervened.
Say: "Forcing it on removes rain's influence on sprinkler selection. Rain can still wet the grass. The known answer is 0.857, close to our fitted result."

# If the faculty asks for the ML model
Open Reasoning lab, select an example and click Evaluate statement. Read the model decision and dataset label honestly, even if they differ. Open Extraction benchmark to show directed-relation results. Do not present a single demo prediction as benchmark accuracy.
If the internet fails, use the local app or explain the saved results. A wrong text prediction is a model error you can discuss. Never promise every example is correct.

---

## Short answers for the faculty

12 / FINAL REVISION

# What is your actual contribution?
"We built a reproducible workflow that separates text classification, reviewed graph construction and Bayesian intervention queries. We compared baselines and controls, and checked the teaching inference against known answers." For individual ownership, name the files and tasks you actually worked on.

# Why not use accuracy alone?
"The labels are imbalanced. Always predicting negative gets about 85 percent accuracy on our reasoning test set but finds no positive cases. Macro F1 reveals this weakness."

# Does a DAG prove causation?
"No. A DAG is a valid structure for this model. Causal interpretation additionally needs justified directions, relevant observed confounders and stable mechanisms."

# Why use synthetic data?
"It gives us a known generating process. We can check intervention calculations against its exact answer. This does not validate real-world causal discovery."

# What does the title's neuro-symbolic idea mean here?
"The overall idea combines language processing with structured causal reasoning. Our Phase 1 implementation uses classical text models and Bayesian networks. We have not implemented a neural language model."

# What is the main limitation?
"Strong text scores can reflect shortcuts. Entity spans are supplied for extraction. Human review and causal assumptions remain necessary, and Bayesian validation uses a small synthetic domain."

# Final rehearsal
One full run with the slides. One run with the app. Then each person explains observation versus intervention without notes. If you forget a term, use the plain-language meaning. If you do not know an answer, say what you know and what further check is needed.

# Where the facts come from
Source: github.com/Mystic-commits/AML-DL_Project. Results: artifacts/metrics.json and the verified extraction run. Background: Corr2Cause, arxiv.org/abs/2306.05836; SemEval, aclanthology.org/S10-1006/; Pearl (1988).

---
