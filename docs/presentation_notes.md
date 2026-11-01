# Faculty Presentation Script

Use the existing eight-slide deck. About six minutes, plus an optional two-minute demonstration. Presenters: Ishita (1-3), Aditya (4-6), Yug (7-8 and demo). Vidit is absent.

## Slide 1: Causal Reasoning

Ishita | about 35 seconds

Good morning. We are Team T094 from Section E. Our project is about causal reasoning from text. A sentence may describe a cause and an effect, but recognising those words is not enough to answer what happens when we take an action. We study the language with classical machine learning, then use reviewed graphs and Bayesian networks for formal probability queries.

## Slide 2: System Components

Ishita | about 40 seconds

Our system has separate language and reasoning parts. One classifier checks whether a claim follows from a premise. Another predicts a causal direction between two given entities. The graph stage accepts reviewed candidate edges and checks that they form a directed acyclic graph. Matching observations provide the probability tables. This separation helps us distinguish a language prediction from a causal conclusion.

## Slide 3: Datasets and Split Integrity

Ishita | about 40 seconds

We use Corr2Cause for claim reasoning and SemEval for directed relation classification. Our retained training sets contain twelve thousand and sixty-four hundred examples respectively. We keep repeated premise or sentence groups separate across splits. We learn text features only from training data, use development data to select settings, and use test data for evaluation. The Bayesian demonstration uses six thousand synthetic observations with a known generating process. Aditya will now explain the models and results.

## Slide 4: Classical Text Models

Aditya | about 45 seconds

We first convert text into numerical features using TF-IDF. It gives weights to words and short phrases. Our comparisons include majority prediction, Naive Bayes and Logistic Regression. Majority prediction always chooses the most common class. Naive Bayes is a simple probabilistic baseline. Our selected reasoning classifier is Logistic Regression. We train a separate three-class Logistic Regression model for extraction. That model masks entity names and uses surrounding context to predict the relation direction.

## Slide 5: Reasoning Results

Aditya | about 40 seconds

On Corr2Cause, our full-text Logistic Regression model achieves test macro F1 of zero point seven zero zero. The majority baseline achieves about zero point four six zero. However, the hypothesis-only control reaches zero point six five three without reading the premise. This shows that wording shortcuts provide useful signals. We therefore describe this as a text-classification baseline, rather than claiming that it has learned complete causal logic.

## Slide 6: Directed Extraction

Aditya | about 40 seconds

For directed relation extraction, the context model achieves edge F1 of about zero point eight two zero. Lexical rules achieve zero point seven four two, while the entity-only control achieves zero point five zero three. This supports the value of sentence context. Direction matters: reversing an edge gives both a false positive and a false negative. The task supplies entity spans, and predicted edges still require review. Yug will now explain the Bayesian stage.

## Slide 7: Observation and Intervention

Yug | about 50 seconds

Observation asks what happens among cases where the sprinkler is on. Intervention asks what happens if we force it on. In our teaching system, rain affects both sprinkler use and wet grass. Observing the sprinkler gives information about rain. An intervention removes the incoming rain-to-sprinkler edge and fixes the sprinkler state. We fit the network's conditional probability tables from observations using Bayesian smoothing. Causal interpretation depends on the supplied structure and assumptions about confounding and stable mechanisms.

## Slide 8: Results and Conclusions

Yug | about 45 seconds

We check the Bayesian calculation against the known synthetic generator. When we force the sprinkler on, the exact wet-grass probability is zero point eight five seven and the fitted result is about zero point eight four six. Our main finding is that context helps classify causal language, while intervention queries also require reviewed structure, observations and explicit assumptions. The text benchmarks and the synthetic validation measure different parts of the project. Thank you. We can now show the live demonstration or take questions.

## Live demonstration

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

## Your final memory line

Text suggests. People review. The graph checks structure. Data supplies probabilities. Intervention asks what changes when we force an action.
