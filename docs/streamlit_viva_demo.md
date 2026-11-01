# Streamlit viva demo: click sequence and speaking script

Live app: https://causal-workbench-t094.streamlit.app/

## What the system actually does

There are three separate capabilities. Corr2Cause claim reasoning uses TF-IDF and logistic regression to predict whether a proposed statement follows from a statistical premise. SemEval directed extraction uses given entity spans and a separate TF-IDF/logistic-regression model to classify the direction of a causal relation. The graph and Bayesian stage takes reviewed edges, checks DAG validity, and fits conditional probability tables from matching binary observations. Text predictions alone do not identify causal effects.

The guided Viva demo runs the lexical extraction → reviewed DAG → fitted Bayesian network → intervention path using the actual implementation. Its rain/sprinkler dataset is synthetic and seeded; its generating DAG is supplied. The trained classifiers are demonstrated separately in the other labs. Phase 1 uses no neural model or LLM API.

## Prepare before entering

Open the app early and keep it awake. Keep the eight-slide presentation, the app, and GitHub in separate tabs. Start on Overview with a fresh browser session. Use a single person to operate the demo; other speakers explain the parts they understand. Assign speaking roles by rehearsal, and describe past contributions only as they actually happened.

## Main demo: approximately three minutes

1. Open **Viva demo** from the sidebar. Say: “We separate language-based candidate extraction from formal graph validation and probabilistic inference. I will demonstrate the graph pipeline on a controlled teaching example, then show our independently evaluated ML results.”
2. Show **01 / Read the source**. Say: “The extractor recognises simple explicit patterns such as ‘causes’. It normalises names, keeps the source sentence and proposes four edges. The final edge deliberately conflicts with the earlier structure. These sentences are input hypotheses, not evidence that causality is established.”
3. Tick **I reviewed these four teaching candidates; check them for graph validity**. Show the graph and 3 accepted / 1 rejected. Say: “The directed Bayesian network must be acyclic. In source order, wet grass to rain would create a cycle, so it is rejected. This is structural validation; it does not prove which causal direction is true.”
4. In **03 / Fit probabilities from data**, acknowledge the supplied teaching assumptions and press **Fit teaching network**. Say: “We now fit conditional probability tables from 6,000 synthetic observations using Bayesian estimation. These probabilities come from data, not from how often a phrase appears or from classifier confidence. The causal interpretation relies on the known generator, observed confounder and stable mechanisms.”
5. In **04 / Observe versus intervene**, begin at **OFF (0)**. Point to approximately **0.452 observed**, **0.275 intervened**, and **0.284 known generator**. Say: “Among cases where the sprinkler is off, rain is relatively common, so wet grass is still common. If we force the sprinkler off, we remove rain’s influence on sprinkler use. This changes the population we are reasoning about.”
6. Click **ON (1)**. Point to approximately **0.801 observed**, **0.846 intervened**, and **0.857 known generator**. Say: “Forcing the sprinkler on differs from merely observing it on. Intervention cuts rain to sprinkler, while rain can still wet the grass. The fitted result is close to the known generator; finite sampling and smoothing explain the small difference.”
7. Open **Evaluation**. Say: “Our Corr2Cause claim-reasoning model achieved test macro F1 0.700. We select settings using development data, keep the test set for evaluation and compare baselines and shortcut diagnostics.”
8. Open **Extraction benchmark**. Say: “This is a separate directed relation task with given entity spans. Its test edge F1 is approximately 0.820. We do not present that number as graph correctness or real-world causal identification.”

## Optional live ML interaction: one minute

Open **Reasoning lab**, choose a held-out example, read its premise and hypothesis, then click **Evaluate statement**. Compare the decision with the displayed dataset label. Say: “This is an entailment classifier. A wrong example is an error to discuss, not a reason to claim the demo failed. Its positive score is uncalibrated and is not the probability that a real-world causal claim is true.” Do not promise that every example will be correct.

Open **Extraction benchmark**, use the marked sentence `<e1>Heat</e1> causes <e2>evaporation</e2>.`, and click **Classify entity pair**. The result is an out-of-benchmark illustrative prediction. Explain whatever result appears. Labels are 0 = no causal relation, 1 = e1 causes e2, 2 = e2 causes e1. If an edge is proposed, it can be included for human review in Text → DAG; it is not automatically inserted into a fitted causal network.

## If asked to use your own text

In Text → DAG, enter simple affirmative clauses, such as `Rain causes wet grass. Sprinkler causes wet grass. Rain causes sprinkler.` Tick Approve for each desired row. Do not use negation, hedging, long unrestricted prose, or arbitrary entity names with the synthetic rain data. Fit with matching complete binary CSV observations; give reviewer/rationale and acknowledge assumptions. The fit button stays disabled until prerequisites are satisfied. If using different variables, upload the appropriate data and justify the graph. For a short viva, the guided demo removes these setup steps by explicitly fixing the teaching domain.

## Questions you should be ready for

- Why is rain → sprinkler a causal edge if rain reduces sprinkler use? An edge indicates dependence of the child's mechanism on the parent. Its sign and strength appear in the conditional probabilities, not in arrow direction.
- Why is a cycle rejected? This chosen Bayesian network factorisation requires a DAG. Feedback systems require another modelling approach or time-indexed variables.
- What is do()? It replaces the treatment's generating mechanism with a fixed value, removing incoming edges. Merely conditioning preserves the generating mechanism.
- Why synthetic data? It gives a known generating process against which intervention calculations can be checked. It does not validate deployment in a scientific or clinical domain.
- What does the demo train? It estimates Bayesian probability tables on demand. The text classifiers were trained offline; the app loads frozen portable snapshots for prediction.
- Are you answering counterfactuals? No individual counterfactual capability is claimed. This demo answers population intervention queries under stated assumptions.
- Is Streamlit the ML model? No. It is the interactive Python interface. The model, graph checks, fitting and queries live in the source modules.

## Internet fallback

Use the local project environment: `.venv/bin/python -m streamlit run app.py`. Keep the seeded observations and bundled snapshots already available. If hosting sleeps, wake it before presenting; do not spend presentation time installing dependencies or retraining models.
