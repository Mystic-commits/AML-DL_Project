# Application usage

Open https://causal-workbench-t094.streamlit.app/ or run `python -m streamlit run app.py` locally. Select a workspace in the sidebar. The guided example works without uploads or API keys.

## Guided example

1. Select **Walkthrough** and read the four source statements.
2. Tick the review checkbox. The builder accepts three edges and rejects wet_grass → rain because it would close a cycle in source order.
3. Confirm the supplied synthetic-generator assumptions and press **Learn from the example data**.
4. Toggle the sprinkler OFF and ON. Compare observed and intervention probabilities with the known generating-process answer.
5. Expand the fitted tables or download the network audit if needed.

The example uses 6,000 synthetic binary observations. It runs the actual extraction, graph validation, Bayesian estimation and query functions. Its fixed structure is a teaching assumption, not a graph discovered from real-world data.

## Reasoning and extraction

In **Reasoning lab**, select a held-out example, optionally edit the premise/hypothesis, then press **Evaluate statement**. The score is uncalibrated and is not the probability that a real-world causal claim is true.

In **Extraction benchmark**, mark exactly one entity with `<e1>...</e1>` and one with `<e2>...</e2>`, then press **Classify entity pair**. Labels are 0 for no causal relation, 1 for e1 causes e2 and 2 for e2 causes e1. Proposed edges can be included for review in Text → DAG. A prediction does not automatically enter a causal network.

## Custom graph and observations

In **Text → DAG**, enter simple affirmative clauses using 'causes', 'leads to' or 'results in'. The lexical extractor does not support unrestricted prose and skips uncertainty/negation cues. Approve desired candidate rows individually. The builder rejects self-loops, duplicate edges and cycles.

To fit the accepted graph, upload complete binary CSV observations with columns matching canonical node names. Alternatively choose the labelled synthetic rain/sprinkler/wet_grass observations for a compatible teaching graph. Supply reviewer, causal rationale and both assumptions. Fitting remains disabled until these prerequisites are complete. Fitted probabilities come from observations, never classifier scores.

**Intervention lab** provides the fixed rain/sprinkler example directly. Observation retains the original incoming mechanisms; intervention removes incoming edges to the fixed variable. Results depend on the supplied causal assumptions and answer population intervention questions rather than individual counterfactuals.
