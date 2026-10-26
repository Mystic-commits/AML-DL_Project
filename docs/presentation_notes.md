# Presentation notes

Use presentation.html offline; scroll one page per slide.

## 1. Causal Workbench

Neuro-Symbolic Causal Inference

Team T094 · Section E; Yug Johri · Vidit Sachan · Aditya Verma · Kaavya Gala · Ishita Thakur; Phase 1 · 2 November 2026

## 2. The problem

A fluent causal claim needs formal scrutiny.

Separate reasoning entailment, explicit relation extraction and causal queries. Each task needs its own evidence.

## 3. Data before models

12,000 train / 1,040 dev / 1,044 test

Official Corr2Cause, pinned source revision. Exact premise overlap removed before training. Training is a bounded prefix; test includes only 18 premise groups.

## 4. Classical baseline

TF-IDF → logistic regression

Train-only vocabulary · word unigrams and bigrams · balanced class weights · C = 1 · seed 42. Compare majority, naive Bayes and a unigram ablation.

## 5. Measured results

0.700 test macro F1

Majority: 0.460 · Naive Bayes: 0.465 · Unigram ablation: 0.521; Positive precision: 0.392 · Recall: 0.923 · F1: 0.551; Approximate group-bootstrap interval: 0.670–0.734

## 6. Text into structure

Extract → review → validate

Keep document and sentence provenance. Normalize small entity aliases. Reject self loops, duplicates and cycle-closing edges. Authored fixture only: 16 documents, F1 0.870.

## 7. Probability needs data

Rain → Sprinkler → Wet grass

Also Rain → Wet grass. Fit CPDs to 6,000 seeded synthetic observations using BDeu smoothing. This supplied causal graph is separate from user-reviewed text.

## 8. Observation versus action

P(W | S) ≠ P(W | do(S))

Sprinkler on: observation 0.801 · intervention 0.846 · oracle 0.857; Sprinkler off: observation 0.452 · intervention 0.275 · oracle 0.284; Intervening removes incoming edges to Sprinkler.

## 9. What the evidence supports

A transparent Phase 1 baseline

No general causal-understanding claim. Nonrandom training prefix; few test groups; lexical parser; authored extraction fixture; synthetic observational data. Individual counterfactuals are not implemented.

## 10. Demonstration and questions

Reproduce every claim.

Open the local Streamlit workspace. Review errors, approve edges, inspect cycle rejection and compare intervention queries. Source data, outputs, notebook and report are included.