"""
Causal Workbench — Phase 1 Streamlit Application
Team T094, Section E

Seven-page interactive workspace for causal reasoning, DAG extraction,
and Bayesian intervention queries. Uses classical ML only (no neural models).
"""

from pathlib import Path
import json
import base64
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from src.runtime import FrozenReasoner
from src.reviewed_network import fit_reviewed_network
from src.causal import (
    extract,
    build_graph,
    query,
    oracle,
    fitted_network,
    observations,
)

ROOT = Path(__file__).parent

# page config has to come before any other st call
st.set_page_config(
    page_title="Causal Workbench",
    page_icon="◈",
    layout="wide",
)

# The presentation and application share the same locally bundled design assets.
font_regular = base64.b64encode((ROOT / "assets/fonts/Poppins-Regular.ttf").read_bytes()).decode()
font_bold = base64.b64encode((ROOT / "assets/fonts/Poppins-Bold.ttf").read_bytes()).decode()
st.markdown(f"<style>@font-face{{font-family:Poppins;src:url(data:font/ttf;base64,{font_regular})}}"
            f"@font-face{{font-family:Poppins;src:url(data:font/ttf;base64,{font_bold});font-weight:700}}</style>",
            unsafe_allow_html=True)
st.markdown((ROOT / "assets/workbench.css").read_text().join(["<style>", "</style>"]), unsafe_allow_html=True)


# ── sidebar ──

st.sidebar.markdown('<div class="brand">CAUSAL<span>WORKBENCH.</span></div>', unsafe_allow_html=True)
st.sidebar.caption("T094 / SECTION E / PHASE 01")

page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Reasoning lab",
        "Text → DAG",
        "Intervention lab",
        "Evaluation",
        "Team & viva",
        "Extraction benchmark",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(
    "CAUSAL RESEARCH WORKSPACE\n\n"
    "Classical ML · Bayesian networks\n\n"
    "T094 · FOUR RESEARCHERS · ONE WORKSPACE"
)


# ── cached loaders ──

@st.cache_data
def load_metrics(_modified):
    """Load the experiment metrics once and cache until file changes."""
    return json.loads((ROOT / "artifacts/metrics.json").read_text())


@st.cache_resource
def load_reasoner(_modified):
    """Load the frozen model snapshot once."""
    return FrozenReasoner(ROOT / "artifacts/reasoner_snapshot.json")


@st.cache_resource
def get_fitted_network():
    """Fit the Bayesian network once and reuse across queries."""
    return fitted_network()


metrics_data = load_metrics(
    (ROOT / "artifacts/metrics.json").stat().st_mtime_ns
)
selected_model_name = metrics_data["selected_model"]
selected_test = metrics_data["results"][selected_model_name]["test"]


# ── top eyebrow ──

st.markdown(
    '<div class="eyebrow">NEURO-SYMBOLIC CAUSAL INFERENCE / PHASE 01</div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════
#  PAGE: Overview
# ═══════════════════════════════════════════════════

if page == "Overview":
    st.markdown('<div class="hero-kicker">LANGUAGE × STRUCTURE × INFERENCE</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">From language.<br>To causal structure.</h1>', unsafe_allow_html=True)

    st.markdown(
        "A research workspace for testing causal reasoning, "
        "reviewing extracted relations, and asking explicit intervention questions."
    )

    st.image(str(ROOT / "assets/research-hands.png"), use_container_width=True)
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    macro_f1 = selected_test["macro_f1"]
    train_n = metrics_data["audit"]["train"]["retained"]
    test_n = metrics_data["audit"]["test"]["retained"]

    # stat cards using html grid
    st.markdown(
        f"""<div class="stat-grid">
        <div class="stat"><small>TRAINING EXAMPLES</small><strong>{train_n:,}</strong></div>
        <div class="stat"><small>HELD-OUT EXAMPLES</small><strong>{test_n:,}</strong></div>
        <div class="stat"><small>TEST MACRO F1</small><strong>{macro_f1:.3f}</strong></div>
        <div class="stat"><small>RESEARCH SCOPE</small><strong>Classical ML</strong></div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.write("")  # spacer

    left_col, right_col = st.columns([1.3, 1])

    with left_col:
        st.subheader("One pipeline. Three distinct questions.")
        st.markdown(
            "**01 / Reasoning**  \n"
            "Does the proposed causal claim follow from the stated statistical relationships?\n\n"
            "**02 / Structure**  \n"
            "Which explicitly stated text relations survive human review and graph checks?\n\n"
            "**03 / Intervention**  \n"
            "What changes when an action replaces a variable's generating mechanism?"
        )

    with right_col:
        st.subheader("Research boundaries")
        st.markdown(
            '<div class="stat note">'
            "Corr2Cause measures claim reasoning; SemEval measures directed relation extraction. "
            "Candidate edges require review before graph construction. "
            "Bayesian probabilities are fitted to a labelled synthetic observational dataset, "
            "under its stated causal assumptions."
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Phase 1 uses classical ML and Bayesian inference. "
            "No LLM API or neural model is used."
        )


# ═══════════════════════════════════════════════════
#  PAGE: Reasoning lab
# ═══════════════════════════════════════════════════

elif page == "Reasoning lab":
    st.title("Test the claim.")
    st.caption(
        "Binary entailment on Corr2Cause. "
        "A model score is not a probability that a causal relationship is true in the world."
    )

    # load the 20 demo examples
    examples = pd.read_csv(ROOT / "data/demo/reasoning_examples.csv")
    num_examples = min(20, len(examples))

    idx = st.selectbox(
        "Held-out example",
        range(num_examples),
        format_func=lambda i: f"Example {i + 1:02d}",
    )

    user_text = st.text_area(
        "Premise and hypothesis",
        examples.iloc[idx]["input"],
        height=230,
    )

    if st.button("Evaluate statement", type="primary"):
        model = load_reasoner(
            (ROOT / "artifacts/reasoner_snapshot.json").stat().st_mtime_ns
        )
        prediction = int(model.predict([user_text])[0])
        score = float(model.predict_proba([user_text])[0, 1])

        col_a, col_b = st.columns(2)
        col_a.metric(
            "Model decision",
            "Entailed" if prediction else "Not entailed",
        )
        col_b.metric("Uncalibrated positive score", f"{score:.3f}")

        # show the gold label if user hasn't edited the text
        if user_text == examples.iloc[idx]["input"]:
            gold = int(examples.iloc[idx]["label"])
            st.caption(
                f"Dataset label: {gold} · 1 = entailed, 0 = not entailed"
            )

        st.caption("Selected on development macro F1: " + selected_model_name)


# ═══════════════════════════════════════════════════
#  PAGE: Text → DAG
# ═══════════════════════════════════════════════════

elif page == "Text → DAG":
    st.title("Review the structure.")
    st.caption(
        'Supported patterns: "X causes Y", "X leads to Y", "X results in Y". '
        "Simple affirmative clauses only; "
        "the extractor does not understand unrestricted prose."
    )

    source_text = st.text_area(
        "Source document",
        "Rain causes wet grass. Sprinkler causes wet grass. "
        "Rain causes sprinkler. Wet grass causes rain.",
        height=130,
    )

    candidates = extract(source_text)
    pending = st.session_state.get("classified_candidate")
    if pending and st.checkbox("Include latest classified entity-pair candidate for review"):
        candidates = pd.concat([candidates, pd.DataFrame([pending])], ignore_index=True)

    if candidates.empty:
        st.info("No supported explicit relations found.")
    else:
        # add an approval checkbox column
        candidates.insert(0, "approved", False)

        st.write(
            "Approve candidate edges before they enter the graph. "
            "Text mentions alone do not establish causality."
        )

        # let user check/uncheck which edges to keep
        edited = st.data_editor(
            candidates,
            disabled=list(candidates.columns[1:]),
            hide_index=True,
            use_container_width=True,
            key="review",
        )

        # build the DAG from approved edges only
        approved_rows = edited[edited.approved].drop(columns="approved")
        graph, accepted, rejected = build_graph(
            approved_rows.to_dict("records")
        )

        left_col, right_col = st.columns([1.2, 1])

        with left_col:
            if graph.nodes:
                fig, ax = plt.subplots(figsize=(8, 4), facecolor="#F8F4EC")
                ax.set_facecolor("#F8F4EC")
                pos = nx.spring_layout(graph, seed=42)
                nx.draw_networkx(
                    graph,
                    pos=pos,
                    ax=ax,
                    node_color="#BF092F",
                    edge_color="#594832",
                    font_size=10,
                    font_color="#ffffff",
                    node_size=2600,
                    arrowsize=24,
                )
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Approve at least one relation to build the DAG.")

        with right_col:
            st.metric("Accepted edges", len(accepted))
            st.metric("Rejected edges", len(rejected))

            if rejected:
                st.dataframe(
                    pd.DataFrame(rejected)[["cause", "effect", "reason"]],
                    hide_index=True,
                )

            st.download_button(
                "Export reviewed provenance",
                json.dumps(
                    {"accepted": accepted, "rejected": rejected}, indent=2
                ),
                "reviewed_edges.json",
                "application/json",
            )

        st.caption(
            "Greedy cycle handling follows source order. "
            "Rejection is a structural safeguard, not proof that an edge is causally false. "
            "Fit this reviewed graph below, or use the separate synthetic network in Intervention lab."
        )


        st.subheader("Fit the reviewed graph")
        st.caption("Upload complete binary observations with column names matching graph nodes. The teaching example uses synthetic observations. Text alone cannot estimate probabilities.")
        uploaded = st.file_uploader("Binary observations CSV", type=["csv"])
        use_demo = st.checkbox("Use synthetic rain/sprinkler/wet_grass observations")
        reviewer = st.text_input("Reviewer name")
        rationale = st.text_input("Why are these directions causal for this dataset?")
        confounders = st.checkbox("I assume all relevant confounders are observed")
        stable = st.checkbox("I assume mechanisms remain stable under intervention")
        if st.button("Fit approved graph"):
            try:
                if uploaded is None and not use_demo:
                    raise ValueError("Upload observations or select the synthetic example")
                data = observations() if use_demo else pd.read_csv(uploaded)
                reviewed = [dict(r, approved=True) for r in accepted]
                network, audit = fit_reviewed_network(reviewed, data, dict(
                    reviewed_by=reviewer, causal_rationale=rationale,
                    observed_confounders=confounders, stable_mechanisms=stable))
                st.session_state["reviewed_fit"] = (network, audit, json.dumps(accepted, sort_keys=True))
                st.success("Fitted reviewed DAG. Queries below rely on your recorded assumptions.")
            except (ValueError, KeyError) as error:
                st.error(str(error))
        fit = st.session_state.get("reviewed_fit")
        if fit and fit[2] == json.dumps(accepted, sort_keys=True):
            network, audit, _ = fit
            nodes = list(network.nodes())
            target = st.selectbox("Reviewed network target", nodes)
            variable = st.selectbox("Reviewed network treatment", [n for n in nodes if n != target])
            value = st.radio("Reviewed treatment state", [0, 1], horizontal=True)
            st.write({"observational_probability": query(target=target, variable=variable, value=value, network=network),
                "intervention_probability": query(target=target, variable=variable, value=value, intervention=True, network=network)})
            st.download_button("Export fitted network audit", json.dumps(audit, indent=2), "fitted_network.json", "application/json")


# ═══════════════════════════════════════════════════
#  PAGE: Intervention lab
# ═══════════════════════════════════════════════════

elif page == "Intervention lab":
    st.title("Observe. Intervene. Compare.")
    st.caption(
        "Synthetic teaching system: Rain → Sprinkler, Rain → Wet grass, "
        "Sprinkler → Wet grass. Rain suppresses sprinkler use and also wets grass, "
        "creating confounding."
    )

    variable = st.selectbox(
        "Condition or intervene on", ["sprinkler", "rain"]
    )
    value = st.radio("Set state", [0, 1], horizontal=True)

    # use the cached fitted network instead of refitting each time
    cached_net = get_fitted_network()

    obs_prob = query(
        variable=variable, value=value, network=cached_net
    )
    int_prob = query(
        variable=variable, value=value, intervention=True, network=cached_net
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("P(wet | observed)", f"{obs_prob:.3f}")
    col_b.metric("P(wet | do)", f"{int_prob:.3f}")
    col_c.metric("Intervention − observation", f"{int_prob - obs_prob:+.3f}")

    st.markdown(
        "Intervention deletes incoming edges to the chosen variable, "
        "then conditions on its fixed state. Observation retains those edges. "
        "Identification assumes the supplied DAG is correct, "
        "all confounders are observed, and the generating mechanisms remain stable."
    )

    if variable == "sprinkler":
        oracle_val = oracle(value)
        st.caption(
            f"Known generating-process intervention answer: {oracle_val:.3f}. "
            "The fitted answer differs slightly because it uses finite sampled data "
            "and Bayesian smoothing."
        )

    with st.expander("Fitted conditional probability tables"):
        for cpd in cached_net.get_cpds():
            st.code(str(cpd), language=None)

    st.download_button(
        "Download 6,000 synthetic observations",
        observations().to_csv(index=False),
        "synthetic_observations.csv",
    )

    st.caption(
        "Population intervention queries only. Individual counterfactuals "
        "require additional structural assumptions and are not claimed here."
    )


# ═══════════════════════════════════════════════════
#  PAGE: Evaluation
# ═══════════════════════════════════════════════════

elif page == "Evaluation":
    st.title("Evidence over impressions.")

    # build model comparison table
    rows = []
    for name, result in metrics_data["results"].items():
        row = {"Model": name}
        # grab test metrics (skip the confusion matrix for the table)
        for k, v in result["test"].items():
            if k != "confusion_matrix":
                row[k] = v
        row["Dev macro F1"] = result["dev"]["macro_f1"]
        rows.append(row)

    comparison_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        comparison_df.style.format("{:.3f}"),
        use_container_width=True,
    )
    st.caption(
        metrics_data["selection"]
        + ". Unigram ablation is reported separately and excluded from model selection."
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Split audit")
        audit_df = pd.DataFrame(metrics_data["audit"]).T
        st.dataframe(audit_df, use_container_width=True)

        ci_lo, ci_hi = metrics_data["test_macro_f1_group_bootstrap_95ci"]
        st.caption(
            f"Selected model test macro F1: group-bootstrap 95% interval "
            f"[{ci_lo:.3f}, {ci_hi:.3f}], 300 resamples, seed 42."
        )

    with right_col:
        st.subheader("Confusion matrix")
        cm = metrics_data["results"][selected_model_name]["test"]["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.dataframe(cm_df)

    # dataset exploration
    with st.expander("Dataset exploration and error analysis"):
        st.image(str(ROOT / "artifacts/eda.png"))
        st.dataframe(
            pd.read_csv(ROOT / "artifacts/eda_summary.csv"),
            hide_index=True,
        )
        st.dataframe(
            pd.read_csv(ROOT / "artifacts/template_metrics.csv"),
            hide_index=True,
        )
        st.caption(
            "Template slices reveal where the selected model fails. "
            "Small slices have substantial uncertainty."
        )

    # Shortcut learning / premise ablation diagnostic
    shortcut_path = ROOT / "artifacts/shortcut_audit.json"
    if shortcut_path.exists():
        with st.expander("Does the model use the premise? / Shortcut diagnostic"):
            shortcut_data = json.loads(shortcut_path.read_text())
            control_f1 = shortcut_data["results"]["test"]["macro_f1"]
            full_f1 = selected_test["macro_f1"]

            comparison_data = pd.DataFrame([
                {"Input Representation": "Full premise and hypothesis", "Test Macro F1": full_f1},
                {"Input Representation": "Hypothesis only (Premise masked)", "Test Macro F1": control_f1},
            ])
            st.dataframe(
                comparison_data.style.format({"Test Macro F1": "{:.3f}"}),
                hide_index=True,
                use_container_width=True,
            )
            st.write(shortcut_data["interpretation"])
            st.caption(
                "This diagnostic is excluded from model selection. It measures shortcut risk, not a newly tuned model."
            )

    # extraction and graph checks
    with st.expander("Extraction and graph regression checks"):
        checks = json.loads(
            (ROOT / "artifacts/component_validation.json").read_text()
        )
        st.caption(checks["fixture_notice"])
        st.json({
            "extraction": checks["extraction"],
            "graph": checks["graph"],
            "interventions": checks["interventions"],
        })

    st.subheader("What these results do and do not establish")
    for limitation in metrics_data["limitations"]:
        st.markdown("• " + limitation)

    st.download_button(
        "Download experiment record",
        json.dumps(metrics_data, indent=2),
        "metrics.json",
        "application/json",
    )


# ═══════════════════════════════════════════════════
#  PAGE: Team & viva
# ═══════════════════════════════════════════════════

elif page == "Team & viva":
    st.title("Prepared to explain it.")
    st.write(
        "Team T094 · Section E · Phase 1 · Four current members"
    )

    team = pd.DataFrame(
        [
            ["Yug Johri", "2401010520"],
            ["Vidit Sachan", "2401010502"],
            ["Aditya Verma", "2401010040"],
            ["Ishita Thakur", "2401010195"],
        ],
        columns=["Member", "Enrollment"],
    )
    st.dataframe(team, hide_index=True, use_container_width=True)


    # viva prep Q&A
    viva_questions = {
        "Why does prediction not establish causation?": (
            "Text classifiers learn associations with labels. "
            "Causal identification requires assumptions about mechanisms, "
            "confounding, and interventions."
        ),
        "Why split before fitting TF-IDF?": (
            "Vocabulary and inverse document frequencies are learned only on training examples. "
            "Exact premise groups are removed from later splits to reduce leakage."
        ),
        "Why report macro F1?": (
            "It weights both classes equally and exposes poor minority-class performance "
            "that accuracy can hide."
        ),
        "Why reject cycles?": (
            "The chosen Bayesian network factorization requires a DAG. "
            "Greedy rejection does not identify the true causal direction."
        ),
        "Where do the probabilities come from?": (
            "The demonstration fits Bayesian CPDs to seeded synthetic observations, "
            "not to text frequency or classifier scores."
        ),
        "What is the key limitation?": (
            "The reasoning classifier, lexical extraction demo, and synthetic Bayesian system "
            "are separate evaluated components; "
            "no end-to-end real-world causal discovery is claimed."
        ),
    }

    for question, answer in viva_questions.items():
        with st.expander(question):
            st.write(answer)


if page == "Extraction benchmark":
    st.title("Directed relation classification")
    st.caption("SemEval-2010 Task 8 extension. Given entity spans. Labels: no relation, e1 causes e2, e2 causes e1. No neural model in Phase 1.")
    result_path = ROOT / "artifacts/runs/phase1_extraction_verified_20261101/metrics.json"
    if result_path.exists():
        results = json.loads(result_path.read_text())
        st.dataframe(pd.DataFrame(results["test"]).T[["edge_precision","edge_recall","edge_f1","macro_f1"]])
        st.write("Selected regularization C:", results["selected_C"])
        st.write("Test edge F1 bootstrap interval:", results["test_edge_f1_bootstrap_95ci"])
        st.caption("Dev selects C. Entity strings are masked in the full context model. This is separate from Corr2Cause entailment and from document-level DAG correctness.")
    marked = st.text_area("Sentence with given entity spans", "<e1>Heat</e1> causes <e2>evaporation</e2>.")
    if st.button("Classify entity pair"):
        import re
        from src.extraction import lexical_label, predicted_edge, features
        if not all(len(re.findall(fr"<e{n}>.+?</e{n}>", marked)) == 1 for n in [1,2]):
            st.error("Provide exactly one e1 span and one e2 span.")
        else:
            label = lexical_label(marked)
            method = "explicit lexical rules"
            model_path = result_path.parent / "model_snapshot.json"
            if model_path.exists():
                from src.extraction_runtime import FrozenExtractor
                label = int(FrozenExtractor(model_path).predict([features(marked)])[0])
                method = "TF-IDF + Logistic Regression"
            edge = predicted_edge(marked, label)
            st.session_state.pop("classified_candidate", None)
            if edge:
                st.session_state["classified_candidate"] = dict(cause=edge[0], effect=edge[1],
                    document_id="marked_pair_interactive", sentence_id=1, sentence=marked, method=method)
                st.info("Candidate available in Text → DAG. Select include latest classified candidate, then review it.")
            st.write({"method": method, "label": label, "candidate_edge": edge})
            st.caption("A predicted edge still needs review before graph construction.")
