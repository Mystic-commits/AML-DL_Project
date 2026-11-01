"""A fixed, transparent teaching walkthrough using the actual pipeline functions."""
import json
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from src.causal import extract, build_graph, observations, query, oracle
from src.reviewed_network import fit_reviewed_network


def render_demo():
    st.title("A causal question. Step by step.")
    st.write("A three-minute viva walkthrough. Start here; use the individual labs for deeper experiments.")
    st.info("This is a synthetic teaching example with a supplied causal structure. Text proposes edges; observations estimate probabilities. It does not discover real-world causality automatically.")
    st.subheader("01 / Read the source")
    st.code("Rain causes sprinkler.\nRain causes wet grass.\nSprinkler causes wet grass.\nWet grass causes rain.", language=None)
    st.caption("‘Causes’ encodes a proposed dependency, not a positive effect. In this generator, rain reduces sprinkler use. The last sentence is a deliberately conflicting candidate to demonstrate cycle rejection.")
    candidates = extract("Rain causes sprinkler. Rain causes wet grass. Sprinkler causes wet grass. Wet grass causes rain.", document_id="viva_teaching_example")
    st.dataframe(candidates[["cause", "effect", "sentence_id", "sentence"]], hide_index=True, use_container_width=True)
    st.subheader("02 / Review and validate")
    reviewed = st.checkbox("I reviewed these four teaching candidates; check them for graph validity", key="demo_review")
    if not reviewed:
        st.caption("Next: tick the review box. No edges enter the graph before this action.")
        return
    graph, accepted, rejected = build_graph(candidates.to_dict("records"))
    a,b = st.columns([1.5,1])
    with a:
        fig, ax = plt.subplots(figsize=(7,3), facecolor="#F8F4EC")
        positions = {"rain":(0,1), "sprinkler":(1,1), "wet_grass":(.5,0)}
        nx.draw_networkx(graph, positions, ax=ax, node_color="#BF092F", font_color="white", edge_color="#594832", node_size=2600, arrowsize=22, font_size=10)
        ax.margins(.25); ax.axis("off"); st.pyplot(fig); plt.close(fig)
    with b:
        st.metric("Accepted edges", len(accepted))
        st.metric("Rejected cycle", len(rejected))
        st.write("**wet_grass → rain** would create a cycle. The builder rejects it in source order.")
    st.caption("Acyclicity is a structural check, not proof of the true causal directions. In your own experiments, review each edge individually in Text → DAG.")
    st.subheader("03 / Fit probabilities from data")
    st.write("Fit Bayesian conditional probability tables using 6,000 seeded binary observations from the rain/sprinkler teaching generator. No classifier score is used as a probability table.")
    assumptions = st.checkbox("For this supplied teaching generator, use its known DAG, observed confounder (rain), and stable mechanisms", key="demo_assumptions")
    if st.button("Fit teaching network", type="primary", disabled=not assumptions):
        network, audit = fit_reviewed_network([dict(r, approved=True) for r in accepted], observations(), dict(reviewed_by="Interactive teaching walkthrough", causal_rationale="Supplied synthetic generator: rain suppresses sprinkler use; rain and sprinkler affect wet grass.", observed_confounders=True, stable_mechanisms=True))
        st.session_state["demo_network"] = (network, audit)
    if not assumptions or "demo_network" not in st.session_state:
        st.caption("Next: acknowledge the teaching assumptions, then press Fit teaching network.")
        return
    network, audit = st.session_state["demo_network"]
    st.success("Network fitted and validated. Now change the sprinkler state below.")
    st.subheader("04 / Observe versus intervene")
    value = st.radio("Sprinkler state", [0,1], format_func=lambda x: "OFF (0)" if x == 0 else "ON (1)", horizontal=True, key="demo_state")
    observed = query(variable="sprinkler", value=value, network=network)
    intervened = query(variable="sprinkler", value=value, intervention=True, network=network)
    a,b,c = st.columns(3)
    a.metric("Observed: P(wet | sprinkler)", f"{observed:.3f}")
    b.metric("Intervened: P(wet | do(sprinkler))", f"{intervened:.3f}")
    c.metric("Known generator answer", f"{oracle(value):.3f}")
    st.write("**Observed:** look at cases where the sprinkler happened to be in this state; rain still influences who belongs to that group. **Intervened:** force the sprinkler state, cutting rain → sprinkler while retaining rain → wet grass.")
    st.caption("Small differences from the generator answer come from finite sampling and Bayesian smoothing. These are population intervention queries, not individual counterfactuals.")
    with st.expander("Show fitted probability tables and audit"):
        for cpd in network.get_cpds():
            st.code(str(cpd), language=None)
        st.download_button("Download teaching network audit", json.dumps(audit, indent=2), "teaching_network_audit.json", "application/json")
    st.subheader("Finish with the measured results")
    st.write("Open Evaluation for Corr2Cause claim reasoning (test macro F1 0.700), then Extraction benchmark for SemEval directed extraction (test edge F1 0.820). These are separate tasks; the teaching example is not a held-out benchmark score.")
