"""A fixed, transparent teaching walkthrough using the actual pipeline functions."""
import json
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from src.causal import extract, build_graph, observations, query, oracle
from src.reviewed_network import fit_reviewed_network


def render_demo():
    st.title("How the project works")
    st.write("Follow four steps: read a few sentences, check their relationships, learn probabilities from example data, and try changing the sprinkler.")
    st.info("We use made-up rain and sprinkler data so we know the correct answers. This example shows how the system works. Real-world causal claims need evidence and justified assumptions.")
    st.subheader("1. Read the example sentences")
    st.code("Rain causes sprinkler.\nRain causes wet grass.\nSprinkler causes wet grass.\nWet grass causes rain.", language=None)
    st.caption("Here, rain affects whether the sprinkler runs: people use it less when it rains. The last sentence is an intentional mistake so you can see the graph check in action.")
    candidates = extract("Rain causes sprinkler. Rain causes wet grass. Sprinkler causes wet grass. Wet grass causes rain.", document_id="walkthrough_teaching_example")
    st.write("The text extractor finds these four possible relationships:")
    st.dataframe(candidates[["cause", "effect", "sentence_id", "sentence"]].rename(columns={"cause": "Possible cause", "effect": "Possible effect", "sentence_id": "Sentence number", "sentence": "Source sentence"}), hide_index=True, use_container_width=True)
    st.subheader("2. Check the suggested relationships")
    reviewed = st.checkbox("I have read the four relationships. Check the graph.", key="demo_review")
    if not reviewed:
        st.caption("Tick the box above to continue. The app waits for your review before adding relationships to the graph.")
        return
    graph, accepted, rejected = build_graph(candidates.to_dict("records"))
    a,b = st.columns([1.5,1])
    with a:
        fig, ax = plt.subplots(figsize=(7,3), facecolor="#F8F4EC")
        positions = {"rain":(0,1), "sprinkler":(1,1), "wet_grass":(.5,0)}
        nx.draw_networkx(graph, positions, ax=ax, node_color="#BF092F", font_color="white", edge_color="#594832", node_size=2600, arrowsize=22, font_size=10)
        ax.margins(.25); ax.axis("off"); st.pyplot(fig); plt.close(fig)
    with b:
        st.metric("Relationships kept", len(accepted))
        st.metric("Relationships rejected", len(rejected))
        st.write("The app rejects **wet grass → rain** because it would make a loop: rain → wet grass → rain.")
    st.caption("An arrow shows a suggested cause-and-effect direction. This type of network cannot contain loops. Passing this check does not prove a relationship is true. For your own text, review each suggestion in Text → DAG.")
    st.subheader("3. Learn probabilities from example data")
    st.write("The app uses 6,000 made-up records of rain, sprinkler use, and wet grass. It learns how often the grass is wet under different conditions. Each record uses 0 for no/off and 1 for yes/on.")
    st.caption("We assume rain is included as a shared cause of sprinkler use and wet grass, and that forcing the sprinkler does not change how rain or wet grass behave otherwise. The made-up data was generated using these assumptions.")
    assumptions = st.checkbox("Use the example’s known relationships and assumptions.", key="demo_assumptions")
    if st.button("Learn from the example data", type="primary", disabled=not assumptions):
        network, audit = fit_reviewed_network([dict(r, approved=True) for r in accepted], observations(), dict(reviewed_by="Interactive teaching walkthrough", causal_rationale="Supplied synthetic generator: rain suppresses sprinkler use; rain and sprinkler affect wet grass.", observed_confounders=True, stable_mechanisms=True))
        st.session_state["demo_network"] = (network, audit)
    if not assumptions or "demo_network" not in st.session_state:
        st.caption("Tick the assumptions box, then press Learn from the example data to continue.")
        return
    network, audit = st.session_state["demo_network"]
    st.success("Ready! Choose OFF or ON below to compare the two questions.")
    st.subheader("4. Compare watching with taking action")
    value = st.radio("Choose a sprinkler setting", [0,1], format_func=lambda x: "OFF (0)" if x == 0 else "ON (1)", horizontal=True, key="demo_state")
    observed = query(variable="sprinkler", value=value, network=network)
    intervened = query(variable="sprinkler", value=value, intervention=True, network=network)
    a,b,c = st.columns(3)
    a.metric("When we see this setting", f"{observed:.3f}")
    b.metric("When we force this setting", f"{intervened:.3f}")
    c.metric("Known answer for forcing it", f"{oracle(value):.3f}")
    setting = "on" if value else "off"
    st.caption(f"These values are chances between 0 and 1. For example, {intervened:.3f} means about {intervened * 100:.1f}%.")
    st.write(f"**Watching:** Among records where the sprinkler happened to be {setting}, how often was the grass wet? Rain influences both sprinkler use and wet grass.")
    st.write(f"**Taking action:** If we force the sprinkler {setting}, how often would the grass be wet? We stop rain from deciding the sprinkler setting, but rain can still wet the grass.")
    st.caption("The learned answer is close to the known answer. Small differences are expected because the app learns from a sample of records and smooths its probability estimates. These numbers describe the example population, not a particular lawn.")
    with st.expander("See the learned probability tables and details"):
        for cpd in network.get_cpds():
            st.code(str(cpd), language=None)
        st.download_button("Download the network details", json.dumps(audit, indent=2), "teaching_network_audit.json", "application/json")
    st.subheader("Explore the model results")
    st.write("Open Evaluation to see how well the text model checks claims (macro F1: 0.700). Open Extraction benchmark to see how well a separate model predicts relation directions (edge F1: 0.820). Those scores come from held-out text tests, separate from this made-up example.")
