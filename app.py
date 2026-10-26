from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from src.causal import extract,build_graph,query,oracle,fitted_network,observations
ROOT=Path(__file__).parent
st.set_page_config(page_title='Causal / Workbench',page_icon='◈',layout='wide')
st.markdown("""<style>
html,body,[class*="css"],.stApp{font-family:Inter,Arial,sans-serif} .block-container{max-width:1440px;padding-top:2.8rem}
h1{font-weight:500!important;letter-spacing:-2px!important;font-size:3.4rem!important} h2{font-weight:500!important;letter-spacing:-.7px}
[data-testid="stMetric"]{border:1px solid #303337;padding:22px;background:#101214;border-radius:0}
[data-testid="stMetricLabel"]{color:#92999b} [data-testid="stMetricValue"]{font-size:2rem}
[data-testid="stToolbar"]{display:none}.stAlert{background:#151a18!important;color:#b8c7be!important;border:1px solid #303a34}.stat-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.stat{border:1px solid #303337;padding:18px;background:#101214}.stat small{font-size:11px;letter-spacing:1px;color:#92999b}.stat strong{display:block;font-size:27px;font-weight:500;margin-top:12px}.stButton>button,.stTextArea textarea,.stSelectbox>div>div{border-radius:0!important}
[data-testid="stSidebar"]{border-right:1px solid #303337} .eyebrow{font-size:11px;letter-spacing:3px;color:#a8b8ae;margin-bottom:18px}
.rule{border-top:1px solid #303337;margin:24px 0}.note{color:#92999b;font-size:14px;line-height:1.7}
</style>""",unsafe_allow_html=True)
st.sidebar.markdown('### ◈ CAUSAL\nWORKBENCH')
st.sidebar.caption('T094 / SECTION E / PHASE 01')
page=st.sidebar.radio('Workspace',['Overview','Reasoning lab','Text → DAG','Intervention lab','Evaluation','Team & viva'],label_visibility='collapsed')
st.sidebar.divider();st.sidebar.caption('LOCAL RESEARCH ENVIRONMENT\n\nClassical ML · Bayesian networks\n\n02 NOV 2026 / 15:20–15:45 / A403')
@st.cache_data
def metrics(modified):return json.loads((ROOT/'artifacts/metrics.json').read_text())
@st.cache_resource
def reasoner(modified):return joblib.load(ROOT/'artifacts/reasoner.joblib')
m=metrics((ROOT/"artifacts/metrics.json").stat().st_mtime_ns)
st.markdown('<div class="eyebrow">NEURO-SYMBOLIC CAUSAL INFERENCE / PHASE 01</div>',unsafe_allow_html=True)
if page=='Overview':
    st.title('From language.\nTo causal structure.')
    st.markdown('A research workspace for testing causal reasoning, reviewing extracted relations, and asking explicit intervention questions.')
    st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
    selected=m['results'][m['selected_model']]['test']['macro_f1']
    st.markdown(f'<div class="stat-grid"><div class="stat"><small>TRAINING EXAMPLES</small><strong>{m["audit"]["train"]["retained"]:,}</strong></div><div class="stat"><small>HELD-OUT EXAMPLES</small><strong>{m["audit"]["test"]["retained"]:,}</strong></div><div class="stat"><small>TEST MACRO F1</small><strong>{selected:.3f}</strong></div><div class="stat"><small>RESEARCH SCOPE</small><strong>Classical ML</strong></div></div>',unsafe_allow_html=True)
    st.write('');left,right=st.columns([1.3,1])
    with left:
        st.subheader('One pipeline. Three distinct questions.')
        st.markdown('**01 / Reasoning**  \nDoes the proposed causal claim follow from the stated statistical relationships?\n\n**02 / Structure**  \nWhich explicitly stated text relations survive human review and graph checks?\n\n**03 / Intervention**  \nWhat changes when an action replaces a variable’s generating mechanism?')
    with right:
        st.subheader('Research boundaries')
        st.markdown('<div class="stat note">Corr2Cause is a reasoning benchmark. The lexical extractor is a separate demonstration. Bayesian probabilities are fitted to a labelled synthetic observational dataset, under its stated causal assumptions.</div>',unsafe_allow_html=True)
        st.caption('No transformer, LLM API, deep learning, or hosted service is used.')
elif page=='Reasoning lab':
    st.title('Test the claim.')
    st.caption('Binary entailment on Corr2Cause. A model score is not a probability that a causal relationship is true in the world.')
    examples=pd.read_csv(ROOT/'artifacts/test_predictions.csv')
    idx=st.selectbox('Held-out example',range(min(20,len(examples))),format_func=lambda i:f'Example {i+1:02d}')
    text=st.text_area('Premise and hypothesis',examples.iloc[idx]['input'],height=230)
    if st.button('Evaluate statement',type='primary'):
        model=reasoner((ROOT/"artifacts/reasoner.joblib").stat().st_mtime_ns);prediction=int(model.predict([text])[0]);score=float(model.predict_proba([text])[0,1])
        a,b=st.columns(2);a.metric('Model decision','Entailed' if prediction else 'Not entailed');b.metric('Uncalibrated positive score',f'{score:.3f}')
        if text==examples.iloc[idx]['input']:st.caption(f"Dataset label: {int(examples.iloc[idx]['label'])} · 1 = entailed, 0 = not entailed")
        st.caption('Selected on development macro F1: '+m['selected_model'])
elif page=='Text → DAG':
    st.title('Review the structure.')
    st.caption('Supported patterns: “X causes Y”, “X leads to Y”, “X results in Y”. Simple affirmative clauses only; the extractor does not understand unrestricted prose.')
    text=st.text_area('Source document','Rain causes wet grass. Sprinkler causes wet grass. Rain causes sprinkler. Wet grass causes rain.',height=130)
    candidates=extract(text)
    if candidates.empty:st.info('No supported explicit relations found.')
    else:
        candidates.insert(0,'approved',False)
        st.write('Approve candidate edges before they enter the graph. Text mentions alone do not establish causality.')
        edited=st.data_editor(candidates,disabled=list(candidates.columns[1:]),hide_index=True,use_container_width=True,key='review')
        graph,accepted,rejected=build_graph(edited[edited.approved].drop(columns='approved').to_dict('records'))
        a,b=st.columns([1.2,1])
        with a:
            if graph.nodes:
                fig,ax=plt.subplots(figsize=(8,4),facecolor='#101214');ax.set_facecolor('#101214')
                nx.draw_networkx(graph,pos=nx.spring_layout(graph,seed=42),ax=ax,node_color='#b7c5bb',edge_color='#8b9591',font_size=10,node_size=2600,arrowsize=24);ax.axis('off');st.pyplot(fig);plt.close(fig)
            else:st.info('Approve at least one relation to build the DAG.')
        with b:
            st.metric('Accepted edges',len(accepted));st.metric('Rejected edges',len(rejected))
            if rejected:st.dataframe(pd.DataFrame(rejected)[['cause','effect','reason']],hide_index=True)
            st.download_button('Export reviewed provenance',json.dumps({'accepted':accepted,'rejected':rejected},indent=2),'reviewed_edges.json','application/json')
        st.caption('Greedy cycle handling follows source order. Rejection is a structural safeguard, not proof that an edge is causally false. This reviewed graph is separate from the fixed synthetic network in Intervention lab.')
elif page=='Intervention lab':
    st.title('Observe. Intervene. Compare.')
    st.caption('Synthetic teaching system: Rain → Sprinkler, Rain → Wet grass, Sprinkler → Wet grass. Rain suppresses sprinkler use and also wets grass, creating confounding.')
    variable=st.selectbox('Condition or intervene on',['sprinkler','rain'])
    value=st.radio('Set state',[0,1],horizontal=True)
    obs=query(variable=variable,value=value);inter=query(variable=variable,value=value,intervention=True)
    a,b,c=st.columns(3);a.metric('P(wet | observed)',f'{obs:.3f}');b.metric('P(wet | do)',f'{inter:.3f}');c.metric('Intervention − observation',f'{inter-obs:+.3f}')
    st.markdown('Intervention deletes incoming edges to the chosen variable, then conditions on its fixed state. Observation retains those edges. Identification assumes the supplied DAG is correct, all confounders are observed, and the generating mechanisms remain stable.')
    if variable=='sprinkler':st.caption(f'Known generating-process intervention answer: {oracle(value):.3f}. The fitted answer differs slightly because it uses finite sampled data and Bayesian smoothing.')
    with st.expander('Fitted conditional probability tables'):
        for cpd in fitted_network().get_cpds():st.code(str(cpd),language=None)
    st.download_button('Download 6,000 synthetic observations',observations().to_csv(index=False),'synthetic_observations.csv')
    st.caption('Population intervention queries only. Individual counterfactuals require additional structural assumptions and are not claimed here.')
elif page=='Evaluation':
    st.title('Evidence over impressions.')
    rows=[]
    for name,result in m['results'].items():
        rows.append({'Model':name,**{k:v for k,v in result['test'].items() if k!='confusion_matrix'},'Dev macro F1':result['dev']['macro_f1']})
    st.dataframe(pd.DataFrame(rows).set_index('Model').style.format('{:.3f}'),use_container_width=True)
    st.caption(m['selection']+'. Unigram ablation is reported separately and excluded from model selection.')
    a,b=st.columns(2)
    with a:
        st.subheader('Split audit');st.dataframe(pd.DataFrame(m['audit']).T,use_container_width=True)
        lo,hi=m['test_macro_f1_group_bootstrap_95ci'];st.caption(f'Selected model test macro F1: group-bootstrap 95% interval [{lo:.3f}, {hi:.3f}], 300 resamples, seed 42.')
    with b:
        st.subheader('Confusion matrix');st.dataframe(pd.DataFrame(m['results'][m['selected_model']]['test']['confusion_matrix'],index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1']))
    with st.expander('Dataset exploration and error analysis'):
        st.image(str(ROOT/'artifacts/eda.png'))
        st.dataframe(pd.read_csv(ROOT/'artifacts/eda_summary.csv'),hide_index=True)
        st.dataframe(pd.read_csv(ROOT/'artifacts/template_metrics.csv'),hide_index=True)
        st.caption('Template slices reveal where the selected model fails. Small slices have substantial uncertainty.')
    with st.expander('Extraction and graph regression checks'):
        checks=json.loads((ROOT/'artifacts/component_validation.json').read_text())
        st.caption(checks['fixture_notice'])
        st.json({'extraction':checks['extraction'],'graph':checks['graph'],'interventions':checks['interventions']})
    st.subheader('What these results do and do not establish')
    for limitation in m['limitations']:st.markdown('• '+limitation)
    st.download_button('Download experiment record',json.dumps(m,indent=2),'metrics.json','application/json')
else:
    st.title('Prepared to explain it.')
    st.write('Team T094 · Section E · Monday, 2 November 2026 · 3:20–3:45 PM · A403')
    st.caption('Evaluators: Sumit Kumar and Indrapal')
    st.dataframe(pd.DataFrame([['Yug Johri','2401010520'],['Vidit Sachan','2401010502'],['Aditya Verma','2401010040'],['Kaavya Gala','2401010204'],['Ishita Thakur','2401010195']],columns=['Member','Enrollment']),hide_index=True,use_container_width=True)
    st.caption('Contribution ownership must reflect the team’s actual work; this project does not fabricate individual contribution records.')
    for q,a in {
      'Why does prediction not establish causation?':'Text classifiers learn associations with labels. Causal identification requires assumptions about mechanisms, confounding, and interventions.',
      'Why split before fitting TF-IDF?':'Vocabulary and inverse document frequencies are learned only on training examples. Exact premise groups are removed from later splits to reduce leakage.',
      'Why report macro F1?':'It weights both classes equally and exposes poor minority-class performance that accuracy can hide.',
      'Why reject cycles?':'The chosen Bayesian network factorization requires a DAG. Greedy rejection does not identify the true causal direction.',
      'Where do the probabilities come from?':'The demonstration fits Bayesian CPDs to seeded synthetic observations, not to text frequency or classifier scores.',
      'What is the key limitation?':'The reasoning classifier, lexical extraction demo, and synthetic Bayesian system are separate evaluated components; no end-to-end real-world causal discovery is claimed.'}.items():
        with st.expander(q):st.write(a)
