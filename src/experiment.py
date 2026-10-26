"""Classical text baselines, fitted on training text only."""
import hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import joblib
ROOT=Path(__file__).resolve().parents[1]
def premise(text):
    return re.split(r"Hypothesis:|Conclusion:", str(text),maxsplit=1,flags=re.I)[0].strip().lower()
def load_splits():
    frames={}; seen=set(); audit={}
    for split in ['train','dev','test']:
        df=pd.read_csv(ROOT/f'data/raw/{split}.csv')
        df['label']=df.label.astype(int)
        df['group']=df.input.map(lambda t:hashlib.sha256(premise(t).encode()).hexdigest())
        before=len(df)
        df=df.drop_duplicates(subset=['input'])
        df=df[~df.group.isin(seen)].copy()
        seen.update(df.group)
        audit[split]={'raw':before,'retained':len(df),'removed':before-len(df),'groups':df.group.nunique()}
        frames[split]=df
    return frames,audit
def scores(y,p):
    pr,rc,f1,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    _,_,macro,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
    return dict(accuracy=accuracy_score(y,p),precision=pr,recall=rc,f1=f1,macro_f1=macro,confusion_matrix=confusion_matrix(y,p,labels=[0,1]).tolist())
def run():
    frames,audit=load_splits();train,dev,test=[frames[k] for k in ['train','dev','test']]
    models={
      'Majority baseline':Pipeline([('tfidf',TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',max_features=18000)),('model',DummyClassifier(strategy='most_frequent'))]),
      'TF-IDF + Naive Bayes':Pipeline([('tfidf',TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',ngram_range=(1,2),max_features=18000)),('model',MultinomialNB(alpha=1.0))]),
      'TF-IDF + Logistic Regression':Pipeline([('tfidf',TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',ngram_range=(1,2),max_features=18000,sublinear_tf=True)),('model',LogisticRegression(C=1,class_weight='balanced',max_iter=1500,random_state=42))]),
      'Unigram ablation':Pipeline([('tfidf',TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',max_features=18000,sublinear_tf=True)),('model',LogisticRegression(C=1,class_weight='balanced',max_iter=1500,random_state=42))])}
    results={}
    for name,model in models.items():
        model.fit(train.input,train.label)
        results[name]={'dev':scores(dev.label,model.predict(dev.input))}
    eligible=[k for k in models if k!='Unigram ablation']
    winner=max(eligible,key=lambda k:results[k]['dev']['macro_f1'])
    for name,model in models.items():
        pred=model.predict(test.input)
        results[name]['test']=scores(test.label,pred)
        if name==winner:
            out=test.copy();out['prediction']=pred
            out['probability']=model.predict_proba(test.input)[:,1]
            out.to_csv(ROOT/'artifacts/test_predictions.csv',index=False)
            joblib.dump(model,ROOT/'artifacts/reasoner.joblib')
    rng=np.random.default_rng(42);pred=models[winner].predict(test.input)
    groups=test.group.unique(); bootstrap=[]
    for _ in range(300):
        sampled=rng.choice(groups,size=len(groups),replace=True)
        indices=np.concatenate([np.flatnonzero(test.group.to_numpy()==g) for g in sampled])
        bootstrap.append(scores(test.label.to_numpy()[indices],pred[indices])['macro_f1'])
    report={'seed':42,'selection':'highest dev macro F1; test never used for selection','selected_model':winner,'audit':audit,'results':results,'test_macro_f1_group_bootstrap_95ci':np.quantile(bootstrap,[.025,.975]).tolist(),'limitations':['Bounded prefix of training data; not a random full-corpus benchmark.','Premise text grouping blocks exact premise overlap; semantic equivalence under variable renaming is not guaranteed.','Corr2Cause measures reasoning entailment, not open-domain relation extraction.','Classifier probabilities are not calibrated causal confidence.']}
    (ROOT/'artifacts/metrics.json').write_text(json.dumps(report,indent=2))
    for split,df in frames.items():df.to_csv(ROOT/f'data/processed/{split}.csv',index=False)
    print(json.dumps(report,indent=2))
    return report
if __name__=='__main__':run()
