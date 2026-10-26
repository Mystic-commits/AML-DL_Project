"""Save diagnostics and error slices from the completed experiment."""
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.experiment import load_splits,scores
ROOT=Path(__file__).resolve().parents[1]
def main():
    frames,_=load_splits();records=[]
    for split,df in frames.items():
        lengths=df.input.str.split().str.len()
        records.append(dict(split=split,rows=len(df),positive=int(df.label.sum()),negative=int((df.label==0).sum()),positive_rate=float(df.label.mean()),median_words=float(lengths.median()),max_words=int(lengths.max()),missing_input=int(df.input.isna().sum())))
    pd.DataFrame(records).to_csv(ROOT/'artifacts/eda_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for split,df in frames.items():axes[0].hist(df.input.str.split().str.len(),bins=25,alpha=.4,label=split,density=True)
    axes[0].set(xlabel='Words per example',ylabel='Density',title='Input length');axes[0].legend()
    pd.DataFrame(records).set_index('split')[['negative','positive']].plot.bar(stacked=True,ax=axes[1],color=['#535d67','#a8b9ae'])
    axes[1].set(title='Class imbalance',ylabel='Examples',xlabel='Split');fig.tight_layout();fig.savefig(ROOT/'artifacts/eda.png',dpi=180);plt.close(fig)
    predictions=pd.read_csv(ROOT/'artifacts/test_predictions.csv')
    predictions[predictions.label!=predictions.prediction].sort_values('probability',ascending=False).to_csv(ROOT/'artifacts/error_examples.csv',index=False)
    slices=[]
    for name,group in predictions.groupby('template'):
        slices.append({'template':name,'n':len(group),**{k:v for k,v in scores(group.label,group.prediction).items() if k!='confusion_matrix'}})
    pd.DataFrame(slices).to_csv(ROOT/'artifacts/template_metrics.csv',index=False)
    model=joblib.load(ROOT/'artifacts/reasoner.joblib')
    if hasattr(model['model'],'coef_'):
        names=model['tfidf'].get_feature_names_out();weights=model['model'].coef_[0];order=np.argsort(weights);idx=np.r_[order[:15],order[-15:]]
        pd.DataFrame({'feature':names[idx],'weight':weights[idx]}).to_csv(ROOT/'artifacts/feature_weights.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))
if __name__=='__main__':main()
