from src.experiment import load_splits

def test_no_exact_premise_overlap():
    splits,_=load_splits()
    groups=[set(splits[s].group) for s in ['train','dev','test']]
    assert all(not groups[a]&groups[b] for a,b in [(0,1),(0,2),(1,2)])
    assert all(set(splits[s].label)=={0,1} for s in splits)
