"""Build small inference assets after reproducing the training experiment."""
import json
from pathlib import Path
import joblib
import pandas as pd
from src.runtime import export_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = joblib.load(ROOT / 'artifacts/reasoner.joblib')
    export_snapshot(model, ROOT / 'artifacts/reasoner_snapshot.json')
    examples = pd.read_csv(ROOT / 'artifacts/test_predictions.csv').head(20)
    folder = ROOT / 'data/demo'
    folder.mkdir(parents=True, exist_ok=True)
    examples[['input', 'label']].to_csv(folder / 'reasoning_examples.csv', index=False)
    manifest = json.loads((ROOT / 'data/raw/manifest.json').read_text())
    (folder / 'README.md').write_text(
        '# Reasoning demonstration examples\n\n'
        'Twenty retained held-out Corr2Cause examples for the interactive app. '
        'These are demonstration inputs, not a replacement evaluation split.\n\n'
        f"Source: https://huggingface.co/datasets/{manifest['dataset']}\n\n"
        f"Revision: `{manifest['revision']}`. Labels retain the source meaning: "
        '1 means entailed and 0 means not entailed.\n'
    )
    print('Exported portable model and 20 held-out demonstration inputs.')


if __name__ == '__main__':
    main()
