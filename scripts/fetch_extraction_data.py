"""Download immutable SemEval Task 8 files for a separate extraction evaluation."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'd58325907b9bf84a7c81f81fe5372791d911c10c'
DATASET = 'SemEvalWorkshop/sem_eval_2010_task_8'


def main():
    dest = ROOT / 'data/raw/semeval'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset': DATASET, 'revision': REVISION, 'files': {}}
    for split in ['train', 'test']:
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{split}-00000-of-00001.parquet'
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        path = dest / f'{split}.parquet'
        path.with_suffix('.tmp').write_bytes(content)
        path.with_suffix('.tmp').replace(path)
        manifest['files'][split] = {'url': url, 'sha256': hashlib.sha256(content).hexdigest()}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Downloaded pinned SemEval train/test parquet files.')


if __name__ == '__main__':
    main()
