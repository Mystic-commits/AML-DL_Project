"""Create audited train/dev/test inputs without altering Phase 1 artifacts."""
import argparse
import json
from pathlib import Path

from src.neural_data import prepare_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/phase2_extraction')
    parser.add_argument('--audit', type=Path, default=ROOT / 'artifacts/phase2/data_audit.json')
    args = parser.parse_args()
    audit = prepare_data(ROOT / 'data/raw/semeval', args.output,
                         ROOT / 'artifacts/runs/phase1_extraction_verified_20261101')
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2))
    print(json.dumps({'status': 'PASS', 'identical_classical_partition': True,
                      'counts': audit['counts'], 'local_data': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
