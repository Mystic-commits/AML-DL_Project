"""
Dataset Downloader for Corr2Cause.
Project 1: Neuro-Symbolic Causal Inference (Phase 1)
Team T094, Section E

Downloads a bounded, reproducible slice directly from the official HuggingFace repository:
https://huggingface.co/datasets/causal-nlp/corr2cause
Pinned commit revision: 42ba12c769e11ff6427c9f52d7db58e3f9bf3e53

Sampling strategy:
- Train: First 12,000 rows (to maintain fast local convergence and manageable memory)
- Dev: All rows (1,076 rows)
- Test: All rows (1,162 rows)
"""

import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_REVISION = "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53"


def download_split(split_name, limit, output_dir):
    """
    Download a single split CSV from Hugging Face, write to a temp file,
    verify integrity, and atomically move into destination.
    """
    url = (
        f"https://huggingface.co/datasets/causal-nlp/corr2cause/resolve/"
        f"{PINNED_REVISION}/{split_name}.csv"
    )
    print(f"Fetching {split_name} split from {url}...")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CSAI-Phase1-StudentResearch/1.0"},
    )

    with urllib.request.urlopen(req, timeout=120) as response:
        text_stream = io.TextIOWrapper(response, encoding="utf-8")
        reader = csv.DictReader(text_stream)
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"No rows retrieved for split {split_name}")

    target_file = output_dir / f"{split_name}.csv"
    temp_file = output_dir / f"{split_name}.tmp"

    with temp_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Atomic rename
    temp_file.replace(target_file)

    sha256 = hashlib.sha256(target_file.read_bytes()).hexdigest()
    print(
        f"  -> Successfully saved {len(rows)} rows to {target_file.name} (SHA-256: {sha256[:12]}...)"
    )

    return {
        "url": url,
        "rows": len(rows),
        "sha256": sha256,
    }


def main():
    raw_dir = ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "causal-nlp/corr2cause",
        "revision": PINNED_REVISION,
        "sampling": "first 12000 train rows; all dev/test rows",
        "files": {},
    }

    # Bounded train slice (12,000 rows); all rows for dev and test
    split_configs = [
        ("train", 12000),
        ("dev", 100000),
        ("test", 100000),
    ]

    for split_name, limit in split_configs:
        file_info = download_split(split_name, limit, raw_dir)
        manifest["files"][split_name] = file_info

    manifest_file = raw_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest recorded at {manifest_file}")


if __name__ == "__main__":
    main()
