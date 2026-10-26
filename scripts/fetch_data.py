"""Download a bounded, reproducible slice directly from the official dataset."""
import csv, hashlib, io, json, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REV="42ba12c769e11ff6427c9f52d7db58e3f9bf3e53"
def main():
    manifest={"dataset":"causal-nlp/corr2cause", "revision":REV,"sampling":"first 12000 train rows; all dev/test rows", "files":{}}
    for split, limit in [("train",12000),("dev",100000),("test",100000)]:
        url=f"https://huggingface.co/datasets/causal-nlp/corr2cause/resolve/{REV}/{split}.csv"
        request=urllib.request.Request(url,headers={"User-Agent":"CSAI-Phase1/1.0"})
        with urllib.request.urlopen(request,timeout=120) as response:
            reader=csv.DictReader(io.TextIOWrapper(response,encoding="utf-8"))
            rows=[]
            for row in reader:
                rows.append(row)
                if len(rows)>=limit: break
        target=ROOT/f"data/raw/{split}.csv"
        with target.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        manifest["files"][split]={"url":url,"rows":len(rows),"sha256":hashlib.sha256(target.read_bytes()).hexdigest()}
        print(split,len(rows),list(rows[0]),flush=True)
    (ROOT/"data/raw/manifest.json").write_text(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
