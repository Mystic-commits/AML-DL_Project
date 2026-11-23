"""Lightweight, hash-verified ONNX inference. No PyTorch or Transformers runtime."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

import numpy as np

from src.neural_data import masked_text


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_bundle(manifest, cache=None):
    """Only load the exact public release archive identified by a committed hash."""
    cache = Path(cache or Path(tempfile.gettempdir()) / 'causal-workbench-neural')
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / manifest['archive_sha256'][:20]
    if (target / 'model_card.json').exists():
        return target
    url = manifest['archive_url']
    if not url.startswith('https://github.com/Mystic-commits/AML-DL_Project/releases/download/'):
        raise ValueError('Unexpected model release URL')
    scratch = Path(tempfile.mkdtemp(dir=cache))
    try:
        archive = scratch / 'download.zip'
        request = urllib.request.Request(url, headers={'User-Agent':'CausalWorkbench/2.0'})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open('wb') as out:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 160 * 1024 * 1024:
                    raise ValueError('Model download exceeds size limit')
                out.write(chunk)
        if sha(archive) != manifest['archive_sha256']:
            raise ValueError('Model archive checksum mismatch')
        folder = scratch / 'bundle'; folder.mkdir()
        allowed = {'model.int8.onnx','tokenizer.json','model_card.json'}
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist()) != allowed or sum(x.file_size for x in z.infolist()) > 220*1024*1024:
                raise ValueError('Unexpected model archive contents')
            for name in allowed:
                with z.open(name) as source, (folder/name).open('wb') as dest:
                    shutil.copyfileobj(source,dest)
        try:
            folder.rename(target)
        except FileExistsError:
            pass  # Another session atomically finished the identical bundle.
        return target
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


class OnnxExtractor:
    def __init__(self, folder):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.folder = Path(folder)
        self.metadata = json.loads((self.folder/'model_card.json').read_text())
        for name, digest in self.metadata['files'].items():
            if name not in {'model.int8.onnx','tokenizer.json'} or sha(self.folder/name) != digest:
                raise ValueError(f'Model bundle checksum mismatch: {name}')
        self.tokenizer = Tokenizer.from_file(str(self.folder/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=self.metadata['max_length'])
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id('[PAD]'), pad_token='[PAD]')
        self.marker_ids = [self.tokenizer.token_to_id(t) for t in ['[E1]','[E2]']]
        if None in self.marker_ids or len(set(self.marker_ids)) != 2:
            raise ValueError('Invalid entity marker vocabulary')
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(self.folder/'model.int8.onnx'), options,
                                             providers=['CPUExecutionProvider'])

    def predict_proba(self, marked_sentences):
        if not marked_sentences:
            return np.empty((0,3))
        encoded = self.tokenizer.encode_batch([masked_text(t) for t in marked_sentences])
        for row in encoded:
            if any(row.ids.count(marker) != 1 for marker in self.marker_ids):
                raise ValueError('Sentence is too long: an entity marker would be lost. Shorten the sentence.')
        inputs = {'input_ids':np.asarray([e.ids for e in encoded],dtype=np.int64),
                  'attention_mask':np.asarray([e.attention_mask for e in encoded],dtype=np.int64)}
        logits = self.session.run(['logits'],inputs)[0].astype(np.float64)
        exp = np.exp(logits-logits.max(axis=1,keepdims=True))
        return exp/exp.sum(axis=1,keepdims=True)

    def predict(self, marked_sentences):
        return self.predict_proba(marked_sentences).argmax(axis=1)
