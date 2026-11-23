"""Export the development-selected primary checkpoint for lightweight CPU hosting."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from src.neural_data import file_sha, masked_text


def main():
    import torch
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = json.loads((args.run / 'run.json').read_text())
    checkpoint = args.run / 'checkpoint'
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
        local_files_only=True, trust_remote_code=False, use_safetensors=True, attn_implementation="eager").eval()
    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits
    text = [masked_text('<e1>Heat</e1> causes <e2>evaporation</e2>.'),
            masked_text('<e1>Flooding</e1> is caused by <e2>rain</e2>.')]
    encoded = tok(text, return_tensors='pt', padding=True)
    fp32 = args.output / 'model.fp32.onnx'
    torch.onnx.export(Logits(model).eval(), (encoded['input_ids'], encoded['attention_mask']), str(fp32),
        input_names=['input_ids','attention_mask'], output_names=['logits'], opset_version=17,
        dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},'logits':{0:'batch'}},
        dynamo=False, external_data=False)
    onnx.checker.check_model(str(fp32))
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(fp32), options, providers=['CPUExecutionProvider'])
    inputs = {k: encoded[k].numpy() for k in ['input_ids','attention_mask']}
    model.eval()
    with torch.no_grad(): reference = model(**encoded).logits.numpy()
    actual = session.run(None, inputs)[0]
    error = float(np.abs(actual-reference).max())
    if not np.allclose(reference, actual, atol=1e-4):
        raise ValueError(f'FP32 export parity failed: {error}')
    quantized = args.output / 'model.int8.onnx'
    quantize_dynamic(str(fp32), str(quantized), per_channel=False,
                     weight_type=QuantType.QUInt8, op_types_to_quantize=['MatMul'])
    onnx.checker.check_model(str(quantized))
    ort.InferenceSession(str(quantized), options, providers=['CPUExecutionProvider'])
    shutil.copy2(checkpoint / 'tokenizer.json', args.output / 'tokenizer.json')
    metadata = {'format':1, 'architecture':'DistilBERT', 'variant':run['variant'], 'seed':run['seed'],
        'selected_epoch':run['selected_epoch'], 'model_id':run['config']['model_id'],
        'upstream_revision':run['config']['model_revision'], 'max_length':run['config']['max_length'],
        'labels':run['config']['labels'], 'export':'dynamic_uint8_matmul_onnx_cpu',
        'primary_selection':'Predeclared seed 42 weighted model; selected by development edge F1',
        'fp32_sample_logits_max_abs_error':error,
        'files':{p.name:file_sha(p) for p in [quantized,args.output/'tokenizer.json']},
        'bytes':{'fp32':fp32.stat().st_size,'int8':quantized.stat().st_size},
        'limitations':['Scores uncalibrated', 'Given entity spans', 'Review required before DAG fitting'],
        'license':'Fine-tuned from Apache-2.0 distilbert/distilbert-base-uncased'}
    (args.output/'model_card.json').write_text(json.dumps(metadata,indent=2))
    archive = args.output/'causal-distilbert-int8.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in [quantized,args.output/'tokenizer.json',args.output/'model_card.json']:
            z.write(p,p.name)
    (args.output/'export_receipt.json').write_text(json.dumps({'status':'PASS',**metadata,
        'archive_sha256':file_sha(archive),'archive_bytes':archive.stat().st_size},indent=2))
    print((args.output/'export_receipt.json').read_text())


if __name__ == '__main__': main()
