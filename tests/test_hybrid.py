import json
import pytest
from src.hybrid import candidate_record
from src.neural_runtime import OnnxExtractor, ensure_bundle


def test_neural_candidates_keep_provenance_without_auto_approval():
    text='<e1>Flooding</e1> is caused by <e2>rain</e2>.'
    row=candidate_record(text,2,[.01,.02,.97],'document-1',3)
    assert (row['cause'],row['effect'])==('rain','flooding')
    assert row['document_id']=='document-1' and row['sentence_id']==3
    assert row['sentence']==text and 'approved' not in row
    assert candidate_record(text,0,[.9,.05,.05],'document-1',3) is None
    with pytest.raises(ValueError):candidate_record(text,2,[.01,.02,.97],'',3)
    with pytest.raises(ValueError):candidate_record(text,5,[.01,.02,.97],'document-1',3)


def test_runtime_rejects_modified_weights_before_loading(tmp_path):
    (tmp_path/'model.int8.onnx').write_bytes(b'changed')
    (tmp_path/'model_card.json').write_text(json.dumps({'files':{'model.int8.onnx':'wronghash'}}))
    with pytest.raises(ValueError,match='checksum'):
        OnnxExtractor(tmp_path)


def test_downloader_rejects_unexpected_destination(tmp_path):
    with pytest.raises(ValueError,match='Unexpected model release URL'):
        ensure_bundle({'archive_sha256':'a'*64,'archive_url':'https://example.com/model.zip'},tmp_path)
