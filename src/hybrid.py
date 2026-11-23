"""Bridge neural labels to the existing reviewed graph contract, preserving provenance."""
import numpy as np
from src.extraction import predicted_edge


def candidate_record(sentence, label, probabilities, document_id, sentence_id):
    probabilities = np.asarray(probabilities, dtype=float)
    if int(label) not in [0,1,2] or probabilities.shape != (3,) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1):
        raise ValueError('Expected a valid three-class prediction')
    if not document_id.strip() or int(sentence_id) < 1:
        raise ValueError('Document and sentence provenance are required')
    edge = predicted_edge(sentence, label)
    if edge is None:
        return None
    return {'cause':edge[0], 'effect':edge[1], 'sentence':sentence,
            'document_id':document_id.strip(), 'sentence_id':int(sentence_id),
            'method':'DistilBERT dynamic INT8 ONNX', 'model_label':int(label),
            'uncalibrated_score':float(probabilities[int(label)])}
