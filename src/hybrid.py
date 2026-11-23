"""Bridge neural labels to the existing reviewed graph contract, preserving provenance."""
from src.extraction import predicted_edge


def candidate_record(sentence, label, probabilities, document_id, sentence_id):
    if not document_id.strip() or int(sentence_id) < 1:
        raise ValueError('Document and sentence provenance are required')
    edge = predicted_edge(sentence, label)
    if edge is None:
        return None
    return {'cause':edge[0], 'effect':edge[1], 'sentence':sentence,
            'document_id':document_id.strip(), 'sentence_id':int(sentence_id),
            'method':'DistilBERT dynamic INT8 ONNX', 'model_label':int(label),
            'uncalibrated_score':float(probabilities[int(label)])}
