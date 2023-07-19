from bert_score import BERTScorer
import numpy as np


bert_scorer = None

"""
    BERTScore for a batch
"""
def evalBERTScore(ref_summaries, ext_summaries, model="allenai/longformer-base-4096"):
    global bert_scorer
    if bert_scorer is None: bert_scorer = BERTScorer(model_type=model)
    prec, recall, f1 = bert_scorer.score(ext_summaries, ref_summaries)
    
    return {
        "fmeasure": np.average(f1),
        "precision": np.average(prec),
        "recall": np.average(recall)
    }