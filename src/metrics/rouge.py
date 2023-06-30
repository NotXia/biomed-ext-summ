from rouge_score.rouge_scorer import RougeScorer

def evalROUGE(model, tokenizer, documents, labels, predictions):
    total_rouge1 = { "precision": 0, "recall": 0, "fmeasure": 0 }
    total_rouge2 = { "precision": 0, "recall": 0, "fmeasure": 0 }
    total_rougeL = { "precision": 0, "recall": 0, "fmeasure": 0 }
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"])

    for i in range(len(labels)): # Batch handling
        ref_summary = documents["ref_summary"][i]
        ext_summary_size = len( (labels[i] == 1).nonzero(as_tuple=True)[0] )
        ext_summary = "\n".join( model.buildSummary(predictions[i], ext_summary_size, documents["ids"][i], documents["clss_mask"][i], tokenizer) )

        rouge_scores = scorer.score(ref_summary, ext_summary)
        total_rouge1["fmeasure"] += rouge_scores["rouge1"].fmeasure
        total_rouge1["precision"] += rouge_scores["rouge1"].precision
        total_rouge1["recall"] += rouge_scores["rouge1"].recall
        total_rouge2["fmeasure"] += rouge_scores["rouge2"].fmeasure
        total_rouge2["precision"] += rouge_scores["rouge2"].precision
        total_rouge2["recall"] += rouge_scores["rouge2"].recall
        total_rougeL["fmeasure"] += rouge_scores["rougeL"].fmeasure
        total_rougeL["precision"] += rouge_scores["rougeL"].precision
        total_rougeL["recall"] += rouge_scores["rougeL"].recall
    
    return {
        "rouge1": {
            "fmeasure": total_rouge1["fmeasure"] / len(labels),
            "precision": total_rouge1["precision"] / len(labels),
            "recall": total_rouge1["recall"] / len(labels)
        },
        "rouge2": {
            "fmeasure": total_rouge2["fmeasure"] / len(labels),
            "precision": total_rouge2["precision"] / len(labels),
            "recall": total_rouge2["recall"] / len(labels)
        },
        "rougeL": {
            "fmeasure": total_rougeL["fmeasure"] / len(labels),
            "precision": total_rougeL["precision"] / len(labels),
            "recall": total_rougeL["recall"] / len(labels)
        }
    }