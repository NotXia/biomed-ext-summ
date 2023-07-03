from rouge_score.rouge_scorer import RougeScorer


"""
    ROUGE scores for a batch
"""
def evalROUGE(ref_summaries, ext_summaries):
    total_rouge1 = { "precision": 0, "recall": 0, "fmeasure": 0 }
    total_rouge2 = { "precision": 0, "recall": 0, "fmeasure": 0 }
    total_rougeL = { "precision": 0, "recall": 0, "fmeasure": 0 }
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"])
    batch_size = len(ref_summaries)

    for i in range(batch_size): # Batch handling
        ref_summary = ref_summaries[i]
        ext_summary = ext_summaries[i]

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
            "fmeasure": total_rouge1["fmeasure"] / batch_size,
            "precision": total_rouge1["precision"] / batch_size,
            "recall": total_rouge1["recall"] / batch_size
        },
        "rouge2": {
            "fmeasure": total_rouge2["fmeasure"] / batch_size,
            "precision": total_rouge2["precision"] / batch_size,
            "recall": total_rouge2["recall"] / batch_size
        },
        "rougeL": {
            "fmeasure": total_rougeL["fmeasure"] / batch_size,
            "precision": total_rougeL["precision"] / batch_size,
            "recall": total_rougeL["recall"] / batch_size
        }
    }