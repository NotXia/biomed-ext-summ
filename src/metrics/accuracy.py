import torch
import numpy as np

def accuracy(labels, predictions):
    accuracies = []

    for i in range(len(labels)): # Batch handling
        summary_sentences_idxs = (labels[i] == 1).nonzero(as_tuple=True)[0]
        if len(summary_sentences_idxs) == 0: continue # Not able to evaluate if none of the sentences are part of the sentence
        selected_sentences_idxs = sorted(torch.topk(predictions[i], len(summary_sentences_idxs)).indices)

        idx_ref, idx_sel = 0, 0
        correct_choices = 0
        while idx_ref < len(summary_sentences_idxs) and idx_sel < len(selected_sentences_idxs):
            if summary_sentences_idxs[idx_ref] == selected_sentences_idxs[idx_sel]:
                correct_choices += 1
                idx_ref += 1
                idx_sel += 1
            elif summary_sentences_idxs[idx_ref] > selected_sentences_idxs[idx_sel]:
                idx_sel += 1
            else:
                idx_ref += 1

        accuracies.append( correct_choices / len(summary_sentences_idxs) )

    return np.average(accuracies)