import torch


def _selectStrategyLength(sentences, predictions, max_length):
    selected_sents = []
    sents_priority = torch.argsort(predictions, descending=True)
    summary_len = 0
    i = 0

    while (summary_len < max_length) and (i < len(sents_priority)):
        if summary_len + len(sentences[sents_priority[i]]) < max_length:
            selected_sents.append(sents_priority[i])
            summary_len += len(sentences[sents_priority[i]])
        i += 1

    return sorted(selected_sents)


def _selectStrategyCount(sentences, predictions, num_sents):
    return sorted(torch.topk(predictions, min(len(predictions), num_sents)).indices)


def _selectStrategyRatio(sentences, predictions, ratio):
    doc_length = sum([ len(sent) for sent in sentences ])
    return _selectStrategyLength(sentences, predictions, doc_length*ratio)


def _selectStrategyThreshold(sentences, predictions, threshold):
    return [i for i, score in enumerate(predictions) if score >= threshold]


def select(sentences, predictions, strategy, strategy_args):
    selected_sents = []

    if strategy == "length":
        selected_sents = _selectStrategyLength(sentences, predictions, strategy_args)
    elif strategy == "count":
        selected_sents = _selectStrategyCount(sentences, predictions, strategy_args)
    elif strategy == "ratio":
        selected_sents = _selectStrategyRatio(sentences, predictions, strategy_args)
    elif strategy == "threshold":
        selected_sents = _selectStrategyThreshold(sentences, predictions, strategy_args)
    else:
        raise NotImplementedError(f"Unknown strategy {strategy}")
    
    return [sentences[i] for i in selected_sents], selected_sents