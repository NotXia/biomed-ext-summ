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



"""
    Splits a document in chunks of maximum a given size.

    Parameters
    ----------
        doc_tokens : str[]
            List of the tokens of the document.

        bos_token : str
            Begin of sentence token.

        eos_token : str
            End of sentence token.

        max_size : int
            Maximum size of a chunk.
    Returns
    -------
        chunks : str[][]
            Splitted document.
"""
def splitDocument(doc_tokens, bos_token, eos_token, max_size):
    def _findNextBOSFrom(start_idx):
        for i in range(start_idx, len(doc_tokens)):
            if doc_tokens[i] == bos_token:
                return i
        return -1
    
    def _findPreviousEOSFrom(start_idx):
        for i in range(start_idx, -1, -1):
            if doc_tokens[i] == eos_token:
                return i
        return -1
    
    chunks = []
    
    while len(doc_tokens) > max_size:
        # Splits at the eos token
        eos_idx = _findPreviousEOSFrom(max_size - 1)

        if eos_idx == -1: 
            # The sentence is too long.
            # Find the next bos in front of the current sentence (if exists) and truncate the current sentence.
            next_bos_idx = _findNextBOSFrom(max_size)
            if next_bos_idx != -1:
                doc_tokens = doc_tokens[:max_size-1] + [eos_token] + doc_tokens[next_bos_idx:]
            else:
                doc_tokens = doc_tokens[:max_size-1] + [eos_token]
            eos_idx = max_size - 1
            break

        chunks.append(doc_tokens[:eos_idx+1])
        doc_tokens = doc_tokens[eos_idx+1:]

    if len(doc_tokens) > 0: chunks.append(doc_tokens) # Remaining part of the document
    
    return chunks