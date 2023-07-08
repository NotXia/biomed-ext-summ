import random



"""
    Reduces the tokens in a document.
    It starts by removing sentences that are not part of the summary.
    If not enough, summary sentences are removed starting from the bottom.

    Parameters
    ----------
        tokenized_sentences : str[][]
            Tokenized sentences of the document.

        labels : bool[]
            Label for each sentence of the document.

        max_tokens : int
            Maximum number of tokens the document should have.

        tokenizer : Tokenizer
            
    Returns
    -------
        reduced_sentences : str[][]
        reduced_labels : bool[]
"""
def reduceTokens(tokenized_sentences, labels, max_tokens, tokenizer):
    total_tokens = sum([ len(sent) for sent in tokenized_sentences ])

    # Randomly selects a sentence that is not part of the summary and removes it
    while total_tokens > max_tokens:
        removable_sentences = [i for i, selected in enumerate(labels) if not selected]
        if len(removable_sentences) == 0: break
        to_remove_sentence_idx = random.choice(removable_sentences)
        
        total_tokens -= len(tokenized_sentences[to_remove_sentence_idx])
        del tokenized_sentences[to_remove_sentence_idx]
        del labels[to_remove_sentence_idx]
    
    # Remove sentences (from the end) that make the document too long
    while total_tokens - len(tokenized_sentences[-1]) > max_tokens:
        total_tokens -= len(tokenized_sentences[-1])
        del tokenized_sentences[-1]
        del labels[-1]

    # Last sentence have to be truncated
    if total_tokens > max_tokens:
        tokenized_sentences[-1] = tokenized_sentences[-1][:len(tokenized_sentences[-1])-total_tokens-max_tokens-2] + [tokenizer.sep_token]
        # Truncated sentence is empty
        if len(tokenized_sentences[-1]) == 1 or len(tokenized_sentences[-1]) == 2:
            del tokenized_sentences[-1]
            del labels[-1]

    assert sum([ len(sent) for sent in tokenized_sentences ]) <= max_tokens
    return tokenized_sentences, labels