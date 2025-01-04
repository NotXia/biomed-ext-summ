"""
Preprocessing for T5
"""

from datasets import Dataset
from utilities.preprocess import reduceTokens
import itertools



"""
    Parses the summary for T5.

    Parameters
    ----------
        sentences : str[]
        labels : bool[]
        tokenizer : T5Tokenizer
        max_tokens : int

    Returns
    -------
        doc_ids : str[]
            Tokenized document.

        cls_idxs : int[]
            Position of classification tokens.

        labels : bool[]
            Label for each sentence.
"""
def _parseForT5(sentences, labels, tokenizer, max_tokens=512):
    tokenized_sentences = [tokenizer.tokenize(sent) + [tokenizer.eos_token] for sent in sentences]

    # Reduces the number of tokens in the document
    reduced_sentences, reduced_labels = reduceTokens(tokenized_sentences, labels, max_tokens, tokenizer)
    doc_tokens = list(itertools.chain.from_iterable(reduced_sentences))

    doc_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
    cls_idxs = [i for i, token in enumerate(doc_ids) if token == tokenizer.convert_tokens_to_ids(tokenizer.eos_token)]

    return doc_ids, cls_idxs, reduced_labels



"""
    Return the utilities to parse T5

    Returns
    -------
        parseDataset : (dataset-row) -> dataset-row
            Function to use with datasets `map`

        filterDataset : (dataset) -> dataset
            Function to remove unnecessary columns from the dataset
"""
def preprocessUtilitiesT5(tokenizer):
    def parseDataset(data):
        out = {}
        doc_ids, cls_idxs, labels = _parseForT5(data["sentences"], data["labels"], tokenizer)
        out["__doc_ids"] = doc_ids
        out["__cls_idxs"] = cls_idxs
        out["__labels"] = labels
        assert len(out["__labels"]) == len(out["__cls_idxs"])
        return out

    def filterDataset(dataset):
        dataset_content = {
            "id": dataset["id"],    
            "ref_summary": dataset["ref_summary"],    
            "labels": dataset["__labels"],
            "doc_ids": dataset["__doc_ids"],
            "cls_idxs": dataset["__cls_idxs"]
        }
        return Dataset.from_dict(dataset_content)

    return parseDataset, filterDataset