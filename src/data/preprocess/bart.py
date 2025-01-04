"""
Preprocessing for BART
"""

from datasets import Dataset
from utilities.preprocess import reduceTokens
import itertools



"""
    Parses the summary for BART.

    Parameters
    ----------
        sentences : str[]
        labels : bool[]
        tokenizer : BartTokenizer
        max_tokens : int

    Returns
    -------
        doc_ids : str[]
            Tokenized document.

        cls_idxs : int[]
            Position of [CLS] tokens.

        labels : bool[]
            Label for each sentence.
"""
def _parseForBART(sentences, labels, tokenizer, max_tokens=1024):
    tokenized_sentences = [[tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token] for sent in sentences]

    # Reduces the number of tokens in the document
    reduced_sentences, reduced_labels = reduceTokens(tokenized_sentences, labels, max_tokens, tokenizer)
    doc_tokens = list(itertools.chain.from_iterable(reduced_sentences))

    doc_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
    cls_idxs = [i for i, token in enumerate(doc_ids) if token == tokenizer.cls_token_id]
    return doc_ids, cls_idxs, reduced_labels



"""
    Return the utilities to parse BART

    Returns
    -------
        parseDataset : (dataset-row) -> dataset-row
            Function to use with datasets `map`

        filterDataset : (dataset) -> dataset
            Function to remove unnecessary columns from the dataset
"""
def preprocessUtilitiesBART(tokenizer):
    def parseDataset(data):
        out = {}
        doc_ids, cls_idxs, labels = _parseForBART(data["sentences"], data["labels"], tokenizer)
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