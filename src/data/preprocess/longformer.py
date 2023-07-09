"""
Preprocessing for Longformer
"""

from datasets import Dataset
from data.preprocess.roberta import _parseForRoBERTa



"""
    Parses the summary.

    Parameters
    ----------
        sentences : str[]
        labels : bool[]
        tokenizer : BertTokenizer
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
def _parseForLongformer(sentences, labels, tokenizer, max_tokens=4096):
    return _parseForRoBERTa(sentences, labels, tokenizer, max_tokens)



"""
    Return the utilities to parse BERT

    Returns
    -------
        parseDataset : (dataset-row) -> dataset-row
            Function to use with datasets `map`

        filterDataset : (dataset) -> dataset
            Function to remove unnecessary columns from the dataset
"""
def preprocessUtilitiesLongformer(tokenizer):
    def parseDataset(data):
        out = {}
        doc_ids, cls_idxs, labels = _parseForLongformer(data["sentences"], data["labels"], tokenizer)
        out["__longformer_doc_ids"] = doc_ids
        out["__longformer_cls_idxs"] = cls_idxs
        out["__labels"] = labels
        assert len(out["__labels"]) == len(out["__longformer_cls_idxs"])
        return out

    def filterDataset(dataset):
        dataset_content = {
            "id": dataset["id"],    
            "ref_summary": dataset["ref_summary"],    
            "labels": dataset["__labels"],
            "longformer_doc_ids": dataset["__longformer_doc_ids"],
            "longformer_cls_idxs": dataset["__longformer_cls_idxs"]
        }
        return Dataset.from_dict(dataset_content)

    return parseDataset, filterDataset