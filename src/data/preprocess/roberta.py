"""
Preprocessing for RoBERTa
"""

from datasets import Dataset
from utilities.preprocess import reduceTokens
import itertools



"""
    Parses the summary for RoBERTa.

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
def _parseForRoBERTa(sentences, labels, tokenizer, max_tokens=512):
    tokenized_sentences = [[tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token] for sent in sentences]

    # Reduces the number of tokens in the document
    reduced_sentences, reduced_labels = reduceTokens(tokenized_sentences, labels, max_tokens, tokenizer)
    doc_tokens = list(itertools.chain.from_iterable(reduced_sentences))

    doc_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
    cls_idxs = [i for i, token in enumerate(doc_ids) if token == tokenizer.cls_token_id]
    return doc_ids, cls_idxs, reduced_labels



"""
    Return the utilities to parse BERT

    Returns
    -------
        parseDataset : (dataset-row) -> dataset-row
            Function to use with datasets `map`

        filterDataset : (dataset) -> dataset
            Function to remove unnecessary columns from the dataset
"""
def preprocessUtilitiesRoBERTa(tokenizer):
    def parseDataset(data):
        out = {}
        doc_ids, cls_idxs, labels = _parseForRoBERTa(data["sentences"], data["labels"], tokenizer)
        out["__roberta_doc_ids"] = doc_ids
        out["__roberta_cls_idxs"] = cls_idxs
        out["__labels"] = labels
        return out

    def filterDataset(dataset):
        for data in dataset:
            assert len(data["__labels"]) == len(data["__roberta_cls_idxs"])
            
        dataset_content = {
            "id": dataset["id"],    
            "ref_summary": dataset["ref_summary"],    
            "labels": dataset["__labels"],
            "roberta_doc_ids": dataset["__roberta_doc_ids"],
            "roberta_cls_idxs": dataset["__roberta_cls_idxs"]
        }
        return Dataset.from_dict(dataset_content)

    return parseDataset, filterDataset