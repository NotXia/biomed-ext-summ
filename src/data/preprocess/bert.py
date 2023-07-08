"""
Preprocessing for BERT
"""

from datasets import Dataset
from utilities.preprocess import reduceTokens
import itertools
from data.BERTDataset import generateSegmentIds



"""
    Parses the summary for BERT.

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

        segments_ids : int[]
            Position of each sentence (alternating 0s and 1s to distinguish sentences).

        cls_idxs : int[]
            Position of [CLS] tokens.

        labels : bool[]
            Label for each sentence.
"""
def _parseForBERT(sentences, labels, tokenizer, max_tokens=512):
    # Each sentence has its own [CLS] (e.g. [CLS] sent1 [SEP] [CLS] sent2 [SEP] [CLS] sent3 [SEP])
    tokenized_sentences = [[tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token] for sent in sentences]

    # Reduces the number of tokens in the document
    reduced_sentences, reduced_labels = reduceTokens(tokenized_sentences, labels, max_tokens, tokenizer)
    doc_tokens = list(itertools.chain.from_iterable(reduced_sentences))

    doc_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
    segment_ids = generateSegmentIds(doc_ids, tokenizer)
    cls_idxs = [i for i, token in enumerate(doc_ids) if token == tokenizer.vocab["[CLS]"]]

    return doc_ids, segment_ids, cls_idxs, reduced_labels



"""
    Return the utilities to parse BERT

    Returns
    -------
        parseDataset : (dataset-row) -> dataset-row
            Function to use with datasets `map`

        filterDataset : (dataset) -> dataset
            Function to remove unnecessary columns from the dataset
"""
def preprocessUtilitiesBERT(tokenizer):
    def parseDataset(data):
        out = {}
        doc_ids, segments_ids, cls_idxs, labels = _parseForBERT(data["sentences"], data["labels"], tokenizer)
        out["__bert_doc_ids"] = doc_ids
        out["__bert_segments_ids"] = segments_ids
        out["__bert_cls_idxs"] = cls_idxs
        out["__labels"] = labels
        assert len(out["__labels"]) == len(out["__bert_cls_idxs"])
        return out

    def filterDataset(dataset):
        dataset_content = {
            "id": dataset["id"],    
            "ref_summary": dataset["ref_summary"],    
            "labels": dataset["__labels"],
            "bert_doc_ids": dataset["__bert_doc_ids"],
            "bert_segments_ids": dataset["__bert_segments_ids"],
            "bert_cls_idxs": dataset["__bert_cls_idxs"]
        }
        return Dataset.from_dict(dataset_content)

    return parseDataset, filterDataset