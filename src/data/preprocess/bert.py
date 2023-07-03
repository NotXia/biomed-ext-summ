"""
Preprocessing for BERT
"""

import argparse
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import BertTokenizer
import itertools
import random
from data.BERTDataset import generateSegmentIds



"""
    Reduces the tokens of a document by removing sentences that are not part of the summary.

    Parameters
    ----------
        sentences : str[]
            Sentences of the document.

        labels : bool[]
            Label for each sentence of the document.

        max_tokens : int
            Maximum number of tokens the document should have.

    Returns
    -------
        reduced_sentences : str[]
        reduced_labels : bool[]
"""
def reduceTokens(sentences, labels, max_tokens):
    total_tokens = sum([ len(sent) for sent in sentences ])

    while total_tokens > max_tokens:
        # Randomly selects a sentence that is not part of the summary
        removable_sentences = [i for i, selected in enumerate(labels) if not selected]
        if len(removable_sentences) == 0: break
        to_remove_sentence_idx = random.choice(removable_sentences)
        
        total_tokens -= len(sentences[to_remove_sentence_idx])
        del sentences[to_remove_sentence_idx]
        del labels[to_remove_sentence_idx]

    return sentences, labels


"""
    Parses the summary for BERT.
    Based on the paper: Text Summarization with Pretrained Encoders.

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
def parseForBERT(sentences, labels, tokenizer, max_tokens=512):
    # Each sentence has its own [CLS] (e.g. [CLS] sent1 [SEP] [CLS] sent2 [SEP] [CLS] sent3 [SEP])
    doc_tokens = ["[CLS]"] + tokenizer.tokenize( " [SEP] [CLS] ".join(sentences) ) + ["[SEP]"]
    
    # Reduces the number of tokens of the document
    tokenized_sentences = [["[CLS]"] + list(y) for x, y in itertools.groupby(doc_tokens, lambda t: t == "[CLS]") if not x]
    reduced_sentences, reduced_labels = reduceTokens(tokenized_sentences, labels, max_tokens)
    doc_tokens = list(itertools.chain.from_iterable(reduced_sentences))
    if len(doc_tokens) > max_tokens:  # In case the document can't be reduced further
        doc_tokens = doc_tokens[:max_tokens-1] + ["[SEP]"]
        reduced_labels = reduced_labels[:doc_tokens.count("[CLS]")]

    # doc_tokens = tokenizer.tokenize( " [SEP] [CLS] ".join(sentences) )
    # doc_tokens = ["[CLS]"] + doc_tokens[:max_tokens-2] + ["[SEP]"]
    # reduced_labels = labels

    doc_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
    segment_ids = generateSegmentIds(doc_ids, tokenizer)
    # Position of [CLS] tokens
    cls_idxs = [i for i, token in enumerate(doc_ids) if token == tokenizer.vocab["[CLS]"]]

    return doc_ids, segment_ids, cls_idxs, reduced_labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dataset preprocessing - Parse for BERT")
    parser.add_argument("--dataset-dir", type=str, help="Directory of the dataset (output of abs2ext.py)")
    parser.add_argument("--output", type=str, help="Directory where the dataset will be exported to")
    parser.add_argument("--proc", type=int, default=1, help="Number of processes to create")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model the dataset targets")
    args = parser.parse_args()
    
    random.seed(42)


    tokenizer = BertTokenizer.from_pretrained(args.model)
    dataset = load_from_disk(args.dataset_dir)

    # Dataset parsing
    def _parseDataset(data):
        out = {}
        doc_ids, segments_ids, cls_idxs, labels = parseForBERT(data["sentences"], data["labels"], tokenizer)
        out["__bert_doc_ids"] = doc_ids
        out["__bert_segments_ids"] = segments_ids
        out["__bert_cls_idxs"] = cls_idxs
        out["__labels"] = labels
        return out
    dataset = dataset.map(_parseDataset, num_proc=args.proc)


    # Splitting
    def _filterDataset(dataset):
        for data in dataset:
            assert len(data["__labels"]) == len(data["__bert_cls_idxs"])
            
        dataset_content = {
            "id": dataset["id"],    
            "ref_summary": dataset["ref_summary"],    
            "labels": dataset["__labels"],
            "bert_doc_ids": dataset["__bert_doc_ids"],
            "bert_segments_ids": dataset["__bert_segments_ids"],
            "bert_cls_idxs": dataset["__bert_cls_idxs"]
        }
        return Dataset.from_dict(dataset_content)
    
    parsed_dataset = {
        "train": _filterDataset(dataset["train"]),
        "test": _filterDataset(dataset["test"]),
        "validation": _filterDataset(dataset["validation"])
    }
    parsed_dataset = DatasetDict(parsed_dataset)


    # Save dataset locally
    if args.output != None:
        parsed_dataset.save_to_disk(args.output)