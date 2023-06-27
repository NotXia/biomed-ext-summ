"""
Data preprocessing
"""

import argparse
import spacy
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from rouge_score.rouge_scorer import RougeScorer
from transformers import BertTokenizer
import itertools
import random

nlp = spacy.load("en_core_web_sm")



def _splitTextToSentences(text):
    return [sent.text for sent in nlp(text).sents]


"""
    Converts an abstractive summarization dataset in an extractive dataset.
    Uses a greedy approach, iterating over the sentences of the document
    and selecting the one that maximize the ROUGE score.

    Parameters
    ----------
        document : str
            Source document to summarize.

        summary : str
            Summary of the document.

        extractive_size : int
            Number of sentences to include in the summary.

    Returns
    ----------
        doc_sentences : str[]
            Sentences of the document.

        labels : bool[]         
            Label for each sentence of the document.
            (True if the sentence is part of the summary, False otherwise).
"""
def abstractiveToExtractive(document: str, summary: str, extractive_size=3):
    doc_sentences = _splitTextToSentences(document)
    selected_sentences_idx = [] # Indexes of the sentences considered as part of the summary
    max_rouge = 0.0             # Current best ROUGE score
    scorer = RougeScorer(["rouge1", "rouge2"])

    for _ in range(extractive_size):
        curr_selected_i = -1

        for i, sentence in enumerate(doc_sentences):
            if i in selected_sentences_idx: continue

            # Evaluates the ROUGE score of the summary that consideres this sentence (order is not relevant)
            tmp_summary = " ".join([doc_sentences[j] for j in selected_sentences_idx] + [sentence])
            rouge_scores = scorer.score(summary, tmp_summary)
            rouge_with_sentence = rouge_scores["rouge1"].recall + rouge_scores["rouge2"].recall

            if rouge_with_sentence > max_rouge:
                curr_selected_i = i
                max_rouge = rouge_with_sentence

        # None of the selectable sentences improves ROUGE
        if curr_selected_i < 0: break

        selected_sentences_idx.append(curr_selected_i)

    selected_sentences_idx = sorted(selected_sentences_idx)
    labels = [ (i in selected_sentences_idx) for i in range(len(doc_sentences)) ]

    return doc_sentences, labels


"""
    Return the required dataset and utilities functions to extract the relevant data.

    Returns
    -------
        dataset : Dataset
            The required dataset. 
        
        dataExtractor : (dataset row) -> str, str
            Function that extracts the column containing the document and the summary from a row of the dataset.
"""
def _getDatasetAndUtilities(name, dataset_dir):
    dataset, dataExtractor = None, None

    if dataset_dir != None: dataset = load_from_disk(dataset_dir)

    if name == "cnn_dailymail":
        if dataset_dir == None: dataset = load_dataset("cnn_dailymail", "3.0.0")
        dataExtractor = lambda dataset_row: (dataset_row["article"], dataset_row["highlights"])
    
    elif name == "ms2":
        if dataset_dir == None: dataset = load_dataset("allenai/mslr2022", "ms2")
        def extractor(dataset_row):
            for i in range(len(dataset_row["abstract"])):
                dataset_row["abstract"][i] = dataset_row["abstract"][i].strip()
                if dataset_row["abstract"][i][-1] != ".": 
                    dataset_row["abstract"][i] += "."
            return " ".join(dataset_row["abstract"]), dataset_row["target"]
        dataExtractor = extractor
    
    else:
        raise NotImplementedError(f"Dataset {name} not available")
    
    return dataset, dataExtractor


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

    # doc_tokens = tokenizer.tokenize( " [SEP] [CLS] ".join(sentences) )
    # doc_tokens = ["[CLS]"] + doc_tokens[:max_tokens-2] + ["[SEP]"]
    # reduced_labels = labels

    doc_ids = tokenizer.convert_tokens_to_ids(doc_tokens)

    # Segments with alternating 0s and 1s
    segments_ids = [0] * len(doc_ids)
    curr_segment = 0
    for i, token in enumerate(doc_ids):
        segments_ids[i] = curr_segment
        if token == tokenizer.vocab["[SEP]"]: curr_segment = 1 - curr_segment
    segments_ids = segments_ids
    
    # Position of [CLS] tokens
    cls_idxs = [i for i, token in enumerate(doc_ids) if token == tokenizer.vocab["[CLS]"]]

    return doc_ids, segments_ids, cls_idxs, reduced_labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Dataset preprocessing")
    parser.add_argument("--dataset", type=str, choices=["cnn_dailymail", "ms2"], required=True)
    parser.add_argument("--dataset-dir", type=str, help="Directory of the dataset, if not specified, it will be downloaded")
    parser.add_argument("--head", action="store_true", help="Show some rows of the parsed dataset")
    parser.add_argument("--output", type=str, help="Directory where the dataset will be exported to")
    parser.add_argument("--proc", type=int, default=1, help="Number of processes to create")
    parser.add_argument("--model", type=str, choices=["bert"], help="Model the dataset targets", required=True)
    args = parser.parse_args()
    
    random.seed(42)

    dataset, dataExtractor = _getDatasetAndUtilities(args.dataset, args.dataset_dir)


    tokenizer = None
    if args.model == "bert": tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            
    # Dataset parsing
    def parseDataset(data):
        out = {}
        document, summary = dataExtractor(data)
        sents, labels = abstractiveToExtractive(document, summary)
        out["__full_sentences"] = sents  # Sentences of the document to summarize
        out["__labels"] = labels    # Boolean associated to each sentence of the document
        out["__summary"] = summary  # Reference abstractive summary

        if args.model == "bert":
            doc_ids, segments_ids, cls_idxs, labels = parseForBERT(sents, labels, tokenizer)
            out["__bert_doc_ids"] = doc_ids
            out["__bert_segments_ids"] = segments_ids
            out["__bert_cls_idxs"] = cls_idxs
            out["__labels"] = labels

        return out
    dataset = dataset.map(parseDataset, num_proc=args.proc)

    # Recreate original split structure
    parsed_dataset = {}
    for split_name in dataset:
        dataset_content = {
            "full_sentences": dataset[split_name]["__full_sentences"],    
            "ref_summary": dataset[split_name]["__summary"],    
            "labels": dataset[split_name]["__labels"]           
        }

        if args.model == "bert":
            dataset_content["bert_doc_ids"] = dataset[split_name]["__bert_doc_ids"]
            dataset_content["bert_segments_ids"] = dataset[split_name]["__bert_segments_ids"]
            dataset_content["bert_cls_idxs"] = dataset[split_name]["__bert_cls_idxs"]

        parsed_dataset[split_name] = Dataset.from_dict(dataset_content, split=split_name)
    parsed_dataset = DatasetDict(parsed_dataset)


    # Save dataset locally
    if args.output != None:
        parsed_dataset.save_to_disk(args.output)


    # Print some of the parsed data
    if args.head:
        scorer = RougeScorer(["rouge1", "rouge2", "rougeL"])
        dataset = parsed_dataset[ list(parsed_dataset.keys())[0] ]
        
        for i in range(min(5, len(dataset))):
            sentences = []
            if args.model == "bert":
                sentences = tokenizer.decode(dataset[i]["bert_doc_ids"]).replace("[SEP]", "").split("[CLS]")
                del sentences[0]
            else:
                sentences = dataset[i]["full_sentences"]

            extractive_sum = "\n".join([ s for j, s in enumerate(sentences) if dataset[i]["labels"][j] ])
            rouge = scorer.score(dataset[i]["ref_summary"], extractive_sum)
            
            print("--- Document sentences ---")
            print("\n".join(sentences))
            print("--------------------------\n")
            print("--- Abstractive summary ---")
            print(dataset[i]["ref_summary"])
            print("---------------------------\n")
            print("--- Extractive summary ---")
            print(extractive_sum)
            print("--------------------------\n")
            print("--- ROUGE ---")
            print(f"ROUGE-1 \t P={rouge['rouge1'].precision:.4f}\t R={rouge['rouge1'].recall:.4f} \t F1={rouge['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-2 \t P={rouge['rouge2'].precision:.4f}\t R={rouge['rouge2'].recall:.4f} \t F1={rouge['rouge2'].fmeasure:.4f}")
            print(f"ROUGE-L \t P={rouge['rougeL'].precision:.4f}\t R={rouge['rougeL'].recall:.4f} \t F1={rouge['rougeL'].fmeasure:.4f}")
            print("-------------")
            print()
