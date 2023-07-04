"""
Abstractive to extractive conversion.
"""

import spacy
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from rouge_score.rouge_scorer import RougeScorer

doc2sentences = spacy.load("en_core_web_sm")



"""
    Converts an abstractive summarization dataset into an extractive dataset.
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
            The maximum between this value and the number of sentences in the abstractive summary will be considered.

    Returns
    ----------
        doc_sentences : str[]
            Sentences of the document.

        labels : bool[]         
            Label for each sentence of the document.
            (True if the sentence is part of the summary, False otherwise).
"""
def _abstractiveToExtractive(document: str, summary: str, extractive_size=3):
    doc_sentences = [sent.text for sent in doc2sentences(document).sents]
    summ_sentences = [sent.text for sent in doc2sentences(summary).sents]
    selected_sentences_idx = [] # Indexes of the sentences considered as part of the summary
    max_rouge = 0.0             # Current best ROUGE score
    scorer = RougeScorer(["rouge1", "rouge2"])
    extractive_size = max( extractive_size, len(summ_sentences) )

    for _ in range(extractive_size):
        curr_selected_i = -1

        for i, sentence in enumerate(doc_sentences):
            if i in selected_sentences_idx: continue

            # Evaluates the ROUGE score of the summary that consideres this sentence (order is not relevant)
            tmp_summary = " ".join([doc_sentences[j] for j in selected_sentences_idx] + [sentence])
            rouge_scores = scorer.score(summary, tmp_summary)
            rouge_with_sentence = 0.1*rouge_scores["rouge1"].recall + 0.9*rouge_scores["rouge2"].recall

            if rouge_with_sentence > max_rouge+1e-3: # To prevent adding sentences that may be just redundant
                curr_selected_i = i
                max_rouge = rouge_with_sentence

        # None of the selectable sentences improves ROUGE
        if curr_selected_i < 0: break

        selected_sentences_idx.append(curr_selected_i)

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
    
    elif name == "ms2" or name == "cochrane":
        if dataset_dir == None: dataset = load_dataset("allenai/mslr2022", name)
        def extractor(dataset_row):
            for i in range(len(dataset_row["abstract"])):
                dataset_row["abstract"][i] = dataset_row["abstract"][i].strip()
                if dataset_row["abstract"][i][-1] != ".": 
                    dataset_row["abstract"][i] += "."
            return " ".join(dataset_row["abstract"]), dataset_row["target"]
        dataExtractor = extractor
    
    elif name == "sumpubmed":
        if dataset_dir == None: dataset = load_dataset("Blaise-g/SumPubmed")
        dataExtractor = lambda dataset_row: (dataset_row["text"], dataset_row["abstract"])

    else:
        raise NotImplementedError(f"Dataset {name} not available")
    
    return dataset, dataExtractor


"""
    Converts an abstractive summarization dataset into an extractive dataset.

    Parameters
    ----------
        dataset_name : str
            Name of the dataset to process.

        dataset_dir : str|None
            If set, the dataset will be loaded from the filesystem. Otherwise it will be downlaoded from Hugging Face.

        selection_size : int
            Number of sentences to include in the summary.
            The maximum between this value and the number of sentences in the abstractive summary will be considered.

        num_proc : int
            Number of processes to use.

    Returns
    ----------
        dataset : DatasetDict
            Dataset with train, test and validation splits.
            Note: it keeps the original splits of the dataset, renamed if needed.
"""
def parseAbs2Ext(dataset_name, dataset_dir, selection_size, num_proc):
    dataset, dataExtractor = _getDatasetAndUtilities(dataset_name, dataset_dir)
            
    # Dataset parsing
    def _parseDataset(data):
        document, summary = dataExtractor(data)
        sents, labels = _abstractiveToExtractive(document, summary, extractive_size=selection_size)
        return {
            "__sentences": sents,       # Sentences of the document to summarize
            "__labels": labels,         # Boolean associated to each sentence of the document
            "__summary": summary        # Reference abstractive summary
        }           
    dataset = dataset.map(_parseDataset, num_proc=num_proc)


    # Splitting
    def _filterDataset(dataset):
        dataset_content = {
            "id": [i for i in range(len(dataset["__sentences"]))],
            "sentences": dataset["__sentences"],    
            "ref_summary": dataset["__summary"],    
            "labels": dataset["__labels"]           
        }
        return Dataset.from_dict(dataset_content)
    
    parsed_dataset = {}
    if dataset_name in ["cnn_dailymail", "ms2", "cochrane"]:
        parsed_dataset["train"] = _filterDataset(dataset["train"])
        parsed_dataset["test"] = _filterDataset(dataset["test"])
        parsed_dataset["validation"] = _filterDataset(dataset["validation"])
    elif dataset_name in ["sumpubmed"]:
        parsed_dataset["train"] = _filterDataset(dataset["train"])
        parsed_dataset["test"] = _filterDataset(dataset["test"])
        parsed_dataset["validation"] = _filterDataset(dataset["dev"])
    else:
        raise NotImplementedError
    parsed_dataset = DatasetDict(parsed_dataset)

    return parsed_dataset