"""
Dataset loading utilities
"""

import torch
from datasets import load_from_disk
from utilities.utilities import padToSize



class SummDatasetBERT(torch.utils.data.Dataset):
    def __init__(self, ext_dataset, tokenizer, input_size=512):
        self.labels = []
        self.documents = []

        for data in ext_dataset:
            labels = [1 if label else 0 for label in data["labels"]]
            ids = data["bert_doc_ids"]
            segments_ids = data["bert_segments_ids"]
            clss_mask = [True if i in data["bert_cls_idxs"] else False for i in range(input_size)]
            bert_mask = [1 for _ in range(len(data["bert_doc_ids"]))]
            
            self.labels.append( torch.tensor(padToSize(labels, input_size, 0)) ) 
            self.documents.append({
                "ids":          torch.tensor( padToSize(ids, input_size, tokenizer.vocab["[PAD]"]) ),
                "segments_ids": torch.tensor( padToSize(segments_ids, input_size, 0) ),
                "clss_mask":    torch.tensor( clss_mask ),
                "bert_mask":    torch.tensor( padToSize(bert_mask, input_size, 0) ).unsqueeze(0)
            })

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx], self.labels[idx]


"""
    Loads a dataset for a specific model.

    Parameters
    ----------
        path : str
            Path to the preprocessed dataset.

        model : bool
            Model the dataset will be created for.

        splits : str[]
            Splits to parse from the source dataset. If not specified, all splits will be processed.

        tokenizer : Tokenizer
            Tokenizer for the model (required for BERT)

    Returns
    -------
        datasets : dict<str, Dataset>
            Dictionary mapping the split to the Dataset object.

"""
def load_dataset(path, model, splits=[], tokenizer=None):
    dataset = load_from_disk(path)
    out = {}

    if model == "bert":
        if tokenizer == None: raise ValueError("Missing BERT tokenizer")
        for split_name in (splits if len(splits) > 0 else dataset):
            out[split_name] = SummDatasetBERT(dataset[split_name], tokenizer)
    else:
        raise NotImplementedError(f"Model {model} not available")

    return out