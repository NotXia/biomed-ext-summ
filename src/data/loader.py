"""
Dataset loading utilities
"""

from datasets import load_from_disk
from transformers import BertTokenizerFast, BertTokenizer, RobertaTokenizer, RobertaTokenizerFast
from data.BERTDataset import BERTDataset
from data.RoBERTaDataset import RoBERTaDataset



"""
    Loads a dataset for a specific model.

    Parameters
    ----------
        path : str
            Path to the preprocessed dataset.

        tokenizer : Tokenizer
            Tokenizer of the model 

        splits : str[]
            Splits to parse from the source dataset. If not specified, all splits will be processed.

    Returns
    -------
        datasets : dict<str, Dataset>
            Dictionary mapping the split to the Dataset object.

"""
def loadDataset(path, tokenizer, splits=[]):
    dataset = load_from_disk(path)
    out = {}

    ModelDataset = None

    if isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, BertTokenizerFast):
        ModelDataset = BERTDataset
    elif isinstance(tokenizer, RobertaTokenizer) or isinstance(tokenizer, RobertaTokenizerFast):
        ModelDataset = RoBERTaDataset
    else:
        raise NotImplementedError(f"{tokenizer.name_or_path} not available")

    for split_name in (splits if len(splits) > 0 else dataset):
        out[split_name] = ModelDataset(dataset[split_name], tokenizer)
    return out