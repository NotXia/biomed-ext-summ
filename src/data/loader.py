"""
Dataset loading utilities
"""

from datasets import load_from_disk
from data.BERTDataset import BERTDataset



"""
    Loads a dataset for a specific model.

    Parameters
    ----------
        path : str
            Path to the preprocessed dataset.

        model_family : str
            Type of model (e.g. bert).

        tokenizer : Tokenizer
            Tokenizer of the model 

        splits : str[]
            Splits to parse from the source dataset. If not specified, all splits will be processed.

    Returns
    -------
        datasets : dict<str, Dataset>
            Dictionary mapping the split to the Dataset object.

"""
def loadDataset(path, model_family, tokenizer, splits=[]):
    dataset = load_from_disk(path)
    out = {}

    if model_family == "bert":
        for split_name in (splits if len(splits) > 0 else dataset):
            out[split_name] = BERTDataset(dataset[split_name], tokenizer)
    else:
        raise NotImplementedError(f"{model_family} not available")

    return out