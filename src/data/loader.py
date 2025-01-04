"""
Dataset loading utilities
"""

from datasets import load_from_disk
from transformers import BertTokenizerFast, BertTokenizer, \
                        RobertaTokenizer, RobertaTokenizerFast, \
                        LongformerTokenizer, LongformerTokenizerFast, \
                        MobileBertTokenizer, MobileBertTokenizerFast, \
                        DistilBertTokenizer, DistilBertTokenizerFast, \
                        BartTokenizer, BartTokenizerFast, \
                        T5Tokenizer, T5TokenizerFast
from data.BERTDataset import BERTDataset
from data.RoBERTaDataset import RoBERTaDataset
from data.LongformerDataset import LongformerDataset
from data.DistilBERTDataset import DistilBERTDataset
from data.BARTDataset import BARTDataset
from data.T5Dataset import T5Dataset



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

    if isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, BertTokenizerFast) or \
       isinstance(tokenizer, MobileBertTokenizer) or isinstance(tokenizer, MobileBertTokenizerFast):
        ModelDataset = BERTDataset
    elif isinstance(tokenizer, RobertaTokenizer) or isinstance(tokenizer, RobertaTokenizerFast):
        ModelDataset = RoBERTaDataset
    elif isinstance(tokenizer, LongformerTokenizer) or isinstance(tokenizer, LongformerTokenizerFast):
        ModelDataset = LongformerDataset
    elif isinstance(tokenizer, DistilBertTokenizer) or isinstance(tokenizer, DistilBertTokenizerFast):
        ModelDataset = DistilBERTDataset
    elif isinstance(tokenizer, BartTokenizer) or isinstance(tokenizer, BartTokenizerFast):
        ModelDataset = BARTDataset
    elif isinstance(tokenizer, T5Tokenizer) or isinstance(tokenizer, T5TokenizerFast):
        ModelDataset = T5Dataset
    else:
        raise NotImplementedError(f"{tokenizer.name_or_path} not available")

    for split_name in (splits if len(splits) > 0 else dataset):
        out[split_name] = ModelDataset(dataset[split_name], tokenizer)
    return out