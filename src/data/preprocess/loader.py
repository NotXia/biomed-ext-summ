from transformers import AutoTokenizer, \
    BertTokenizer, BertTokenizerFast, \
    RobertaTokenizer, RobertaTokenizerFast, \
    LongformerTokenizer, LongformerTokenizerFast, \
    MobileBertTokenizer, MobileBertTokenizerFast, \
    DistilBertTokenizer, DistilBertTokenizerFast
from data.preprocess.bert import preprocessUtilitiesBERT
from data.preprocess.roberta import preprocessUtilitiesRoBERTa
from data.preprocess.longformer import preprocessUtilitiesLongformer



"""
    Loads the functions to preprocess a specific model.

    Parameters
    ----------
        model_name : str
            Base pretrained model (e.g. bert-base-uncased).

    Returns
    -------
        parseDataset : (dataset-row) -> dataset-row
            Function to use with datasets' `map`

        filterDataset : (dataset) -> dataset
            Function to remove unnecessary columns from the dataset

"""
def loadPreprocessUtilities(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, BertTokenizerFast) or \
       isinstance(tokenizer, MobileBertTokenizer) or isinstance(tokenizer, MobileBertTokenizerFast):
        return preprocessUtilitiesBERT(tokenizer)
    elif isinstance(tokenizer, RobertaTokenizer) or isinstance(tokenizer, RobertaTokenizerFast) or \
         isinstance(tokenizer, DistilBertTokenizer) or isinstance(tokenizer, DistilBertTokenizerFast):
        return preprocessUtilitiesRoBERTa(tokenizer)
    elif isinstance(tokenizer, LongformerTokenizer) or isinstance(tokenizer, LongformerTokenizerFast):
        return preprocessUtilitiesLongformer(tokenizer)
    else:
        raise NotImplementedError