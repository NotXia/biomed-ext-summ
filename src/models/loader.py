from transformers import AutoModel, AutoTokenizer, BertModel, RobertaModel, LongformerModel, MobileBertModel
from models.BERTSummarizer import BERTSummarizer
from models.RoBERTaSummarizer import RoBERTaSummarizer
from models.LongformerSummarizer import LongformerSummarizer



"""
    Loads a specific model.

    Parameters
    ----------
        model_name : str
            Base pretrained model (e.g. bert-base-uncased).

    Returns
    -------
        datasets : dict<str, Dataset>
            Dictionary mapping the split to the Dataset object.

"""
def loadModel(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if isinstance(model, BertModel) or isinstance(model, MobileBertModel):
        return BERTSummarizer(model, tokenizer)
    elif isinstance(model, RobertaModel):
        return RoBERTaSummarizer(model, tokenizer)
    elif isinstance(model, LongformerModel):
        return LongformerSummarizer(model, tokenizer)
    else:
        raise NotImplementedError(f"{model_name} not available")