from transformers import AutoModel, AutoTokenizer, BertModel, RobertaModel
from models.BERTSummarizer import BERTSummarizer
from models.RoBERTaSummarizer import RoBERTaSummarizer



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

    if isinstance(model, BertModel):
        return BERTSummarizer(model, tokenizer)
    elif isinstance(model, RobertaModel):
        return RoBERTaSummarizer(model, tokenizer)
    else:
        raise NotImplementedError(f"{model_name} not available")