from transformers import AutoModel, AutoTokenizer, BertModel, RobertaModel, LongformerModel, MobileBertModel, DistilBertModel, BartModel, T5Model, T5EncoderModel
from models.BERTSummarizer import BERTSummarizer
from models.RoBERTaSummarizer import RoBERTaSummarizer
from models.LongformerSummarizer import LongformerSummarizer
from models.DistilBERTSummarizer import DistilBERTSummarizer
from models.BARTSummarizer import BARTSummarizer
from models.T5Summarizer import T5Summarizer



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
    if model_name in [ "google-t5/t5-base", "razent/SciFive-base-Pubmed", "google/flan-t5-base" ]:
        model = T5EncoderModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if isinstance(model, BertModel) or isinstance(model, MobileBertModel):
        return BERTSummarizer(model, tokenizer)
    elif isinstance(model, RobertaModel):
        return RoBERTaSummarizer(model, tokenizer)
    elif isinstance(model, LongformerModel):
        return LongformerSummarizer(model, tokenizer)
    elif isinstance(model, DistilBertModel):
        return DistilBERTSummarizer(model, tokenizer)
    elif isinstance(model, BartModel):
        return BARTSummarizer(model, tokenizer)
    elif isinstance(model, T5Model):
        return T5Summarizer(model, tokenizer)
    else:
        raise NotImplementedError(f"{model_name} not available")