from models.BERTSummarizer import BERTSummarizer



"""
    Loads a specific model.

    Parameters
    ----------
        model_name : str
            Base pretrained model (e.g. bert-base-uncased).

        model_family : str
            Type of model (e.g. bert).

    Returns
    -------
        datasets : dict<str, Dataset>
            Dictionary mapping the split to the Dataset object.

"""
def loadModel(model_name, model_family):
    if model_family == "bert":
        return BERTSummarizer(model_name)
    else:
        raise NotImplementedError(f"{model_family} not available")