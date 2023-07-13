from data.RoBERTaDataset import RoBERTaDataset


class DistilBERTDataset(RoBERTaDataset):
    def __init__(self, ext_dataset, tokenizer, input_size=512):
        super().__init__(ext_dataset, tokenizer, input_size)
