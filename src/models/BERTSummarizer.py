import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
from models.BaseSummarizer import BaseSummarizer
from utilities.transformer import TransformerInterEncoder
from utilities.utilities import padToSize
from utilities.summary import select
import itertools
from data.BERTDataset import generateSegmentIds



class BERTSummarizer(BaseSummarizer):
    def __init__(self, bert_model="bert-base-uncased", input_size=512):
        super().__init__(bert_model, "bert", input_size)
        self.bert = BertModel.from_pretrained(bert_model)
        self.encoder = TransformerInterEncoder(self.bert.config.hidden_size)


    def _createTokenizer(self):
        return BertTokenizer.from_pretrained(self.model_name)


    def forward(self, batch):
        document_ids = batch["ids"].to(self.bert.device)
        segments_ids = batch["segments_ids"].to(self.bert.device)
        clss_mask = batch["clss_mask"].to(self.bert.device)
        bert_mask = batch["bert_mask"].to(self.bert.device)

        tokens_out, _ = self.bert(input_ids=document_ids, token_type_ids=segments_ids, attention_mask=bert_mask, return_dict=False)
        out = []

        for i in range(len(tokens_out)): # Batch handling
            clss_out = tokens_out[i][clss_mask[i], :]
            sentences_scores = self.encoder(clss_out)
            padding = torch.zeros(self.input_size - sentences_scores.shape[0]).to(sentences_scores.device)
            
            out.append( torch.cat((sentences_scores, padding)) )

        return torch.stack(out)
    

    def summarizeFromDataset(self, predictions, doc_ids, summary_size):
        doc_ids = [id for id in doc_ids if id != self.tokenizer.vocab["[PAD]"]]
        doc_sentences = [self.tokenizer.decode(list(ids)[:-1]) 
                            for x, ids in itertools.groupby(doc_ids, lambda id: id == self.tokenizer.vocab["[CLS]"]) if not x]
        return self.summarizeSentences(doc_sentences, "count", summary_size, predictions=predictions)
    

    def predict(self, sentences):
        def _splitDocument(doc_tokens):
            splits = []
            
            while len(doc_tokens) > self.input_size:
                # Splits at the [SEP] token
                sep_idx = self.input_size - 1
                while doc_tokens[sep_idx] != "[SEP]":
                    if sep_idx <= 0: # The sentence is too long, truncate it
                        # Finds the next [CLS] (if exists)
                        next_cls_idx = None
                        i = self.input_size
                        for i in range(self.input_size, len(doc_tokens)):
                            if doc_tokens[i] == "[CLS]":
                                next_cls_idx = i
                                break

                        if next_cls_idx != None:
                            doc_tokens = doc_tokens[:self.input_size-1] + ["[SEP]"] + doc_tokens[next_cls_idx:]
                        else:
                            doc_tokens = doc_tokens[:self.input_size-1] + ["[SEP]"]
                        sep_idx = self.input_size - 1
                        break
                    sep_idx -= 1
                splits.append(doc_tokens[:sep_idx+1])
                doc_tokens = doc_tokens[sep_idx+1:]
            if len(doc_tokens) != 0: splits.append(doc_tokens) # Remaining part of the document
            
            return splits

        doc_tokens = self.tokenizer.tokenize( " [SEP] [CLS] ".join(sentences) )
        doc_tokens = ["[CLS]"] + doc_tokens + ["[SEP]"]
        # If the document is too long, it is split and processed separately. 
        # The resulting predictions are then concatenated.
        doc_splits = _splitDocument(doc_tokens)
        predictions = torch.as_tensor([]).to(self.bert.device)

        for split in doc_splits:
            doc_ids = self.tokenizer.convert_tokens_to_ids(split)

            segment_ids = generateSegmentIds(doc_ids, self.tokenizer)
            clss_mask = [True if token == self.tokenizer.vocab["[CLS]"] else False for token in doc_ids]
            bert_mask = [1 for _ in range(len(doc_ids))]

            # Simulates a batch of size 1
            batch = {}
            batch["ids"] = torch.as_tensor( [padToSize(doc_ids, 512, self.tokenizer.vocab["[PAD]"])] ).to(self.bert.device)
            batch["segments_ids"] = torch.as_tensor( [padToSize(segment_ids, 512, 0)] ).to(self.bert.device)
            batch["clss_mask"] = torch.as_tensor( [padToSize(clss_mask, 512, False)] ).to(self.bert.device)
            batch["bert_mask"] = torch.as_tensor( [padToSize(bert_mask, 512, 0)] ).to(self.bert.device)

            self.eval()
            with torch.no_grad():
                split_preds = self(batch)[0][:split.count("[CLS]")]
                predictions = torch.cat((predictions, split_preds))

        return predictions