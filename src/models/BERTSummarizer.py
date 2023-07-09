import torch
from models.BaseSummarizer import BaseSummarizer
from utilities.transformer import TransformerInterEncoder
from utilities.utilities import padToSize
from data.BERTDataset import generateSegmentIds



class BERTSummarizer(BaseSummarizer):
    def __init__(self, bert_model, bert_tokenizer, input_size=512):
        super().__init__(bert_model.name_or_path, input_size)
        self.bert = bert_model
        self.tokenizer = bert_tokenizer
        self.encoder = TransformerInterEncoder(self.bert.config.hidden_size)


    def forward(self, batch):
        document_ids = batch["ids"].to(self.bert.device)
        segments_ids = batch["segments_ids"].to(self.bert.device)
        clss_mask = batch["clss_mask"].to(self.bert.device)
        bert_mask = batch["bert_mask"].to(self.bert.device)

        tokens_out, _ = self.bert(input_ids=document_ids, token_type_ids=segments_ids, attention_mask=bert_mask, return_dict=False)
        out = []
        logits_out = []

        for i in range(len(tokens_out)): # Batch handling
            clss_out = tokens_out[i][clss_mask[i], :]
            sentences_scores, logits = self.encoder(clss_out)
            padding = torch.zeros(self.input_size - sentences_scores.shape[0]).to(sentences_scores.device)
            
            out.append( torch.cat((sentences_scores, padding)) )
            logits_out.append( torch.cat((logits, padding)) )

        return torch.stack(out), torch.stack(logits_out)
    

    def predictChunk(self, chunk_tokens):
        doc_ids = self.tokenizer.convert_tokens_to_ids(chunk_tokens)
        segment_ids = generateSegmentIds(doc_ids, self.tokenizer)
        clss_mask = [True if token == self.tokenizer.cls_token_id else False for token in doc_ids]
        bert_mask = [1 for _ in range(len(doc_ids))]

        # Simulates a batch of size 1
        batch = {}
        batch["ids"] = torch.as_tensor( [padToSize(doc_ids, 512, self.tokenizer.pad_token_id)] ).to(self.bert.device)
        batch["segments_ids"] = torch.as_tensor( [padToSize(segment_ids, 512, 0)] ).to(self.bert.device)
        batch["clss_mask"] = torch.as_tensor( [padToSize(clss_mask, 512, False)] ).to(self.bert.device)
        batch["bert_mask"] = torch.as_tensor( [padToSize(bert_mask, 512, 0)] ).to(self.bert.device)

        self.eval()
        with torch.no_grad():
            predictions = self(batch)[0][:chunk_tokens.count(self.tokenizer.cls_token)]
            return predictions