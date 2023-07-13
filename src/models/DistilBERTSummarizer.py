import torch
from models.BaseSummarizer import BaseSummarizer
from utilities.transformer import TransformerInterEncoder
from utilities.utilities import padToSize



class DistilBERTSummarizer(BaseSummarizer):
    def __init__(self, distilbert_model, distilbert_tokenizer, input_size=512):
        super().__init__(distilbert_model.name_or_path, input_size)
        self.distilbert = distilbert_model
        self.tokenizer = distilbert_tokenizer
        self.interSentenceEncoder = TransformerInterEncoder(self.distilbert.config.hidden_size)


    def forward(self, batch):
        document_ids = batch["ids"].to(self.distilbert.device)
        clss_mask = batch["clss_mask"].to(self.distilbert.device)
        attn_mask = batch["attn_mask"].to(self.distilbert.device)

        tokens_out, = self.distilbert(input_ids=document_ids, attention_mask=attn_mask, return_dict=False)
        out = []
        logits_out = []

        for i in range(len(tokens_out)): # Batch handling
            clss_out = tokens_out[i][clss_mask[i], :]
            sentences_scores, logits = self.interSentenceEncoder(clss_out)
            padding = torch.zeros(self.input_size - sentences_scores.shape[0]).to(sentences_scores.device)
            
            out.append( torch.cat((sentences_scores, padding)) )
            logits_out.append( torch.cat((logits, padding)) )

        return torch.stack(out), torch.stack(logits_out)
    

    def predictChunk(self, chunk_tokens):
        doc_ids = self.tokenizer.convert_tokens_to_ids(chunk_tokens)
        clss_mask = [True if token == self.tokenizer.cls_token_id else False for token in doc_ids]
        attn_mask = [1 for _ in range(len(doc_ids))]

        # Simulates a batch of size 1
        batch = {}
        batch["ids"] = torch.as_tensor( [padToSize(doc_ids, 512, self.tokenizer.pad_token_id)] ).to(self.distilbert.device)
        batch["clss_mask"] = torch.as_tensor( [padToSize(clss_mask, 512, False)] ).to(self.distilbert.device)
        batch["attn_mask"] = torch.as_tensor( [padToSize(attn_mask, 512, 0)] ).to(self.distilbert.device)

        self.eval()
        with torch.no_grad():
            predictions, _ = self(batch)
            return predictions[0][:chunk_tokens.count(self.tokenizer.cls_token)]