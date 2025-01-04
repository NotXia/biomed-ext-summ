import torch
from models.BaseSummarizer import BaseSummarizer
from utilities.transformer import TransformerInterEncoder
from utilities.utilities import padToSize



class BARTSummarizer(BaseSummarizer):
    def __init__(self, bart_model, bart_tokenizer, input_size=1024):
        super().__init__(bart_model.name_or_path, input_size)
        self.bart = bart_model
        self.tokenizer = bart_tokenizer
        self.interSentenceEncoder = TransformerInterEncoder(self.bart.config.hidden_size, max_len=input_size)


    def forward(self, batch):
        document_ids = batch["ids"].to(self.bart.device)
        clss_mask = batch["clss_mask"].to(self.bart.device)
        attn_mask = batch["attn_mask"].to(self.bart.device)

        if attn_mask.ndim == 3: attn_mask = torch.squeeze(attn_mask, dim=1)

        
        tokens_out = self.bart.encoder(input_ids=document_ids, attention_mask=attn_mask, return_dict=False)[0]
        
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
        batch["ids"] = torch.as_tensor( [padToSize(doc_ids, self.input_size, self.tokenizer.pad_token_id)] ).to(self.bart.device)
        batch["clss_mask"] = torch.as_tensor( [padToSize(clss_mask, self.input_size, False)] ).to(self.bart.device)
        batch["attn_mask"] = torch.as_tensor( [padToSize(attn_mask, self.input_size, 0)] ).to(self.bart.device)

        self.eval()
        with torch.no_grad():
            predictions, _ = self(batch)
            return predictions[0][:chunk_tokens.count(self.tokenizer.cls_token)]