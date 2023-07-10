import torch
from models.BaseSummarizer import BaseSummarizer
from utilities.transformer import TransformerInterEncoder
from utilities.utilities import padToSize



class LongformerSummarizer(BaseSummarizer):
    def __init__(self, longformer_model, longformer_tokenizer, input_size=4096):
        super().__init__(longformer_model.name_or_path, input_size)
        self.longformer = longformer_model
        self.tokenizer = longformer_tokenizer
        self.interSentenceEncoder = TransformerInterEncoder(self.longformer.config.hidden_size, max_len=input_size)


    def forward(self, batch):
        document_ids = batch["ids"].to(self.longformer.device)
        clss_mask = batch["clss_mask"].to(self.longformer.device)
        attn_mask = batch["attn_mask"].to(self.longformer.device)
        global_attn_mask = batch["global_attn_mask"].to(self.longformer.device)

        tokens_out, _ = self.longformer(input_ids=document_ids, attention_mask=attn_mask, global_attention_mask=global_attn_mask, return_dict=False)
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
        global_attn_mask = [1 if token == self.tokenizer.cls_token_id else 0 for token in doc_ids]

        # Simulates a batch of size 1
        batch = {}
        batch["ids"] =              torch.as_tensor( [padToSize(doc_ids, self.input_size, self.tokenizer.pad_token_id)] ).to(self.longformer.device)
        batch["clss_mask"] =        torch.as_tensor( [padToSize(clss_mask, self.input_size, False)] ).to(self.longformer.device)
        batch["attn_mask"] =        torch.as_tensor( [padToSize(attn_mask, self.input_size, 0)] ).to(self.longformer.device)
        batch["global_attn_mask"] = torch.as_tensor( [padToSize(global_attn_mask, self.input_size, 0)] ).to(self.longformer.device)

        self.eval()
        with torch.no_grad():
            predictions, _ = self(batch)
            return predictions[0][:chunk_tokens.count(self.tokenizer.cls_token)]