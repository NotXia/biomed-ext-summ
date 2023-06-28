import torch
import torch.nn as nn
from transformers import BertModel
from utilities.transformer import PositionalEncoding



class TransformerInterEncoder(nn.Module):
    def __init__(self, d_model, d_ff=2048, nheads=8, num_encoders=2, dropout=0.1, max_len=512):
        super().__init__()
        self.positional_enc = PositionalEncoding(d_model, dropout, max_len)
        self.encoders = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nheads, dim_feedforward=d_ff), 
            num_layers=num_encoders
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.positional_enc(x)
        x = self.encoders(x)
        x = self.layer_norm(x)
        sentences_scores = self.sigmoid(self.linear(x))

        return sentences_scores.squeeze(-1) 


class BERTSummarizer(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", input_size=512):
        super().__init__()
        self.input_size = input_size
        self.bert = BertModel.from_pretrained(bert_model)
        self.encoder = TransformerInterEncoder(self.bert.config.hidden_size)
        self.scheduler_d_model = self.bert.config.hidden_size # For Noam scheduler

    def forward(self, document_ids, segments_ids, cls_mask, bert_mask):
        tokens_out, _ = self.bert(input_ids=document_ids, token_type_ids=segments_ids, attention_mask=bert_mask, return_dict=False)
        out = []

        for i in range(len(tokens_out)): # Batch handling
            clss_out = tokens_out[i][cls_mask[i], :]
            
            sentences_scores = self.encoder(clss_out)
            padding = torch.zeros(self.input_size - sentences_scores.shape[0]).to(sentences_scores.device)
            
            out.append( torch.cat((sentences_scores, padding)) )

        return torch.stack(out)