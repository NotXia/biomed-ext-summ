import torch
import torch.nn as nn
import math



# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


"""
    Same scheduler as in "Attention Is All You Need"
"""
class NoamScheduler():
    def __init__(self, optimizer, warmup, model_size):
        self.epoch = 0
        self.optimizer = optimizer
        self.warmup = warmup
        self.model_size = model_size

    def step(self):
        self.epoch += 1
        new_lr = self.model_size**(-0.5) * min(self.epoch**(-0.5), self.epoch * self.warmup**(-1.5))

        for param in self.optimizer.param_groups:
            param["lr"] = new_lr
