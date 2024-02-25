import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnablePosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=31):
        super(LearnablePosEmbedding, self).__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        i = torch.arange(x.size(1), device=x.device)
        return x + self.pos_embed(i)
