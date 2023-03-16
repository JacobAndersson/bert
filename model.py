import torch.nn as nn
import dataclasses
import torch
import torch.nn.functional as F
import math

@dataclasses.dataclass
class BertConfig:
    model_dim: int = 768
    vocab_size: int = 2**15
    n_layers: int = 6
    n_heads: int = 12
    ff_dim: int = 3072
    bias: bool = True
    dropout: float = 0.1
    seq_len: int = 128

class SinusoidalPositionalEmbedding(nn.Module):
    # TODO rewrite!
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_len = self.config.seq_len

        model_dim = self.config.model_dim
            
        pos = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim))

        pe = torch.zeros(1, self.max_len, model_dim)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]

        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.k = nn.Linear(config.model_dim, config.model_dim, bias=config.bias)
        self.q = nn.Linear(config.model_dim, config.model_dim, bias=config.bias)
        self.v = nn.Linear(config.model_dim, config.model_dim, bias=config.bias)

        self.w = nn.Linear(config.model_dim, config.model_dim, bias=config.bias)

        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        self.n_heads = config.n_heads
        self.dim = config.model_dim // config.n_heads

    def forward(self, x):
        B, T, C = x.size()

        key = self.k(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        query = self.q(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        value = self.v(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)

        attention = (query @ key.transpose(-2, -1))*1/(math.sqrt(self.dim))
        attention = F.softmax(attention, dim=-1)
        attention = attention@value
        attention = self.attention_dropout(attention)
        y = attention.transpose(1,2).contiguous().view(B, T, C)

        y = self.output_dropout(self.w(y))

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.model_dim, config.ff_dim)
        self.fc2 = nn.Linear(config.ff_dim, config.model_dim)

        self.dropout1 = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlp = MLP(config)
        self.attention = MultiHeadAttention(config)

        self.norm1 = nn.LayerNorm(config.model_dim)
        self.norm2 = nn.LayerNorm(config.model_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.mlp(x))
        return x

class Bert(nn.Module):
    def __init__(self, config):
        # TODO: weight initialization

        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.model_dim
        )

        self.pos_embedding = SinusoidalPositionalEmbedding(config)

        self.encoder_layers = nn.ModuleList(
            [ EncoderLayer(config) for _ in range(config.n_layers) ]
        )

        self.norm = nn.LayerNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.vocab_size)

        self.embedding.weight = self.output.weight

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_embedding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)
        x = self.output(x)
        return x
