import torch
import torch.nn as nn
from transformers import PretrainedConfig
import math


class Sentiment140Config(PretrainedConfig):
    model_type = "sentiment140_transformer"
    def __init__(self, vocab_size=119547, d_model=256, num_heads=8, d_ff=1024, num_layers=6, num_classes=3, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

class Rotary_Positional_Encoding:
    def __init__(self, dim, device):
        assert dim % 2 == 0
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def get_position_angles(self, seq_len, device):
        positions = torch.arange(seq_len, dtype=torch.float, device=device)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        return torch.cat((freqs.sin(), freqs.cos()), dim=-1)

    def apply_rotary(self, x, seq_len=None):
        bsz, seqlen, dim = x.shape
        assert dim == self.dim
        if seq_len is None:
            seq_len = seqlen

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        freqs = self.get_position_angles(seq_len, x.device).unsqueeze(0)
        sin = freqs[..., :self.dim // 2]
        cos = freqs[..., self.dim // 2:]

        x_rotated_even = x1 * cos - x2 * sin
        x_rotated_odd = x1 * sin + x2 * cos

        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1)
        return x_rotated.flatten(-2)

class SentenceEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, device='cpu'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = Rotary_Positional_Encoding(embed_dim, device)

    def forward(self, x):
        embedded = self.embedding(x)
        return self.pos_encoding.apply_rotary(embedded, x.size(1))

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        output, _ = scaled_dot_product_attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.out_linear(output)

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.expand(-1, -1, mask.size(-1), -1)
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn

class RMSNorm_Add(nn.Module):
    def __init__(self, d_model, eps=1e-6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        residual = x + self.dropout(sublayer_output)
        rms = torch.sqrt(torch.mean(residual ** 2, dim=-1, keepdim=True) + self.eps)
        normalized = residual / rms
        output = self.scale * normalized
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_model, d_ff)
        self.linear_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = torch.sigmoid(self.linear2(x))
        x = self.linear1(x) * gate
        x = self.dropout(x)
        x = self.linear_out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = Multi_Head_Attention(d_model, num_heads)
        self.norm1 = RMSNorm_Add(d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm2 = RMSNorm_Add(d_model, dropout=dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x, attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x, ff_output)
        return x

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = SentenceEmbedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        cls_output = x[:, 0, :]  # Use [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits