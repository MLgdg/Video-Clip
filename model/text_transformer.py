import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from mask import PadMasking, FutureMasking
from track import RMSNorm as LayerNorm


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class FeedForward(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_model * 4)
        self.ac = QuickGELU()
        self.l2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        return self.l2(self.ac(self.l1(x)))

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model)
        self.ln_2 = LayerNorm(d_model)
        #self.attn_mask = attn_mask
    def attention(self, x, attn_mask=None):

        #self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x, attn_mask=None):
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width, layers, heads):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
        
    def forward(self, x, attn_mask=None):
        return self.resblocks(x, attn_mask=None)
