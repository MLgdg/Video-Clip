
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PadMasking(nn.Module):

    def __init__(self, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x):
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        shifted = torch.zeros(x.size()[:-1] + (1, 0,),
                              dtype=torch.bool, device=x.device)

        mask = torch.cat((shifted, is_pad), dim=-1)
        return mask.expand(x.shape + mask.shape[-1:])

class FutureMasking(nn.Module):

    def forward(self, x):
        seq_len = x.size(-1)

        # Create shifted upper triangular matrix.
        future = torch.ones((seq_len, seq_len),
                            dtype=torch.bool, device=x.device)
        future = future.triu(1)

        mask = future.view((1,) * (x.ndim - 1) + future.size())
        return mask.expand(x.shape + mask.shape[-1:])
