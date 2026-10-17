"""Conversion-friendly wrappers around the PyTorch modules (same weights, no packing/mask plumbing)."""
import torch, torch.nn as nn

class ReaderFull(nn.Module):
    """Reader.forward with lengths == T: mask is all ones, pack_padded_sequence is the identity, so it is dropped."""
    def __init__(self, r): super().__init__(); self.r = r
    def forward(self, x):
        r = self.r
        h = r.pre(x)
        for b in r.blocks:
            h = b(h, torch.ones_like(h[..., :1]))
        h, _ = r.rnn(h)
        return r.out(h)
