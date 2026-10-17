"""Shared pieces: random-init SmallNet builders and the streaming step module.

PIPELINE-ONLY: no TingMix-Small checkpoint exists (HF returns 401). Weights here are random (seeded); every number
produced from them validates the export/runtime plumbing, never recognition accuracy."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tingmix.models.small.model import N_MELS, SmallNet

# Placeholder shapes. TINY is for committed-free fixtures and fast tests; PROD mirrors SmallNet's default sizes
# (d=512, hidden=640, layers=5) with a guessed vocabulary size, for size/latency measurements only.
TINY = dict(vocab_size=40, d=32, hidden=48, layers=2)
PROD = dict(vocab_size=4096, d=512, hidden=640, layers=5)


def make_net(cfg: dict, seed: int = 0) -> SmallNet:
    """Random SmallNet with peaky, input-dependent outputs (default init gives near-uniform logits whose argmax is noise)."""
    torch.manual_seed(seed)
    net = SmallNet(**cfg).eval()
    with torch.no_grad():
        net.out.weight.mul_(6.0)
        net.out.bias.normal_(0, 1.0)
        net.out.bias[0] += 1.0  # blank slightly favoured, like a trained CTC model
    return net


class StepNet(nn.Module):
    """One streaming step with explicit state, exactly equal to SmallNet.forward on the concatenated audio.

    Inputs : mel [B,K,80] raw log-mel (K multiple of 4), c1 [B,80,1], c2 [B,d,1], h [layers,B,hidden]
    Outputs: logp [B,K/4,V], c1', c2', h'
    State  : c1 = last normalized mel frame, c2 = last conv1 output frame (post-GELU); h = GRU hidden.
             All zero at the start of an utterance, which reproduces the offline left zero-padding.
    The offline pass is the same graph with zero state and K = all frames."""

    def __init__(self, net: SmallNet):
        super().__init__()
        self.net = net

    def forward(self, mel, c1, c2, h):
        n = self.net
        # Two-pass LayerNorm written out: ORT's fused LayerNormalization computes the variance as E[x^2]-E[x]^2, which
        # cancels catastrophically on near-constant frames (clamped digital silence is exactly log(1e-6) in all 80 bins)
        # and drifted logp by 0.05 against a float64 reference. torch's own LayerNorm is two-pass and fine.
        d = mel - mel.mean(-1, keepdim=True)
        x = (d / torch.sqrt((d * d).mean(-1, keepdim=True) + n.norm.eps) * n.norm.weight + n.norm.bias).transpose(1, 2)  # [B,80,K]
        y1 = F.gelu(n.conv1(torch.cat([c1, x], dim=2)))  # [B,d,K/2]
        y2 = F.gelu(n.conv2(torch.cat([c2, y1], dim=2)))  # [B,d,K/4]
        y, h2 = n.gru(y2.transpose(1, 2), h)
        return n.out(y).log_softmax(-1), x[:, :, -1:], y1[:, :, -1:], h2


def zero_state(cfg: dict, batch: int = 1):
    return (
        torch.zeros(batch, N_MELS, 1),
        torch.zeros(batch, cfg["d"], 1),
        torch.zeros(cfg["layers"], batch, cfg["hidden"]),
    )


def load_wav(path) -> "torch.Tensor":
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0)
