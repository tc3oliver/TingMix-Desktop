"""TingMix-Large on MLX: mlx-vlm gemma4 AudioEncoder (1536-d last_hidden_state) + GRU-CTC Reader."""
import json

import mlx.core as mx
import mlx.nn as nn
from common import BASE_DIR, OUT
from mlx.utils import tree_flatten
from mlx_vlm.models.gemma4.audio import AudioEncoder
from mlx_vlm.models.gemma4.config import AudioConfig

ART = OUT / "tingmix-large-mlx-fp32.safetensors"


class Block(nn.Module):
    def __init__(self, d, k=5):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, padding=k // 2)
        self.norm = nn.LayerNorm(d, eps=1e-5)

    def __call__(self, x):
        return x + nn.gelu(self.norm(self.conv(x)))


class GRUDir(nn.Module):
    """One direction of a torch-layout GRU (gate order r,z,n); Wx/Wh/b/bhn as built by convert.py."""

    def __init__(self, i, h):
        super().__init__()
        self.Wx, self.Wh, self.b, self.bhn = mx.zeros((3 * h, i)), mx.zeros((3 * h, h)), mx.zeros(3 * h), mx.zeros(h)
        self.h = h

    def __call__(self, x, reverse=False):  # x [B,T,I] -> [B,T,H]
        xp = x @ self.Wx.T + self.b
        if reverse:
            xp = xp[:, ::-1]
        h = mx.zeros((x.shape[0], self.h), dtype=x.dtype)
        hs = []
        WhT = self.Wh.T
        for t in range(xp.shape[1]):
            xt, hp = xp[:, t], h @ WhT
            r = mx.sigmoid(xt[:, : self.h] + hp[:, : self.h])
            z = mx.sigmoid(xt[:, self.h : 2 * self.h] + hp[:, self.h : 2 * self.h])
            n = mx.tanh(xt[:, 2 * self.h :] + r * (hp[:, 2 * self.h :] + self.bhn))
            h = (1 - z) * n + z * h
            hs.append(h)
        out = mx.stack(hs, 1)
        return out[:, ::-1] if reverse else out


class Reader(nn.Module):
    def __init__(self, in_dim=1536, vocab=5025, d=768, gru=384, blocks=4):
        super().__init__()
        self.pre_norm, self.pre_lin = nn.LayerNorm(in_dim, eps=1e-5), nn.Linear(in_dim, d)
        self.blocks = [Block(d) for _ in range(blocks)]
        self.rnn0_f, self.rnn0_b = GRUDir(d, gru), GRUDir(d, gru)
        self.rnn1_f, self.rnn1_b = GRUDir(2 * gru, gru), GRUDir(2 * gru, gru)
        self.out = nn.Linear(2 * gru, vocab)

    def __call__(self, x):  # all frames valid (batch 1, unpadded)
        h = nn.gelu(self.pre_lin(self.pre_norm(x)))
        for b in self.blocks:
            h = b(h)
        h = mx.concatenate([self.rnn0_f(h), self.rnn0_b(h, True)], -1)
        h = mx.concatenate([self.rnn1_f(h), self.rnn1_b(h, True)], -1)
        return self.out(h)


class TingMixLarge(nn.Module):
    def __init__(self):
        super().__init__()
        ac = json.loads((BASE_DIR / "config.json").read_text())["audio_config"]
        self.audio_tower = AudioEncoder(AudioConfig.from_dict(ac))
        self.reader = Reader()

    def features(self, mel, valid=None):  # mel [B,T,128], valid [B,T] bool -> [B,T',1536] (padded frames zeroed)
        pad = mx.zeros(mel.shape[:2], dtype=mx.bool_) if valid is None else ~valid  # True = padding
        f, _ = self.audio_tower(mel, pad)
        return f

    def __call__(self, mel, valid=None, n=None):
        """Batch 1. `n` = valid output frames (default: all); the reader only sees frames [:n]."""
        f = self.features(mel, valid)
        f = f if n is None else f[:, :n]
        return f, self.reader(f.astype(mx.float32))


def load(dtype=mx.float32, path=ART):
    """dtype applies to the tower and reader matmul weights; clip bounds, norms' scale and per_dim_scale stay fp32."""
    m = TingMixLarge()
    w = mx.load(str(path))
    m.load_weights(list(w.items()), strict=True)
    if dtype != mx.float32:
        keep = ("input_min", "input_max", "output_min", "output_max", "per_dim_scale")
        m.update(tree_flatten_cast(m, dtype, keep))
    mx.eval(m.parameters())
    return m


def tree_flatten_cast(m, dtype, keep):
    from mlx.utils import tree_unflatten

    flat = [(k, v if k.endswith(keep) else v.astype(dtype)) for k, v in tree_flatten(m.parameters())]
    return tree_unflatten(flat)
