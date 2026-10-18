"""TingMix-Large mel -> CTC logits with a DYNAMIC frame count, written for ONNX export.

Same numerics as the PyTorch CPU fp32 oracle for an all-valid input (the Rust Gemma4 frontend hands the model only the valid
frames). Differences from the HF modules, all exact rewrites (see docs/research/coreml.md, which found them for Core ML):
- local attention without Tensor.unfold (two static slices of the chunked, left-padded sequence); chunk/context sizes are
  config constants, only the number of blocks depends on the input length;
- the blocked 5-D attention mask is computed in the graph from the length instead of being a trace-time constant;
- mask fill -3e4 instead of -1e9 (fp16-safe), RMSNorm on x/16 (power of two, bit-exact), ReLU as clamp(min=0), no
  pack_padded_sequence in the Reader (lengths are exact)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_VALUE = -3.0e4
RMS_PRESCALE = 1.0 / 16


def scaled_rmsnorm_forward(self, x):
    c = RMS_PRESCALE
    xs = x.float() * c
    y = xs * torch.pow(xs.pow(2).mean(-1, keepdim=True) + self.eps * c * c, -0.5)
    if self.with_scale:
        y = y * self.weight.float()
    return y.type_as(x)


class SubsampleDyn(nn.Module):
    """subsample_conv_projection for an all-valid input; ReLU as clamp (see coreml.md; harmless for ONNX)."""

    def __init__(self, tower):
        super().__init__()
        self.s = tower.subsample_conv_projection

    def _layer(self, l, h):
        return torch.clamp(l.norm(l.conv(h).permute(0, 2, 3, 1)), min=0).permute(0, 3, 1, 2)

    def forward(self, mel):
        h = self._layer(self.s.layer1, self._layer(self.s.layer0, mel.unsqueeze(1)))
        return self.s.input_proj_linear(h.permute(0, 2, 3, 1).flatten(2))


class DynAttn(nn.Module):
    """Gemma4AudioAttention (chunked local attention, batch 1) for any sequence length."""

    def __init__(self, a):
        super().__init__()
        self.a = a
        self.C, self.H, self.D = a.chunk_size, a.num_heads, a.head_dim
        assert a.context_size == 2 * self.C and a.max_future_horizon == 0 and a.max_past_horizon == self.C

    def _ctx(self, x, nb, T):  # [1,T,H,D] -> [1,nb,2C,H,D]: window b = blocks b and b+1 of the C-left-padded sequence
        C, H, D = self.C, self.H, self.D
        x = torch.cat([x.new_zeros(1, C, H, D), x, x.new_zeros(1, nb * C - T, H, D)], dim=1)
        xb = x.reshape(1, nb + 1, C, H, D)
        return torch.cat([xb[:, :-1], xb[:, 1:]], dim=2)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        a, C, H, D = self.a, self.C, self.H, self.D
        T = hidden_states.shape[1]
        nb = (T + C - 1) // C
        q = a.q_proj(hidden_states).float().view(1, T, H, D)
        k = a.k_proj(hidden_states).float().view(1, T, H, D)
        v = a.v_proj(hidden_states).float().view(1, T, H, D)
        q = q * a.q_scale * F.softplus(a.per_dim_scale)
        k = k * a.k_scale
        q = torch.cat([q, q.new_zeros(1, nb * C - T, H, D)], dim=1).reshape(1, nb, C, H, D)
        k, v = self._ctx(k, nb, T), self._ctx(v, nb, T)
        rel = a.relative_k_proj(position_embeddings).view(-1, H, D)  # [P,H,D], P = context//2 + 1
        qs = q.permute(0, 3, 1, 2, 4)  # [1,H,nb,C,D]
        ac = qs @ k.permute(0, 3, 1, 4, 2)  # [1,H,nb,C,2C]
        bd = (qs.reshape(1, H, nb * C, D) @ rel.permute(1, 2, 0)).reshape(1, H, nb, C, -1)
        ctx = a.context_size
        P = ctx // 2 + 1
        bd = F.pad(bd, (0, ctx + 1 - P)).reshape(1, H, nb, C * (ctx + 1))[..., : C * ctx].reshape(1, H, nb, C, ctx)
        w = torch.tanh((ac + bd) / a.softcap) * a.softcap
        w = w.masked_fill(attention_mask.logical_not(), MASK_VALUE)
        w = F.softmax(w, dim=-1)
        o = (w @ v.permute(0, 3, 1, 2, 4)).permute(0, 2, 3, 1, 4).reshape(1, nb * C, -1)[:, :T]
        return a.post(o), None


def blocked_mask(T, nb, C, device):
    """[1,1,nb,C,2C] bool: query bC+i sees keys bC+j-C for j in (i, i+C] that exist (the 12 keys q-11..q) (0 <= key < T). Equals the HF
    sliding-window (left 13, right 0) mask converted to blocks, for an all-valid sequence."""
    b = torch.arange(nb, device=device)[:, None, None]
    i = torch.arange(C, device=device)[None, :, None]
    j = torch.arange(2 * C, device=device)[None, None, :]
    key = b * C + j - C
    m = (j > i) & (j <= i + C) & (key >= 0) & (key < T)
    return m[None, None]


class TowerDyn(nn.Module):
    """mel [1,F,128] (all frames valid) -> features [1,T,1536], T = ceil(ceil(F/2)/2)."""

    def __init__(self, tower, rms_prescale=True):
        super().__init__()
        from tingmix.models.large.encoder import _mask_and_pos

        self.sub = SubsampleDyn(tower)
        self.layers = nn.ModuleList(tower.layers)
        for l in self.layers:
            l.self_attn.forward = DynAttn(l.self_attn).forward
            if rms_prescale:
                for m in l.modules():
                    if type(m).__name__ == "Gemma4RMSNorm":
                        m.forward = scaled_rmsnorm_forward.__get__(m)
        _, pe = _mask_and_pos(tower, torch.zeros(1, 24, tower.config.hidden_size), torch.tensor([24]))
        self.register_buffer("pe", pe.clone())  # depends on the config only
        self.out = tower.output_proj
        self.C = tower.config.attention_chunk_size

    def forward(self, mel):
        h = self.sub(mel)
        T = h.shape[1]
        nb = (T + self.C - 1) // self.C
        am = blocked_mask(T, nb, self.C, h.device)
        for l in self.layers:
            h = l(h, attention_mask=am, position_embeddings=self.pe)
        return self.out(h)


class ReaderDyn(nn.Module):
    """Reader.forward with lengths == T: the mask is all ones and pack_padded_sequence is the identity."""

    def __init__(self, r):
        super().__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        h = r.pre(x)
        for b in r.blocks:
            h = b(h, torch.ones_like(h[..., :1]))
        h, _ = r.rnn(h)
        return r.out(h)


class EncoderReaderDyn(nn.Module):
    """mel [1,F,128] -> CTC logits [1,T,V]."""

    def __init__(self, tower, reader, rms_prescale=True):
        super().__init__()
        self.tower, self.reader = TowerDyn(tower, rms_prescale), ReaderDyn(reader)

    def forward(self, mel):
        return self.reader(self.tower(mel))
