"""Conversion-friendly wrappers around the PyTorch modules (same weights, no packing/mask plumbing)."""
import torch, torch.nn as nn, torch.nn.functional as F

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


class SubsampleFull(nn.Module):
    """subsample_conv_projection with an all-valid mask (mask multiplies and the [::2] mask slices are identities).
    ReLU is written as clamp(min=0): with nn.ReLU the Core ML CPU backend returns wrong fp32 results after
    LayerNorm+permute (rel L2 1.2 on layer 0; GPU/ANE and fp16 CPU are fine). See docs/research/coreml.md."""
    def __init__(self, tower):
        super().__init__(); self.s = tower.subsample_conv_projection
    def _layer(self, l, h):
        h = torch.clamp(l.norm(l.conv(h).permute(0, 2, 3, 1)), min=0)
        return h.permute(0, 3, 1, 2)
    def forward(self, mel):
        h = self._layer(self.s.layer1, self._layer(self.s.layer0, mel.unsqueeze(1)))
        return self.s.input_proj_linear(h.permute(0, 2, 3, 1).flatten(2))


class BlockFixed(nn.Module):
    """Conformer layers[i:j] for a fixed T and all-valid input: blocked local-attention mask and relative position
    embeddings are trace-time constants (they depend only on T and the config)."""
    def __init__(self, tower, i, j, T, proj=False, rms_prescale=True):
        super().__init__()
        from tingmix.models.large.encoder import _mask_and_pos
        h = torch.zeros(1, T, tower.config.hidden_size)
        am, pe = _mask_and_pos(tower, h, torch.tensor([T]))
        self.layers = nn.ModuleList(tower.layers[i:j])
        for l in self.layers:
            l.self_attn.forward = FixedAttn(l.self_attn, T).forward  # shares the original weights
            if rms_prescale:
                for m in l.modules():
                    if type(m).__name__ == "Gemma4RMSNorm":
                        m.forward = scaled_rmsnorm_forward.__get__(m)
        self.out = tower.output_proj if proj else None
        self.register_buffer("pe", pe.clone())
        self.am = None if am is None else am.clone()
        if self.am is not None:
            self.register_buffer("am_", self.am)
            self.am = None
    def forward(self, h):
        am = getattr(self, "am_", None)
        for l in self.layers:
            h = l(h, attention_mask=am, position_embeddings=self.pe)
        return self.out(h) if self.out is not None else h


RMS_PRESCALE = 1.0 / 16


def scaled_rmsnorm_forward(self, x):
    """Gemma4RMSNorm.forward, exact rewrite for fp16 graphs: activations reach ~300 (see actrange.py), so x.pow(2) exceeds the
    fp16 max (65504) on the Core ML CPU/ANE paths. With xs = c*x: x*(mean(x^2)+eps)^-0.5 == xs*(mean(xs^2)+eps*c^2)^-0.5."""
    c = RMS_PRESCALE
    xs = x.float() * c
    y = xs * torch.pow(xs.pow(2).mean(-1, keepdim=True) + self.eps * c * c, -0.5)
    if self.with_scale:
        y = y * self.weight.float()
    return y.type_as(x)


MASK_VALUE = -3.0e4  # fp16-safe; HF uses -1e9 which is -inf in fp16. exp(-3e4 - max) == 0 either way


class FixedAttn(nn.Module):
    """Gemma4AudioAttention for a fixed sequence length T, numerically the same graph with static shapes only.
    Differences from the HF code (both required by coremltools 9.0, see docs/research/coreml.md):
    - Tensor.unfold (unsupported by the torch.export frontend) is replaced by two static slices: the context stride equals
      the chunk size and the context is 2 chunks, so window b = concat(block[b], block[b+1]) of the left-padded sequence.
    - Python ints instead of tensor-derived sizes (the TorchScript frontend fails on `int()` of a traced size)."""
    def __init__(self, a, T):
        super().__init__()
        self.a, self.T = a, T
        self.C, self.H, self.D = a.chunk_size, a.num_heads, a.head_dim
        self.nb = (T + self.C - 1) // self.C
        assert a.context_size == 2 * self.C and a.max_future_horizon == 0 and a.max_past_horizon == self.C

    def _ctx(self, x):  # [1,T,H,D] -> [1,nb,2C,H,D]
        C, nb = self.C, self.nb
        x = F.pad(x, (0, 0, 0, 0, C, nb * C - self.T))  # left C zeros, right up to (nb+1)*C total
        xb = x.reshape(1, nb + 1, C, self.H, self.D)
        return torch.cat([xb[:, :-1], xb[:, 1:]], dim=2)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        a, T, C, H, D, nb = self.a, self.T, self.C, self.H, self.D, self.nb
        shape = (1, T, H, D)
        q = a.q_proj(hidden_states).float().view(shape)
        k = a.k_proj(hidden_states).float().view(shape)
        v = a.v_proj(hidden_states).float().view(shape)
        q = q * a.q_scale * F.softplus(a.per_dim_scale)
        k = k * a.k_scale
        q = F.pad(q, (0, 0, 0, 0, 0, nb * C - T)).reshape(1, nb, C, H, D)
        k, v = self._ctx(k), self._ctx(v)
        rel = a.relative_k_proj(position_embeddings).view(-1, H, D)  # [P,H,D]
        qs = q.permute(0, 3, 1, 2, 4)  # [1,H,nb,C,D]
        ac = qs @ k.permute(0, 3, 1, 4, 2)  # [1,H,nb,C,2C]
        bd = qs.reshape(1, H, -1, D) @ rel.permute(1, 2, 0)  # [1,H,nb*C,P]
        bd = bd.reshape(1, H, nb, C, -1)
        ctx = a.context_size
        P = ctx // 2 + 1  # bd.shape[-1]; a literal so the TorchScript frontend sees no traced int()
        bd = F.pad(bd, (0, ctx + 1 - P)).reshape(1, H, nb, C * (ctx + 1))[..., : C * ctx].reshape(1, H, nb, C, ctx)
        w = torch.tanh((ac + bd) / a.softcap) * a.softcap
        if attention_mask is not None:
            w = w.masked_fill(attention_mask.logical_not(), MASK_VALUE)
        w = F.softmax(w, dim=-1)
        o = (w @ v.permute(0, 3, 1, 2, 4)).permute(0, 2, 3, 1, 4).reshape(1, nb * C, -1)[:, :T]
        return a.post(o), None


class TowerFull(nn.Module):
    """mel [1,F,128] (all frames valid) -> last_hidden_state [1,T,1536]; T = ceil(ceil(F/2)/2)."""
    def __init__(self, tower, F, rms_prescale=True):
        super().__init__()
        self.sub = SubsampleFull(tower)
        self.T = (((F + 1) // 2) + 1) // 2
        self.body = BlockFixed(tower, 0, len(tower.layers), self.T, proj=True, rms_prescale=rms_prescale)
    def forward(self, mel):
        return self.body(self.sub(mel))


class EncoderReader(nn.Module):
    """mel [1,F,128] -> CTC logits [1,T,5025] (full audio tower + Reader, fixed F, all frames valid)."""
    def __init__(self, tower, reader, F):
        super().__init__()
        self.tower, self.reader = TowerFull(tower, F), ReaderFull(reader)
    def forward(self, mel):
        return self.reader(self.tower(mel))
