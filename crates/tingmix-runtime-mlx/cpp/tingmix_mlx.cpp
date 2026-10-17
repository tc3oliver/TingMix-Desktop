// MLX C++ port of experiments/macos/mlx/tingmix_mlx.py (mlx-vlm gemma4 AudioEncoder + GRU-CTC Reader). Same op order as the
// Python graph so parity numbers carry over; weights come from the fp32 artifact written by convert.py.
#include "tingmix_mlx.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <optional>
#include <string>
#include <unordered_map>

#include "mlx/mlx.h"

namespace mx = mlx::core;
using mx::array;
using Shape = mx::Shape;

namespace {
constexpr int D = 1024, NH = 8, HD = 128, CHUNK = 12, PAST = 12, CTX = 24, KCONV = 5;
constexpr float SOFTCAP = 50.f, INVALID = -1e9f, CLIP = 1e10f, RES_W = 0.5f, EPS = 1e-6f;
constexpr int GRU_H = 384;

struct Model {
  std::unordered_map<std::string, array> w;
  array& operator[](const std::string& k) {
    auto it = w.find(k);
    if (it == w.end()) throw std::runtime_error("missing tensor " + k);
    return it->second;
  }
  mx::Dtype dtype = mx::float32;
  array sin_pos = array(0.f);
  array causal = array(false);

  array lin(const array& x, const std::string& n) { return mx::matmul(x, mx::transpose((*this)[n + ".weight"])); }
  array clin(const array& x, const std::string& n) {  // ClippableLinear
    auto y = mx::clip(x, std::optional<array>((*this)[n + ".input_min"]), std::optional<array>((*this)[n + ".input_max"]));
    y = lin(y, n + ".linear");
    return mx::clip(y, std::optional<array>((*this)[n + ".output_min"]), std::optional<array>((*this)[n + ".output_max"]));
  }
  array rms(const array& x, const std::string& n) { return mx::fast::rms_norm(x, std::optional<array>((*this)[n + ".weight"]), EPS); }
  static array gclip(const array& x) {
    return mx::clip(x, std::optional<array>(array(-CLIP, x.dtype())), std::optional<array>(array(CLIP, x.dtype())));
  }

  array ffw(const array& x0, const std::string& p) {
    auto x = rms(gclip(x0), p + ".pre_layer_norm");
    x = clin(x, p + ".ffw_layer_1");
    x = mx::multiply(x, mx::sigmoid(x));  // silu
    x = clin(x, p + ".ffw_layer_2");
    x = rms(gclip(x), p + ".post_layer_norm");
    return mx::add(x0, mx::multiply(x, array(RES_W)));
  }

  array pad1(const array& x, int lo, int hi) {  // pad axis 1
    std::vector<int> axes{1};
    return mx::pad(x, axes, Shape{lo}, Shape{hi}, array(0, x.dtype()));
  }
  array block_ctx(const array& x) {  // [B,T,...] -> [B,U,CTX,...]
    auto p = pad1(x, PAST, CHUNK - 1);
    int U = (p.shape(1) - CTX) / CHUNK + 1;
    auto idx = mx::add(mx::reshape(mx::multiply(mx::arange(U), array(CHUNK)), {U, 1}), mx::reshape(mx::arange(CTX), {1, CTX}));
    return mx::take(p, idx, 1);
  }

  array attn(const array& x, const array& pad, const std::string& p) {
    int B = x.shape(0), T = x.shape(1);
    Shape qs{B, T, NH, HD};
    auto q = mx::reshape(mx::astype(clin(x, p + ".q_proj"), mx::float32), qs);
    auto k = mx::reshape(mx::astype(clin(x, p + ".k_proj"), mx::float32), qs);
    auto v = mx::reshape(mx::astype(clin(x, p + ".v_proj"), mx::float32), qs);
    auto pds = mx::logaddexp((*this)[p + ".per_dim_scale"], array(0.f));  // softplus
    q = mx::multiply(q, mx::multiply(array((float)((1.0 / std::sqrt((double)HD)) / std::log(2.0))), pds));
    k = mx::multiply(k, array((float)(std::log(1.0 + std::exp(1.0)) / std::log(2.0))));
    int U = (T + CHUNK - 1) / CHUNK;
    auto qb = mx::reshape(pad1(q, 0, U * CHUNK - T), {B, U, CHUNK, NH, HD});
    auto kb = block_ctx(k), vb = block_ctx(v);
    auto valid = block_ctx(mx::logical_not(pad));  // [B,U,CTX]
    auto cond = mx::logical_and(mx::reshape(valid, {B, 1, U, 1, CTX}), mx::reshape(causal, {1, 1, 1, CHUNK, CTX}));
    // relative position term
    auto pe = mx::matmul(sin_pos, mx::transpose((*this)[p + ".relative_k_proj.weight"]));  // [13, N*H]
    pe = mx::reshape(pe, {PAST + 1, NH, HD});
    auto qp = mx::transpose(qb, {0, 3, 1, 2, 4});                     // [B,N,U,W,H]
    auto ac = mx::matmul(qp, mx::transpose(kb, {0, 3, 1, 4, 2}));      // [B,N,U,W,C]
    auto bd = mx::matmul(mx::reshape(qp, {B, NH, U * CHUNK, HD}), mx::transpose(pe, {1, 2, 0}));  // [B,N,UW,13]
    bd = mx::reshape(bd, {B, NH, U, CHUNK, PAST + 1});
    bd = mx::pad(bd, std::vector<int>{4}, Shape{0}, Shape{CTX + 1 - (PAST + 1)}, array(0.f));
    bd = mx::reshape(bd, {B, NH, U, CHUNK * (CTX + 1)});
    bd = mx::slice(bd, {0, 0, 0, 0}, {B, NH, U, CHUNK * CTX});
    bd = mx::reshape(bd, {B, NH, U, CHUNK, CTX});
    auto lg = mx::add(ac, bd);
    lg = mx::multiply(mx::tanh(mx::divide(lg, array(SOFTCAP))), array(SOFTCAP));
    lg = mx::where(cond, lg, array(INVALID));
    auto pr = mx::softmax(lg, std::vector<int>{-1});
    auto ctx = mx::matmul(pr, mx::transpose(vb, {0, 3, 1, 2, 4}));  // [B,N,U,W,H]
    ctx = mx::reshape(mx::transpose(ctx, {0, 2, 3, 1, 4}), {B, U * CHUNK, NH, HD});
    ctx = mx::reshape(mx::slice(ctx, {0, 0, 0, 0}, {B, T, NH, HD}), {B, T, D});
    return clin(ctx, p + ".post");
  }

  array lconv(const array& x, const std::string& p) {
    auto h = rms(x, p + ".pre_layer_norm");
    h = clin(h, p + ".linear_start");
    auto parts = mx::split(h, 2, -1);
    h = mx::multiply(parts[0], mx::sigmoid(parts[1]));
    std::vector<int> one{1}, lo{KCONV - 1}, hi{0};
    h = mx::conv_general(h, (*this)[p + ".depthwise_conv1d.weight"], one, lo, hi, one, one, D, false);
    h = rms(gclip(h), p + ".conv_norm");
    h = mx::multiply(h, mx::sigmoid(h));
    h = clin(h, p + ".linear_end");
    return mx::add(h, x);
  }

  array conformer(const array& x0, const array& pad, int i) {
    std::string p = "audio_tower.layers." + std::to_string(i);
    auto x = ffw(x0, p + ".feed_forward1");
    auto r = x;
    x = rms(gclip(x), p + ".norm_pre_attn");
    x = attn(x, pad, p + ".self_attn");
    x = mx::add(r, rms(gclip(x), p + ".norm_post_attn"));
    x = mx::multiply(x, mx::astype(mx::expand_dims(mx::logical_not(pad), 2), x.dtype()));
    x = lconv(x, p + ".lconv1d");
    x = ffw(x, p + ".feed_forward2");
    return rms(gclip(x), p + ".norm_out");
  }

  array sscp(const array& mel, array& pad) {  // mel [B,T,128]
    auto x = mx::expand_dims(mel, -1);
    for (int l = 0; l < 2; ++l) {
      std::string p = "audio_tower.subsample_conv_projection.layer" + std::to_string(l);
      x = mx::where(mx::reshape(pad, {pad.shape(0), pad.shape(1), 1, 1}), array(0.f, x.dtype()), x);
      x = mx::pad(x, std::vector<int>{1, 2}, Shape{1, 1}, Shape{1, 1}, array(0, x.dtype()));
      x = mx::conv2d(x, (*this)[p + ".conv.weight"], {2, 2}, {0, 0});
      int t = x.shape(1);
      pad = mx::slice(pad, {0, 0}, {pad.shape(0), pad.shape(1)}, {1, 2});
      pad = mx::slice(pad, {0, 0}, {pad.shape(0), t});
      x = mx::fast::layer_norm(x, std::optional<array>((*this)[p + ".norm.weight"]), std::nullopt, EPS);
      x = mx::maximum(x, array(0.f, x.dtype()));
    }
    x = mx::reshape(x, {x.shape(0), x.shape(1), x.shape(2) * x.shape(3)});
    return lin(x, "audio_tower.subsample_conv_projection.input_proj_linear");
  }

  array gru_dir(const array& x, const std::string& p, bool rev) {
    auto xp = mx::add(mx::matmul(x, mx::transpose((*this)[p + ".Wx"])), (*this)[p + ".b"]);
    if (rev) xp = mx::take(xp, mx::arange(xp.shape(1) - 1, -1, -1), 1);
    auto WhT = mx::transpose((*this)[p + ".Wh"]);
    auto bhn = (*this)[p + ".bhn"];
    auto h = mx::zeros({x.shape(0), GRU_H}, x.dtype());
    std::vector<array> hs;
    int T = xp.shape(1), H = GRU_H, B = x.shape(0);
    for (int t = 0; t < T; ++t) {
      auto xt = mx::reshape(mx::slice(xp, {0, t, 0}, {B, t + 1, 3 * H}), {B, 3 * H});
      auto hp = mx::matmul(h, WhT);
      auto r = mx::sigmoid(mx::add(mx::slice(xt, {0, 0}, {B, H}), mx::slice(hp, {0, 0}, {B, H})));
      auto z = mx::sigmoid(mx::add(mx::slice(xt, {0, H}, {B, 2 * H}), mx::slice(hp, {0, H}, {B, 2 * H})));
      auto n = mx::tanh(mx::add(mx::slice(xt, {0, 2 * H}, {B, 3 * H}), mx::multiply(r, mx::add(mx::slice(hp, {0, 2 * H}, {B, 3 * H}), bhn))));
      h = mx::add(mx::multiply(mx::subtract(array(1.f, h.dtype()), z), n), mx::multiply(z, h));
      hs.push_back(h);
    }
    auto out = mx::stack(hs, 1);
    if (rev) out = mx::take(out, mx::arange(T - 1, -1, -1), 1);
    return out;
  }

  array reader(const array& f) {
    auto h = mx::fast::layer_norm(f, std::optional<array>((*this)["reader.pre_norm.weight"]), std::optional<array>((*this)["reader.pre_norm.bias"]), 1e-5f);
    h = mx::add(lin(h, "reader.pre_lin"), (*this)["reader.pre_lin.bias"]);
    auto gelu = [](const array& a) { return mx::multiply(mx::multiply(a, array(0.5f)), mx::add(array(1.f), mx::erf(mx::multiply(a, array(0.70710678118f))))); };
    h = gelu(h);
    for (int i = 0; i < 4; ++i) {
      std::string p = "reader.blocks." + std::to_string(i);
      auto c = mx::add(mx::conv1d(h, (*this)[p + ".conv.weight"], 1, 2), (*this)[p + ".conv.bias"]);
      c = mx::fast::layer_norm(c, std::optional<array>((*this)[p + ".norm.weight"]), std::optional<array>((*this)[p + ".norm.bias"]), 1e-5f);
      h = mx::add(h, gelu(c));
    }
    for (int l = 0; l < 2; ++l) {
      std::string p = "reader.rnn" + std::to_string(l);
      h = mx::concatenate({gru_dir(h, p + "_f", false), gru_dir(h, p + "_b", true)}, -1);
    }
    return mx::add(lin(h, "reader.out"), (*this)["reader.out.bias"]);
  }

  array tower(const array& mel, array pad) {
    auto x = sscp(mel, pad);
    for (int i = 0; i < 12; ++i) x = conformer(x, pad, i);
    x = mx::add(lin(x, "audio_tower.output_proj"), (*this)["audio_tower.output_proj.bias"]);
    return mx::where(mx::expand_dims(pad, 2), array(0.f, x.dtype()), x);
  }

  void init() {
    // sinusoidal timing signal for relative positions 12..0 (fp32, computed once)
    int nt = D / 2;
    std::vector<float> sig((PAST + 1) * D);
    double inc = std::log(10000.0) / (nt - 1);
    for (int i = 0; i <= PAST; ++i) {
      double pos = PAST - i;
      for (int j = 0; j < nt; ++j) {
        double a = pos * std::exp(-inc * j);
        sig[i * D + j] = (float)std::sin(a);
        sig[i * D + nt + j] = (float)std::cos(a);
      }
    }
    sin_pos = array(sig.data(), {PAST + 1, D}, mx::float32);
    std::vector<uint8_t> m(CHUNK * CTX);
    for (int q = 0; q < CHUNK; ++q)
      for (int c = 0; c < CTX; ++c) {
        int dist = q + PAST - c;
        m[q * CTX + c] = (dist >= 0 && dist < PAST) ? 1 : 0;  // right context is 0: no future keys
      }
    causal = mx::astype(array(m.data(), {CHUNK, CTX}, mx::uint8), mx::bool_);
  }
};

void put(char* err, size_t n, const std::string& m) {
  if (err && n) std::snprintf(err, n, "%s", m.c_str());
}
}  // namespace

struct tm_model {
  Model m;
};

extern "C" {
tm_model* tm_load(const char* path, int dtype, char* err, size_t err_len) {
  try {
    auto* h = new tm_model();
    auto loaded = mx::load_safetensors(path);
    mx::Dtype dt = dtype == TM_FP16 ? mx::float16 : dtype == TM_BF16 ? mx::bfloat16 : mx::float32;
    h->m.dtype = dt;
    for (auto& [k, v] : loaded.first) {
      bool keep32 = k.size() > 4 && (k.find("_min") != std::string::npos || k.find("_max") != std::string::npos || k.find("per_dim_scale") != std::string::npos);
      h->m.w.insert_or_assign(k, keep32 ? v : mx::astype(v, dt));
    }
    h->m.init();
    std::vector<array> all;
    for (auto& kv : h->m.w) all.push_back(kv.second);
    mx::eval(all);
    return h;
  } catch (const std::exception& e) {
    put(err, err_len, e.what());
    return nullptr;
  }
}
void tm_free(tm_model* h) { delete h; }
int tm_vocab(const tm_model* h) { return const_cast<tm_model*>(h)->m["reader.out.bias"].shape(0); }
int tm_out_frames(int n) { return ((n + 1) / 2 + 1) / 2; }

int tm_run(tm_model* h, const float* mel, int n_mel, int n_valid, float* feats, float* logits, char* err, size_t err_len) {
  try {
    if (n_valid <= 0 || n_valid > n_mel) throw std::runtime_error("bad n_valid");
    auto& m = h->m;
    array a(mel, {1, n_mel, 128}, mx::float32);
    array pad = mx::reshape(mx::greater_equal(mx::arange(n_mel), array(n_valid)), {1, n_mel});
    auto f = m.tower(a, pad);
    int n_out = tm_out_frames(n_valid);
    f = mx::slice(f, {0, 0, 0}, {1, n_out, TM_FEATURE_DIM});
    auto lg = m.reader(mx::astype(f, mx::float32));
    lg = mx::astype(lg, mx::float32);
    auto ff = mx::astype(f, mx::float32);
    mx::eval({lg, ff});
    std::memcpy(logits, lg.data<float>(), sizeof(float) * (size_t)n_out * lg.shape(2));
    if (feats) std::memcpy(feats, ff.data<float>(), sizeof(float) * (size_t)n_out * TM_FEATURE_DIM);
    return 0;
  } catch (const std::exception& e) {
    put(err, err_len, e.what());
    return 1;
  }
}
}
