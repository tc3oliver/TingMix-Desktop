// Parity driver: tingmix_mlx_test <artifact.safetensors> <dtype 0|1|2> <ref_dir> <id>...  (ref_dir from dump_bin.py)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "tingmix_mlx.h"

static std::vector<float> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(2); }
  std::vector<float> v(f.tellg() / 4);
  f.seekg(0);
  f.read((char*)v.data(), v.size() * 4);
  return v;
}

int main(int argc, char** argv) {
  char err[512] = {0};
  tm_model* m = tm_load(argv[1], std::atoi(argv[2]), err, sizeof err);
  if (!m) { std::fprintf(stderr, "load: %s\n", err); return 1; }
  int V = tm_vocab(m);
  for (int i = 4; i < argc; ++i) {
    std::string id = argv[i], d = std::string(argv[3]) + "/" + id;
    auto mel = rd(d + ".mel.f32"), rf = rd(d + ".feats.f32"), rl = rd(d + ".logits.f32");
    int n_mel = mel.size() / 128, n_out = rf.size() / TM_FEATURE_DIM;
    int n_valid = (int)rd(d + ".nvalid.f32")[0];
    std::vector<float> f((size_t)n_out * TM_FEATURE_DIM), l((size_t)n_out * V);
    for (int w = 0; w < 2; ++w) {
      auto t = std::chrono::steady_clock::now();
      if (tm_run(m, mel.data(), n_mel, n_valid, f.data(), l.data(), err, sizeof err)) { std::fprintf(stderr, "run: %s\n", err); return 1; }
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
      if (w == 1) {
        double mf = 0, ml = 0;
        for (size_t k = 0; k < f.size(); ++k) mf = std::fmax(mf, std::fabs(f[k] - rf[k]));
        size_t agree = 0;
        for (size_t k = 0; k < l.size(); ++k) ml = std::fmax(ml, std::fabs(l[k] - rl[k]));
        for (int r = 0; r < n_out; ++r) {
          int a = 0, b = 0;
          for (int c = 1; c < V; ++c) { if (l[r * V + c] > l[r * V + a]) a = c; if (rl[r * V + c] > rl[r * V + b]) b = c; }
          agree += a == b;
        }
        std::printf("{\"id\":\"%s\",\"frames\":%d,\"feat_max_abs\":%.3e,\"logit_max_abs\":%.3e,\"argmax_agree\":%.4f,\"warm_run_ms\":%.1f}\n", id.c_str(), n_out, mf, ml, (double)agree / n_out, ms);
      }
    }
  }
  tm_free(m);
}
