/* Narrow C ABI over the MLX C++ port of TingMix-Large (audio tower + GRU-CTC reader). Batch 1, 128-bin log-mel in. */
#ifndef TINGMIX_MLX_H
#define TINGMIX_MLX_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct tm_model tm_model;
enum { TM_FP32 = 0, TM_FP16 = 1, TM_BF16 = 2 };
#define TM_FEATURE_DIM 1536
/* Returns NULL and writes a message to err on failure. */
tm_model* tm_load(const char* safetensors_path, int dtype, char* err, size_t err_len);
void tm_free(tm_model*);
int tm_vocab(const tm_model*);
/* "<mlx version>\t<gpu|cpu>:<device name>" of the MLX default device, NUL-terminated into buf. Returns 0 on success. */
int tm_runtime_info(char* buf, size_t len);
/* Encoder frames for n_valid mel frames (two stride-2 convs). */
int tm_out_frames(int n_valid_mel);
/* mel: [n_mel, 128] f32 row-major, the first n_valid rows are real (rest padding). feats (optional): [n_out, 1536] f32;
 * logits: [n_out, vocab] f32. Returns 0 on success, nonzero with message in err. Runs on the MLX default device (GPU). */
int tm_run(tm_model*, const float* mel, int n_mel, int n_valid, float* feats, float* logits, char* err, size_t err_len);
#ifdef __cplusplus
}
#endif
#endif
