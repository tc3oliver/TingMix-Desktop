# Qwen3-ASR accuracy probe

Research harness, its own `uv` project, never shipped. It runs Qwen3-ASR 0.6B and 1.7B (the `mlx-qwen3-asr` port, fp16,
language auto, weights at pinned revisions) on the 55 error-attribution clips. The goal is to see whether a Rust backend is
worth building before building one. Results: `benchmarks/qwen3-asr/real-zh-tw-55clips-mlx.json`.

    uv sync
    uv run python eval.py ../error-attribution/out 0.6B 1.7B          # -> out/qwen_<size>_<video>.json (gitignored)
    cd ../error-attribution && uv run python -m ea.other qwen3-asr-0.6B product=../qwen3-asr/out/qwen_0.6B_product.json finance=../qwen3-asr/out/qwen_0.6B_finance.json

## Oracle for the Rust backend

    git clone https://github.com/andrewleech/qwen3-asr-onnx.git out/export-src && git -C out/export-src checkout 4aee2ae
    B=https://huggingface.co/andrewleech/qwen3-asr-0.6b-onnx/resolve/4fc24a1402e74db89c4d2ef256875e71680128c4
    mkdir -p out/onnx-0.6b && for f in encoder.int4.onnx decoder_init.int4.onnx decoder_step.int4.onnx decoder_weights.int4.data embed_tokens.bin vocab.json added_tokens.json config.json; do curl -fL -o out/onnx-0.6b/$f $B/$f; done
    uv run python ref_onnx.py out/onnx-0.6b ../../tests/fixtures/audio/*.wav     # -> out/qwen_ref.json, out/<clip>_mel.npy
    # from the repository root:
    TINGMIX_QWEN3_DIR=$PWD/experiments/qwen3-asr/out/onnx-0.6b cargo test -p tingmix-runtime-onnx --test qwen3 -- --nocapture

Findings: [docs/research/qwen3-asr.md](../../docs/research/qwen3-asr.md).
