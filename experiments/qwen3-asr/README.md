# Qwen3-ASR accuracy probe

Research harness, its own `uv` project, never shipped. It runs Qwen3-ASR 0.6B and 1.7B (the `mlx-qwen3-asr` port, fp16,
language auto, weights at pinned revisions) on the 55 error-attribution clips. The goal is to see whether a Rust backend is
worth building before building one. Results: `benchmarks/qwen3-asr/real-zh-tw-55clips-mlx.json`.

    uv sync
    uv run python eval.py ../error-attribution/out 0.6B 1.7B          # -> out/qwen_<size>_<video>.json (gitignored)
    cd ../error-attribution && uv run python -m ea.other qwen3-asr-0.6B product=../qwen3-asr/out/qwen_0.6B_product.json finance=../qwen3-asr/out/qwen_0.6B_finance.json
