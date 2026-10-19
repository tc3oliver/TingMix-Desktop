# SenseVoice-Small oracle

Research harness, its own `uv` project and never shipped. It runs sherpa-onnx 1.13.8 (the reference implementation of the
SenseVoice ONNX export) and kaldi-native-fbank 1.22.3 on the clips and writes the oracle that
`crates/tingmix-runtime-onnx/tests/sensevoice.rs` compares against. Findings: [docs/research/sensevoice.md](../../docs/research/sensevoice.md).

    # the export (239 MB), into out/model (gitignored)
    B=https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/2365baeacb507f821a0c8120fcee3d484dba7a07
    mkdir -p out/model/test_wavs && for f in model.int8.onnx tokens.txt test_wavs/zh.wav test_wavs/en.wav test_wavs/yue.wav; do curl -fL -o out/model/$f $B/$f; done

    uv sync
    uv run python ref.py out/model out/model/test_wavs/*.wav ../../tests/fixtures/audio/*.wav   # -> out/reference.json, out/*_lfr.npy

    # the Rust backend against it (from the repository root)
    TINGMIX_SENSEVOICE_DIR=$PWD/experiments/sensevoice/out/model cargo test -p tingmix-runtime-onnx --test sensevoice -- --nocapture

`out/reference.json` holds per clip the sherpa-onnx text, the detected language, the LFR frame count and the first three LFR+CMVN
frames. `out/<clip>_lfr.npy` holds every frame.
