# Error attribution on real zh-TW recordings

Research harness (its own `uv` project; tingmix and embed2asr from git at pinned commits). It asks where TingMix-Large's
errors on real Taiwanese Mandarin come from: the acoustic model, the decoder, the LM, chunking, background sound or speaking
rate. Findings: [docs/research/error-attribution.md](../../docs/research/error-attribution.md).

Nothing derived from the recordings is committed: audio, references, hypotheses and the worst-span table stay in `out/`
(gitignored). The recordings are third-party videos; fetch them yourself:

    uvx yt-dlp -f bestaudio -o 'product.%(ext)s' <product-review video URL>
    ffmpeg -i product.webm -ar 16000 -ac 1 product.wav        # same for the finance show

## Steps

    uv sync
    uv run python -m ea.common                                       # scoring self-check
    uv run python -m ea.regions --name product --wav product.wav --vtt product.zh-TW.vtt --regions 5
    uv run python -m ea.regions --name finance --wav finance.wav --regions 6       # no subtitles: Whisper segments
    uv run python -m ea.drafts product finance                       # Whisper large-v3 draft per clip
    uv run python -m ea.review product finance && open out/review.html   # human review, then "Export"
    uv run python -m ea.gt ~/Downloads/tingmix-review.json           # verified clips -> out/NAME/gt.json
    PKG=<MLX Large package dir> MLX_PREFIX=<mlx wheel dir> ./run_all.sh

`run_all.sh` decodes every clip once (encoder + Reader, CPU fp32), runs all decoders from the saved logits, the app's live
path (`app-live/`, MLX, `SessionConfig::live_recommended()`) with several segment caps, the window-length sweep, the
counterfactual audio (Demucs vocals, 0.8x tempo, and a 10 ms shift as the null baseline), the span attribution and the
MLX-vs-PyTorch check of the live segments. Clips without a verified reference are scored
against the Whisper draft and reported as provisional; `EA_REF=subtitles` scores against the uploader's subtitles instead.

| module | output |
|---|---|
| `ea.regions` | `out/NAME/regions.json`, `clips/*.wav`, `regions/*.wav` (seed 20261019, 60 s regions, clips <= 15 s cut at quiet points) |
| `ea.decode`, `ea.decoders` | `out/NAME/logits/<variant>/*.npz`, `out/NAME/hyps_<variant>.json` |
| `ea.score` | decoder and chunking tables, `out/results.json` |
| `ea.livecaps`, `ea.winsweep` | live segment-cap and window-length tables |
| `ea.attribute` | `out/attribution.json`, `out/worst20.md` |
| `ea.liveparity` | `out/liveparity.json` (live segments re-decoded with PyTorch) |

N-best oracles pick the hypothesis closest to the reference. They bound what a better decoder could reach and are never used
to choose an output.
