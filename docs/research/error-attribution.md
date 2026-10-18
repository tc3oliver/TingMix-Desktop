# Error attribution on real zh-TW speech (2026-10-19)

Status: **PROVISIONAL**. The analysis code is final; the reference transcripts are not yet human-verified. Until they are,
clips are scored against a Whisper large-v3 draft made from each clip's own audio. Section 6 repeats the product video
against the uploader's subtitles, an independent reference, and the conclusions hold. Section 8 lists what verification
can still change.

Question: why is TingMix-Large far from usable on real Taiwanese recordings (CER about 32% on a product-review video against
its subtitles), and why does v3's attention rescoring, which helps on the public benchmarks, barely change that?

Short answer: **most errors are not recoverable by any decoder, LM, background or speaking-rate change that was tested.**
83% of the error units are spans that no test recovers: none of the LM beam's 32 best is right, greedy CTC is wrong too, and
vocal separation or slowing the speech fixes them no more often than an inaudible 10 ms shift does. This residual holds the
acoustic model's errors and also any errors in the provisional reference. A perfect choice among the 32 best would remove 2.6
of 21.2 points, and v3 removes 0.6. The largest non-model effect is in the app: the live path is 3.2 points worse than
reference-boundary clips, mostly through deletions, and a 6 s segment cap recovers most of that in both videos.

Harness, commands and module list: [experiments/error-attribution/](../../experiments/error-attribution/README.md).

## 1. Data and references

| | product review | finance talk show |
|---|---|---|
| recording | 527 s, casual zh-TW, background music, English product names | 1988 s, fast broadcast speech over a music bed, English company names |
| sampled | 5 regions, 25 clips, 292 s | 6 regions, 30 clips, 354 s |
| clip boundaries | uploader subtitle cues | Whisper large-v3 segments (the video has no subtitles) |
| reference | Whisper draft (provisional); subtitles for the sensitivity check | Whisper draft (provisional) |

Regions are 60 s, drawn without overlap with seed 20261019, and start and end on cue boundaries. Clips merge cues up to 15 s
and are cut at the quietest 20 ms frame within 0.3 s of the boundary. In total that is 647 s in 55 clips, with 3711 reference
units (3626 CJK characters, 85 English words: 24 in the product video, 61 in the finance show). The subtitles are edited, not
verbatim: they drop fillers that are audible and that Whisper keeps, so they are not the main reference.

Scoring: OpenCC s2t on both sides, digits to spoken Mandarin (embed2asr `numnorm`), punctuation to spaces, then embed2asr
`norm` (NFKC, lowercase, no spaces next to CJK). Units are CJK characters plus English words (MER). Four reference
characters (variant forms the s2t step leaves alone) are not in the 5025-unit vocabulary; vocabulary coverage is not a factor.

## 2. Decoders on the same audio and slices (55 clips)

Encoder and Reader run once per clip (PyTorch CPU fp32; v2 and v3 share these weights), and every decoder reads the same
logits. Oracle rows choose, per clip, the hypothesis closest to the reference among the N best. They bound what a better
decoder or rescorer could reach. They are not inference results.

| decoder | MER | CER zh | WER en | S | D | I |
|---|---|---|---|---|---|---|
| v2 greedy CTC | 0.214 | 0.205 | 0.600 | 0.115 | 0.094 | 0.006 |
| v2 beam 16 + char LM (the app) | 0.212 | 0.203 | 0.600 | 0.094 | 0.111 | 0.006 |
| v3 attention rescoring of the 8-best | 0.206 | 0.197 | 0.588 | 0.094 | 0.105 | 0.006 |
| oracle, LM beam 8 / 16 / 32 best | 0.196 / 0.193 / 0.186 | 0.187 / 0.184 / 0.177 | 0.576 | 0.085-0.089 | 0.097-0.101 | 0.005-0.006 |
| oracle, plain CTC beam 8 / 16 / 32 best | 0.198 / 0.193 / 0.190 | 0.189 / 0.185 / 0.181 | 0.565-0.588 | 0.104-0.108 | 0.082-0.085 | 0.004-0.005 |

- By video (app decoder): product MER 0.273 (CER 0.264, WER 0.917 on 24 English words), finance 0.158 (CER 0.148, WER 0.475 on 61).
- By clip language: Mandarin-only clips 0.191 (oracle-32 0.159), code-switched clips 0.226 (oracle-32 0.204).
- Deletions are the largest error type. The LM trades substitutions for deletions (S 0.115 to 0.094, D 0.094 to 0.111), and
  its net effect is -0.2 points.
- v3: -0.6 points (0.212 to 0.206). No significance test was run, and the sample is two videos.
- The decoder ceiling is low: a perfect pick from the LM beam's 32 best reaches 0.186, which is 2.6 points better than the app.

## 3. Chunking (11 regions, same audio)

| chunking (LM decoder) | MER | D | S |
|---|---|---|---|
| reference-boundary clips, joined | 0.212 | 0.111 | 0.094 |
| fixed 25 s windows | 0.236 | 0.156 | 0.075 |
| 25 s windows overlapping by 5 s, central frames kept | 0.221 | 0.133 | 0.082 |
| app live path (adaptive VAD, quiet-point cut, gap bridge; MLX), segment cap 25 s | 0.244 | 0.150 | 0.087 |

The reference drafts were made on the reference-boundary clips. That row may therefore be favoured, and the 3.2-point gap to
the live path is measured under that bias. All 41 live segments, re-decoded with the PyTorch reference, give identical
text (`ea.liveparity`), so the MLX backend is not a factor.

App live path with other segment caps (only this setting changed):

| cap (s) | 6 | 8 | 10 | 12 | 15 | 20 | 25 (today) |
|---|---|---|---|---|---|---|---|
| MER, both videos | 0.209 | 0.221 | 0.222 | 0.223 | 0.225 | 0.230 | 0.244 |
| MER, product | 0.272 | 0.289 | 0.289 | 0.284 | 0.279 | 0.301 | 0.290 |
| MER, finance | 0.153 | 0.161 | 0.163 | 0.168 | 0.178 | 0.167 | 0.203 |
| deletions, both videos | 0.088 | 0.106 | 0.111 | 0.121 | 0.125 | 0.136 | 0.150 |
| segments, both videos | 145 | 104 | 84 | 73 | 59 | 47 | 41 |

- Within each video the caps between 8 and 20 s do not follow a clean trend. Only two findings hold in both videos: 6 s is
  the best cap, and deletions are lowest at 6 s (product 0.114, finance 0.064) and highest at 20-25 s (0.163-0.182 and
  0.095-0.138). Per-video rows are in `out/livecaps.json`.
- At 6 s the live path scores 0.209, slightly below the reference-boundary clips (0.212). The cause is not known. One
  possibility is that shorter segments help the model on their own, which is consistent with the deletion trend.
- Fixed windows cut without regard to pauses (2 s overlap, central frames kept; `ea.winsweep`), LM decoder: MER 0.212 / 0.219 /
  0.230 / 0.248 / 0.229 / 0.233 / 0.235 / 0.229 at 4 / 6 / 8 / 10 / 12 / 15 / 20 / 25 s, deletions 0.109 at 4 s and
  0.138-0.158 from 8 s up. Short windows have the lowest MER here too, and the trend beyond 8 s is flat or noisy.
- R14 found no effect of the segment cap when scoring against the subtitles. That was a different chunking path (file
  mode, energy VAD) and has not been re-run, so the two results are not directly comparable.

## 4. Counterfactual audio (55 clips, app decoder)

| audio | MER | error units | D | S |
|---|---|---|---|---|
| original | 0.212 | 787 | 0.111 | 0.094 |
| shifted by 10 ms of silence (null perturbation) | 0.212 | 786 | 0.112 | 0.093 |
| vocals only (Demucs htdemucs) | 0.208 | 773 | 0.111 | 0.091 |
| tempo 0.8x (ffmpeg atempo, pitch kept) | 0.248 | 920 | 0.147 | 0.095 |

- **Background.** Separation removes 14 of 787 error units. Span by span (section 5), it recovers 5% of error units, while
  the inaudible shift recovers 7% by chance. There is no measurable background effect on these clips.
- The clip error rate correlates with the vocal-to-rest level in the product video (r = -0.38, 25 clips, 2-21 dB) but not in
  the finance show (r = +0.14, 30 clips, 10-38 dB). Pooled over both videos, the correlation mostly reflects the difference
  between the videos.
- **Rate.** Slower audio is worse overall. Per span it recovers 8% of error units, which is close to the 7% chance level.
  The clip error rate does not correlate with the speaking rate in the product video (r = -0.06). In the finance show the
  correlation is r = -0.37: faster clips have fewer errors. The stretched audio is out of distribution for the model, so this
  test cannot rule a rate effect in or out.
- Separation and stretching both add artifacts. Both tests are lower bounds.

## 5. Error spans (407 spans, 787 error units, app decoder)

Each maximal run of errors is tested against the other decodes of the same clip. The first test that passes is the span's
primary cause:

| primary cause | test that passes | error units | recovered by this test alone |
|---|---|---|---|
| decoder | the span is right in one of the LM beam's 32 best | 69 (9%) | 9% |
| LM | right in greedy CTC, wrong with the LM | 18 (2%) | 4% |
| background | right after vocal separation | 16 (2%) | 5% |
| rate | right at 0.8x tempo | 30 (4%) | 8% |
| unexplained (residual) | none of the above | 654 (83%) | - |
| null baseline | right after the 10 ms shift | - | 7% |

Descriptive tags (features, not causes): 36% of error units are in deletion-only spans, 16% are in spans made only of
function words and fillers, and 12% are in spans that contain English words.

The 20 largest spans are all unexplained. Their text is kept local in `out/worst20.md` because it is third-party speech. The
times are approximate (linear within the clip).

| # | video | time (s) | units | ops | tags | right in greedy | right after shift |
|---|---|---|---|---|---|---|---|
| 1 | product | 382.5 | 12 | D3 S9 | latin | no | no |
| 2 | product | 88.3 | 10 | D3 S7 | - | no | no |
| 3 | finance | 619.0 | 8 | D4 S4 | latin | no | no |
| 4 | product | 112.9 | 7 | D5 S2 | latin | no | no |
| 5 | product | 238.9 | 7 | D2 S5 | - | no | no |
| 6 | product | 334.6 | 7 | D5 S2 | - | no | no |
| 7 | product | 387.2 | 7 | D2 S5 | - | no | no |
| 8 | finance | 976.3 | 7 | D5 S2 | - | no | no |
| 9 | product | 25.9 | 6 | D3 S3 | - | no | no |
| 10 | product | 125.7 | 5 | I1 S4 | - | no | no |
| 11 | product | 219.9 | 5 | D5 | deletion_only | no | no |
| 12 | product | 222.8 | 5 | D5 | deletion_only | no | no |
| 13 | product | 230.6 | 5 | D3 S2 | - | no | no |
| 14 | product | 320.8 | 5 | D3 S2 | - | no | no |
| 15 | product | 340.9 | 5 | D2 S3 | - | no | no |
| 16 | product | 371.1 | 5 | D4 S1 | - | no | no |
| 17 | finance | 1455.7 | 5 | I1 S4 | - | no | no |
| 18 | finance | 1495.5 | 5 | I1 S4 | - | no | no |
| 19 | product | 43.0 | 4 | S4 | - | no | no |
| 20 | product | 46.5 | 4 | D4 | deletion_only | no | no |

What the local table shows:

- Three spans contain English product or company names inside Mandarin speech, spelled as wrong BPE pieces.
- The other seventeen are everyday Mandarin words in fast casual speech, replaced by similar-sounding words or dropped.
- Span 17 also contains a variant character that is not in the vocabulary, a normalization artifact worth one unit.
- Spans 11, 12 and 20 are pure deletions; 11 and 12 are in the same clip. These clips should be checked first in the human
  review, because a draft that invents words would also produce this pattern.

## 6. Sensitivity to the reference (product video, uploader subtitles)

| | Whisper draft | subtitles |
|---|---|---|
| app (beam + LM) MER | 0.273 | 0.277 |
| v3 MER | 0.263 | 0.269 |
| oracle LM 32-best | 0.240 | 0.245 |
| unexplained share of error units | 84% | 82% |
| live path, cap 6 s / 25 s | 0.272 / 0.290 | 0.272 / 0.290 |

Outputs are in `out/*-subtitles.*` (`EA_REF=subtitles`). The two references are independent, and the conclusions on this
video do not depend on which one is used. The human-verified reference sets the final numbers.

## 7. Decision for the next round

| direction | evidence here | expected gain | cost |
|---|---|---|---|
| better decoder or rescoring (v3 style, larger beam, domain LM) | oracle-32 ceiling -2.6 points; v3 -0.6 | at most -2.6, realistically about -1 | no GPU for decoding changes; a domain LM needs licensed text |
| background augmentation (music, noise) | separation recovers no more than the null shift | not shown on these recordings | a fine-tune |
| speed perturbation | no rate effect shown (test limited by stretch artifacts) | not shown | a fine-tune |
| **app chunking: live segment cap 25 s -> 6 s** | best cap in both videos: -1.8 (product) and -5.0 (finance) points; deletions lowest at 6 s in both | about -2 to -5 points on the live path | about 2 h, plus first-final latency and the synthetic fixture suite |
| acoustic adaptation to spontaneous zh-TW | 83% of error units are unexplained by every test that does not retrain the model | unknown; this is the only listed direction that could reach the residual, and the residual also contains reference errors | pilot below |

Recommendation: **one gated pilot on the acoustic model, plus the 6 s cap in the app**, both after the references are verified
and the tables are re-run.

The chunking change is engineering. The cap of 6 s is the one value that wins in both videos. The caps in between are noisy.

The pilot (training team; this repository does not modify the training repositories; drafted as R15 in
`requests-to-training.md`, to be sent after verification):

1. **Data.** The published training list already includes ASCEND (spontaneous Mandarin-English code-switching, Hong Kong
   speakers) and NTUML2021 (Taiwanese lecture speech with English terms). Those are the closest data in the list. It contains
   no casual zh-TW conversation, vlog or podcast speech. The pilot needs at least 5 h of licensed, verbatim-transcribed speech
   of that kind (fillers kept, English words in place), split by source, so that no speaker or channel is in both train and
   test. The two videos analysed here are evaluation material only and must not be trained on. `training/large` at
   `tingmix@34458e0` contains no augmentation code. That is noted, but nothing here shows augmentation would help.
2. **Training.** Continue LoRA r16 plus the Reader from v2. Mix the new pool in at about 30% of each epoch, for 2-3k steps
   (v2 used a 12k-step cosine schedule).
3. **Compute (estimate).** The only published figure is 4 GPU-hours for 28k steps of the frozen-front attention decoder.
   LoRA steps also run the encoder backward, so the pilot needs about 2-5 GPU-hours and a full run about 10-20.
4. **Validation.** At least 60 min of human-verified held-out clips from at least 10 sources outside the training data, scored
   by this harness (unexplained-class error units, deletions, MER), plus the public dev sets.
5. **Go criterion.** At least 15% relative fewer unexplained-class error units on the held-out sources, and at most +0.3
   points on any public dev set. If the pilot falls short, the next question is whether the frozen EmbeddingGemma audio
   encoder limits this speech style. The answer to that is a different base encoder, not more decoder work.

Labelling: an estimated 6-8 person-hours for 60 min of verified evaluation audio. The 11 minutes here are expected to take
about 40 minutes of listening. Labelling the pilot's 5 h training pool is separate.

## 8. Limits and what verification can change

- **Provisional references.** Whisper errors count against TingMix and end up in the unexplained residual. The subtitle check
  bounds this for one video only. If Whisper invents fillers, the deletion rates are overstated.
- **Small sample.** Two videos and 11 minutes. The two videos differ a lot (product 0.273, finance 0.158). The English sample
  is 85 words.
- **What the tests cannot see.** A span that needs both a better LM and cleaner audio stays in the residual. The null shift
  shows that about 7% of error units flip under any small perturbation.
- **Whisper on verified text.** Once Whisper is scored against verified text (`ea.score` table C), it shows what a model
  trained on broader data achieves on the same clips. That is the most direct evidence for or against the data hypothesis.
