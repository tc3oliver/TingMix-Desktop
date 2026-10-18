"""Sample evaluation regions from one recording and cut them into utterance clips.

    uv run python -m ea.regions --name NAME --wav FILE.wav [--vtt SUBS.vtt] [--regions 5] [--region-s 60] [--seed 20261019]

Utterance times come from the subtitle cues when a .vtt is given, otherwise from Whisper large-v3 segments. Regions are
non-overlapping, drawn with a fixed seed, and start and end on cue boundaries. Clips merge consecutive cues up to 15 s and
are cut at the quietest 20 ms frame within 0.3 s of the cue boundary. Writes out/NAME/regions.json and out/NAME/clips/*.wav."""

import argparse
import random
import re

import numpy as np

from .common import SR, read_wav, save_json, to_trad, video_dir, write_wav

WHISPER = "mlx-community/whisper-large-v3-mlx"


def vtt_cues(path):
    def sec(t):
        h, m, s = t.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    cues, cur = [], None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        m = re.match(r"(\d+:\d+:\d+[.,]\d+)\s+-->\s+(\d+:\d+:\d+[.,]\d+)", line)
        if m:
            cur = [sec(m[1]), sec(m[2]), []]
            cues.append(cur)
        elif line and cur is not None and not line.isdigit():
            cur[2].append(re.sub(r"<[^>]+>", "", line))
        elif not line:
            cur = None
    return [(a, b, " ".join(t)) for a, b, t in cues if t and b > a]


def whisper_cues(wav_path):
    import mlx_whisper

    r = mlx_whisper.transcribe(str(wav_path), path_or_hf_repo=WHISPER, language="zh", condition_on_previous_text=False)
    return [(s["start"], s["end"], to_trad(s["text"].strip())) for s in r["segments"] if s["end"] > s["start"]]


def quiet_point(x, t, win=0.3):
    """Sample index of the quietest 20 ms frame within +-win s of time t."""
    f = int(0.02 * SR)
    a, b = max(0, int((t - win) * SR)), min(len(x), int((t + win) * SR))
    if b - a < f:
        return int(t * SR)
    seg = x[a:b]
    n = len(seg) // f
    e = (seg[: n * f].reshape(n, f) ** 2).mean(1)
    return a + int(np.argmin(e)) * f + f // 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--vtt")
    p.add_argument("--regions", type=int, default=5)
    p.add_argument("--region-s", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=20261019)
    a = p.parse_args()
    x = read_wav(a.wav)
    dur = len(x) / SR
    cues = vtt_cues(a.vtt) if a.vtt else whisper_cues(a.wav)
    source = "subtitles" if a.vtt else "whisper-large-v3"
    # candidate region starts: cue starts that leave a full region before the end
    starts = [i for i, c in enumerate(cues) if c[0] + a.region_s <= cues[-1][1]]
    rng = random.Random(a.seed)
    rng.shuffle(starts)
    taken, regions = [], []
    for i in starts:
        s = cues[i][0]
        j = i
        while j + 1 < len(cues) and cues[j + 1][1] <= s + a.region_s:
            j += 1
        e = cues[j][1]
        if any(s < te and e > ts for ts, te in taken) or e - s < 0.8 * a.region_s:
            continue
        taken.append((s, e))
        regions.append((i, j))
        if len(regions) == a.regions:
            break
    regions.sort()
    d = video_dir(a.name)
    out = []
    for r, (i, j) in enumerate(regions):
        clips, k = [], i
        while k <= j:
            m = k
            while m + 1 <= j and cues[m + 1][1] - cues[k][0] <= 15.0:
                m += 1
            clips.append((k, m))
            k = m + 1
        rc = []
        for c, (k, m) in enumerate(clips):
            lo = quiet_point(x, cues[k][0]) if c else int(cues[k][0] * SR)
            hi = quiet_point(x, cues[m][1]) if c < len(clips) - 1 else int(cues[m][1] * SR)
            if c:
                lo = rc[-1]["end_sample"]  # contiguous: the previous clip ends where this one starts
            cid = f"r{r}c{c:02d}"
            write_wav(d / "clips" / f"{cid}.wav", x[lo:hi])
            rc.append(dict(id=cid, start=lo / SR, end=hi / SR, start_sample=lo, end_sample=hi, cue_text=" ".join(cues[t][2] for t in range(k, m + 1))))
        write_wav(d / "regions" / f"r{r}.wav", x[rc[0]["start_sample"] : rc[-1]["end_sample"]])
        out.append(dict(id=f"r{r}", start=rc[0]["start"], end=rc[-1]["end"], clips=rc))
    save_json(d / "regions.json", dict(name=a.name, wav_seconds=dur, cue_source=source, seed=a.seed, region_s=a.region_s, regions=out))
    print(f"{a.name}: {len(out)} regions, {sum(len(r['clips']) for r in out)} clips, {sum(r['end'] - r['start'] for r in out):.0f} s ({source})")


if __name__ == "__main__":
    main()
