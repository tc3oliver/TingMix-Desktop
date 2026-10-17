#!/usr/bin/env python3
"""Regenerate tests/fixtures/audio: synthetic speech from macOS `say` (no third-party recordings, no licence issue).

  python3 tools/fixtures/make_audio.py

16 kHz mono s16 WAV + manifest.jsonl (id, audio, text, lang, voice). TTS speech is a plumbing / parity fixture, not an
accuracy benchmark: recognition quality on real speech is measured on real test sets only."""

import json
import subprocess
import tempfile
import wave
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "tests/fixtures/audio"
ZH, EN = "Meijia", "Samantha"
CLIPS = [
    ("zh_02s", ZH, "zh", "今天天氣很好"),
    ("en_02s", EN, "en", "good morning everyone"),
    ("mix_04s", ZH, "mix", "今天我們來 review 這個 API 的設計"),
    ("zh_08s", ZH, "zh", "立法院今天下午召開會議，討論明年度的中央政府總預算案，以及多項重要的民生法案。"),
    ("en_08s", EN, "en", "the quick brown fox jumps over the lazy dog while the weather stays calm and clear all afternoon"),
    ("mix_08s", ZH, "mix", "我覺得這個 deadline 有點太趕了，我們要不要先跟 PM 討論一下 scope 再決定"),
    ("zh_16s", ZH, "zh", "語音辨識系統需要處理不同的口音、背景噪音以及說話速度。為了讓模型在桌面電腦上即時運作，"
     "我們必須仔細設計串流架構，確保每一段音訊只處理一次，並且在使用者停止說話之後盡快輸出最終結果。"),
    ("mix_29s", ZH, "mix", "大家好，今天的 meeting 主要有三個 topic。第一個是上週的 release 狀況，QA 回報了幾個 bug，"
     "其中兩個已經 fix 了。第二個是新的 feature 規劃，我們希望下個 sprint 可以完成 prototype。第三個是 performance 的問題，"
     "目前在 Windows 上面的 latency 還是太高，需要再做一些 profiling 跟 optimization。最後，請大家記得更新自己的 ticket 狀態，"
     "謝謝大家。"),
]


def say(voice, text, path):
    with tempfile.TemporaryDirectory() as d:
        aiff = Path(d) / "x.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(path)], check=True)


def dur(p):
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid, voice, lang, text in CLIPS:
        p = OUT / f"{cid}.wav"
        say(voice, text, p)
        rows.append({"id": cid, "audio": p.name, "text": text, "lang": lang, "voice": voice, "duration_s": round(dur(p), 3)})
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for r in rows:
        print(r["id"], r["duration_s"])


if __name__ == "__main__":
    main()
