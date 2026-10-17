"""Run the reference Python StreamingSession with a deterministic fake model on fixture audio and write the
event sequences the Rust implementation must reproduce (tests/fixtures/session/*.json).

    cd tools/golden-session && uv sync && uv run python generate.py

The fake model mirrors tingmix_streaming::MockBackend: text is a pure function of the audio length.
"""

import json
import wave
from pathlib import Path

import numpy as np
from tingmix.streaming.session import StreamingSession

ROOT = Path(__file__).resolve().parents[2]
AUDIO = ROOT / "tests/fixtures/audio"
OUT = ROOT / "tests/fixtures/session"
SR = 16000
WORDS_PER_SECOND = 2


def fake_text(n_samples: int) -> str:
    n = n_samples * WORDS_PER_SECOND // SR
    words = [f"w{i}" for i in range(1, n + 1)]
    if n % 2 == 1:
        words[-1] += "~"
    return " ".join(words)


class FakeStream:
    def __init__(self):
        self.total = 0

    def accept(self, pcm):
        self.total += len(pcm)
        return fake_text(self.total)

    def finalize(self):
        return fake_text(self.total)


class FakeModel:
    name, sample_rate, max_window_s = "mock", SR, 30.0

    def __init__(self, native: bool):
        self.streaming_native = native

    def transcribe_batch(self, wavs):
        return [fake_text(len(w)) for w in wavs]

    def new_stream(self):
        return FakeStream()


def load(name: str) -> np.ndarray:
    with wave.open(str(AUDIO / name), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2), name
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


def build(inputs: list[dict]) -> np.ndarray:
    parts = [load(i["wav"]) if "wav" in i else np.zeros(int(i["silence_s"] * SR), dtype=np.float32) for i in inputs]
    return np.concatenate(parts)


SPEECH = [{"wav": "mix_08s.wav"}, {"silence_s": 1.5}, {"wav": "en_02s.wav"}, {"silence_s": 1.0}]
SCENARIOS = [
    {"name": "buffered_two_segments", "native": False, "chunk": 1600, "config": {}, "inputs": SPEECH},
    {"name": "native_two_segments", "native": True, "chunk": 777, "config": {}, "inputs": SPEECH},
    {
        "name": "buffered_max_length_cut",
        "native": False,
        "chunk": 1600,
        "config": {"max_segment_s": 3.0},
        "inputs": [{"wav": "zh_16s.wav"}, {"silence_s": 1.0}],
    },
    {"name": "buffered_flush_open_segment", "native": False, "chunk": 4000, "config": {}, "inputs": [{"wav": "en_08s.wav"}]},
    {
        "name": "native_flush_open_segment",
        "native": True,
        "chunk": 320,
        "config": {"silence_ms": 300, "tail_pad_ms": 100},
        "inputs": [{"wav": "zh_08s.wav"}],
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sc in SCENARIOS:
        audio = build(sc["inputs"])
        session = StreamingSession(FakeModel(sc["native"]), **sc["config"])
        events = []
        for i in range(0, len(audio), sc["chunk"]):
            events += session.feed(audio[i : i + sc["chunk"]])
        events += session.flush()
        doc = {**sc, "sample_rate": SR, "words_per_second": WORDS_PER_SECOND, "events": [e.to_dict() for e in events]}
        (OUT / f"{sc['name']}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
        kinds = [e["type"] for e in doc["events"]]
        print(f"{sc['name']}: {kinds.count('partial')} partial, {kinds.count('final')} final")


if __name__ == "__main__":
    main()
