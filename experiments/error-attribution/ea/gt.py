"""References for scoring.

    uv run python -m ea.gt EXPORT.json      # import the review page export -> out/NAME/gt.json (verified clips only)

`reference(name)` returns {clip_id: (text, status)}: the human-verified text where it exists ("verified"), otherwise the
Whisper draft ("provisional"). Reports state how many clips of each kind were scored."""

import sys

from .common import load_json, save_json, video_dir


def reference(name):
    """EA_REF=subtitles scores against the uploader's subtitles instead of the Whisper draft (sensitivity check only)."""
    import os

    d = video_dir(name)
    meta = load_json(d / "regions.json")
    gt = load_json(d / "gt.json") if (d / "gt.json").exists() else {}
    alt = os.environ.get("EA_REF") == "subtitles" and meta["cue_source"] == "subtitles"
    out = {}
    for r in meta["regions"]:
        for c in r["clips"]:
            if c["id"] in gt:
                out[c["id"]] = (gt[c["id"]], "verified")
            else:
                out[c["id"]] = (c["cue_text"], "subtitles") if alt else (c["whisper_text"], "provisional")
    return out


def main():
    exp = load_json(sys.argv[1])
    per = {}
    for key, v in exp.items():
        name, cid = key.split("/", 1)
        if v.get("done"):
            per.setdefault(name, {})[cid] = v["text"].strip()
    for name, m in per.items():
        save_json(video_dir(name) / "gt.json", m)
        print(name, len(m), "verified clips")


if __name__ == "__main__":
    main()
