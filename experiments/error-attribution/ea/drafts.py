"""Draft transcripts for human review: Whisper large-v3 on each clip's own audio (Traditional characters).

    uv run python -m ea.drafts NAME [NAME ...]

Adds `whisper_text` to every clip in out/NAME/regions.json. Drafts are a starting point for the reviewer, never a reference."""

import sys

from .common import load_json, save_json, to_trad, video_dir
from .regions import WHISPER


def main():
    import mlx_whisper

    for name in sys.argv[1:]:
        d = video_dir(name)
        meta = load_json(d / "regions.json")
        for r in meta["regions"]:
            for c in r["clips"]:
                res = mlx_whisper.transcribe(
                    str(d / "clips" / f"{c['id']}.wav"), path_or_hf_repo=WHISPER, language="zh", condition_on_previous_text=False
                )
                c["whisper_text"] = to_trad(" ".join(s["text"].strip() for s in res["segments"]))
        meta["draft_model"] = WHISPER
        save_json(d / "regions.json", meta)
        print(name, sum(len(r["clips"]) for r in meta["regions"]), "clips drafted")


if __name__ == "__main__":
    main()
