"""Build a local review page for the human-verified references: out/review.html (open it from disk; nothing is uploaded).

    uv run python -m ea.review NAME [NAME ...]

Each clip plays exactly as it is cut, so the reviewer writes what is audible in that clip. The text box starts from the
Whisper draft; the uploader's subtitles (if any) are shown beside it. "Export" downloads tingmix-review.json, which
`python -m ea.gt FILE` turns into out/NAME/gt.json. Progress is kept in the browser's localStorage between visits."""

import html
import json
import sys

from .common import OUT, load_json, video_dir

PAGE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>逐字稿核對</title><style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#d2d2d7;--ok:#1a7f37;--acc:#0a66c2}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f2f2f2;--mut:#a1a1a6;--line:#3a3a3c;--ok:#3fb950;--acc:#4da3ff}}
body{background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,"PingFang TC",sans-serif;margin:0 auto;max-width:880px;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line)}
.top{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:1}
.clip{border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0}.clip.done{border-color:var(--ok)}
.meta{color:var(--mut);font-size:13px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
textarea{width:100%;box-sizing:border-box;font:16px/1.6 inherit;background:transparent;color:inherit;border:1px solid var(--line);border-radius:8px;padding:8px;min-height:84px}
.sub{color:var(--mut);font-size:14px;margin:6px 0}button{font:inherit;padding:4px 12px;border-radius:8px;border:1px solid var(--line);background:transparent;color:inherit;cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}audio{width:100%;margin:6px 0}label{cursor:pointer}
</style></head><body>
<h1>逐字稿核對</h1><p class="meta">逐段聽，把框裡的文字改成「實際聽到的話」，包括「那」「呢」等口語詞；聽不清楚的字寫成 ⁇。英文照念出的寫。數字寫阿拉伯數字或中文都可以。改好就勾「已核對」。</p>
<div class="top"><span id="prog"></span><button class="primary" onclick="exp()">匯出</button><label><input type="checkbox" id="slow"> 0.75 倍速</label></div>
__BODY__
<script>
const KEY="tingmix-review-v1";let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
document.querySelectorAll(".clip").forEach(el=>{const id=el.dataset.id,ta=el.querySelector("textarea"),cb=el.querySelector("input[type=checkbox]");
 if(st[id]){ta.value=st[id].text;cb.checked=st[id].done}el.classList.toggle("done",cb.checked);
 const up=()=>{st[id]={text:ta.value,done:cb.checked};el.classList.toggle("done",cb.checked);save();prog()};ta.oninput=up;cb.onchange=up;
 el.querySelector("audio").onplay=e=>{e.target.playbackRate=document.getElementById("slow").checked?0.75:1}});
function prog(){const a=document.querySelectorAll(".clip").length,d=document.querySelectorAll(".clip.done").length;document.getElementById("prog").textContent=`已核對 ${d} / ${a}`}prog();
function exp(){const out={};document.querySelectorAll(".clip").forEach(el=>{out[el.dataset.id]={text:el.querySelector("textarea").value,done:el.querySelector("input[type=checkbox]").checked}});
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));a.download="tingmix-review.json";a.click()}
</script></body></html>"""


def main():
    body = []
    for name in sys.argv[1:]:
        meta = load_json(video_dir(name) / "regions.json")
        body.append(f"<h2>{html.escape(name)}</h2>")
        for r in meta["regions"]:
            for c in r["clips"]:
                key = f"{name}/{c['id']}"
                sub = f'<div class="sub">字幕：{html.escape(c["cue_text"])}</div>' if meta["cue_source"] == "subtitles" else ""
                body.append(
                    f'<div class="clip" data-id="{html.escape(key)}"><div class="meta"><b>{html.escape(key)}</b>'
                    f'<span>{c["start"]:.1f}–{c["end"]:.1f} s</span><label><input type="checkbox"> 已核對</label></div>'
                    f'<audio controls preload="none" src="{html.escape(name)}/clips/{c["id"]}.wav"></audio>{sub}'
                    f'<textarea>{html.escape(c.get("whisper_text", ""))}</textarea></div>'
                )
    (OUT / "review.html").write_text(PAGE.replace("__BODY__", "\n".join(body)), encoding="utf-8")
    print(OUT / "review.html")


if __name__ == "__main__":
    main()
