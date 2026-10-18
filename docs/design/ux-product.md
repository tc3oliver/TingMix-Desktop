# TingMix Desktop: product UX brief (task #34)

For: implementers of `docs/design/ux-product.md`. It replaces the parts of `docs/design/ui-ux.md` that assume a technical user: the Models tab (`ui-ux.md:60-69`), the "{model} · Experimental" chip (`ui-ux.md:39-45`) and the "Speech engine" settings section (`ui-ux.md:93-94`). The rest of that brief still applies: contrast, caption scrolling, screen-reader behaviour, confirm before delete, and the copy rules at `ui-ux.md:116-122`.

Everything below comes from reading the code. I was not given any renders, so every "currently shows" claim needs one screenshot to confirm it.

## 0. Product stance

- One promise, said in the welcome screen and nowhere else: **"Speech to text on this computer. Nothing is uploaded. After a one-time download, it works offline."**
- The user should never see how recognition works. Words to avoid: model, engine, backend, ONNX, MLX, precision, probe, manifest, registry, serial, revision, hash, mock, demo, experimental, segment, dB, px. "語音模型 / speech model" may appear once, in the welcome screen and in Settings.
- A release build has exactly two states: **getting ready** or **ready**. Anything in between is either progress or one error with one action.

## 1. Defects to fix before any polish (found by reading)

1. **The "Mock" warning leaks into a ready release build.** With no session, `live_snapshot` returns `backend: BACKEND_LABEL` (`src-tauri/src/commands.rs:148-149`), which is `"Mock — not ASR"` (`src-tauri/src/dto.rs:8`). The front end's idle status also has `backend: ""` (`src/features/live/useLive.ts:8`), and `isMock` treats an empty label as mock (`src/lib/provenance.ts:5`).
   - The result: idle Live renders `<Provenance>` (`src/features/live/LivePanel.tsx:119`), i.e. "示範資料，並非真實語音辨識…" (`src/i18n/zh-TW.ts:58`).
   - The footer also says "示範模式" (`LivePanel.tsx:141-144`).
   - This is the single most "experimental-feeling" thing in the app.
2. **Windows users are told "this Mac".**
   - Strings: `en.ts:115,140,150,151,220-223` and the host fallbacks at `src-tauri/src/backends.rs:364,370,376,382`.
   - These reach Windows because the ONNX path is being wired (task #33/#35).
3. **Every live History entry has the same English name,** "Live session" (`commands.rs:88`), and it is shown untranslated in zh-TW (`src/features/history/HistoryPanel.tsx:41`).
4. **Screen-reader chatter during the download.** The stage text updates on every byte and sits in `role="status"` (`src/features/setup/SetupScreen.tsx:66-67`, text from `src/features/models/onboard.ts:24`).
5. **Raw technical details are shown to users** through `{detail}`:
   - `app.load_error` (`en.ts:12`), `host.setup.offline` (`en.ts:248`, detail from `setup.rs:88-92`), `host.live.disconnected` (`en.ts:206`, from `src-tauri/src/live.rs:64-69`), `host.backend.unreadable` (`en.ts:216`) and `files.failed` (`en.ts:85`).
   - Unknown host codes fall back to English (`src/i18n/host.ts:43`).

## 2. Information architecture

**Tabs: 即時字幕 / Live captions · 檔案轉錄 / Transcribe files · 歷史紀錄 / History**, with **設定 / Settings** right-aligned at the end of the same tablist. Settings is text, not an icon; a gear glyph is optional and gets no new dependency.

- **Models is no longer a tab** (`App.tsx:17,169`).
  - A normal user has one model, which the welcome screen installs.
  - Model information moves to Settings › 語音模型 / Speech model: friendly name, size, a status line, and "重新檢查 / Check again", which calls `api.modelsProbe`.
  - Everything else from `ModelsPanel.tsx` goes under one collapsed "進階 / Advanced" disclosure in that section: add folder, .zip or signed package, verify, roll back, remove, technical details, computer info and selection reasons. The handlers stay; it is a move, not a rewrite.
- **Diagnostics** become Settings › 關於 / About › "疑難排解資訊 / Troubleshooting info" (collapsed, as now at `SettingsPanel.tsx:122-125`), plus one "複製資訊 / Copy info" button. This is the only place technical words may appear: engine label, ORT path, sample counters (`DiagnosticsPanel.tsx:19-32`). Row labels still get plain names (see §4).
- **Shortcuts:**
  - ⌘/Ctrl 1-3 switch tabs and ⌘/Ctrl , opens Settings (macOS convention).
  - Today `shortcutFor` binds 1..`tabCount` (`src/lib/shortcuts.ts:21`). Pass 3 and add `,`.
  - ⌘↵ and ⌘O stay. The text at `SettingsPanel.tsx:115-120` becomes "⌘1-⌘3 …".
- **First run** uses the **full window**: no tab bar while nothing is installed (setup states `download`, `offline`, `unavailable`, and an install in progress). Language follows the OS.
- **A broken model** (`problem` state, a model is installed) keeps the tab shell, so History stays reachable. Live and Files show the problem card, as today (`App.tsx:163,166`).
- **Header status chip** (`App.tsx:149-151`):
  - It is **hidden when ready**.
  - It shows only "準備中… / Getting ready…" while `load.loading`, and "語音辨識無法使用 / Speech recognition unavailable" (danger) on `load.error`.
  - Clicking it opens Settings › Speech model, not Models.
  - The model name is never in the header.
- **The demo banner** (`App.tsx:124-129`) exists only in debug builds (`engine.ts:29`). Keep it as is, but the harness must prove it never renders in release.

## 3. Journey map

### 3.1 First launch

| Step | Screen | Primary action | Notes |
|---|---|---|---|
| Checking | Welcome card, title "歡迎使用 TingMix", muted line "正在確認這台電腦…" | none | Show it only after 300 ms, to avoid a flash. |
| Offer | Title, promise line, then "首次使用需要下載語音模型（約 1.4 GB）。下載完成後即可離線使用。" Then a licence checkbox, and the primary button "下載並開始使用 (1.4 GB)". | Download | The button is disabled until the box is checked. The hint below the button stays (`SetupScreen.tsx:94`). |
| Downloading | A determinate bar, then "已下載 0.6 GB / 1.4 GB · 約剩 3 分鐘" (tabular numbers) and the secondary button "暫停 / Pause". | Pause | Pause calls `api.modelsCancel`; the host keeps the bytes (`en.ts:228`). Time left comes from a front-end rolling 10 s rate and is hidden until 5 s of data exist. |
| Paused / failed | "已暫停，下次會從中斷處繼續。" or "下載沒有完成。請檢查網路連線後再試一次。" | **繼續下載 / Resume**, one click | Today Resume only re-checks (`SetupScreen.tsx:75`) and the user must press Download again. Change: after the re-check, if every required licence is already accepted, call `modelsInstall` directly. |
| Finishing | Indeterminate bar, "正在完成設定，請稍候…" | none | This merges `stage.verifying`, `stage.files_verified` and `stage.testing` into one sentence. The user does not need three phases. |
| Ready | Switch straight to the Live tab. Its empty state is the welcome tip (§3.2). | Start | No separate success screen. |
| Offline | "無法連線下載語音模型。請確認網路連線後再試一次。" | 再試一次 | The `detail` goes to Troubleshooting only. |
| Unavailable (`platform`) | "目前還沒有適用於這台電腦的語音模型。" | none | Muted text, no red. |
| Unavailable (`build`) | "這個版本的 TingMix 不適用於這台電腦，請下載適合的版本。" | none | |

- **Licence row:** label "我同意 {name} 的使用條款 / I agree to the {name} terms"; the URL and SPDX id sit below as small,
  selectable muted text (`SetupScreen.tsx:90-91`). A clickable "閱讀條款 / Read terms" link opens the URL in the system browser.

### 3.2 Live captions

- **Empty, ready:** one centred tip, "按「開始」或 ⌘↵，對著麥克風說話，字幕會即時出現。" with the input picker beside Start. No
  provenance line, no engine name, no "demo" text in a release build (fixes §1.1).
- **Running:** Start becomes Stop (red), a recording dot and the elapsed time, and a small level meter. Partial text is shown in a muted
  colour after the final text of the same line; finals are full contrast.
- **Retracted caption** (empty final, `docs/research/vad-noise.md`): the partial fades out over 150 ms. There is no message; it was noise.
- **Silence or no signal:** after `SILENCE_AFTER_S`, an inline, non-blocking hint under the meter: "沒有收到聲音。請確認麥克風沒有靜音，或換一個輸入裝置。"
- **Permission denied:** a card with one action, "開啟系統設定 / Open System Settings", with platform-correct wording (macOS:
  隱私權與安全性 › 麥克風; Windows: 設定 › 隱私權與安全性 › 麥克風). No "this Mac" on Windows (§1.2).
- **Device unplugged:** captions stop, the card says "麥克風已中斷連線。" with "重新開始 / Start again"; the transcript so far is kept.

### 3.3 Transcribe files

- **Idle:** a large drop zone, "將音訊或影片檔拖到這裡，或按「選擇檔案」", listing the accepted types in muted text.
- **Running:** the file name, a determinate bar on audio seconds, Cancel.
- **Done:** the transcript, with export buttons ("匯出文字檔 (.txt) / 匯出字幕檔 (.srt)") next to the title, not at the bottom.
- **Failed:** "無法轉錄這個檔案。檔案可能已損壞，或是不支援的格式。" No raw decoder message (it goes to Troubleshooting).

### 3.4 History

- **Empty:** "轉錄的內容會保存在這台電腦上，並顯示在這裡。"
- **Items:** title = date and time plus source ("即時字幕 10月18日 14:05" or the file name), duration, first line preview. Never the
  untranslated "Live session" (§1.3). Delete asks for confirmation.

### 3.5 Settings

Sections, in order: 一般 (language, appearance) · 麥克風 (default input) · 語音模型 (name, size, status, "重新檢查"; Advanced
disclosure with everything that was the Models tab) · 關於 (version, licences, Troubleshooting info with "複製資訊").

## 4. Copy rules

- zh-TW uses Taiwan usage: 轉錄 (never 轉寫), 檔案, 設定, 資訊, 影片, 麥克風. Sentences are natural, not translated word for word.
- No word from the §0 list in the normal UI. Diagnostics may use technical words, but row labels are plain ("辨識方式 / Recognition"
  instead of "Engine").
- Platform words come from the OS (`isMac`): "這台 Mac" only on macOS, otherwise "這台電腦".
- Error text = what happened + what to do, one sentence each. `{detail}` never appears outside Troubleshooting (§1.5). An unknown host
  code shows a generic localized sentence, never English in zh-TW.
- Progress text in `role="status"` is updated at most every 5 s or on a stage change (§1.4).

## 5. Priorities and acceptance (checked in the screenshot harness, both languages, both themes)

**P0**
1. §1.1: `ready-empty` and `live-captions` show no provenance, demo or engine text in a release build.
2. §2: three tabs plus Settings; Models content reachable under Settings › 語音模型 › 進階.
3. §3.1: first run is full-window with no tab bar; one-click Resume; merged "finishing" step; no "需要設定" chip.
4. §1.2/§4: no "Mac" wording when the mocked OS is Windows (add a Windows variant to `setup-unavailable` and `live-mic-denied`).
5. §1.5: no raw `{detail}` in any card in the `*-failed`, `setup-offline`, `files-done` (failed card) states.

**P1**
6. §3.2 empty-state tip, retract fade, platform-correct permission card.
7. §3.3 drop zone and export placement; §3.4 localized history titles (needs a host change: the live entry name; route to the host owner).
8. §1.4 throttled status text; download size and time-left line.
9. Header chip hidden when ready.

**P2**
10. Motion polish, ⌘1-3 and ⌘, shortcuts, Copy info button.

## 6. Harness states to add

`setup-checking`, `setup-paused`, `setup-finishing`, `setup-unavailable-windows`, `live-mic-denied-windows`, `live-disconnected`,
`settings-model-advanced-open`, `settings-troubleshooting-open`. Each state in the release-build mode (no debug banner).
