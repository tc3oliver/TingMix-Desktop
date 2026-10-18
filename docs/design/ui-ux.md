# UI/UX design principles

Design intent for the desktop app (`src/`). The goals and requirements it serves are in [goals.md](goals.md); how the host and
front end are built is in [../architecture/app.md](../architecture/app.md). This page states what the interface should do and the
acceptance checks for it, not how any one component is written.

## Principles

- Plain language. Say what happened, then what to do; second person, sentence case, calm tone. Jargon ("probe", "backend",
  "dBFS", "Mock") appears only under Advanced and Diagnostics; elsewhere use "test on this Mac", "engine", "Demo mode".
- One status, said once. The engine state is shown in one place and never as concatenated fragments.
- Never present placeholder output as recognition. Demo mode is labelled on every tab and in the transcript's provenance.
- Nothing destructive happens on a single click, and nothing is hidden behind a hover.
- Everything is usable by keyboard and screen reader, readable in light and dark themes, and fits a 640 x 480 window.
- No new UI framework or dependency for visual polish; the interface stays small.

## Information hierarchy per tab

| Tab | Primary action | Main content | Secondary |
|---|---|---|---|
| Live | Start / Stop | Captions | Microphone and level meter, export |
| Files | Drop or choose a file | Job cards, newest first | Per-card actions |
| History | Choose an entry | Opened transcript | Export, delete |
| Models | "In use" summary | One choice card per installed model | Add model, collapsed Advanced |
| Settings | Sections in order: Appearance, History, Speech engine, Updates, About and diagnostics | | |

## Requirements

### Input signal feedback
The Live tab judges the signal instead of showing a bare meter. If the input is silent for about 3 s while recording, or no
samples arrive for about 2 s, a warning names the device, explains the likely cause (muted input; remote-desktop and virtual
audio devices often deliver silence) and offers "Stop and choose another microphone". The warning container uses `role="alert"`,
appears only when the condition becomes true, carries no running counter in its text, and clears after one good reading. The
meter is labelled (`role="meter"`, input level) and sits next to the microphone selector. Thresholds are calibrated on real
hardware (built-in, USB and remote-desktop microphones). Acceptance: a muted or remote-desktop input shows the warning within
3 to 4 s naming the device; a built-in microphone in a quiet room for 60 s never trips it; VoiceOver announces it once.

### One engine status and first-run guidance
The header has an engine chip that opens Models: "Checking engine…", "Demo mode", "Loading {model}…", "Engine failed to load",
"{model} · Experimental" (outlined warning), or "{model}" (neutral, no banner). A full-width banner appears only in demo mode
("Demo mode: TingMix isn't recognising speech yet; output is placeholder words") with a "Set up a speech model" button. Empty
states differ by engine: in demo mode Live and Files say captions are placeholders and link to Models; with a real engine Live says
"Press Start and speak. Captions appear here." The first run of Models shows a three-step onboarding card (get a model package,
add it, start transcribing) with one primary button. Acceptance: Models is one click from Live on a fresh profile; with a validated
model no warning colour appears anywhere; no chip or banner string is built by concatenation.

### Captions, scrolling and screen readers
The caption container is always rendered (the placeholder lives inside it) so layout does not jump. Live and running file jobs
follow the end of the text only while the user is at the bottom; scrolling up stops following and shows "Jump to latest". Only the
caption container scrolls, never the window or an ancestor, and History entries and finished jobs open at the top. Final segments are
in a polite log region (`role="log"`); partial text is outside any live region so revisions are not re-announced; the region has an
accessible label and is focusable. Acceptance: scrolling up keeps the view; VoiceOver reads each final once and never partial
revisions; at 640 x 480 the window does not scroll when captions arrive.

### Contrast
Text and controls meet 4.5:1 in the system, light and dark themes: on-accent and on-danger text tokens differ per theme, the warning
banner uses a tinted background with normal foreground text and a coloured edge, and the unstable tail of a partial uses the muted
colour at full opacity rather than reduced opacity. Acceptance: every pair checked with a contrast tool.

### Models as a "choose a model" view
A summary card says "In use: {model}" or "No model in use: demo mode" with a plain reason. Each installed model has one card:
readable name, one plain status line ("Ready", "Experimental: turn on in Settings or here", "Testing on this Mac…", "Test failed:
{reason}", "Not supported in this build"), "In use" and "Previous version" badges, and exactly one primary "Use this model" button
(the in-use card shows a secondary "Test again"). A collapsed "Technical details" holds id, engine, precision, size, serial,
revision, licences, "Verify files", "Remove" (with confirmation) and, for the in-use model when a previous version exists, "Roll
back". Adding a model offers "Add model folder…" as primary, with ".zip" and signed-manifest installs secondary. The licence
step lists licences that need acceptance with unchecked checkboxes, shows already-included ones as a plain list, moves focus to
its heading and cancels on Esc. Acceptance: first-run Models has exactly one primary button; with two models the user sees which
is in use and switches with one click; no id, serial or revision is visible until Technical details is expanded.

### Confirm before deleting
Deleting a history entry or removing a model asks "Delete "{name}"? This can't be undone." with Delete and Cancel (focus on Cancel,
Esc cancels). No single click deletes data.

### Live toolbar and caption styling
Start / Stop has a fixed slot and does not move when the state changes. State is a static dot plus a word ("Recording",
"Starting…", "Stopping…", "Ready", "Error"); the dot never blinks. Elapsed time is large, monospaced and labelled. The microphone
selector and meter sit on the right; the device list is "loading" until known, so "No microphone found" never flashes at launch.
Caption lines align timestamp and text in two columns (timestamp fixed small and muted); partials have an accent bar and a muted
unstable tail; no italics (CJK italics are synthetic); no transition on text changes. Acceptance: Start never moves at 640 px;
a partial is distinguishable in a grayscale screenshot.

### Files
Dropping a file anywhere in the window shows a full-window overlay, switches to Files and shows the job. The drop zone is large with
no jobs and one compact line afterwards, and lists the supported formats and that processing is local. Each job card shows a
truncated file name (full name in the tooltip), a state label (Transcribing / Done / Cancelled / Failed), audio time transcribed as
mm:ss, Cancel while running, and Export TXT, Export SRT and "Remove from list" afterwards. Empty results say why: running "text
appears as it's recognised", done "No speech found in this file", cancelled "Cancelled before any speech was found", failed shows
the error. Export success shows the file name with the full path in the tooltip. Acceptance: a cancelled job never says "No
speech found".

### Settings
Five sections: Appearance (theme, caption size with a live sample, language), History, Speech engine (experimental switch with an
associated hint and a link to Models), Updates, About and diagnostics (diagnostics folded in, polled only while open). The caption
size control keeps a local value while dragging and persists on release; out-of-order responses never snap it back. Update success
is shown as a status, not as an error.

### Navigation, keyboard and focus
Tabs follow the ARIA tabs pattern (roving tabindex, Left/Right/Home/End, labelled panels). Shortcuts (Cmd on macOS, Ctrl on
Windows): 1 to 5 switch tabs, Enter starts or stops, O chooses a file, = and - change caption size within 12 to 72, Esc closes the
licence card, cancels a pending confirmation or dismisses an error. Shortcuts are listed in each button's tooltip and under About;
they ignore keystrokes in text fields and avoid system or webview combinations (reload, close, quit, minimise, hide, Space). Focus
returns to the opener when a card closes. The global error has a dismiss button. Acceptance: the Live, Files and Models flows can
each be completed with the keyboard only.

### Layout
The app fills the window: header and tabs fixed, the content area scrolls internally, and Live gives the transcript the remaining
height (at least 160 px). Below 720 px the microphone group wraps and the engine chip truncates with its full text in the tooltip.
Long device, model and file names truncate with a tooltip and never push buttons off-screen. Acceptance: at 640 x 480 and 960 x 700
Live shows Start, elapsed time, the transcript and export buttons without page scroll, and strings 40% longer wrap without overlap.

### Motion and theme
Respect `prefers-reduced-motion`: no meter transition, instant programmatic scrolling. Native controls follow the colour scheme in
dark mode. Overlays use `color-mix` tints consistent with the rest of the palette.

### Copy and localisation
The interface ships in English and Traditional Chinese and follows the OS language. One string per whole sentence with named
parameters; no concatenated fragments or raw enum values (enums go through label maps); plurals through `Intl.PluralRules`; numbers,
durations and dates through `Intl` with the UI locale; punctuation inside the string (full-width in zh-Hant); no all-caps
status words; buttons never sized by their English text; the document language is set at run time for CJK glyph selection and
screen-reader voices. Prose produced by the host (status messages, operation messages, error strings, tray items) must be
translatable too and is a known gap to close. Taiwan wording is used for Traditional Chinese (for example 轉錄 for transcription).

## Optional refinements

- Caption size buttons in the Live footer next to the shortcuts.
- History as master-detail at 800 px and wider; durations as mm:ss.
- "Copy text" for results and "Copy diagnostics" (verify clipboard access in the webview, else a selectable text area).
- A warning chip when audio was dropped because the computer could not keep up.

## Known design risks

1. Silence thresholds are estimates and need calibration on real hardware (the level reading covers only the last chunk).
2. The default application menu may swallow some shortcuts; verify they reach the webview.
3. Some checks need a person at the screen: VoiceOver, contrast in practice and the 640 x 480 layout.
4. Windows-specific layout and real screen-reader behaviour have not been evaluated.
