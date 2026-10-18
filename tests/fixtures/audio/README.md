# Sample speech fixtures

The WAV files here (16 kHz mono, 16-bit) are **synthetic speech** generated with the macOS system voices **Meijia** (Mandarin) and
**Samantha** (English) through the `say` command, from sentences written for this project (`manifest.jsonl` lists each clip's
text, language and voice). They contain no recordings of real people and no third-party text.

They are plumbing and parity fixtures (decoder, front end, streaming session and backend comparisons). They are not an accuracy
benchmark: recognition quality on real speech has to be measured on real test sets.

Regenerate them on a Mac with `python3 tools/fixtures/make_audio.py`. The voices are Apple system voices; if you redistribute
the files, check Apple's terms for the macOS version you generated them on.
