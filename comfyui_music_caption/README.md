# Music Caption (ComfyUI node pack)

Measure and describe an audio track the way a music-generation prompt needs it:
**BPM**, **key**, **vocal presence + vocal type**, structure, and — with
[MOSS-Music](https://github.com/OpenMOSS/MOSS-Music) — a full musical caption and lyrics.

Used by [`audio_continue_tk.py`](../audio_continue_tk.py) in this repository, but useful in
any ACE-Step / MiniMax / Stable Audio graph that takes `tags` + `bpm` + `keyscale` +
`lyrics`.

## Nodes (`audio/caption`)

| Node | What it does | Needs |
| --- | --- | --- |
| **Music Caption (measure: bpm / key / vocals)** | numpy DSP: tempo (autocorrelation + octave correction, with candidates), key (chroma → Krumhansl-Schmuckler + confidence + Camelot), vocal presence/register, loudness, brightness, section map | numpy (ships with ComfyUI) |
| **Music Caption (MOSS-Music)** | asks an 8B music-understanding model for caption, vocal details, key, BPM, structure, lyrics | an SGLang server on MOSS-Music-8B-Instruct |
| **Music Caption (build prompt)** | merges both into `tags` / `lyrics` / `bpm` / `keyscale` / `instrumental` | — |
| **Music Caption (collect text)** | collects `STRING` outputs of captioners from *other* packs so tools can read them | — |

Typical graph:

```
LoadAudio ─┬─▶ MusicCaptionAnalyze ─── analysis_json ─┐
           └─▶ MusicCaptionMOSS ────── moss_json ────▶ MusicCaptionMerge ──▶ TextEncodeAceStepAudio1.5
                                                          │
                                                          └─▶ (bpm / keyscale widgets)
```

## MOSS-Music (optional, recommended)

```bash
hf download OpenMOSS-Team/MOSS-Music-8B-Instruct --local-dir ./weights/MOSS-Music-8B-Instruct
sglang serve --model-path ./weights/MOSS-Music-8B-Instruct --trust-remote-code   # port 30000
```

Then set `server_url` on the node (default `http://127.0.0.1:30000`). `transport = path`
hands the server the temp wav it wrote (same machine); use `base64` when the server is
remote. Without a server the node logs the reason and returns empty strings — the graph
keeps running on the DSP numbers alone.

## No pip dependencies

`numpy` comes with ComfyUI; the MOSS client is `urllib`. Nothing else is imported at module
level.

## Limits (read before trusting the numbers)

* tempo is octave-ambiguous by nature — check `bpm_candidates` if the tail drifts;
* the key needs tonal material, `key_confidence` is a margin, not a probability;
* the DSP vocal heuristic cannot tell a plucked synth arpeggio from a syllabic vocal line
  (it says `uncertain`), and the register estimate is a median f0 — MOSS-Music is the tool
  for those cases.

See [`../CAPTIONER.md`](../CAPTIONER.md) for the comparison with other captioner packs and
the reasoning behind this split.
