# Music Caption (ComfyUI node pack)

Measure and describe an audio track inside ComfyUI:
**BPM**, **key / Camelot**, **vocal presence + vocal type**, **loudness / section map** (measured via numpy DSP with zero model downloads), plus — when [MOSS-Music 8B](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct) is placed in `ComfyUI/models/moss_music/` — a full musical **caption**, **vocal/instrument description**, **chord progression**, **structure**, and **lyrics**.

Used by [`audio_continue_tk.py`](../audio_continue_tk.py) and by the ComfyUI workflows in [`../workflows/`](../workflows/):
- [`workflows/music_caption_ui.json`](../workflows/music_caption_ui.json) — standalone ComfyUI captioner & analyzer workflow with an in-canvas report viewer and a `MarkdownNote` containing the HuggingFace download links.
- [`workflows/ace15xl_audio_continue_caption_ui.json`](../workflows/ace15xl_audio_continue_caption_ui.json) — full ACE-Step 1.5 XL continuation workflow with auto-captioning wired directly into `TextEncodeAceStepAudio1.5` and an on-canvas caption viewer.

## Nodes (`audio/caption`)

| Node | Display Name | What it does | Needs |
| --- | --- | --- | --- |
| `MusicCaptionMOSSLoader` | **MOSS-Music Loader** | Loads local HuggingFace `MOSS-Music-8B-Instruct` weights from `ComfyUI/models/moss_music/` (or connects to an optional SGLang server). Defaults to `when_missing = skip (DSP only)` so workflows run cleanly even before downloading the 8B model. | Optional ~17 GB HF weights |
| `MusicCaptionAnalyze` | **Music Caption (measure: bpm / key / vocals)** | Pure numpy DSP: tempo (autocorrelation + octave candidates), key (chroma → Krumhansl-Schmuckler + confidence + Camelot), vocal presence/register, loudness, brightness, section map. | `numpy` (ships with ComfyUI) |
| `MusicCaptionMOSS` | **Music Caption (MOSS-Music)** | Runs MOSS-Music 8B (via `MOSS-Music Loader` in-process or an optional SGLang server) for rich caption, vocal details, chords, structure, key/BPM, and lyrics ASR. | `MOSS-Music Loader` or SGLang URL |
| `MusicCaptionMerge` | **Music Caption (build prompt)** | Merges DSP measurements and MOSS-Music text into `tags`, `lyrics`, `bpm`, `keyscale`, `instrumental`, `vocal_type`, and `summary_json`. | — |
| `MusicCaptionView` | **Music Caption Viewer** | Displays **all** captioner findings directly inside ComfyUI (in-node multi-line readout + `PreviewAny` compatible `report` string + individual typed outputs, with `bpm` safely clamped to `10..300`). | — |
| `MusicCaptionTap` | **Music Caption (collect text)** | Collects `STRING` outputs of third-party captioner nodes so external tools (`audio_continue_tk.py`) can read them from `/history`. | — |

All sink/analysis nodes (`MusicCaptionAnalyze`, `MusicCaptionMOSS`, `MusicCaptionMerge`, `MusicCaptionView`, `MusicCaptionTap`) declare `OUTPUT_NODE = True` so ComfyUI executes them directly without requiring a separate `SaveAudio` node.

## Graph layout

```text
LoadAudio ──────────────┬─▶ MusicCaptionAnalyze (DSP) ── analysis_json ─┐
                        │                                               ▼
MOSS-Music Loader ──────┴─▶ MusicCaptionMOSS (8B LLM) ── moss_json ──▶ MusicCaptionMerge
                                                                        │
                                             ┌──────────────────────────┴───────────────────┐
                                             ▼                                              ▼
                                    MusicCaptionView ──▶ PreviewAny          TextEncodeAceStepAudio1.5
                                 (on-node full report)                        (tags + clamped bpm)
```

## Installing MOSS-Music 8B weights (optional, recommended for rich captions)

Download from HuggingFace (Apache-2.0, ~17 GB across 4 safetensors shards):
- **Instruct (recommended):** <https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct>
- **Thinking (chain-of-thought variant):** <https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking>

Place the folder inside `ComfyUI/models/moss_music/`:

```bash
hf download OpenMOSS-Team/MOSS-Music-8B-Instruct \
  --local-dir /path/to/ComfyUI/models/moss_music/MOSS-Music-8B-Instruct
```

Expected folder layout:

```text
ComfyUI/models/moss_music/MOSS-Music-8B-Instruct/
├── config.json
├── configuration_moss_music.py
├── modeling_moss_music.py
├── processing_moss_music.py
├── preprocessor_config.json
├── generation_config.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── model.safetensors.index.json
└── model-00001-of-00004.safetensors ... model-00004-of-00004.safetensors
```

You can also tick `auto_download = True` on **MOSS-Music Loader** (if `huggingface_hub` is installed in ComfyUI's Python environment), set `MOSS_MUSIC_DIR` / `MOSS_MUSIC_MODEL_PATH`, or select `backend = sglang_server` if you prefer hosting MOSS-Music externally via `sglang serve --model-path MOSS-Music-8B-Instruct --trust-remote-code`.

By default `keep_loaded = False` unloads the 8B model from VRAM right after captioning so ACE-Step 1.5 XL has full GPU memory for generation.

## Limits (read before trusting the numbers)

* Tempo is octave-ambiguous by nature — check `bpm_candidates` in the viewer report if the tail drifts.
* Key estimation requires tonal content; `key_confidence` is a correlation margin over the runner-up.
* The DSP vocal heuristic cannot separate a plucked synth lead from singing (`uncertain`) — **MOSS-Music 8B** resolves those cases and adds genre, instrumentation, chord progression, structure, and lyrics.

See [`../CAPTIONER.md`](../CAPTIONER.md) for full architecture notes.
