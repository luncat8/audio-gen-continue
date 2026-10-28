## 1. Can ComfyUI continue audio? Yes — one model is clearly right

| Model | Continuation? | Verdict |
|---|---|---|
| **ACE-Step 1.5** (2B turbo / 4B XL) | trained for repaint/extend (`src_latents + chunk_masks`), native in ComfyUI ≥0.12 | **use this** — 48 kHz, vocals + lyrics, ~8 GB VRAM, 10 GB AIO checkpoint |
| **Stable Audio 3** medium | native `inpainting/continuation` and ComfyUI's `StableAudio3.concat_cond` *does* feed mask+latent into the DiT | good fallback for **instrumentals only** (no vocals, 6:20 cap, needs CUDA+FA2) |
| **YuE2** | continuation only at **score / semantic-token level**, not waveform — and ComfyUI's tokenizer always appends `[ABC_END]`, so even that path is unreachable there | great vocals, **wrong tool** for "continue this mp3" |
| ACE-Step 1.0, MiniMax Music 3, MusicGen, YuE v1 | each has some extend support | older / different trade-offs, see research doc |

## 2. Your instinct about "VAE-encode → KSampler" is half right

Blindly encoding a track and re-diffusing it *is* destructive (at `denoise < 1` you smear noise over everything, at `1.0` you regenerate everything). But masked latent continuation is legitimate **if all four hold**: an explicit `noise_mask` pins the prefix (`out = out*mask + latent_image*(1-mask)` every step), `denoise = 1.0` on the new region only, the model's **length conditioning** is set to the *total* duration, and the model was trained with masked context. ACE-Step 1.5 satisfies 1–4; the seam is then fixed by crossfading back into your untouched original.

One caveat I verified in the source: ComfyUI's `ACEStep15.forward` passes `chunk_masks = ones` and `src_latents = x`, so the model is never told *which* part to keep — it only attends to your prefix. That's why the stitch node and `repaint_tail` mode exist.

## 3. Existing workflows (shortlist)

- **billwuhao/ComfyUI_ACE-Step** — `ACEStepExtend` (ACE 1.0, pipeline `task="extend"`).
- **starsFriday/ComfyUI-ACEStep** — `ACEStep15XLExtendAudio` + example `ace_step_15xl_extend_audio.json`: closest existing continuation graph, but no stitch-back step.
- **RyanOnTheInside**, **ComfyUI-AceMusic**, **scromfyUI-AceStep** — native ACE 1.5 extend/repaint guiders.
- **diskrotrepo/comfyui-minimax-music3** — Suno-style extend with state replay; **eigenpunk/ComfyUI-audio** — MusicGen continuation.
- Official ACE-Step recipe for long songs = repeated **Repaint**; nothing ships *masked extend + original stitching + GUI*.

## 4. What I built (committed on your branch)

```
comfyui_audio_continue/   2 nodes: AudioContinueExtend (prefix + N s empty frames + mask,
                          outputs total_seconds), AudioContinueStitch (keeps your exact
                          original, crossfades into the generated tail)
workflows/ace15_audio_continue_ui.json    ← drag into ComfyUI and press Queue
workflows/ace15_audio_continue_api.json   ← what the GUI posts to /prompt
audio_continue_tk.py      Tk GUI: open file → add seconds → Generate
tools/build_workflows.py  one graph definition → both workflow files
tests/                    18 client tests (fake ComfyUI server) + 11 node tests
RESEARCH.md, README.md    investigation + install/usage notes
```

The GUI uploads via `/upload/image`, queues `/prompt`, polls `/history`, downloads `/view`, names the result `song_continue+30s_<timestamp>.flac`, and persists everything (last file, last folder, server, all params, window size) to `~/.audio_continue.json`. Destination field defaults to `audio-continue/audio` → `ComfyUI/output/audio-continue/`; **Check setup** validates server, nodes and checkpoint, `--check`/`--dry-run` work headless.

## 5. To run it

```bash
cp -r comfyui_audio_continue /path/to/ComfyUI/custom_nodes/     # restart ComfyUI
# ace_step_1.5_turbo_aio.safetensors -> ComfyUI/models/checkpoints/  (10 GB)
python3 audio_continue_tk.py
```

Pick the file, set **add seconds**, press **Generate**. In the ComfyUI workflow itself, set `add_seconds` on *Add seconds* and put the **real BPM and key of your source** into *Style + lyrics* — that's the biggest continuity win.

**Honest status:** the client is tested end-to-end against a fake ComfyUI server and the node math is unit-tested (lengths, mask placement, crossfade, channel/rate handling), but nothing has run on a real GPU — so the first real run is yours. If a node name differs on your ComfyUI build, `--check` will list it and I can adjust the workflow in minutes.

## 6. Captioner follow-up (the TODO from the TLDR)

* **Prompt files**: sidecar detection (`song.prompt.txt`, `song.caption.txt`, `song.tags.txt`,
  `song.txt`, `song.json`), one checkbox to append or replace, tag-level de-duplication.
* **New pack `comfyui_music_caption/`** (no pip dependencies):
  * `MusicCaptionAnalyze` — numpy DSP: BPM (spectral flux → autocorrelation → octave
    correction, with `bpm_candidates`), key (chroma → Krumhansl-Schmuckler, `key_confidence`,
    Camelot), vocal presence + register (syllabic modulation + harmonic salience), loudness,
    brightness, section map.
  * `MusicCaptionMOSS` — MOSS-Music 8B over an SGLang OpenAI-compatible endpoint: caption,
    vocal description, key, BPM, structure, lyrics. Degrades gracefully when the server is
    absent.
  * `MusicCaptionMerge` — one `tags` / `lyrics` / `bpm` / `keyscale` bundle for the text encoder.
  * `MusicCaptionTap` — makes captioners from *other* packs readable by the app.
* **App**: captioner auto-discovery through `/object_info` (ours first, then any node with an
  AUDIO input, STRING outputs and "caption/analyze/scribe/music info" in its name), an
  `Analyse source` button, a *run before Generate* checkbox, results written back into the
  BPM/key fields + description, and a captioner failure never blocks a render.
* **Tests**: 16 GUI tests against a fake tkinter (including the full generate job with the
  fake ComfyUI server), 29 client/captioner tests, 25 node/DSP tests.
