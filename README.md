# audio-gen-continue

Continue an existing music file with ComfyUI. Pick a track, say how many seconds to add,
press **Generate** — you get `yourtrack_continue+30s_20261027-183000.flac` with your
original audio preserved and a seamless generated tail.

```
┌───────────────┐   ┌──────────────────────────┐   ┌──────────────────────────────┐
│ LoadAudio     │──▶│ AudioContinueExtend      │──▶│ KSampler (denoise 1.0, mask) │
│ (your file)   │   │ prefix + N s empty frames│   │ only the masked tail runs    │
└───────┬───────┘   │ noise_mask [0…0 | 1…1]   │   └──────────────┬───────────────┘
        │           │ total_seconds ───────────┼──▶ TextEncode duration          │
        │           └──────────────────────────┘                  ▼
        │                                            VAEDecodeAudio
        └──────────────────────────▶ AudioContinueStitch (keeps your file, crossfades)
                                                   │
                                            SaveAudio → output/audio-continue/
```

Model: **ACE-Step 1.5 turbo** (AIO checkpoint), 48 kHz, ~8 steps. See
[RESEARCH.md](RESEARCH.md) for why this model, and why the "just VAE-encode it into a
KSampler" idea is only half right.

---

## 1. Requirements

* ComfyUI **v0.12 or newer** (ACE-Step 1.5 nodes are native; check that
  `TextEncodeAceStepAudio1.5` exists in the node search).
* Python 3.9+ with **tkinter** for the GUI (Debian/Ubuntu: `sudo apt install python3-tk`).
  The app itself has **no pip dependencies** — stdlib only.
* GPU: ~8 GB VRAM recommended for the 2B turbo AIO (works on less with the LM disabled —
  untick `generate_audio_codes` in the app). No GPU is used by the Tk app, it only talks
  HTTP to ComfyUI.

## 2. Install the custom nodes (2 nodes, no dependencies)

```bash
cp -r comfyui_audio_continue  /path/to/ComfyUI/custom_nodes/
# or:  ln -s "$PWD/comfyui_audio_continue" /path/to/ComfyUI/custom_nodes/comfyui_audio_continue
```

Restart ComfyUI, then in the node search you should find
`Audio Continue (extend latent)` and `Audio Continue (stitch result)`.

They only use `torch` + `comfy.audio.resample`, i.e. what ComfyUI already ships.

## 3. Download the model (10 GB, one file)

```bash
huggingface-cli download Comfy-Org/ace_step_1.5_ComfyUI_files \
  checkpoints/ace_step_1.5_turbo_aio.safetensors \
  --local-dir /path/to/ComfyUI/models
mv /path/to/ComfyUI/models/checkpoints/ace_step_1.5_turbo_aio.safetensors \
   /path/to/ComfyUI/models/checkpoints/
```

(or download it in the browser:
<https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/tree/main/checkpoints>)

Make sure the final path is `ComfyUI/models/checkpoints/ace_step_1.5_turbo_aio.safetensors`.

## 4. Run the GUI

```bash
python3 audio_continue_tk.py
```

| Field | What it does |
| --- | --- |
| **Server** | ComfyUI address, e.g. `http://127.0.0.1:8188` (a remote box works too). `Check setup` verifies the server, the node pack and the checkpoint. |
| **Music file** | The file to continue. It is uploaded to `ComfyUI/input/` through `/upload/image`. |
| **Add seconds** | How much new music to generate (1–900 s). |
| **Join crossfade** | Where the generated audio takes over from your original file (0.5–2 s is a good range). |
| **Mask mode** | `generate_new_only` = only append. `repaint_tail` = also regenerate the last seconds of the source so the model can re-do the join (set `repaint_seconds` in the workflow) — use it when the seam is audible. |
| **BPM / Key** | BPM and key **of the source track**. This is the biggest continuity win; ACE-Step conditions on both. |
| **Steps / CFG** | 8 / 1.0 for the turbo checkpoint. Raise steps (16–24) for a cleaner tail; keep CFG at 1.0. |
| **Seed** | `new seed each run` keeps rolling; untick to reproduce a take. |
| **LM audio-code planning** | `generate_audio_codes`. Faster and more "prefix-driven" when off; more detailed arrangement when on. |
| **Music description** | Tags describing the *existing* arrangement ("same instruments, same mix, no intro"). Not a new song idea. |
| **Lyrics** | Leave **empty** for instrumental continuation, otherwise the model may start singing. |
| **ComfyUI prefix** | Destination inside `ComfyUI/output`. Default `audio-continue/audio` → `output/audio-continue/audio_00001_.flac`. |
| **Save to folder** | Where the app downloads the result. Defaults to the folder of the source file. |

Everything (last file, last folder, server, all parameters, window size) is saved to
`~/.audio_continue.json` and restored on startup. Override the path with
`AUDIO_CONTINUE_CONFIG=/some/file.json`.

## 5. Test the graph in ComfyUI itself

`workflows/ace15_audio_continue_ui.json` is a normal UI workflow — drag it onto the canvas:

1. `Load Audio` → upload your track.
2. `Audio Continue (extend latent)` → set `add_seconds`.
3. `Style + lyrics` → tags; BPM/key; leave lyrics empty.
4. Queue. The result is written to `ComfyUI/output/audio-continue/`.

The `duration` input of the text encoder is already wired to the extend node's
`total_seconds` output, so the model knows how long the final track is. Longer/shorter
than that and the model ends the song early.

`workflows/ace15_audio_continue_api.json` is the same graph in **API format**; that is the
file the GUI patches and posts to `/prompt`. Edit it (or point the GUI at your own file)
and the app keeps working as long as the node classes are the ones it knows.

Regenerate both files after editing the graph definition:

```bash
python3 tools/build_workflows.py
```

## 6. Command line

```bash
# verify a ComfyUI install without opening a window
python3 audio_continue_tk.py --check --url http://127.0.0.1:8188

# write the patched API workflow and inspect it (no generation)
python3 audio_continue_tk.py --dry-run --audio song.mp3 --add 30 --out patched.json
```

`--check` reports the ComfyUI version, GPU, missing nodes and available checkpoints.

## 7. Files

```
audio_continue_tk.py              the Tk app (GUI + --check / --dry-run)
comfyui_audio_continue/           ComfyUI custom node pack (2 nodes)
workflows/ace15_audio_continue_ui.json    drag & drop into ComfyUI
workflows/ace15_audio_continue_api.json   what the app posts to /prompt
tools/build_workflows.py          generates both workflow files
tests/                            headless tests (fake ComfyUI server + numpy torch stub)
RESEARCH.md                       model/workflow investigation, sources, caveats
```

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `setup check failed` / missing node | The node pack is not in `ComfyUI/custom_nodes/` or ComfyUI was not restarted. Check the startup console for import errors. |
| `TextEncodeAceStepAudio1.5` missing | ComfyUI is older than v0.12 — update ComfyUI. |
| `no ACE-Step checkpoint found` | The AIO file has to be in `models/checkpoints/` with the exact name. |
| Tail starts too loud / quiet, or has a click | Raise `Join crossfade`, or use `repaint_tail` with `repaint_seconds` 2–4. Check that BPM/key match the source. |
| Continuation drifts into a different song | Turn **LM audio-code planning** off, write tags that describe the *current* arrangement, set the real BPM/key, keep `add_seconds` ≤ 60 per run and run the app again on the result. |
| Model ends the song before the requested length | The `duration` link is missing in your workflow — reconnect `total_seconds` → `TextEncodeAceStepAudio1.5.duration`. |
| Out of VRAM | Untick LM audio-code planning (~4 GB), or use the split files with `qwen_0.6b_ace15` instead of the AIO. The masked prefix is attended to at every step, so a 10-minute source is much heavier than a 2-minute one. |
| Long source track is very slow | Continue a tail instead: cut the last ~60 s in any editor, continue that file, then join it to the rest (the join is what the crossfade is for). |
| Upload fails on a remote server | ComfyUI must be reachable over HTTP; if it sits behind a path prefix, put the full URL (including the path) into **Server**. |
| Result is FLAC and large | `SaveAudio` writes FLAC by default; swap it for `SaveAudioAdvanced` (mp3) in the workflow and set `filename_prefix` as usual — the app picks up any audio in the history output. |

## 9. Other models / packs

The app patches nodes by **class name**, so it also works with workflows built on
`starsFriday/ComfyUI-ACEStep` (`ACEStep15XLExtendAudio` + `ACEStep15XLExtendLatent`,
`ACEStep15XLTextEncode`) — point **Workflow...** at such an API workflow and the same fields
are applied. See [RESEARCH.md](RESEARCH.md) §4 for the list of existing continuation
workflows (ACE-Step 1.0 extend, MiniMax Music 3 extend, MusicGen continuation, Stable
Audio 3 inpainting) and §6 for when to reach for them.

## 10. Tests

```bash
python3 tests/test_client.py     # GUI client against a fake ComfyUI server (stdlib only)
python3 tests/test_nodes.py      # node math, needs numpy (torch is stubbed)
```

`tests/fake_comfyui.py` is also handy to click through the GUI without a GPU.

## License

Code in this repository: see [LICENSE](LICENSE).
ACE-Step 1.5 weights are licensed by their authors — check the model card before
commercial use. The workflows and scripts here only call ComfyUI.
