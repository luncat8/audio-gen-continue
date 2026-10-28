# audio-gen-continue

TLDR: it works somehow but quality and continuation is not acceptable, need to try next year with better models. TODO: good captioner, best if capable to detect key, BMP, and vocal type.

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

Model: **ACE-Step 1.5 XL SFT** (split files: 4B DiT + Qwen text encoders + VAE), 48 kHz,
50 steps / CFG 7.0. See [RESEARCH.md](RESEARCH.md) for why this model, why the XL variant,
and why the "just VAE-encode it into a KSampler" idea is only half right.

Two model layouts are supported; the workflow decides which one is used:

| Workflow | Model files | Steps / CFG |
| --- | --- | --- |
| `workflows/ace15xl_audio_continue_api.json` (**default**) | `diffusion_models/acestep_v1.5_xl_sft_bf16` + `text_encoders/qwen_0.6b_ace15` + `qwen_4b_ace15` + `vae/ace_1.5_vae` | 50 / 7.0 |
| `workflows/archive/ace15_audio_continue_turbo_aio_api.json` (archived) | `checkpoints/ace_step_1.5_turbo_aio` | 8 / 1.0 |

The archived workflow is the previous 2B turbo AIO setup. It still works (same nodes, same
`AudioContinueExtend`/`AudioContinueStitch`), it just uses the single checkpoint and a much
lower VRAM footprint. Load it with **Workflow...** in the app.

---

## 1. Requirements

* ComfyUI **v0.12 or newer** (ACE-Step 1.5 nodes are native; check that
  `TextEncodeAceStepAudio1.5` exists in the node search).
* Python 3.9+ with **tkinter** for the GUI (Debian/Ubuntu: `sudo apt install python3-tk`).
  The app itself has **no pip dependencies** — stdlib only.
* GPU: **≥12 GB VRAM** for XL with ComfyUI offloading, **≥20 GB** for a comfortable run
  (XL DiT ≈10 GB bf16 + 4B text encoder ≈8 GB). ~8 GB is enough for the archived 2B turbo
  AIO workflow, especially with the LM disabled (untick `generate_audio_codes`).
  No GPU is used by the Tk app, it only talks HTTP to ComfyUI.

## 2. Install the custom nodes (2 nodes, no dependencies)

```bash
cp -r comfyui_audio_continue  /path/to/ComfyUI/custom_nodes/
# or:  ln -s "$PWD/comfyui_audio_continue" /path/to/ComfyUI/custom_nodes/comfyui_audio_continue
```

Restart ComfyUI, then in the node search you should find
`Audio Continue (extend latent)` and `Audio Continue (stitch result)`.

They only use `torch` + `comfy.audio.resample`, i.e. what ComfyUI already ships.

## 3. Download the model (XL SFT, 4 files ≈ 20 GB)

```bash
cd /path/to/ComfyUI/models

huggingface-cli download Comfy-Org/ace_step_1.5_ComfyUI_files \
  split_files/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors \
  --local-dir /tmp/ace15
mv /tmp/ace15/split_files/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors \
   diffusion_models/

huggingface-cli download Comfy-Org/ace_step_1.5_ComfyUI_files \
  split_files/text_encoders/qwen_0.6b_ace15.safetensors \
  split_files/text_encoders/qwen_4b_ace15.safetensors \
  split_files/vae/ace_1.5_vae.safetensors \
  --local-dir /tmp/ace15
mv /tmp/ace15/split_files/text_encoders/*.safetensors text_encoders/
mv /tmp/ace15/split_files/vae/ace_1.5_vae.safetensors     vae/
```

(or download them in the browser:
<https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/tree/main/split_files>)

Final layout:

```
ComfyUI/models/
├── diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors   (9.97 GB)
├── text_encoders/qwen_0.6b_ace15.safetensors               (1.19 GB, slot 1, mandatory)
├── text_encoders/qwen_4b_ace15.safetensors                 (8.38 GB, slot 2)
└── vae/ace_1.5_vae.safetensors                             (0.34 GB)
```

Small GPU / less patience: swap the diffusion model for
`acestep_v1.5_xl_turbo_bf16.safetensors` (8 steps, CFG 1.0 — press **Sampler defaults for
this workflow** in the app afterwards), or use `qwen_1.7b_ace15` in clip slot 2.

Archived 2B setup (unchanged): the single 10 GB file
`ComfyUI/models/checkpoints/ace_step_1.5_turbo_aio.safetensors` from
<https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/tree/main/checkpoints>.

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
| **Steps / CFG** | 50 / 7.0 for XL SFT (what the shipped workflow uses), 8 / 1.0 for XL turbo or the archived 2B AIO. **Sampler defaults for this workflow** re-reads these from the loaded workflow so you cannot accidentally run an SFT model at turbo settings. |
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

`workflows/ace15xl_audio_continue_ui.json` is a normal UI workflow — drag it onto the canvas:

1. `Load Audio` → upload your track.
2. `Audio Continue (extend latent)` → set `add_seconds`.
3. `Style + lyrics` → tags; BPM/key; leave lyrics empty.
4. Queue. The result is written to `ComfyUI/output/audio-continue/`.

The `duration` input of the text encoder is already wired to the extend node's
`total_seconds` output, so the model knows how long the final track is. Longer/shorter
than that and the model ends the song early.

`workflows/ace15xl_audio_continue_api.json` is the same graph in **API format**; that is
the file the GUI patches and posts to `/prompt`. Edit it (or point the GUI at your own
file) and the app keeps working as long as the node classes are the ones it knows.

`workflows/archive/` holds the previous 2B turbo AIO pair
(`ace15_audio_continue_turbo_aio_ui.json` / `..._api.json`). Drag the UI one onto the
canvas or select the API one with **Workflow...** — remember that it wants 8 steps / CFG 1
and `models/checkpoints/ace_step_1.5_turbo_aio.safetensors`.

Regenerate all four files after editing the graph definitions (`tools/build_workflows.py`
contains the XL and the archived AIO graph):

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
audio_continue_tk.py                            the Tk app (GUI + --check / --dry-run)
comfyui_audio_continue/                         ComfyUI custom node pack (2 nodes)
workflows/ace15xl_audio_continue_ui.json        default: XL SFT, drag & drop into ComfyUI
workflows/ace15xl_audio_continue_api.json       default: what the app posts to /prompt
workflows/archive/ace15_audio_continue_turbo_aio_{ui,api}.json   archived 2B turbo AIO pair
tools/build_workflows.py                        generates all four workflow files
tests/                                          headless tests (fake ComfyUI server + numpy torch stub)
RESEARCH.md                                     model/workflow investigation, sources, caveats
```

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `setup check failed` / missing node | The node pack is not in `ComfyUI/custom_nodes/` or ComfyUI was not restarted. Check the startup console for import errors. |
| `TextEncodeAceStepAudio1.5` missing | ComfyUI is older than v0.12 — update ComfyUI. |
| `'...' is not installed -> put it in models/...` | The XL files are not where the workflow looks: `diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors`, `text_encoders/qwen_0.6b_ace15.safetensors` + `qwen_4b_ace15.safetensors`, `vae/ace_1.5_vae.safetensors`. For the archived AIO workflow: `checkpoints/ace_step_1.5_turbo_aio.safetensors`. |
| Errors mentioning `clip_name2` / garbage text conditioning | `qwen_0.6b_ace15` must be in slot **1** (`clip_name1`); XL does not accept 4B in slot 1. |
| Tail starts too loud / quiet, or has a click | Raise `Join crossfade`, or use `repaint_tail` with `repaint_seconds` 2–4. Check that BPM/key match the source. |
| Continuation drifts into a different song | Turn **LM audio-code planning** off, write tags that describe the *current* arrangement, set the real BPM/key, keep `add_seconds` ≤ 60 per run and run the app again on the result. |
| Model ends the song before the requested length | The `duration` link is missing in your workflow — reconnect `total_seconds` → `TextEncodeAceStepAudio1.5.duration`. |
| Out of VRAM | XL needs ≥12 GB (≥20 GB comfortable): swap `qwen_4b_ace15` for `qwen_1.7b_ace15`, untick LM audio-code planning, use `acestep_v1.5_xl_turbo_bf16` (8 steps), or fall back to the archived 2B AIO workflow (~8 GB). The masked prefix is attended to at every step, so a 10-minute source is much heavier than a 2-minute one. |
| Long source track is very slow | Continue a tail instead: cut the last ~60 s in any editor, continue that file, then join it to the rest (the join is what the crossfade is for). |
| Upload fails on a remote server | ComfyUI must be reachable over HTTP; if it sits behind a path prefix, put the full URL (including the path) into **Server**. |
| Result is FLAC and large | `SaveAudioAdvanced` writes FLAC in the shipped workflows; change its `format` widget to `mp3` (or swap in `SaveAudio`) and keep `filename_prefix` — the app picks up any audio in the history output. |

## 9. Other models / packs

The app patches nodes by **class name**, so it also works with workflows built on
`starsFriday/ComfyUI-ACEStep` (`ACEStep15XLExtendAudio` + `ACEStep15XLExtendLatent`,
`ACEStep15XLTextEncode`) — point **Workflow...** at such an API workflow and the same fields
are applied. The XL split layout (`UNETLoader` + `DualCLIPLoader` + `VAELoader`) is what the
default workflow uses; `CheckpointLoaderSimple` (the archived AIO workflow) and any other
ACE-Step 1.5 loader combination work the same way in the app. See [RESEARCH.md](RESEARCH.md) §4 for the list of existing continuation
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
