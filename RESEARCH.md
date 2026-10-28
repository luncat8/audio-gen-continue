# Continuing existing music locally — models, mechanisms, ComfyUI workflows

Research notes for `audio_continue_tk.py`. Everything below was checked against the
current ComfyUI source (`Comfy-Org/ComfyUI`), the official workflow templates and the
model repositories. ComfyUI moves fast; node names/limits can change.

**Short answer**

| Question | Answer |
| --- | --- |
| Can ComfyUI continue an existing track? | Yes — with **ACE-Step 1.5** + a masked-latent "extend" step. It is the only current local music model that is (a) natively supported in ComfyUI, (b) trained for repaint/extend style masked generation, and (c) good enough quality-wise to glue onto a real song. |
| Is YuE2 a candidate? | For *quality of vocals*, yes; for *continuation*, no. YuE2 continues at **score/semantic-token level**, not waveform level, and ComfyUI's YuE2 implementation always closes the score (`[ABC_END]`) before rendering, so score continuation is not reachable from ComfyUI at all. Third-party forks (audio.cpp, HOT-Step-CPP, YuE2-Studio, Wan2GP) implement "make it longer" by replaying a semantic-token prefix; none of that is exposed in the ComfyUI node. |
| Is Stable Audio a candidate? | Yes for **instrumental** music: Stable Audio 3 (medium) is natively trained for inpainting/continuation and ComfyUI's `StableAudio3` model *does* consume a mask + latent (`concat_cond`), so a latent-mask workflow is conceptually the *right* path there. But the ComfyUI integration is text-to-audio only (6:20 cap, no vocals, medium checkpoint needs a CUDA GPU + FlashAttention 2), so it is a secondary option, not the default. |
| Is "VAE-encode → KSampler → decode" enough? | Not by itself — your instinct is right, but the reason is subtle: it *works* as a mechanism only when (1) an explicit **noise mask** pins the original part, (2) sampling runs at **denoise = 1.0** on the masked region, and (3) the **length conditioning** describes the new total duration. Blindly encoding a whole track and re-diffusing at `denoise 0.4–0.7` destroys it. With those three conditions, masked latent continuation is exactly how "extend" is implemented for ACE-Step 1.5 today. |

---

## 1. Four different things are all called "continue"

| Mechanism | What actually happens | Which models do it natively |
| --- | --- | --- |
| **Token / score continuation** | The autoregressive planner is given a token prefix (ABC score or semantic codes) and keeps writing | YuE2 (score + semantic prefix), YuE v1 (`YuE-extend`), MiniMax Music 3 (state replay) |
| **Audio-prompt continuation** | The model gets the last N seconds as an audio prompt and predicts the next window | MusicGen (`generate_continuation`), Stable Audio 3, ACE-Step |
| **Masked latent inpainting** | Latent is extended with empty frames, a binary mask marks the new region, the DiT denoises only there while attending to the fixed prefix | ACE-Step 1.0/1.5 (repaint/extend task), Stable Audio 3 (inpaint mode) |
| **Repaint / edit** | A *time range inside* the track is regenerated (fix a bad bar, redo the join) | ACE-Step 1.5, Stable Audio 3 |

Only the last two are reachable from a ComfyUI `KSampler` graph, and they are the only
ones this project uses.

---

## 2. Model comparison for "continue my file"

| Model | Local? | Continuation support | Vocals | Max length | Quality for stitching onto real music | ComfyUI |
| --- | --- | --- | --- | --- | --- | --- |
| **ACE-Step 1.5** (2B turbo / 4B XL) | yes, ~4–8 GB VRAM (LM off), 10.0 GB AIO checkpoint | repaint/extend task trained with `src_latents + chunk_masks`; also official "infinite duration" via repeated repaint | **yes**, lyrics + 50 languages | 10 min | **best available local option** — musical, keeps tempo/key if you tell it | **native** (v0.12+): `TextEncodeAceStepAudio1.5`, `EmptyAceStep1.5LatentAudio`, `ModelSamplingAuraFlow` |
| ACE-Step 1.0 (3.5B) | yes | `ACEStepExtend` node exists in `billwuhao/ComfyUI_ACE-Step` (pipeline `task="extend"`, left/right seconds) | yes | ~4 min | decent, clearly below 1.5 | native `TextEncodeAceStepAudio` + custom nodes |
| **Stable Audio 3** medium (1.4B) / small (433M) | yes, medium needs CUDA + FA2; small runs on CPU | **native inpainting/continuation** (`inpaint_mask_start_seconds` = source length, longer duration) | **no intelligible vocals** | 6:20 (medium) / 2:00 (small) | good timbre/atmosphere, weaker long-form structure; 6:20 hard cap is a real limit | native model (`StableAudio3`, consumes `denoise_mask` + latent) but **only text-to-audio is shipped as a template**; also a reported VAE round-trip bug (#16087) |
| **YuE2** (3B) | yes, ~24 GB unquantized | score-prefix / semantic-prefix only, and **not in the ComfyUI node** | **best open vocals** | 5:30 hard cap | great for "another song in this style", cannot append to an existing waveform | native nodes (`YuE2GenerateABC`, `YuE2GenerateMusic`, `EmptyYuE2LatentAudio`); tokenizer appends `[ABC_END]` → no continuation |
| YuE v1 | yes | `YuE-extend` (Mozer) and YuE-UI do continuation via audio prompt + incremental generation | yes | ~5 min | superseded by YuE2 | not in core |
| MiniMax Music 3 | yes (core `MiniMaxMusic3` model) | state-replay "Extend (continue)": `diskrotrepo/comfyui-minimax-music3` | yes | long songs | good, but heavier and license-gated | custom nodes; core has `MiniMaxMusic3TextEncode` / `EmptyMiniMaxMusic3LatentAudio` |
| MusicGen small/medium | yes, small ~1 GB VRAM | `generate_continuation` (audio prompt, 30 s windows) | no | 30 s window, chainable | 2023-era quality, smeary tails — fine for ambience, not for songs | `eigenpunk/ComfyUI-audio` (`MusicgenGenerate` with continuation) |
| Stable Audio Open 1.0 | yes | no (fixed ~47 s; `seconds_start` conditioning only) | no | 47 s | not usable for continuation | native (legacy) |

**Verdict: use ACE-Step 1.5 XL (SFT, split files) — that is the default workflow of this
repo since the XL release.** The 2B turbo AIO checkpoint still works and its workflow is
archived in `workflows/archive/` for people with <12 GB VRAM. Keep Stable Audio 3 medium
as the instrumental-only fallback if ACE-Step's continuation drifts stylistically.

Why XL: it is the same ACE-Step 1.5 architecture (same VAE, same text encoders, same
`TextEncodeAceStepAudio1.5` / `KSampler` / `ModelSamplingAuraFlow` nodes, same
repaint/extend training task) with a 4B DiT instead of 2B — better prompt adherence,
cleaner timbre and fewer metallic artefacts in generated tails. Nothing in the
continuation mechanism changes: mask + `denoise 1.0` + `duration` conditioning are
handled by ComfyUI core and the two custom nodes, not by the checkpoint. Only the
**loader layout** (split files instead of the AIO checkpoint) and the **sampler numbers**
(SFT = 50 steps / CFG 7, turbo = 8 steps / CFG 1) differ.

| Variant | File (`models/diffusion_models/`) | Steps | CFG | Notes |
| --- | --- | --- | --- | --- |
| **xl_sft** (default) | `acestep_v1.5_xl_sft_bf16.safetensors` | 50 | 7.0 | best quality, slowest; official template numbers |
| xl_turbo | `acestep_v1.5_xl_turbo_bf16.safetensors` | 8 | 1.0 | ~6× faster, slightly less refined |
| xl_base | `acestep_v1.5_xl_base_bf16.safetensors` | 50 | 7.0 | non-distilled base |
| turbo AIO (archived) | `models/checkpoints/ace_step_1.5_turbo_aio.safetensors` | 8 | 1.0 | 2B, ~8 GB VRAM |

Text encoders are the same in all cases: `DualCLIPLoader` type `ace` with
`qwen_0.6b_ace15` (slot 1, mandatory) + `qwen_4b_ace15` (slot 2; `qwen_1.7b_ace15` if VRAM
is tight). VAE: `ace_1.5_vae.safetensors`, i.e. the same 48 kHz / 25 fps latent space, so
`AudioContinueExtend`/`AudioContinueStitch` need no changes for XL.

VRAM: XL DiT ≈10 GB bf16, 4B text encoder ≈8 GB — ≥12 GB with ComfyUI offloading, ≥20 GB
comfortable. That is the only real trade-off against the archived 2B AIO workflow.

---

## 3. Why naive "VAE encode → KSampler" is wrong, and what makes it right

Encoding a finished track into latents and denoising it again is *not* continuation, it
is destructive audio-to-audio: at `denoise < 1.0` the sampler mixes fresh noise into the
whole clip, and with `denoise = 1.0` it re-invents everything. The model has no idea that
"the first 60 s must stay".

What makes masked continuation legitimate:

1. **A mask decides what is generated.** `AudioContinueExtend` returns
   `{"samples": [prefix | empty frames], "noise_mask": [0 … 0 | 1 … 1]}`. In
   `comfy.samplers.KSamplerX0Inpaint.__call__` the unmasked half is re-imposed from the
   encoded latent at every step:
   `out = out * mask + latent_image * (1 - mask)`. The prefix is therefore *bit-exact*
   the original encoding, not a re-generation.
2. **`denoise = 1.0`.** The new region must be full noise, otherwise it starts from
   "silence-ish latent + noise" and the model tries to continue from nothing.
3. **Length conditioning.** ACE-Step 1.5's text encoder takes `duration`; the DiT is
   trained to render exactly that much. If you leave it at 120 s while generating 200 s,
   the model wraps the song up early (or runs past the end). That is why the extend node
   outputs `total_seconds` and the shipped workflow wires it into
   `TextEncodeAceStepAudio1.5.duration`.
4. **Model support matters.** ACE-Step 1.5 was trained with repaint-style conditioning
   (`context_latents = cat([src_latents, chunk_masks])` in
   `comfy/ldm/ace/ace_step15.py`). Stable Audio 3 also takes mask+latent
   (`StableAudio3.concat_cond`). A model without any masked/context training (e.g. Stable
   Audio Open 1.0) will produce a jump at the seam no matter what the sampler does.
5. **Seam handling.** Even with a perfect mask the decoded prefix goes through the VAE,
   so it is not sample-identical to your FLAC. `AudioContinueStitch` throws that part away,
   keeps your original file and crossfades into the generated tail.

### ComfyUI caveat you should know about

In ComfyUI, `comfy/model_base.py::ACEStep15.forward` reaches the DiT with
`src_latents = x` and `chunk_masks = torch.ones_like(x)` — the repaint mask is **never
passed into the model**. So the model thinks it is doing a normal full generation and only
"sees" your prefix through self-attention over the noisy latent. It works (this is what
every ACE-Step 1.5 extend node does today), but the model is not told "keep this part",
which is why long continuations can drift in level/timbre and why the join sometimes needs
`mask_mode = repaint_tail` (mask the last 2–4 s too, let the model re-render the join, then
crossfade back into the original).

---

## 4. Existing ComfyUI workflows for this task

| Workflow / pack | Model | What it gives you | Notes |
| --- | --- | --- | --- |
| **Official template** `audio_ace_step1_5_xl_sft` (Comfy-Org/workflow_templates) | ACE-Step 1.5 XL | text→song, 50 steps, cfg 7, `euler/simple`, shift 3 | the base graph the default XL workflow is built from; no extend node |
| **Official template** `audio_ace_step_1_5_checkpoint` (same repo) | ACE-Step 1.5 (2B AIO) | text→song, 8 steps, cfg 1 | base graph of the archived AIO workflow |
| **starsFriday/ComfyUI-ACEStep** | ACE-Step 1.5 / XL | `ACEStep15XLExtendAudio` (append silence + mask), `ACEStep15XLRepaintAudio`, `ReferenceLatent`, example workflow `workflows/ace_step_15xl_extend_audio.json` | **closest existing "continuation" workflow**; does *not* stitch the original back, and the reference-latent node needs the same mask plumbing; requires the pack |
| **billwuhao/ComfyUI_ACE-Step** | ACE-Step 1.0 | `ACEStepExtend` (left/right seconds, overlapped decode) → calls the official pipeline `task="extend"` | real model-native extend (the pipeline does the repaint boundaries for you) but ACE 1.0 quality |
| **ComfyUI_RyanOnTheInside** | ACE-Step 1.5 | native guiders incl. **Extend/Repaint** and Cover, plus audio KSamplers | good if you already use that pack |
| **ComfyUI-AceMusic** | ACE-Step 1.5 | 15 nodes: cover, repaint, extend, edit, retake | broad feature coverage |
| **scromfyUI-AceStep** | ACE-Step 1.5 | 30+ nodes, audio KSamplers, masking/inpainting | kitchen sink |
| **ComfyUI-ACEStep** issue #1 | — | report of "extend produces silence" | worth reading before you debug your own masks |
| **diskrotrepo/comfyui-minimax-music3** | MiniMax Music 3 | `Extend (continue)` with `continue_from_seconds` (negative = back up), song states | best "Suno-style" extend UX outside ACE-Step |
| **ttulttul/ComfyUI-Minimax-H3-Continuation** | MiniMax H3 | rolling-overlap continuation for joint video+audio latents | same idea (hidden overlap + append) applied to H3 |
| **eigenpunk/ComfyUI-audio** (`MusicgenGenerate`) | MusicGen / AudioGen | `continuation` flag (audio prompt) | old but genuinely natively supported |
| **endman100/ComfyUI-YuE2** | YuE2 | composition/edit nodes; README explicitly says *no* waveform inpainting/continuation | confirms the YuE2 gap |
| **filliptm/ComfyUI-FL-YuE2** | YuE2 | generation + editable piano roll | no waveform continuation |
| **Comfy-Org/workflow_templates** `audio_stable_audio_3_*` | Stable Audio 3 | text→audio subgraph (medium/base) | no inpaint/continuation graph shipped yet |
| **ACE-Step 1.5 official repo** | ACE-Step 1.5 | "infinite duration" recipe = repeated **Repaint** at the end of the previous segment | the reference way to do a long continuation |

Nothing shipped combines *masked extend* + *original-file stitching* + *GUI*, which is
what `workflows/` + `audio_continue_tk.py` add here.

---

## 5. What this repo does, exactly

```
LoadAudio ──┬─────────────────────────────────────────────┐
            │                                             │
            └─► AudioContinueExtend(VAE, add_seconds) ─┐    │
                    latent = [encoded prefix | zeros]   │    │
                    noise_mask = [0…0 | 1…1]            │    │
                    total_seconds ──────┐               │    │
                                        ▼               ▼    ▼
                              TextEncodeAceStepAudio1.5   KSampler (denoise 1.0)
                                        │                   │
                                 ConditioningZeroOut        │
                                        └──── negative ─────┤
                                                            ▼
                                            VAEDecodeAudio ─► AudioContinueStitch(original, decoded, crossfade) ─► SaveAudio
                                                                                                     └─ output/audio-continue/
```

Verified numbers for ACE-Step 1.5 in ComfyUI (from the source, not guessed):

* audio is 48 kHz, the VAE (`AudioOobleckVAE`, strides `2·4·4·6·10 = 1920`) gives
  **25 latent frames per second**, latent shape `[B, 64, T]` (`latent_formats.ACEAudio15`);
* `EmptyAceStep1.5LatentAudio` uses the same 25 fps (`round(seconds*48000/1920)`);
* the VAE round-trips through `vae.audio_sample_rate = 48000`, so an mp3/wav source is
  resampled by ComfyUI before encoding — lengths stay in seconds, not samples;
* turbo = **8 steps, cfg 1.0, euler + simple, `ModelSamplingAuraFlow` shift 3.0**;
* **XL SFT/base = 50 steps, cfg 7.0**, euler + simple, shift 3.0, `ConditioningZeroOut` as
  negative (the official `audio_ace_step1_5_xl_sft` template — CFG 7 makes the negative
  branch actually matter, unlike turbo's cfg 1); XL turbo keeps 8 steps / cfg 1.0;
* `ConditioningZeroOut` is used as the negative because cfg 1 makes it irrelevant;
* `filename_prefix = "audio-continue/audio"` → `ComfyUI/output/audio-continue/audio_00001_.flac`.

Practical settings that matter more than the sampler:

* **BPM and key of the source** — ACE-Step conditions on both; giving it 96 / A minor
  instead of the default 120 / C major is the single biggest continuity win.
* **tags** — describe the *existing* arrangement, not a new song ("continues the existing
  song, same instruments, same mix, no intro, no build-up").
* **empty lyrics** for instrumental sources, otherwise the model may start singing.
* **`generate_audio_codes`** (LM planning) helps detail but plans a *new* track; if the
  seam drifts, turn it off so the DiT relies purely on the prefix.
* **`mask_mode = repaint_tail`, `repaint_seconds` 2–4 s** when the join is audible.
* **crossfade 0.5–2 s** — the stitch node keeps your original up to that point.

---

## 6. If ACE-Step quality is not enough

1. **Stable Audio 3 medium, inpainting route (instrumental only).** Encode the source with
   the SA3 VAE, concatenate silence to the requested total length, set a `noise_mask` for
   the new part, `denoise = 1.0`. `StableAudio3.concat_cond` feeds `1 - mask` and the
   latent into the DiT, i.e. the model gets the *real* inpainting conditioning — this is
   more faithful to the training than what ComfyUI does for ACE-Step. Open issues: the
   integration is text-to-audio only, the 6:20 cap, and VAE round-trip bug #16087. Worth
   trying with the same `AudioContinueExtend`/`AudioContinueStitch` nodes and
   `stable_audio_3_medium_base.safetensors` (+ `t5gemma_b_b_ul2` text encoder).
2. **MiniMax Music 3 extend nodes** if you want Suno-style "replay + branch" semantics and
   do not mind the bigger checkpoint.
3. **YuE2 outside ComfyUI** (`YuE2Studio`, `audio.cpp` `yue2_semantic_prefix_continue`,
   `HOT-Step-CPP`, `Wan2GP` score extend) when you need *vocals that sound real* and are
   willing to accept "a new song that opens with your clip" instead of an exact splice.
4. **Chain short segments.** ACE-Step 1.5's own recipe for long songs is repeated repaint;
   the app can do that by running it twice (second run on the already continued file).

---

## 7. Sources

ComfyUI code (checked directly):

* `comfy_extras/nodes_ace.py`, `comfy_extras/nodes_audio.py`, `comfy_extras/nodes_yue2.py`
* `comfy/model_base.py` (`ACEStep15`, `StableAudio3`, `YuE2`), `comfy/ldm/ace/ace_step15.py`
* `comfy/samplers.py` (`KSamplerX0Inpaint`), `comfy/sample.py`, `comfy/sd.py` (VAE detection)
* `comfy/latent_formats.py` (`ACEAudio15`, `StableAudio3`, `YuE2`)
* `execution.py::get_input_data` (link into a widget input is resolved like any other link)

Official templates / docs:

* https://github.com/Comfy-Org/workflow_templates — `audio_ace_step_1_5_checkpoint.json`, `audio_ace_step_1_5_split.json`, `audio_stable_audio_3_medium_base.json`
* https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1-5 (also lists Cover/Repaint as "coming soon" to ComfyUI)
* https://docs.comfy.org/tutorials/audio/stable-audio/stable-audio-3
* https://github.com/Comfy-Org/ace_step_1.5_ComfyUI_files (AIO checkpoint 10.0 GB + split
  files: XL base/sft/turbo ≈9.97 GB, `qwen_0.6b_ace15` 1.19 GB, `qwen_4b_ace15` 8.38 GB,
  `ace_1.5_vae` 0.34 GB)

Models / workflows:

* https://github.com/ace-step/ACE-Step-1.5 — infinite duration via repeated repaint
* https://github.com/starsFriday/ComfyUI-ACEStep — extend/repaint/reference nodes + example workflows
* https://github.com/billwuhao/ComfyUI_ACE-Step — `ACEStepExtend` for ACE-Step 1.0
* https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside — native ACE-Step 1.5 extend/repaint guiders
* https://github.com/ace-step/awesome-ace-step — index of ComfyUI integrations
* https://github.com/diskrotrepo/comfyui-minimax-music3 — extend/continue with state replay
* https://github.com/eigenpunk/ComfyUI-audio — MusicGen continuation
* https://github.com/endman100/ComfyUI-YuE2 — states "no waveform inpainting/continuation"
* https://github.com/Stability-AI/stable-audio-3 — inpainting/continuation (`inpaint_mask_start_seconds`, duration)
* https://github.com/multimodal-art-projection/YuE — YuE2, score-prefix editing; `YuE-extend` (Mozer) for v1
* https://github.com/0xShug0/audio.cpp/pull/641 — `yue2_semantic_prefix_continue`
* https://github.com/hybra/YuE2Studio , https://github.com/scragnog/HOT-Step-CPP/releases/tag/v1.4.0 , https://github.com/timoncool/YuE2-Studio — "make it longer" implementations
* https://github.com/Comfy-Org/ComfyUI/issues/16087 — Stable Audio 3 VAE round-trip bug
