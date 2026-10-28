#!/usr/bin/env python3
"""
Generate the workflow files used by this project from clean graph definitions.

    python3 tools/build_workflows.py

Outputs (relative to the repo root):

    workflows/music_caption_ui.json                       <- standalone captioner + viewer (DSP + MOSS-Music Loader)
    workflows/music_caption_api.json
    workflows/ace15xl_audio_continue_caption_ui.json      <- captioner + ACE-Step 1.5 XL continuation in one graph
    workflows/ace15xl_audio_continue_caption_api.json
    workflows/ace15xl_audio_continue_ui.json              <- default continuation graph (what the Tk app posts)
    workflows/ace15xl_audio_continue_api.json
    workflows/archive/ace15_audio_continue_turbo_aio_ui.json   <- archived: 2B turbo AIO
    workflows/archive/ace15_audio_continue_turbo_aio_api.json
"""

import json
import os
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "workflows")
ARCHIVE_DIR = os.path.join(OUT_DIR, "archive")

SAVE_PREFIX = "audio-continue/audio"

# XL (default) -------------------------------------------------------------------------
XL_UNET = "acestep_v1.5_xl_sft_bf16.safetensors"
XL_CLIP_LOW = "qwen_0.6b_ace15.safetensors"
XL_CLIP_HIGH = "qwen_4b_ace15.safetensors"
XL_VAE = "ace_1.5_vae.safetensors"

# archived AIO -------------------------------------------------------------------------
AIO_CKPT = "ace_step_1.5_turbo_aio.safetensors"

# MOSS-Music ---------------------------------------------------------------------------
MOSS_MODEL_NAME = "MOSS-Music-8B-Instruct"
MOSS_HF_URL = "https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct"
MOSS_HF_THINKING_URL = "https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking"
MOSS_DEFAULT_SERVER = "http://127.0.0.1:30000"

DEFAULT_TAGS = (
    "instrumental, continues the existing song, same tempo, same key, same instruments, "
    "same mix and reverb, natural musical development, smooth transition, studio quality"
)
CONTINUE_EXTRA_TAGS = (
    "continues the existing song, same tempo, same key, same instruments, "
    "same mix and reverb, natural musical development, smooth transition, studio quality"
)
DEFAULT_LYRICS = ""

CAPTION_NOTE = f"""## Music Captioner — DSP + MOSS-Music 8B

### 1. Works immediately (even before downloading MOSS-Music!)
1. Upload a song in **1. Load Music File** and click **Queue**.
2. **3a. DSP Measure** needs **no model download** and measures **BPM** (+ octave candidates), **Key / Scale** (+ Camelot code + confidence), **Vocal presence & register**, **Loudness**, **Brightness**, and a **5 s section map**.
3. `when_missing` on **2. MOSS-Music Loader** defaults to `skip (DSP only)`, so pressing Queue before downloading MOSS succeeds cleanly and shows all DSP data in **6a. Analysis Report**.

### 2. Required HuggingFace model for full AI captioning (MOSS-Music 8B)
To unlock full genre / instrumentation / mix description, accurate vocal gender + register + timbre, chord progression, song structure and singing-voice lyrics ASR:

* **Required model (~17 GB, Apache-2.0):**
  [{MOSS_HF_URL}]({MOSS_HF_URL})
* *(Optional chain-of-thought variant):*
  [{MOSS_HF_THINKING_URL}]({MOSS_HF_THINKING_URL})

Place the HuggingFace repo folder inside ComfyUI at:
```
ComfyUI/models/moss_music/MOSS-Music-8B-Instruct/
```

**CLI download command:**
```bash
hf download OpenMOSS-Team/MOSS-Music-8B-Instruct \\
  --local-dir ComfyUI/models/moss_music/MOSS-Music-8B-Instruct
```
*(or set **`auto_download = true`** on **2. MOSS-Music Loader** and press Queue once).*

### 3. Viewing & using the outputs
* **6a. Analysis Report** — full breakdown of BPM, Key, Camelot, Vocals, Chords, Structure, Loudness, Sections and MOSS status.
* **6b. Prompt / Tags** — merged tag string ready for `TextEncodeAceStepAudio1.5` or MiniMax Music.
* **6c. Lyrics** — transcribed singing lyrics (when MOSS-Music is installed and vocals are present).
* Individual typed slots (`bpm`, `keyscale`, `vocal_type`, `chords`, `structure`, `tags`, `lyrics`) are exposed on **4. Build Prompt** and **5. Music Caption Viewer** for wiring into any audio generation graph (see `workflows/ace15xl_audio_continue_caption_ui.json`).
"""

XL_CAPTION_NOTE = f"""## Audio Continue + Music Captioner — ACE-Step 1.5 XL + MOSS-Music 8B

This workflow combines **music captioning / measurement** (bottom-left) with **ACE-Step 1.5 XL continuation** (top-right):
* **DSP Measure** + **MOSS-Music Loader** analyse your uploaded file, show the full report in **Analysis Report**, and feed **`tags`** and **`bpm`** straight into **Style + lyrics** (`TextEncodeAceStepAudio1.5`).
* Note: ComfyUI's `keyscale` widget on `TextEncodeAceStepAudio1.5` is a dropdown — read the detected key in **Analysis Report** and select it on `Style + lyrics` (or run through `audio_continue_tk.py`, which sets `keyscale` automatically).

### Required models
1. **ACE-Step 1.5 XL SFT** ([Comfy-Org/ace_step_1.5_ComfyUI_files](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/tree/main/split_files)):
   * `ComfyUI/models/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors`
   * `ComfyUI/models/text_encoders/qwen_0.6b_ace15.safetensors` (slot 1)
   * `ComfyUI/models/text_encoders/qwen_4b_ace15.safetensors`   (slot 2)
   * `ComfyUI/models/vae/ace_1.5_vae.safetensors`
2. **MOSS-Music 8B Instruct (optional, ~17 GB, Apache-2.0):**
   * Download: [{MOSS_HF_URL}]({MOSS_HF_URL})
   * Put in: `ComfyUI/models/moss_music/MOSS-Music-8B-Instruct/`
   * Command: `hf download OpenMOSS-Team/MOSS-Music-8B-Instruct --local-dir ComfyUI/models/moss_music/MOSS-Music-8B-Instruct`
   * Without MOSS-Music downloaded yet, `when_missing = skip (DSP only)` keeps the workflow running on DSP BPM/key/vocal measurements, and `keep_loaded = false` unloads MOSS from VRAM before ACE-Step samples.
"""

XL_NOTE = f"""## Audio Continue - ACE-Step 1.5 XL (SFT)

1. Install the **comfyui_audio_continue** and **comfyui_music_caption** custom nodes (this repo) plus the split XL files ([Comfy-Org/ace_step_1.5_ComfyUI_files](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/tree/main/split_files)):
   * `models/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors`
   * `models/text_encoders/qwen_0.6b_ace15.safetensors`  (must be clip slot 1)
   * `models/text_encoders/qwen_4b_ace15.safetensors`    (clip slot 2)
   * `models/vae/ace_1.5_vae.safetensors`
2. Load your track in **Load Audio** (upload button).
3. Set **add_seconds** on *Audio Continue (extend latent)* = how much new music to add.
4. Describe the music in **tags** (and set the BPM / key of the source — or open `workflows/music_caption_ui.json` / `workflows/ace15xl_audio_continue_caption_ui.json` to measure & caption automatically with DSP + [{MOSS_MODEL_NAME}]({MOSS_HF_URL})).
5. Queue. The **duration** input of the text encoder is wired to `total_seconds`.

Sampler: **50 steps, CFG 7.0**, euler + simple, `ModelSamplingAuraFlow` shift 3.0 - that is
what `xl_sft` wants (the `xl_turbo` file wants 8 steps / CFG 1.0 instead).

How it works: the source is VAE-encoded, N seconds of *empty* latent frames are
appended and masked, and the sampler only denoises the masked part. The unmasked
prefix is pinned to the encoded original every step, so the model continues the
actual recording instead of inventing a new song. **Stitch** then puts your exact
original file back and crossfades into the generated tail.

VRAM: the XL DiT is ~10 GB in bf16 and the 4B text encoder another ~8 GB, so expect
>=12 GB with ComfyUI offloading and >=20 GB for a comfortable run. On a smaller card
switch `qwen_4b` for `qwen_1.7b` or use `acestep_v1.5_xl_turbo_bf16.safetensors`
(8 steps, CFG 1.0). If the join sounds rough: raise `crossfade_seconds`, or set
`mask_mode` to `repaint_tail` with `repaint_seconds` ~2-4 s so the model re-renders
the seam.
"""

AIO_NOTE = """## Audio Continue - ACE-Step 1.5 turbo (AIO, archived)

Archived workflow. Install the model
`models/checkpoints/ace_step_1.5_turbo_aio.safetensors` and load this file with
**Workflow...** in the app (or drag it onto the ComfyUI canvas).

Sampler: **8 steps, CFG 1.0**, euler + simple, `ModelSamplingAuraFlow` shift 3.0.
That is the turbo recipe; do not copy the XL SFT numbers (50 / 7.0) here.

Same mechanics as the XL workflow: LoadAudio -> AudioContinueExtend (masked tail) ->
KSampler -> VAEDecodeAudio -> AudioContinueStitch -> SaveAudio.
"""

ANALYZE_OUT_SPECS = [
    ("tags", "STRING"),
    ("bpm", "INT"),
    ("keyscale", "STRING"),
    ("key_confidence", "FLOAT"),
    ("bpm_confidence", "FLOAT"),
    ("vocal_presence", "STRING"),
    ("vocal_type", "STRING"),
    ("duration_seconds", "FLOAT"),
    ("analysis_json", "STRING"),
]

MOSS_OUT_SPECS = [
    ("caption", "STRING"),
    ("lyrics", "STRING"),
    ("vocal", "STRING"),
    ("bpm", "INT"),
    ("keyscale", "STRING"),
    ("instrumental", "BOOLEAN"),
    ("structure", "STRING"),
    ("moss_json", "STRING"),
]

MERGE_OUT_SPECS = [
    ("tags", "STRING"),
    ("lyrics", "STRING"),
    ("bpm", "INT"),
    ("keyscale", "STRING"),
    ("instrumental", "BOOLEAN"),
    ("vocal_type", "STRING"),
    ("summary_json", "STRING"),
]

VIEW_OUT_SPECS = [
    ("report", "STRING"),
    ("tags", "STRING"),
    ("caption", "STRING"),
    ("lyrics", "STRING"),
    ("bpm", "INT"),
    ("keyscale", "STRING"),
    ("vocal_type", "STRING"),
    ("chords", "STRING"),
    ("structure", "STRING"),
]


# --------------------------------------------------------------------------------------
# tiny graph builder
#   node(id, class_type, title, pos, size, widgets, inputs, outputs)
#     inputs:  [(name, type, origin_id, origin_slot, is_widget)]
#     outputs: [type_str | (slot_name, type_str), ...]
# --------------------------------------------------------------------------------------


class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []

    def node(self, nid, ctype, title, pos, size, widgets=None, inputs=None, outputs=None):
        self.nodes.append(
            {
                "id": nid,
                "type": ctype,
                "title": title,
                "pos": pos,
                "size": size,
                "widgets": widgets or [],
                "inputs": inputs or [],
                "outputs": outputs or [],
            }
        )
        return nid


def _common_nodes(g, extend_vae, encode_clip, sampler_settings, extra_encode_links=None):
    """Load audio -> extend -> encode -> sample -> decode -> stitch -> save."""
    steps, cfg = sampler_settings

    g.node(3, "LoadAudio", "Music file", [40, 560], [420, 140], ["input.wav"], [], ["AUDIO"])

    g.node(
        4,
        "AudioContinueExtend",
        "Add seconds",
        [40, 740],
        [420, 170],
        [30.0, "generate_new_only", 0.0],
        [("audio", "AUDIO", 3, 0, False), ("vae", "VAE", extend_vae, 0, False)],
        ["LATENT", "FLOAT", "FLOAT", "FLOAT", "INT"],
    )

    encode_inputs = [("clip", "CLIP", encode_clip, 0, False)]
    if extra_encode_links:
        encode_inputs.extend(extra_encode_links)
    encode_inputs.append(("duration", "FLOAT", 4, 3, True))

    g.node(
        5,
        "TextEncodeAceStepAudio1.5",
        "Style + lyrics",
        [500, 300],
        [600, 540],
        [
            DEFAULT_TAGS,
            DEFAULT_LYRICS,
            31,
            "randomize",
            120,
            120.0,
            "4",
            "en",
            "C major",
            True,
            2.0,
            0.85,
            0.9,
            0,
            0.0,
        ],
        encode_inputs,
        ["CONDITIONING"],
    )

    g.node(
        6,
        "ConditioningZeroOut",
        "Negative (zero out)",
        [1140, 300],
        [260, 60],
        [],
        [("conditioning", "CONDITIONING", 5, 0, False)],
        ["CONDITIONING"],
    )

    g.node(
        7,
        "KSampler",
        "Sampler",
        [1140, 400],
        [330, 290],
        [31, "randomize", steps, cfg, "euler", "simple", 1.0],
        [
            ("model", "MODEL", 2, 0, False),
            ("positive", "CONDITIONING", 5, 0, False),
            ("negative", "CONDITIONING", 6, 0, False),
            ("latent_image", "LATENT", 4, 0, False),
        ],
        ["LATENT"],
    )

    g.node(
        8,
        "VAEDecodeAudio",
        "Decode",
        [1520, 400],
        [250, 60],
        [],
        [("samples", "LATENT", 7, 0, False), ("vae", "VAE", extend_vae, 0, False)],
        ["AUDIO"],
    )

    g.node(
        9,
        "AudioContinueStitch",
        "Stitch original + continuation",
        [1520, 500],
        [330, 110],
        [1.0, "highest"],
        [("source_audio", "AUDIO", 3, 0, False), ("continued_audio", "AUDIO", 8, 0, False)],
        ["AUDIO"],
    )

    g.node(
        10,
        "SaveAudioAdvanced",
        "Save",
        [1900, 500],
        [420, 130],
        [SAVE_PREFIX, "flac"],
        [("audio", "AUDIO", 9, 0, False)],
        ["AUDIO"],
    )


def caption_graph():
    """Standalone ComfyUI music captioner & analyzer workflow (DSP + MOSS-Music Loader)."""
    g = Graph()

    g.node(
        1,
        "LoadAudio",
        "1. Load Music File",
        [40, 80],
        [440, 150],
        ["input.wav"],
        [],
        [("AUDIO", "AUDIO")],
    )

    g.node(
        2,
        "MusicCaptionMOSSLoader",
        "2. MOSS-Music Loader (optional 8B model)",
        [40, 270],
        [440, 260],
        [
            MOSS_MODEL_NAME,
            "auto",
            "auto",
            "auto",
            False,
            False,
            "skip (DSP only)",
            MOSS_DEFAULT_SERVER,
        ],
        [],
        [("moss_model", "MOSS_MODEL"), ("status", "STRING")],
    )

    g.node(
        3,
        "MusicCaptionAnalyze",
        "3a. DSP Measure (BPM / Key / Vocals — no model)",
        [520, 80],
        [420, 260],
        [90.0, True, ""],
        [("audio", "AUDIO", 1, 0, False)],
        ANALYZE_OUT_SPECS,
    )

    g.node(
        4,
        "MusicCaptionMOSS",
        "3b. MOSS-Music 8B (caption / vocals / chords / lyrics)",
        [520, 380],
        [420, 340],
        [MOSS_DEFAULT_SERVER, "auto", "default", "path", "tail", 120.0, 1024, 0.0, 300.0, ""],
        [("audio", "AUDIO", 1, 0, False), ("moss_model", "MOSS_MODEL", 2, 0, False)],
        MOSS_OUT_SPECS,
    )

    g.node(
        5,
        "MusicCaptionMerge",
        "4. Build Prompt (merge DSP + MOSS)",
        [980, 120],
        [380, 240],
        ["dsp", "", False],
        [
            ("analysis_json", "STRING", 3, 8, False),
            ("moss_json", "STRING", 4, 7, False),
            ("lyrics", "STRING", 4, 1, False),
        ],
        MERGE_OUT_SPECS,
    )

    g.node(
        6,
        "MusicCaptionView",
        "5. Music Caption Viewer (all fields)",
        [980, 410],
        [380, 270],
        [True],
        [
            ("summary_json", "STRING", 5, 6, False),
            ("analysis_json", "STRING", 3, 8, False),
            ("moss_json", "STRING", 4, 7, False),
        ],
        VIEW_OUT_SPECS,
    )

    g.node(
        7,
        "PreviewAny",
        "6a. Analysis Report (BPM / Key / Vocals / Chords / Sections)",
        [1400, 80],
        [580, 440],
        [],
        [("source", "*", 6, 0, False)],
        [("STRING", "STRING")],
    )

    g.node(
        9,
        "PreviewAny",
        "6b. Prompt / Tags (for ACE-Step / MiniMax)",
        [1400, 560],
        [580, 200],
        [],
        [("source", "*", 6, 1, False)],
        [("STRING", "STRING")],
    )

    g.node(
        10,
        "PreviewAny",
        "6c. Lyrics (MOSS-Music ASR)",
        [1400, 800],
        [580, 220],
        [],
        [("source", "*", 6, 3, False)],
        [("STRING", "STRING")],
    )

    g.node(
        8,
        "MarkdownNote",
        "Read me — Download MOSS-Music & How to Use",
        [40, 570],
        [440, 560],
        [CAPTION_NOTE],
        [],
        [],
    )
    return g


def xl_caption_graph():
    """All-in-one graph: MOSS-Music Loader + Captioner + ACE-Step 1.5 XL Continuation."""
    g = Graph()
    g.node(
        1,
        "UNETLoader",
        "ACE-Step 1.5 XL SFT (diffusion model)",
        [40, 120],
        [420, 110],
        [XL_UNET, "default"],
        [],
        ["MODEL"],
    )
    g.node(
        12,
        "DualCLIPLoader",
        "Text encoders (Qwen 0.6B + 4B)",
        [40, 260],
        [420, 140],
        [XL_CLIP_LOW, XL_CLIP_HIGH, "ace", "default"],
        [],
        ["CLIP"],
    )
    g.node(13, "VAELoader", "ACE-Step 1.5 VAE", [40, 430], [420, 80], [XL_VAE], [], ["VAE"])

    g.node(
        2,
        "ModelSamplingAuraFlow",
        "Model sampling (shift 3)",
        [500, 120],
        [320, 90],
        [3.0, "flow"],
        [("model", "MODEL", 1, 0, False)],
        ["MODEL"],
    )

    # Captioner branch (nodes 14..19) wired from LoadAudio (node 3) into Style + lyrics (node 5)
    g.node(
        14,
        "MusicCaptionMOSSLoader",
        "MOSS-Music Loader (optional 8B model)",
        [40, 960],
        [420, 260],
        [
            MOSS_MODEL_NAME,
            "auto",
            "auto",
            "auto",
            False,
            False,
            "skip (DSP only)",
            MOSS_DEFAULT_SERVER,
        ],
        [],
        [("moss_model", "MOSS_MODEL"), ("status", "STRING")],
    )
    g.node(
        15,
        "MusicCaptionAnalyze",
        "DSP Measure (BPM / Key / Vocals)",
        [500, 900],
        [400, 250],
        [90.0, True, ""],
        [("audio", "AUDIO", 3, 0, False)],
        ANALYZE_OUT_SPECS,
    )
    g.node(
        16,
        "MusicCaptionMOSS",
        "MOSS-Music 8B Captioner",
        [500, 1190],
        [400, 330],
        [MOSS_DEFAULT_SERVER, "auto", "default", "path", "tail", 120.0, 1024, 0.0, 300.0, ""],
        [("audio", "AUDIO", 3, 0, False), ("moss_model", "MOSS_MODEL", 14, 0, False)],
        MOSS_OUT_SPECS,
    )
    g.node(
        17,
        "MusicCaptionMerge",
        "Build Continuation Prompt",
        [940, 900],
        [380, 240],
        ["dsp", CONTINUE_EXTRA_TAGS, False],
        [
            ("analysis_json", "STRING", 15, 8, False),
            ("moss_json", "STRING", 16, 7, False),
            ("lyrics", "STRING", 16, 1, False),
        ],
        MERGE_OUT_SPECS,
    )
    g.node(
        18,
        "MusicCaptionView",
        "Music Caption Viewer",
        [940, 1180],
        [380, 260],
        [True],
        [
            ("summary_json", "STRING", 17, 6, False),
            ("analysis_json", "STRING", 15, 8, False),
            ("moss_json", "STRING", 16, 7, False),
        ],
        VIEW_OUT_SPECS,
    )
    g.node(
        19,
        "PreviewAny",
        "Analysis Report (BPM / Key / Vocals / Chords)",
        [1360, 900],
        [540, 440],
        [],
        [("source", "*", 18, 0, False)],
        [("STRING", "STRING")],
    )

    _common_nodes(
        g,
        extend_vae=13,
        encode_clip=12,
        sampler_settings=(50, 7.0),
        extra_encode_links=[
            ("tags", "STRING", 17, 0, True),
            ("bpm", "INT", 18, 4, True),
        ],
    )

    g.node(11, "MarkdownNote", "Read me — Models & Setup", [1940, 900], [620, 520], [XL_CAPTION_NOTE], [], [])
    return g


def xl_graph():
    """ACE-Step 1.5 XL SFT: split diffusion model / dual text encoder / VAE."""
    g = Graph()
    g.node(
        1,
        "UNETLoader",
        "ACE-Step 1.5 XL SFT (diffusion model)",
        [40, 120],
        [420, 110],
        [XL_UNET, "default"],
        [],
        ["MODEL"],
    )
    g.node(
        12,
        "DualCLIPLoader",
        "Text encoders (Qwen 0.6B + 4B)",
        [40, 260],
        [420, 140],
        [XL_CLIP_LOW, XL_CLIP_HIGH, "ace", "default"],
        [],
        ["CLIP"],
    )
    g.node(13, "VAELoader", "ACE-Step 1.5 VAE", [40, 430], [420, 80], [XL_VAE], [], ["VAE"])

    g.node(
        2,
        "ModelSamplingAuraFlow",
        "Model sampling (shift 3)",
        [500, 120],
        [320, 90],
        [3.0, "flow"],
        [("model", "MODEL", 1, 0, False)],
        ["MODEL"],
    )
    _common_nodes(g, extend_vae=13, encode_clip=12, sampler_settings=(50, 7.0))
    g.node(11, "MarkdownNote", "Read me", [500, 900], [760, 500], [XL_NOTE], [], [])
    return g


def turbo_aio_graph():
    """Archived: ACE-Step 1.5 2B turbo, single all-in-one checkpoint."""
    g = Graph()
    g.node(
        1,
        "CheckpointLoaderSimple",
        "ACE-Step 1.5 turbo (AIO)",
        [40, 120],
        [420, 100],
        [AIO_CKPT],
        [],
        ["MODEL", "CLIP", "VAE"],
    )
    g.node(
        2,
        "ModelSamplingAuraFlow",
        "Model sampling (shift 3)",
        [500, 120],
        [320, 90],
        [3.0, "flow"],
        [("model", "MODEL", 1, 0, False)],
        ["MODEL"],
    )
    _common_nodes(g, extend_vae=1, encode_clip=1, sampler_settings=(8, 1.0))
    g.node(11, "MarkdownNote", "Read me", [500, 900], [760, 500], [AIO_NOTE], [], [])
    return g


# --------------------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------------------

# widget position -> API input name, per node class (None = frontend-only widget)
WIDGET_KEYS = {
    "CheckpointLoaderSimple": ["ckpt_name"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type", None],  # `device` stays default
    "VAELoader": ["vae_name"],
    # only `shift` is sent: `sampling` is an optional widget in newer builds and sending it
    # would break older ComfyUI versions
    "ModelSamplingAuraFlow": ["shift"],
    "LoadAudio": ["audio"],
    "TextEncodeAceStepAudio1.5": [
        "tags",
        "lyrics",
        "seed",
        None,  # control_after_generate: frontend only
        "bpm",
        "duration",
        "timesignature",
        "language",
        "keyscale",
        "generate_audio_codes",
        "cfg_scale",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
    ],
    "KSampler": ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "AudioContinueExtend": ["add_seconds", "mask_mode", "repaint_seconds"],
    "AudioContinueStitch": ["crossfade_seconds", "output_sample_rate"],
    "SaveAudio": ["filename_prefix"],
    "SaveAudioAdvanced": ["filename_prefix", "format"],
    "MusicCaptionMOSSLoader": [
        "model_name",
        "backend",
        "device",
        "dtype",
        "keep_loaded",
        "auto_download",
        "when_missing",
        "server_url",
    ],
    "MusicCaptionAnalyze": ["max_seconds", "sections", "extra_tags"],
    "MusicCaptionMOSS": [
        "server_url",
        "mode",
        "model",
        "transport",
        "segment",
        "max_seconds",
        "max_tokens",
        "temperature",
        "timeout",
        "extra_tags",
    ],
    "MusicCaptionMerge": ["prefer_numeric", "extra_tags", "keep_lyrics"],
    "MusicCaptionView": ["include_sections"],
    "PreviewAny": [],
}


def build_ui(graph, name=""):
    nodes = []
    links = []
    link_id = 0
    out_links = {}  # (origin_id, origin_slot) -> [link_id, ...]

    for n in graph.nodes:
        n["_in_links"] = []
        for slot, (iname, ntype, origin, origin_slot, is_widget) in enumerate(n["inputs"]):
            link_id += 1
            links.append([link_id, origin, origin_slot, n["id"], slot, ntype])
            out_links.setdefault((origin, origin_slot), []).append(link_id)
            n["_in_links"].append(
                {"name": iname, "type": ntype, "link": link_id, "widget": is_widget}
            )

    for n in graph.nodes:
        inputs = []
        for entry in n.get("_in_links", []):
            item = {"name": entry["name"], "type": entry["type"], "link": entry["link"]}
            if entry["widget"]:
                item["widget"] = {"name": entry["name"]}
            inputs.append(item)
        outputs = []
        for slot, out_spec in enumerate(n["outputs"]):
            if isinstance(out_spec, (tuple, list)) and len(out_spec) == 2:
                oname, otype = out_spec
            else:
                oname = otype = out_spec
            outputs.append(
                {
                    "name": oname,
                    "type": otype,
                    "links": out_links.get((n["id"], slot), None),
                    "slot_index": slot,
                }
            )
        ui = {
            "id": n["id"],
            "type": n["type"],
            "pos": n["pos"],
            "size": n["size"],
            "flags": {},
            "order": n["id"],
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "title": n["title"],
            "properties": {"Node name for S&R": n["type"]},
        }
        if n["widgets"]:
            ui["widgets_values"] = n["widgets"]
        nodes.append(ui)

    return {
        # deterministic id: rebuilding must not churn the file in git
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "audio-gen-continue/" + name)),
        "revision": 0,
        "last_node_id": max(n["id"] for n in graph.nodes),
        "last_link_id": link_id,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 1.0, "offset": [0, 0]}},
        "version": 0.4,
    }


def build_api(graph):
    api = {}
    for n in graph.nodes:
        if n["type"] == "MarkdownNote":
            continue
        inputs = {}
        keys = WIDGET_KEYS.get(n["type"])
        if keys is None:
            if n["widgets"]:
                raise SystemExit(f"no widget mapping for node class {n['type']}")
            keys = []  # link-only node (ConditioningZeroOut, VAEDecodeAudio, ...)
        for key, value in zip(keys, n["widgets"]):
            if key is not None:
                inputs[key] = value

        # links win over widget values (a linked input ignores its widget value)
        for name, _ntype, origin, slot, _is_widget in n["inputs"]:
            inputs[name] = [str(origin), slot]

        api[str(n["id"])] = {
            "class_type": n["type"],
            "inputs": inputs,
            "_meta": {"title": n["title"]},
        }
    return api


def write_pair(graph, out_dir, base_name):
    os.makedirs(out_dir, exist_ok=True)
    ui_path = os.path.join(out_dir, base_name + "_ui.json")
    api_path = os.path.join(out_dir, base_name + "_api.json")
    with open(ui_path, "w", encoding="utf-8") as fh:
        json.dump(build_ui(graph, base_name), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with open(api_path, "w", encoding="utf-8") as fh:
        json.dump(build_api(graph), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return ui_path, api_path


def main():
    written = []
    written += write_pair(caption_graph(), OUT_DIR, "music_caption")
    written += write_pair(xl_caption_graph(), OUT_DIR, "ace15xl_audio_continue_caption")
    written += write_pair(xl_graph(), OUT_DIR, "ace15xl_audio_continue")
    written += write_pair(turbo_aio_graph(), ARCHIVE_DIR, "ace15_audio_continue_turbo_aio")
    for path in written:
        print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
