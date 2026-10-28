#!/usr/bin/env python3
"""
Generate the workflow files used by this project from two graph definitions.

    python3 tools/build_workflows.py

Outputs (relative to the repo root):

    workflows/ace15xl_audio_continue_ui.json              <- default: ACE-Step 1.5 XL SFT
    workflows/ace15xl_audio_continue_api.json                (drag & drop / what the app posts)
    workflows/archive/ace15_audio_continue_turbo_aio_ui.json   <- archived: 2B turbo AIO
    workflows/archive/ace15_audio_continue_turbo_aio_api.json

Graph (both): LoadAudio -> AudioContinueExtend -> masked KSampler -> Decode -> Stitch -> Save

The XL graph is the split-file layout the official template uses:
    UNETLoader(acestep_v1.5_xl_sft_bf16) + DualCLIPLoader(qwen_0.6b + qwen_4b) + VAELoader
    KSampler: 50 steps, cfg 7.0, euler + simple, ModelSamplingAuraFlow shift 3.0

The AIO graph is the original single-checkpoint turbo setup. It is kept for people who
already have the 10 GB `ace_step_1.5_turbo_aio.safetensors` and a small GPU:
    CheckpointLoaderSimple(ace_step_1.5_turbo_aio) -> model/clip/vae
    KSampler: 8 steps, cfg 1.0, euler + simple, shift 3.0
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

DEFAULT_TAGS = (
    "instrumental, continues the existing song, same tempo, same key, same instruments, "
    "same mix and reverb, natural musical development, smooth transition, studio quality"
)
DEFAULT_LYRICS = ""

XL_NOTE = """## Audio Continue - ACE-Step 1.5 XL (SFT)

1. Install the **audio_continue** custom nodes (this repo) plus the split XL files:
   * `models/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors`
   * `models/text_encoders/qwen_0.6b_ace15.safetensors`  (must be clip slot 1)
   * `models/text_encoders/qwen_4b_ace15.safetensors`    (clip slot 2)
   * `models/vae/ace_1.5_vae.safetensors`
2. Load your track in **Load Audio** (upload button).
3. Set **add_seconds** on *Audio Continue (extend latent)* = how much new music to add.
4. Describe the music in **tags** (and paste the BPM / key of the source - it really helps).
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


# --------------------------------------------------------------------------------------
# tiny graph builder
#   node(id, class_type, title, pos, size, widgets, inputs, outputs)
#     inputs:  [(name, type, origin_id, origin_slot, is_widget)]
#     outputs: [type, ...]
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


def _common_nodes(g, extend_vae, encode_clip, sampler_settings):
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
        [("clip", "CLIP", encode_clip, 0, False), ("duration", "FLOAT", 4, 3, True)],
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
}


def build_ui(graph, name=""):
    nodes = []
    links = []
    link_id = 0
    out_links = {}  # (origin_id, origin_slot) -> [link_id, ...]

    for n in graph.nodes:
        n["_in_links"] = []
        for slot, (name, ntype, origin, origin_slot, is_widget) in enumerate(n["inputs"]):
            link_id += 1
            links.append([link_id, origin, origin_slot, n["id"], slot, ntype])
            out_links.setdefault((origin, origin_slot), []).append(link_id)
            n["_in_links"].append(
                {"name": name, "type": ntype, "link": link_id, "widget": is_widget}
            )

    for n in graph.nodes:
        inputs = []
        for entry in n.get("_in_links", []):
            item = {"name": entry["name"], "type": entry["type"], "link": entry["link"]}
            if entry["widget"]:
                item["widget"] = {"name": entry["name"]}
            inputs.append(item)
        outputs = []
        for slot, otype in enumerate(n["outputs"]):
            outputs.append(
                {
                    "name": otype,
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
    written += write_pair(xl_graph(), OUT_DIR, "ace15xl_audio_continue")
    written += write_pair(turbo_aio_graph(), ARCHIVE_DIR, "ace15_audio_continue_turbo_aio")
    for path in written:
        print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
