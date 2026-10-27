#!/usr/bin/env python3
"""
Generate the two workflow files used by this project from a single graph definition.

    python3 tools/build_workflows.py

Outputs (relative to the repo root):
    workflows/ace15_audio_continue_ui.json    - drag & drop into the ComfyUI canvas
    workflows/ace15_audio_continue_api.json   - what audio_continue_tk.py posts to /prompt

Graph: ACE-Step 1.5 (turbo, all-in-one checkpoint) + AudioContinueExtend/Stitch.
    Checkpoint -> LoadAudio -> Extend -> masked KSampler -> Decode -> Stitch -> Save
"""

import json
import os
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "workflows")

CKPT = "ace_step_1.5_turbo_aio.safetensors"
SAVE_PREFIX = "audio-continue/audio"

DEFAULT_TAGS = (
    "instrumental, continues the existing song, same tempo, same key, same instruments, "
    "same mix and reverb, natural musical development, smooth transition, studio quality"
)
DEFAULT_LYRICS = ""

NOTE = """## Audio Continue - ACE-Step 1.5

1. Install the **audio_continue** custom nodes (this repo) and the model
   `ace_step_1.5_turbo_aio.safetensors` in `ComfyUI/models/checkpoints/`.
2. Load your track in **Load Audio** (upload button).
3. Set **add_seconds** on *Audio Continue (extend latent)* = how much new music to add.
4. Describe the music in **tags** (and paste the BPM / key of the source - it really helps).
5. Queue. The **duration** input of the text encoder is wired to `total_seconds`.

How it works: the source is VAE-encoded, N seconds of *empty* latent frames are
appended and masked, and the sampler only denoises the masked part. The unmasked
prefix is pinned to the encoded original every step, so the model continues the
actual recording instead of inventing a new song. **Stitch** then puts your exact
original file back and crossfades into the generated tail.

If the join sounds rough: raise `crossfade_seconds`, or set `mask_mode` to
`repaint_tail` with `repaint_seconds` ~2-4 s so the model re-renders the seam.
"""

# --------------------------------------------------------------------------------------
# graph definition: (id, class_type, title, pos, size, widgets, inputs, outputs)
#   inputs:  [(name, type, origin_id, origin_slot, is_widget)]
#   outputs: [type, ...]
# --------------------------------------------------------------------------------------

G = []


def node(nid, ctype, title, pos, size, widgets=None, inputs=None, outputs=None):
    G.append(
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


# 1 - model / clip / vae
node(
    1,
    "CheckpointLoaderSimple",
    "ACE-Step 1.5 turbo (AIO)",
    [40, 120],
    [400, 100],
    [CKPT],
    [],
    ["MODEL", "CLIP", "VAE"],
)

# 2 - sampling patch (turbo wants shift 3)
node(
    2,
    "ModelSamplingAuraFlow",
    "Model sampling (shift 3)",
    [480, 120],
    [300, 90],
    [3.0, "flow"],
    [("model", "MODEL", 1, 0, False)],
    ["MODEL"],
)

# 3 - source audio
node(
    3,
    "LoadAudio",
    "Music file",
    [40, 270],
    [400, 140],
    ["input.wav"],
    [],
    ["AUDIO"],
)

# 4 - extend + mask
node(
    4,
    "AudioContinueExtend",
    "Add seconds",
    [40, 450],
    [400, 170],
    [30.0, "generate_new_only", 0.0],
    [("audio", "AUDIO", 3, 0, False), ("vae", "VAE", 1, 2, False)],
    ["LATENT", "FLOAT", "FLOAT", "FLOAT", "INT"],
)

# 5 - text encode (duration is driven by the extend node)
node(
    5,
    "TextEncodeAceStepAudio1.5",
    "Style + lyrics",
    [480, 270],
    [560, 520],
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
    [("clip", "CLIP", 1, 1, False), ("duration", "FLOAT", 4, 3, True)],
    ["CONDITIONING"],
)

# 6 - negative conditioning (cfg 1 with turbo, but keep the graph complete)
node(
    6,
    "ConditioningZeroOut",
    "Negative (zero out)",
    [1080, 270],
    [240, 60],
    [],
    [("conditioning", "CONDITIONING", 5, 0, False)],
    ["CONDITIONING"],
)

# 7 - sampler
node(
    7,
    "KSampler",
    "Sampler",
    [1080, 370],
    [320, 290],
    [31, "randomize", 8, 1.0, "euler", "simple", 1.0],
    [
        ("model", "MODEL", 2, 0, False),
        ("positive", "CONDITIONING", 5, 0, False),
        ("negative", "CONDITIONING", 6, 0, False),
        ("latent_image", "LATENT", 4, 0, False),
    ],
    ["LATENT"],
)

# 8 - decode
node(
    8,
    "VAEDecodeAudio",
    "Decode",
    [1440, 370],
    [240, 60],
    [],
    [("samples", "LATENT", 7, 0, False), ("vae", "VAE", 1, 2, False)],
    ["AUDIO"],
)

# 9 - stitch source + generated
node(
    9,
    "AudioContinueStitch",
    "Stitch original + continuation",
    [1440, 470],
    [320, 110],
    [1.0, "highest"],
    [("source_audio", "AUDIO", 3, 0, False), ("continued_audio", "AUDIO", 8, 0, False)],
    ["AUDIO"],
)

# 10 - save into output/audio-continue/
node(
    10,
    "SaveAudio",
    "Save",
    [1800, 470],
    [400, 120],
    [SAVE_PREFIX],
    [("audio", "AUDIO", 9, 0, False)],
    ["AUDIO"],
)

# 11 - help note
node(
    11,
    "MarkdownNote",
    "Read me",
    [40, 660],
    [700, 420],
    [NOTE],
    [],
    [],
)


def build_ui():
    nodes = []
    links = []
    link_id = 0
    out_links = {}  # (origin_id, origin_slot) -> [link_id, ...]

    # first pass: create links (target slot = index in the node's input list)
    for n in G:
        n["_in_links"] = []
        for slot, (name, ntype, origin, origin_slot, is_widget) in enumerate(n["inputs"]):
            link_id += 1
            links.append([link_id, origin, origin_slot, n["id"], slot, ntype])
            out_links.setdefault((origin, origin_slot), []).append(link_id)
            n["_in_links"].append(
                {"name": name, "type": ntype, "link": link_id, "widget": is_widget}
            )

    for n in G:
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
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": max(n["id"] for n in G),
        "last_link_id": link_id,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 1.0, "offset": [0, 0]}},
        "version": 0.4,
    }


def build_api():
    api = {}
    for n in G:
        if n["type"] == "MarkdownNote":
            continue
        inputs = {}
        widgets = n["widgets"]

        if n["type"] == "CheckpointLoaderSimple":
            inputs["ckpt_name"] = widgets[0]
        elif n["type"] == "ModelSamplingAuraFlow":
            # only `shift`: `sampling` is an optional advanced widget in newer builds and
            # defaulting it here would break older ComfyUI versions
            inputs["shift"] = widgets[0]
        elif n["type"] == "LoadAudio":
            inputs["audio"] = widgets[0]
        elif n["type"] == "TextEncodeAceStepAudio1.5":
            keys = [
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
            ]
            for key, value in zip(keys, widgets):
                if key is not None:
                    inputs[key] = value
        elif n["type"] == "KSampler":
            keys = ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"]
            for key, value in zip(keys, widgets):
                if key is not None:
                    inputs[key] = value
        elif n["type"] == "AudioContinueExtend":
            inputs["add_seconds"] = widgets[0]
            inputs["mask_mode"] = widgets[1]
            inputs["repaint_seconds"] = widgets[2]
        elif n["type"] == "AudioContinueStitch":
            inputs["crossfade_seconds"] = widgets[0]
            inputs["output_sample_rate"] = widgets[1]
        elif n["type"] == "SaveAudio":
            inputs["filename_prefix"] = widgets[0]

        # links win over widget values (a linked input ignores its widget value)
        for name, _ntype, origin, slot, _is_widget in n["inputs"]:
            inputs[name] = [str(origin), slot]

        api[str(n["id"])] = {
            "class_type": n["type"],
            "inputs": inputs,
            "_meta": {"title": n["title"]},
        }
    return api


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ui_path = os.path.join(OUT_DIR, "ace15_audio_continue_ui.json")
    api_path = os.path.join(OUT_DIR, "ace15_audio_continue_api.json")
    with open(ui_path, "w", encoding="utf-8") as fh:
        json.dump(build_ui(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with open(api_path, "w", encoding="utf-8") as fh:
        json.dump(build_api(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print("wrote", ui_path)
    print("wrote", api_path)


if __name__ == "__main__":
    main()
