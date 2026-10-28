#!/usr/bin/env python3
"""
audio_continue_tk.py - small Tk GUI that continues an existing music file with a
ComfyUI server (ACE-Step 1.5 by default).

    python3 audio_continue_tk.py

Workflow
    1. pick a music file
    2. set how many seconds to add
    3. press Generate

The app uploads the file to ComfyUI, patches workflows/ace15xl_audio_continue_api.json
(ACE-Step 1.5 XL SFT; the archived 2B turbo AIO workflow in workflows/archive/ works too),
queues it, waits for the result, downloads it next to your source file and (optionally)
opens it. Every field, the last file and the last folder are remembered on startup.

Only the Python standard library is used (tkinter + urllib), no pip install needed.

Command line helpers (no GUI):
    python3 audio_continue_tk.py --check --url http://127.0.0.1:8188
    python3 audio_continue_tk.py --dry-run --audio song.mp3 --add 30 --out patched.json
"""

import argparse
import copy
import json
import mimetypes
import os
import random
import re
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass

try:  # tkinter is optional: --check / --dry-run work on headless servers too
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - depends on the python build
    tk = None
    filedialog = messagebox = ttk = None

APP_NAME = "Audio Continue"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORKFLOW = os.path.join(HERE, "workflows", "ace15xl_audio_continue_api.json")
ARCHIVE_DIR = os.path.join(HERE, "workflows", "archive")
GENERIC_WORKFLOW = os.path.join(ARCHIVE_DIR, "ace15_audio_continue_turbo_aio_api.json")
CONFIG_PATH = os.environ.get(
    "AUDIO_CONTINUE_CONFIG", os.path.join(os.path.expanduser("~"), ".audio_continue.json")
)

REQUIRED_NODES = [
    "UNETLoader",
    "DualCLIPLoader",
    "VAELoader",
    "ModelSamplingAuraFlow",
    "LoadAudio",
    "AudioContinueExtend",
    "AudioContinueStitch",
    "TextEncodeAceStepAudio1.5",
    "ConditioningZeroOut",
    "KSampler",
    "VAEDecodeAudio",
    "SaveAudioAdvanced",
]

# ACE-Step 1.5 checkpoints want very different sampler settings. The app keeps the numbers
# in the workflow file, but shows these defaults so "Steps / CFG" does not silently turn an
# SFT run into a turbo run.
MODEL_PRESETS = {
    "xl_sft": {
        "label": "ACE-Step 1.5 XL SFT",
        "steps": 50,
        "cfg": 7.0,
        "generate_audio_codes": True,
        "match": ("xl_sft", "xl-base", "xl_base", "_sft", "_base"),
    },
    "xl_turbo": {
        "label": "ACE-Step 1.5 XL Turbo",
        "steps": 8,
        "cfg": 1.0,
        "generate_audio_codes": True,
        "match": ("xl_turbo", "xl-turbo"),
    },
    "turbo_aio": {
        "label": "ACE-Step 1.5 turbo (2B AIO)",
        "steps": 8,
        "cfg": 1.0,
        "generate_audio_codes": True,
        "match": ("turbo_aio", "1.5_turbo"),
    },
}

# where each loader looks for its file, used by `Check setup` to say something useful
LOADER_HINTS = {
    "UNETLoader": (("unet_name",), "models/diffusion_models/"),
    "CheckpointLoaderSimple": (("ckpt_name",), "models/checkpoints/"),
    "DualCLIPLoader": (("clip_name1", "clip_name2"), "models/text_encoders/"),
    "CLIPLoader": (("clip_name",), "models/text_encoders/"),
    "VAELoader": (("vae_name",), "models/vae/"),
}

AUDIO_EXTENSIONS = (
    ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aif", ".aiff", ".wma",
)

# files next to the track that the app offers as the prompt, in this order
PROMPT_SIDECAR_SUFFIXES = (
    ".prompt.txt", ".caption.txt", ".description.txt", ".tags.txt", ".txt",
)
PROMPT_SIDECAR_JSON = (".json", ".prompt.json")
PROMPT_JSON_KEYS = (
    "prompt", "caption", "tags", "description", "music_description", "style", "genre",
)

# captioner nodes shipped by this repo, best first (see CAPTIONER.md)
CAPTIONER_PREFERENCE = ("MusicCaptionMerge", "MusicCaptionAnalyze", "MusicCaptionMOSS")
CAPTIONER_TAP = "MusicCaptionTap"
# any third-party node whose name/display name/category mentions one of these and
# that takes an AUDIO input and returns STRINGs is offered as a captioner
CAPTIONER_HINTS = ("caption", "scribe", "analy", "music info", "music_info", "describe", "tagger")
MOSS_DEFAULT_URL = "http://127.0.0.1:30000"

DEFAULT_TAGS = (
    "instrumental, continues the existing song, same tempo, same key, same instruments, "
    "same mix and reverb, natural musical development, smooth transition, studio quality"
)

KEYS = [f"{root} {quality}" for quality in ("major", "minor") for root in (
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"
)]
LANGUAGES = [
    "en", "zh", "ja", "ko", "es", "fr", "de", "pt", "it", "ru", "nl", "pl", "tr", "uk",
    "ar", "hi", "id", "vi", "th", "sv", "fi", "no", "da", "cs", "el", "he", "hu", "ro",
]


# --------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------


@dataclass
class Settings:
    comfy_url: str = "http://127.0.0.1:8188"
    workflow: str = DEFAULT_WORKFLOW
    audio_file: str = ""
    prompt_file: str = ""
    prompt_append: bool = True
    auto_prompt_file: bool = True
    captioner: str = "auto"
    captioner_url: str = MOSS_DEFAULT_URL
    auto_caption: bool = False
    caption_prefer_numeric: str = "dsp"
    caption_fill_lyrics: bool = False
    add_seconds: float = 30.0
    tags: str = DEFAULT_TAGS
    lyrics: str = ""
    bpm: int = 120
    keyscale: str = "C major"
    language: str = "en"
    seed: int = 31
    randomize_seed: bool = True
    steps: int = 50
    cfg: float = 7.0
    generate_audio_codes: bool = True
    mask_mode: str = "generate_new_only"
    repaint_seconds: float = 0.0
    crossfade: float = 1.0
    save_prefix: str = "audio-continue/audio"
    output_dir: str = ""
    open_when_done: bool = True
    window_geometry: str = ""


def load_settings(path=None):
    path = path or CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return Settings()
    if not isinstance(data, dict):
        return Settings()
    known = {f for f in Settings().__dataclass_fields__}
    return Settings(**{k: v for k, v in data.items() if k in known})


def save_settings(settings, path=None):
    path = path or CONFIG_PATH
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(asdict(settings), fh, indent=2)
        os.replace(tmp, path)
        return True
    except OSError as exc:
        print(f"could not save settings to {path}: {exc}", file=sys.stderr)
        return False


# --------------------------------------------------------------------------------------
# ComfyUI client (stdlib only)
# --------------------------------------------------------------------------------------


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url, timeout=60):
        base_url = (base_url or "").strip()
        if not base_url.startswith(("http://", "https://")):
            base_url = "http://" + base_url
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- low level ---------------------------------------------------------------

    def _request(self, path, data=None, headers=None, method=None, timeout=None):
        url = self.base_url + path
        req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace").strip()
            raise ComfyError(f"HTTP {exc.code} from {path}: {body[:2000]}") from exc
        except urllib.error.URLError as exc:
            raise ComfyError(
                f"cannot reach {self.base_url} ({exc.reason}). Is ComfyUI running?"
            ) from exc
        except TimeoutError as exc:
            raise ComfyError(f"timeout talking to {self.base_url}{path}") from exc

    def _json(self, path, payload=None):
        if payload is None:
            raw = self._request(path)
        else:
            body = json.dumps(payload).encode("utf-8")
            raw = self._request(path, data=body, headers={"Content-Type": "application/json"})
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise ComfyError(f"invalid JSON from {path}: {raw[:400]!r}") from exc

    # -- api ---------------------------------------------------------------------

    def system_stats(self):
        return self._json("/system_stats")

    def object_info(self, node_class=None):
        path = "/object_info" + (f"/{urllib.parse.quote(node_class)}" if node_class else "")
        return self._json(path)

    def queue_length(self):
        return self._json("/queue")

    def interrupt(self):
        self._request("/interrupt", data=b"", method="POST")

    def upload_audio(self, path):
        """Upload a local file into ComfyUI/input and return the LoadAudio value."""
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise ComfyError(f"file not found: {path}")
        filename = os.path.basename(path)
        stem, ext = os.path.splitext(filename)
        ascii_stem = (
            unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
        )
        safe_stem = "".join(ch for ch in ascii_stem if ch.isalnum() or ch in " ._-()[]").strip()
        safe_stem = safe_stem or "audio"
        safe = safe_stem + (ext if ext else ".wav")

        with open(path, "rb") as fh:
            content = fh.read()

        boundary = "----audiocontinue" + uuid.uuid4().hex
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts = []
        for name, value in (("type", "input"), ("overwrite", "true")):
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                .encode("utf-8")
            )
        parts.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
                f"filename=\"{safe}\"\r\nContent-Type: {mime}\r\n\r\n"
            ).encode("utf-8")
        )
        parts.append(content)
        parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(parts)

        raw = self._request(
            "/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=max(self.timeout, 300),
        )
        info = json.loads(raw.decode("utf-8"))
        name = info.get("name") or safe
        subfolder = info.get("subfolder") or ""
        return f"{name} [{subfolder}]" if subfolder else name

    def queue_prompt(self, prompt, client_id=None):
        payload = {"prompt": prompt, "client_id": client_id or uuid.uuid4().hex}
        result = self._json("/prompt", payload)
        if result.get("node_errors"):
            raise ComfyError("workflow rejected:\n" + json.dumps(result["node_errors"], indent=2)[:3000])
        if "prompt_id" not in result:
            raise ComfyError("unexpected /prompt answer: " + json.dumps(result)[:2000])
        return result["prompt_id"]

    def wait(self, prompt_id, on_progress=None, poll=1.0, max_wait=6 * 3600):
        started = time.time()
        while True:
            history = self._json(f"/history/{prompt_id}")
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {}) or {}
                if status.get("status_str") == "error" or status.get("completed") is False:
                    messages = status.get("messages") or []
                    raise ComfyError("generation failed:\n" + json.dumps(messages[-2:], indent=2)[:3000])
                return entry
            if time.time() - started > max_wait:
                raise ComfyError("timed out waiting for the prompt to finish")

            if on_progress:
                try:
                    queue = self.queue_length()
                    running = {item[1] for item in queue.get("queue_running", []) if len(item) > 1}
                    pending = {item[1] for item in queue.get("queue_pending", []) if len(item) > 1}
                    if prompt_id in running:
                        on_progress("running")
                    elif prompt_id in pending:
                        on_progress("queued")
                    elif queue.get("queue_running") or queue.get("queue_pending"):
                        on_progress("waiting")
                except ComfyError:
                    pass
            time.sleep(poll)

    def download_output(self, file_info, destination):
        query = urllib.parse.urlencode(
            {
                "filename": file_info["filename"],
                "subfolder": file_info.get("subfolder", ""),
                "type": file_info.get("type", "output"),
            }
        )
        data = self._request("/view?" + query, timeout=max(self.timeout, 300))
        os.makedirs(os.path.dirname(os.path.abspath(destination)) or ".", exist_ok=True)
        with open(destination, "wb") as fh:
            fh.write(data)
        return destination


def find_audio_outputs(history_entry):
    """Collect {'filename','subfolder','type'} dicts from a /history entry."""
    found = []
    for node_output in (history_entry.get("outputs") or {}).values():
        for value in node_output.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict) and item.get("filename"):
                    if item["filename"].lower().endswith(AUDIO_EXTENSIONS):
                        found.append(item)
    return found


# --------------------------------------------------------------------------------------
# workflow patching
# --------------------------------------------------------------------------------------


def _set(inputs, key, value):
    if key in inputs:
        inputs[key] = value
        return True
    return False


def patch_workflow(workflow, settings, uploaded_name=None):
    """Return a copy of an API workflow with the GUI values applied.

    Nodes are matched by class_type, so titles can be renamed freely. Unknown nodes are
    left untouched, which keeps the app usable with slightly different workflows
    (for example the starsFriday ComfyUI-ACEStep extend nodes).
    """
    patched = copy.deepcopy(workflow)
    touched = set()

    for node in patched.values():
        ctype = node.get("class_type", "")
        inputs = node.setdefault("inputs", {})

        if ctype == "LoadAudio" and uploaded_name:
            inputs["audio"] = uploaded_name
            touched.add("audio")

        elif ctype == "AudioContinueExtend":
            _set(inputs, "add_seconds", float(settings.add_seconds))
            _set(inputs, "mask_mode", settings.mask_mode)
            _set(inputs, "repaint_seconds", float(settings.repaint_seconds))
            touched.add("extend")

        elif ctype in ("ACEStep15XLExtendAudio", "ACEStep15XLExtendLatent"):
            # starsFriday/ComfyUI-ACEStep variant
            _set(inputs, "right_extend_seconds", float(settings.add_seconds))
            touched.add("extend")

        elif ctype in ("TextEncodeAceStepAudio1.5", "TextEncodeAceStepAudio", "ACEStep15XLTextEncode"):
            _set(inputs, "tags", settings.tags)
            _set(inputs, "lyrics", settings.lyrics)
            _set(inputs, "bpm", int(settings.bpm))
            _set(inputs, "keyscale", settings.keyscale)
            _set(inputs, "language", settings.language)
            _set(inputs, "seed", int(settings.seed))
            _set(inputs, "generate_audio_codes", bool(settings.generate_audio_codes))
            touched.add("text")

        elif ctype in ("KSampler", "KSamplerAdvanced"):
            _set(inputs, "seed", int(settings.seed))
            _set(inputs, "noise_seed", int(settings.seed))
            _set(inputs, "steps", int(settings.steps))
            _set(inputs, "cfg", float(settings.cfg))
            touched.add("sampler")

        elif ctype in ("SaveAudio", "SaveAudioMP3", "SaveAudioOpus", "SaveAudioAdvanced"):
            _set(inputs, "filename_prefix", settings.save_prefix)
            touched.add("save")

        elif ctype == "AudioContinueStitch":
            _set(inputs, "crossfade_seconds", float(settings.crossfade))
            touched.add("stitch")

    missing = []
    if uploaded_name and "audio" not in touched:
        missing.append("LoadAudio")
    for key, label in (
        ("extend", "AudioContinueExtend (or an *Extend* node)"),
        ("text", "TextEncodeAceStepAudio1.5"),
        ("sampler", "KSampler"),
        ("save", "SaveAudio"),
    ):
        if key not in touched:
            missing.append(label)
    if missing:
        raise ComfyError(
            "this workflow is missing required nodes: " + ", ".join(missing) +
            "\nCheck 'Workflow (API)' in the app; see README for the shipped JSON."
        )
    return patched


# --------------------------------------------------------------------------------------
# prompt files
# --------------------------------------------------------------------------------------


def find_prompt_file(audio_path):
    """Look for a prompt sidecar next to `audio_path`; return the first hit or "".

    `song.flac` -> song.prompt.txt, song.caption.txt, song.description.txt,
    song.tags.txt, song.txt, song.json (with a prompt/caption/tags key).
    """
    if not audio_path:
        return ""
    stem = os.path.splitext(os.path.abspath(audio_path))[0]
    candidates = [stem + suffix for suffix in PROMPT_SIDECAR_SUFFIXES]
    candidates += [stem + suffix for suffix in PROMPT_SIDECAR_JSON]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) == 0:
                continue
        except OSError:
            continue
        if read_prompt_file(path):
            return path
    return ""


def read_prompt_file(path):
    """Read a prompt from a .txt/.md or a metadata .json file. Returns "" if unusable."""
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError:
        return ""
    text = raw.strip()
    if not text:
        return ""
    if path.lower().endswith(".json") or text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            return text
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, list):
            joined = ", ".join(str(item) for item in data if str(item).strip())
            return joined.strip()
        if isinstance(data, dict):
            for key in PROMPT_JSON_KEYS:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, list) and value and isinstance(value[0], str):
                    return ", ".join(item.strip() for item in value if item.strip())
            return ""
    return text


def merge_prompt_text(base, extra, append=True):
    """Merge two tag/description strings, de-duplicating comma tokens.

    `append=False` means `extra` replaces `base` (that is what the checkbox in
    the GUI switches between).
    """
    base = (base or "").strip()
    extra = (extra or "").strip()
    if not extra:
        return base
    if not base or not append:
        return extra
    seen = set()
    parts = []
    for chunk in (base, extra):
        for raw in re.split(r"[,\n]", chunk):
            tag = re.sub(r"\s+", " ", raw).strip().strip(",;")
            if not tag or tag.lower() in seen:
                continue
            seen.add(tag.lower())
            parts.append(tag)
    return ", ".join(parts)


def _pitches():
    return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def normalize_keyscale(value):
    """'am', 'A min', 'A minor' -> 'A minor'; 'Bb' -> 'A# major' (ACE-Step spelling)."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in ("auto", "none", "null", "unknown", "n/a", "atonal"):
        return ""
    match = re.search(
        r"(?<![a-z#])([a-g])\s*(#|♯|b|♭)?\s*(major|maj|minor|min|ionian|aeolian|m|dur|moll)?\b",
        text.lower(),
    )
    if not match:
        return ""
    letter, accidental, quality = match.groups()
    pitch = letter.upper()
    if accidental in ("#", "♯"):
        pitch += "#"
    elif accidental in ("b", "♭"):
        flat = (letter + "b").upper()
        mapping = {"CB": "B", "DB": "C#", "EB": "D#", "FB": "E", "GB": "F#",
                   "AB": "G#", "BB": "A#"}
        if flat not in mapping:
            return ""
        pitch = mapping[flat]
    if pitch not in _pitches():
        return ""
    mode = "minor" if quality in ("minor", "min", "aeolian", "m", "moll") else "major"
    candidate = f"{pitch} {mode}"
    return candidate if candidate in KEYS else f"{pitch} major"


def normalize_bpm(value, default=0):
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = re.search(r"(\d+(?:[.,]\d+)?)", str(value))
        if not match:
            return default
        number = float(match.group(1).replace(",", "."))
    if number <= 0:
        return default
    number = int(round(number))
    return number if 20 <= number <= 400 else default


# --------------------------------------------------------------------------------------
# captioner discovery (node names come from the live ComfyUI /object_info)
# --------------------------------------------------------------------------------------


_MISSING = object()


def _spec_type(spec):
    """The declared type of an object_info input spec ('AUDIO', 'STRING', ...)."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return ""
    kind = spec[0]
    if isinstance(kind, str):
        return kind.upper()
    if isinstance(kind, list):
        return "COMBO"
    return ""


def _spec_default(spec):
    kind = _spec_type(spec)
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if kind == "COMBO":
        choices = spec[0]
        default = options.get("default")
        if default is not None and default in choices:
            return default
        if choices:
            return choices[0]
        return _MISSING
    if kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
        if "default" in options:
            return options["default"]
        return {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}[kind]
    return _MISSING


def _node_specs(entry):
    inputs = (entry or {}).get("input") or {}
    specs = {}
    for section in ("required", "optional"):
        for name, spec in (inputs.get(section) or {}).items():
            specs[name] = spec
    return specs


def _required_names(entry):
    return set(((entry or {}).get("input") or {}).get("required") or {})


def _string_output_indices(entry):
    outputs = (entry or {}).get("output") or []
    names = (entry or {}).get("output_name") or []
    found = []
    for index, kind in enumerate(outputs):
        if str(kind).upper() in ("STRING", "*"):
            found.append((index, names[index] if index < len(names) else f"output_{index}"))
    return found


def captioner_candidates(object_info):
    """Third-party captioner nodes present on the server, best guess first."""
    found = []
    for name, entry in (object_info or {}).items():
        if name in CAPTIONER_PREFERENCE or name == CAPTIONER_TAP:
            continue
        specs = _node_specs(entry)
        if not any(_spec_type(spec) == "AUDIO" for spec in specs.values()):
            continue
        if not _string_output_indices(entry):
            continue
        label = str(entry.get("display_name") or name)
        haystack = " ".join(
            (name, label, str(entry.get("category") or ""), str(entry.get("description") or ""))
        ).lower()
        if not any(hint in haystack for hint in CAPTIONER_HINTS):
            continue
        score = 0
        if "key" in haystack or "bpm" in haystack:
            score += 2
        if "caption" in haystack:
            score += 1
        outputs = " ".join(str(n).lower() for _, n in _string_output_indices(entry))
        if "bpm" in outputs:
            score += 2
        if "key" in outputs:
            score += 2
        found.append({"class": name, "label": label, "score": score})
    found.sort(key=lambda item: (-item["score"], item["class"].lower()))
    return found


def captioner_plan(object_info, requested="auto"):
    """Decide how to caption: {'mode': native|tap|none, 'class', 'label', 'reason'}."""
    object_info = object_info or {}
    ours = [name for name in CAPTIONER_PREFERENCE if name in object_info]
    if requested in ("off", "none", "disabled"):
        return {"mode": "none", "class": "", "label": "", "reason": "captioner disabled"}
    if requested and requested not in ("auto", "", None):
        if requested in object_info:
            if requested in CAPTIONER_PREFERENCE:
                return {"mode": "native", "class": requested, "label": requested,
                        "reason": "selected captioner"}
            if CAPTIONER_TAP in object_info:
                return {"mode": "tap", "class": requested, "label": requested,
                        "reason": "selected captioner + MusicCaptionTap"}
            return {"mode": "none", "class": requested, "label": requested,
                    "reason": f"'{requested}' needs {CAPTIONER_TAP} from comfyui_music_caption"}
        return {"mode": "none", "class": requested, "label": requested,
                "reason": f"captioner node '{requested}' is not installed"}

    if ours:
        return {"mode": "native", "class": ours[0], "label": ours[0],
                "reason": "comfyui_music_caption found"}
    third_party = captioner_candidates(object_info)
    if third_party:
        if CAPTIONER_TAP in object_info:
            best = third_party[0]
            return {"mode": "tap", "class": best["class"], "label": best["label"],
                    "reason": "third-party captioner + MusicCaptionTap"}
        return {"mode": "tap", "class": third_party[0]["class"], "label": third_party[0]["label"],
                "reason": f"'{third_party[0]['class']}' found but {CAPTIONER_TAP} "
                          "(comfyui_music_caption) is missing"}
    return {"mode": "none", "class": "", "label": "",
            "reason": "no captioner node - install comfyui_music_caption/ from this repo"}


def _fill_inputs(entry, overrides=None, links=None, skip=()):
    """Build an API-format inputs dict from object_info defaults + our overrides."""
    overrides = overrides or {}
    links = links or {}
    required = _required_names(entry)
    inputs = {}
    for name, spec in _node_specs(entry).items():
        if name in skip:
            continue
        if name in links:
            inputs[name] = links[name]
            continue
        if name in overrides:
            inputs[name] = overrides[name]
            continue
        default = _spec_default(spec)
        if default is _MISSING:
            if name in required:
                raise ComfyError(
                    f"captioner input '{name}' ({_spec_type(spec) or 'link'}) cannot be filled "
                    "by the app - pick another captioner node"
                )
            continue
        inputs[name] = default
    return inputs


def _captioner_inputs(entry, uploaded_node, moss_url="", moss_mode="auto", max_seconds=90.0):
    """Defaults for a captioner node, with the audio wired to a LoadAudio node."""
    specs = _node_specs(entry)
    audio_name = next((name for name, spec in specs.items() if _spec_type(spec) == "AUDIO"), None)
    if audio_name is None:
        raise ComfyError("captioner node has no AUDIO input")
    overrides = {}
    for name in specs:
        lowered = name.lower()
        if lowered in ("server_url", "base_url", "api_url", "endpoint", "server") and moss_url:
            overrides[name] = moss_url
        elif lowered in ("mode", "task", "question") and moss_mode and _spec_type(specs[name]) == "COMBO":
            choices = specs[name][0] if isinstance(specs[name][0], list) else []
            if moss_mode in choices:
                overrides[name] = moss_mode
        elif lowered in ("max_seconds", "audio_seconds", "duration", "max_duration"):
            if _spec_type(specs[name]) in ("FLOAT", "INT") and max_seconds:
                overrides[name] = float(max_seconds) if _spec_type(specs[name]) == "FLOAT" else int(max_seconds)
    return _fill_inputs(entry, overrides=overrides, links={audio_name: [uploaded_node, 0]})


def build_caption_prompt(object_info, plan, uploaded_name, moss_url=MOSS_DEFAULT_URL,
                         moss_mode="auto", max_seconds=90.0, prefer_numeric="dsp"):
    """Build the small API workflow that captions `uploaded_name`.

    Returns (prompt, meta) where meta tells `caption_result_from_history` where
    to read the answer.
    """
    object_info = object_info or {}
    prompt = {"1": {"class_type": "LoadAudio", "inputs": {"audio": uploaded_name}}}
    meta = {"kind": plan.get("mode"), "captioner": plan.get("class"), "read_node": ""}

    if plan.get("mode") == "native":
        requested = plan.get("class") or "auto"
        use_analyze = "MusicCaptionAnalyze" in object_info and requested in (
            "auto", "MusicCaptionAnalyze", "MusicCaptionMerge"
        )
        use_moss = "MusicCaptionMOSS" in object_info and requested in (
            "auto", "MusicCaptionMOSS", "MusicCaptionMerge"
        )
        if not (use_analyze or use_moss):
            # a requested node is missing: fall back to whatever is installed
            use_analyze = "MusicCaptionAnalyze" in object_info
            use_moss = "MusicCaptionMOSS" in object_info
        if use_analyze:
            prompt["2"] = {
                "class_type": "MusicCaptionAnalyze",
                "inputs": _fill_inputs(
                    object_info["MusicCaptionAnalyze"],
                    overrides={"max_seconds": float(max_seconds)},
                    links={"audio": ["1", 0]},
                ),
            }
        if use_moss:
            prompt["3"] = {
                "class_type": "MusicCaptionMOSS",
                "inputs": _fill_inputs(
                    object_info["MusicCaptionMOSS"],
                    overrides={"server_url": moss_url, "mode": moss_mode,
                               "max_seconds": float(max_seconds)},
                    links={"audio": ["1", 0]},
                ),
            }
        if "2" in prompt and "3" in prompt and "MusicCaptionMerge" in object_info:
            prompt["4"] = {
                "class_type": "MusicCaptionMerge",
                "inputs": _fill_inputs(
                    object_info["MusicCaptionMerge"],
                    overrides={"prefer_numeric": prefer_numeric},
                    links={"analysis_json": ["2", 8], "moss_json": ["3", 7]},
                ),
            }
            meta["read_node"] = "4"
        elif "2" in prompt:
            meta["read_node"] = "2"
        else:
            meta["read_node"] = "3"
        if not meta["read_node"]:
            raise ComfyError("comfyui_music_caption is installed but no usable node was found")
        return prompt, meta

    if plan.get("mode") == "tap":
        captioner = plan.get("class")
        if captioner not in object_info:
            raise ComfyError(f"captioner node '{captioner}' is not installed")
        if CAPTIONER_TAP not in object_info:
            raise ComfyError(
                f"{CAPTIONER_TAP} is missing - install comfyui_music_caption/ from this repo"
            )
        prompt["2"] = {
            "class_type": captioner,
            "inputs": _captioner_inputs(
                object_info[captioner], "1", moss_url, moss_mode, max_seconds
            ),
        }
        links = {}
        names = []
        for position, (index, name) in enumerate(_string_output_indices(object_info[captioner])[:12]):
            links[f"text_{position + 1}"] = ["2", index]
            names.append(name)
        if not names:
            raise ComfyError(f"captioner '{captioner}' returns no STRING outputs")
        prompt["3"] = {
            "class_type": CAPTIONER_TAP,
            "inputs": _fill_inputs(
                object_info[CAPTIONER_TAP], overrides={"label": captioner},
                links=links, skip=("text_1", "text_2", "text_3", "text_4", "text_5", "text_6",
                                   "text_7", "text_8", "text_9", "text_10", "text_11", "text_12"),
            ),
        }
        prompt["3"]["inputs"].update(links)
        meta["read_node"] = "3"
        meta["output_names"] = names
        return prompt, meta

    raise ComfyError(plan.get("reason") or "no captioner available")


def describe_captioners(object_info, plan=None):
    """Human-readable lines about the captioners the server offers."""
    ours = [name for name in CAPTIONER_PREFERENCE if name in (object_info or {})]
    third_party = captioner_candidates(object_info)
    lines = []
    if ours:
        lines.append("captioner: comfyui_music_caption installed (" + ", ".join(ours) + ")")
    else:
        lines.append("captioner: comfyui_music_caption is NOT installed "
                     "(copy comfyui_music_caption/ into ComfyUI/custom_nodes/ and restart)")
    if third_party:
        lines.append("third-party captioners: " + ", ".join(
            f"{item['class']}" for item in third_party[:6]))
    if plan:
        if plan.get("mode") == "none":
            lines.append("  -> captioning off: " + plan.get("reason", ""))
        else:
            lines.append(f"  -> will caption with {plan.get('class')} ({plan.get('mode')})")
    if ours and "MusicCaptionMOSS" in ours:
        lines.append(f"  MOSS-Music server (optional, better captions): {MOSS_DEFAULT_URL} "
                     "- `sglang serve --model-path MOSS-Music-8B-Instruct --trust-remote-code`")
    return lines


def _history_texts(history_entry, node_id):
    outputs = (history_entry or {}).get("outputs") or {}
    entry = outputs.get(str(node_id)) or {}
    blocks = entry.get("text") or []
    return [block for block in blocks if isinstance(block, str)]


def interpret_caption_payload(payload):
    """Normalise whatever a captioner returned into the app's caption dict."""
    result = {
        "caption": "", "tags": "", "bpm": 0, "keyscale": "", "lyrics": "",
        "vocal_type": "", "vocal_presence": "", "instrumental": None,
        "structure": "", "notes": [], "raw": payload,
    }
    if isinstance(payload, str):
        payload = {"caption": payload}
    if not isinstance(payload, dict):
        return result

    def pick(*names):
        for name in names:
            for key, value in payload.items():
                if str(key).lower() == name and value not in (None, "", [], {}):
                    return value
        return None

    measured = payload.get("measured") if isinstance(payload.get("measured"), dict) else {}
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}

    caption = pick("caption", "description") or ""
    tags = pick("tags") or ""
    if not caption and model:
        caption = model.get("caption") or ""
    if not caption and measured:
        caption = measured.get("caption") or ""

    bpm = normalize_bpm(payload.get("bpm"))
    if not bpm and model:
        bpm = normalize_bpm(model.get("bpm"))
    if not bpm and measured:
        bpm = normalize_bpm(measured.get("bpm"))
    if not bpm:
        for key in ("tempo", "tempo_bpm", "bpm_value"):
            bpm = normalize_bpm(pick(key))
            if bpm:
                break

    keyscale = normalize_keyscale(pick("keyscale", "key", "key_scale", "scale", "keyscale_key"))
    if not keyscale and model:
        keyscale = normalize_keyscale(model.get("keyscale") or model.get("key"))
    if not keyscale and measured:
        keyscale = normalize_keyscale(measured.get("key"))

    lyrics = pick("lyrics", "lyric") or ""
    vocal_type = pick("vocal_type", "voice", "vocal_details") or ""
    if not vocal_type and model:
        vocal_type = model.get("vocal_type") or ""
    presence = pick("vocal_presence", "vocal") or ""
    if isinstance(presence, dict):
        vocal_type = vocal_type or str(presence.get("type") or "")
        presence = presence.get("presence") or ""
    if not vocal_type and measured.get("vocal"):
        vocal_type = str((measured.get("vocal") or {}).get("type") or "")
    if not presence and measured.get("vocal"):
        presence = str((measured.get("vocal") or {}).get("presence") or "")

    instrumental = payload.get("instrumental")
    if instrumental is None and model.get("instrumental") is not None:
        instrumental = bool(model["instrumental"])
    if instrumental is None and str(presence).lower() in ("instrumental", "vocals", "uncertain"):
        instrumental = str(presence).lower() == "instrumental"

    result.update(
        caption=str(caption).strip(),
        tags=str(tags).strip(),
        bpm=bpm,
        keyscale=keyscale,
        lyrics=str(lyrics).strip(),
        vocal_type=str(vocal_type).strip(),
        vocal_presence=str(presence).strip(),
        instrumental=instrumental,
        structure=str(pick("structure") or (model.get("structure") or "")).strip(),
    )
    for key in ("analysis_json", "moss_json", "summary_json"):
        if isinstance(payload.get(key), str):
            try:
                payload[key] = json.loads(payload[key])
            except ValueError:
                pass
    if result["bpm"]:
        result["notes"].append(f"{result['bpm']} bpm")
    if result["keyscale"]:
        result["notes"].append(result["keyscale"])
    if result["vocal_type"]:
        result["notes"].append(f"vocals: {result['vocal_type']}")
    elif result["instrumental"]:
        result["notes"].append("instrumental")
    return result


def caption_result_from_history(history_entry, meta):
    """Read the captioner answer out of a /history entry."""
    node_id = str(meta.get("read_node") or "")
    blocks = _history_texts(history_entry, node_id) if node_id else []
    if not blocks:
        for candidate in ((history_entry or {}).get("outputs") or {}):
            blocks = _history_texts(history_entry, candidate)
            if blocks:
                node_id = candidate
                break
    if not blocks:
        raise ComfyError(
            "the captioner ran but returned no text - it has to write its result into the "
            f"history 'text' output (node {node_id or '?'})"
        )
    payload = blocks[-1]
    try:
        loaded = json.loads(payload)
    except ValueError:
        loaded = payload
    if isinstance(loaded, list):
        names = meta.get("output_names") or []
        fields = {}
        for index, value in enumerate(loaded):
            name = names[index] if index < len(names) else f"output_{index}"
            fields[str(name)] = value
        return interpret_caption_payload(fields)
    return interpret_caption_payload(loaded)


def merge_caption_into_settings(settings, caption_result, fill_lyrics=False):
    """Apply a caption result to the settings; returns the list of changes (for the log)."""
    changes = []
    bpm = caption_result.get("bpm")
    if bpm and str(bpm) != str(settings.bpm):
        changes.append(f"bpm {settings.bpm} -> {bpm}")
        settings.bpm = int(bpm)
    keyscale = caption_result.get("keyscale")
    if keyscale and keyscale != settings.keyscale:
        changes.append(f"key {settings.keyscale} -> {keyscale}")
        settings.keyscale = keyscale
    if fill_lyrics and not settings.lyrics.strip() and not caption_result.get("instrumental"):
        lyrics = caption_result.get("lyrics") or ""
        if lyrics and not re.match(r"^\s*\[instrumental\]\s*$", lyrics, re.I):
            settings.lyrics = lyrics
            changes.append(f"lyrics filled from the captioner ({len(lyrics)} chars)")
    return changes


def caption_tags_from_result(caption_result):
    """The part of a caption result that belongs in the tag prompt.

    `tags` (from our merge node) is the finished string, so it is used as the base
    and only what is missing is added - no "96 bpm" twice.
    """
    merged = caption_result.get("tags") or ""
    pieces = [caption_result.get("caption") or ""]
    if caption_result.get("vocal_type") and caption_result.get("vocal_presence") != "instrumental":
        pieces.append(str(caption_result["vocal_type"]))
    if caption_result.get("structure"):
        pieces.append(str(caption_result["structure"]))
    if caption_result.get("bpm"):
        pieces.append(f"{int(caption_result['bpm'])} bpm")
    if caption_result.get("keyscale"):
        pieces.append(caption_result["keyscale"])
    for piece in pieces:
        if not piece:
            continue
        if piece.lower() in merged.lower():
            continue
        merged = merge_prompt_text(merged, piece, append=True)
    return merged


def effective_tags(settings, prompt_text="", caption_tags=""):
    """description text widget + prompt file + captioner, in that order."""
    tags = merge_prompt_text(settings.tags, prompt_text, append=settings.prompt_append)
    if caption_tags:
        tags = merge_prompt_text(tags, caption_tags, append=True)
    return tags


def make_output_name(audio_file, add_seconds, server_name):
    stem = os.path.splitext(os.path.basename(audio_file))[0] if audio_file else "audio"
    ext = os.path.splitext(server_name)[1] or ".flac"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stem}_continue+{int(round(add_seconds))}s_{stamp}{ext}"


# --------------------------------------------------------------------------------------
# headless helpers (--check / --dry-run)
# --------------------------------------------------------------------------------------


def workflow_node_classes(workflow):
    return sorted({n.get("class_type") for n in workflow.values() if n.get("class_type")})


def model_preset(workflow):
    """Guess the ACE-Step 1.5 variant from the workflow's model files.

    Returns a MODEL_PRESETS entry (plus its key under "name") or None.
    """
    for node in workflow.values():
        inputs = node.get("inputs") or {}
        for key in ("unet_name", "ckpt_name"):
            value = inputs.get(key)
            if not isinstance(value, str):
                continue
            lowered = value.lower()
            for name, preset in MODEL_PRESETS.items():
                if any(token in lowered for token in preset["match"]):
                    return dict(preset, name=name)
    return None


def workflow_assets(workflow):
    """{class_type: [file names]} for every loader node in an API workflow."""
    found = {}
    for node in workflow.values():
        ctype = node.get("class_type")
        entry = LOADER_HINTS.get(ctype)
        if not entry:
            continue
        keys = entry[0]
        inputs = node.get("inputs") or {}
        names = [inputs.get(key) for key in keys if isinstance(inputs.get(key), str)]
        if names:
            found.setdefault(ctype, []).extend(names)
    return found


def _loader_options(info, node_class, key):
    """The list of installed files ComfyUI reports for a loader input."""
    entry = info.get(node_class) or {}
    required = (entry.get("input") or {}).get("required") or {}
    value = required.get(key)
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return []


def check_setup(client, node_classes=None, assets=None):
    lines = []
    try:
        stats = client.system_stats()
        devices = stats.get("devices") or []
        gpu = devices[0].get("name") if devices else "unknown"
        lines.append(f"ComfyUI OK at {client.base_url} (version {stats.get('system', {}).get('comfyui_version', '?')})")
        lines.append(f"device: {gpu}")
    except ComfyError as exc:
        lines.append(f"ERROR: {exc}")
        return lines, False

    required = list(node_classes) if node_classes else list(REQUIRED_NODES)
    info = client.object_info()
    missing = [name for name in required if name not in info]
    if missing:
        lines.append("MISSING NODES: " + ", ".join(missing))
        lines.append("  -> copy comfyui_audio_continue/ into ComfyUI/custom_nodes/ and restart ComfyUI")
        ok = False
    else:
        lines.append("all required nodes present")
        ok = True

    if assets is None:
        # no workflow at hand: just report what is installed
        assets = {"UNETLoader": [], "CheckpointLoaderSimple": []}

    for node_class, names in assets.items():
        keys, folder = LOADER_HINTS[node_class]
        options = []
        for key in keys:
            for option in _loader_options(info, node_class, key):
                if option not in options:  # DualCLIPLoader reports two lists
                    options.append(option)
        if not options:
            continue
        lines.append(f"{node_class} files: " + ", ".join(map(str, options[:12])))
        for name in names:
            if name not in options:
                lines.append(f"  ! '{name}' is not installed -> put it in {folder}")
                ok = False

    if not any(names for names in assets.values()):
        # nothing declared (custom workflow): fall back to "is any ACE-Step model there?"
        models = []
        for node_class in ("UNETLoader", "CheckpointLoaderSimple"):
            for key in LOADER_HINTS[node_class][0]:
                models += _loader_options(info, node_class, key)
        if models and not any("ace" in str(m).lower() for m in models):
            lines.append(
                "  ! no ACE-Step model found (expected acestep_v1.5_xl_sft_bf16.safetensors in "
                "models/diffusion_models/ or ace_step_1.5_turbo_aio.safetensors in models/checkpoints/)"
            )
    return lines, ok


def _cli_caption(settings):
    """--caption: run the captioner on the source file and print the result."""
    if not settings.audio_file or not os.path.isfile(settings.audio_file):
        print("--caption needs --audio with an existing file", file=sys.stderr)
        return 2
    client = ComfyClient(settings.comfy_url)
    try:
        info = client.object_info()
    except ComfyError as exc:
        print(f"cannot read /object_info: {exc}", file=sys.stderr)
        return 1
    plan = captioner_plan(info, settings.captioner)
    if plan["mode"] == "none" or settings.captioner == "off":
        print("no captioner: " + (plan.get("reason") or "disabled"), file=sys.stderr)
        for line in describe_captioners(info, plan):
            print(line, file=sys.stderr)
        return 1
    print(f"captioner: {plan['label']} ({plan['mode']})")
    uploaded = client.upload_audio(settings.audio_file)
    prompt, meta = build_caption_prompt(
        info, plan, uploaded, moss_url=settings.captioner_url,
        prefer_numeric=settings.caption_prefer_numeric,
    )
    prompt_id = client.queue_prompt(prompt)
    entry = client.wait(prompt_id)
    result = caption_result_from_history(entry, meta)
    changes = merge_caption_into_settings(settings, result, fill_lyrics=settings.caption_fill_lyrics)
    print(f"bpm:      {result.get('bpm') or '-'}")
    print(f"keyscale: {result.get('keyscale') or '-'}")
    print(f"vocals:   {result.get('vocal_type') or result.get('vocal_presence') or '-'}")
    print(f"tags:     {caption_tags_from_result(result) or '-'}")
    if result.get("structure"):
        print(f"structure: {result['structure']}")
    for change in changes:
        print("  " + change)
    return 0


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description="Continue a music file with ComfyUI (ACE-Step 1.5).")
    parser.add_argument("--url", default=None, help="ComfyUI base url")
    parser.add_argument("--audio", default=None, help="music file to continue")
    parser.add_argument("--add", type=float, default=None, help="seconds to add")
    parser.add_argument("--workflow", default=None, help="API workflow json")
    parser.add_argument("--check", action="store_true", help="only test the ComfyUI setup, no GUI")
    parser.add_argument("--dry-run", action="store_true", help="write the patched workflow, do not run it")
    parser.add_argument("--out", default=None, help="output path for --dry-run")
    parser.add_argument("--prompt-file", default=None,
                        help="prompt/description file (default: auto-detected next to the audio)")
    parser.add_argument("--append-prompt", dest="append_prompt", action="store_true", default=None,
                        help="append the prompt file to the description (default)")
    parser.add_argument("--replace-prompt", dest="append_prompt", action="store_false",
                        help="let the prompt file replace the description")
    parser.add_argument("--captioner", default=None,
                        help="captioner node class, 'auto' (default) or 'off'")
    parser.add_argument("--captioner-url", default=None, help="MOSS-Music SGLang url")
    parser.add_argument("--caption", action="store_true",
                        help="caption the audio and print bpm/key/vocal tags, no generation")
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.url:
        settings.comfy_url = args.url
    if args.audio:
        settings.audio_file = args.audio
    if args.add:
        settings.add_seconds = args.add
    if args.workflow:
        settings.workflow = args.workflow
    if args.prompt_file:
        settings.prompt_file = args.prompt_file
    elif settings.audio_file and settings.auto_prompt_file:
        settings.prompt_file = find_prompt_file(settings.audio_file)
    if args.append_prompt is not None:
        settings.prompt_append = args.append_prompt
    if args.captioner:
        settings.captioner = args.captioner
    if args.captioner_url:
        settings.captioner_url = args.captioner_url

    if args.caption:
        return _cli_caption(settings)

    if args.check:
        node_classes = assets = preset = None
        try:
            with open(settings.workflow, "r", encoding="utf-8") as fh:
                workflow = json.load(fh)
            node_classes = workflow_node_classes(workflow)
            assets = workflow_assets(workflow)
            preset = model_preset(workflow)
        except (OSError, ValueError) as exc:
            print(f"(could not read workflow: {exc}; checking the default node set)")
        client = ComfyClient(settings.comfy_url)
        lines, ok = check_setup(client, node_classes, assets)
        if preset:
            lines.append(
                f"model preset: {preset['label']} -> {preset['steps']} steps, cfg {preset['cfg']}"
            )
        try:
            info = client.object_info()
            lines += describe_captioners(info, captioner_plan(info, settings.captioner))
        except ComfyError as exc:
            lines.append(f"(captioners not checked: {exc})")
        print("\n".join(lines))
        return 0 if ok else 1

    if args.dry_run:
        if not os.path.isfile(settings.workflow):
            print(f"workflow not found: {settings.workflow}", file=sys.stderr)
            return 2
        with open(settings.workflow, "r", encoding="utf-8") as fh:
            workflow = json.load(fh)
        prompt_text = read_prompt_file(settings.prompt_file) if settings.prompt_file else ""
        if prompt_text:
            print(f"prompt file: {settings.prompt_file} "
                  f"({'append' if settings.prompt_append else 'replace'})")
        settings.tags = effective_tags(settings, prompt_text)
        patched = patch_workflow(workflow, settings, uploaded_name=os.path.basename(settings.audio_file or "input.wav"))
        out = args.out or os.path.join(HERE, "patched_workflow.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(patched, fh, indent=2)
        print("wrote", out)
        return 0

    return None


# --------------------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------------------


class App:
    def __init__(self, root):
        self.root = root
        self.settings = load_settings()
        self.queue = []
        self.worker = None
        self.cancel_flag = threading.Event()
        self.last_output = None

        root.title(APP_NAME + " - ComfyUI / ACE-Step 1.5")
        root.minsize(820, 860)
        self._build_ui()
        self._apply_settings()
        if self.settings.window_geometry:
            try:
                root.geometry(self.settings.window_geometry)
            except tk.TclError:
                pass
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(120, self._drain_queue)

    # -- ui ----------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 6, "pady": 3}
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)

        # ComfyUI
        box = ttk.LabelFrame(outer, text="ComfyUI", padding=6)
        box.grid(row=0, column=0, columnspan=2, sticky="ew")
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="Server").grid(row=0, column=0, sticky="w", **pad)
        self.url_var = tk.StringVar()
        ttk.Entry(box, textvariable=self.url_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(box, text="Check setup", command=self.on_check).grid(row=0, column=2, **pad)
        ttk.Button(box, text="Workflow...", command=self.on_pick_workflow).grid(row=0, column=3, **pad)
        self.workflow_var = tk.StringVar()
        ttk.Label(box, textvariable=self.workflow_var, foreground="#666").grid(
            row=1, column=0, columnspan=4, sticky="w", padx=6
        )

        # input
        box = ttk.LabelFrame(outer, text="Input", padding=6)
        box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="Music file").grid(row=0, column=0, sticky="w", **pad)
        self.audio_var = tk.StringVar()
        ttk.Entry(box, textvariable=self.audio_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(box, text="Open...", command=self.on_open_audio).grid(row=0, column=2, **pad)
        self.audio_info = ttk.Label(box, text="-", foreground="#666")
        self.audio_info.grid(row=1, column=0, columnspan=3, sticky="w", padx=6)

        # continuation
        box = ttk.LabelFrame(outer, text="Continuation", padding=6)
        box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for col in (1, 3, 5):
            box.columnconfigure(col, weight=1)

        ttk.Label(box, text="Add seconds").grid(row=0, column=0, sticky="w", **pad)
        self.add_var = tk.StringVar(value="30")
        ttk.Spinbox(box, from_=1, to=900, increment=5, width=8, textvariable=self.add_var).grid(
            row=0, column=1, sticky="w", **pad
        )
        ttk.Label(box, text="Join crossfade (s)").grid(row=0, column=2, sticky="w", **pad)
        self.crossfade_var = tk.StringVar(value="1.0")
        ttk.Spinbox(box, from_=0, to=10, increment=0.25, width=8, textvariable=self.crossfade_var).grid(
            row=0, column=3, sticky="w", **pad
        )
        ttk.Label(box, text="Mask mode").grid(row=0, column=4, sticky="w", **pad)
        self.mask_var = tk.StringVar(value="generate_new_only")
        ttk.Combobox(
            box, textvariable=self.mask_var, width=18, state="readonly",
            values=["generate_new_only", "repaint_tail"],
        ).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(box, text="BPM (of the source)").grid(row=1, column=0, sticky="w", **pad)
        self.bpm_var = tk.StringVar(value="120")
        ttk.Spinbox(box, from_=10, to=300, width=8, textvariable=self.bpm_var).grid(
            row=1, column=1, sticky="w", **pad
        )
        ttk.Label(box, text="Key").grid(row=1, column=2, sticky="w", **pad)
        self.key_var = tk.StringVar(value="C major")
        ttk.Combobox(box, textvariable=self.key_var, width=10, values=KEYS).grid(
            row=1, column=3, sticky="w", **pad
        )
        ttk.Label(box, text="Language").grid(row=1, column=4, sticky="w", **pad)
        self.lang_var = tk.StringVar(value="en")
        ttk.Combobox(box, textvariable=self.lang_var, width=18, values=LANGUAGES).grid(
            row=1, column=5, sticky="w", **pad
        )

        ttk.Label(box, text="Steps").grid(row=2, column=0, sticky="w", **pad)
        self.steps_var = tk.StringVar(value="50")
        ttk.Spinbox(box, from_=1, to=150, width=8, textvariable=self.steps_var).grid(
            row=2, column=1, sticky="w", **pad
        )
        ttk.Label(box, text="CFG").grid(row=2, column=2, sticky="w", **pad)
        self.cfg_var = tk.StringVar(value="7.0")
        ttk.Spinbox(box, from_=0, to=20, increment=0.5, width=8, textvariable=self.cfg_var).grid(
            row=2, column=3, sticky="w", **pad
        )
        ttk.Label(box, text="Seed").grid(row=2, column=4, sticky="w", **pad)
        seedframe = ttk.Frame(box)
        seedframe.grid(row=2, column=5, sticky="w", **pad)
        self.seed_var = tk.StringVar(value="31")
        ttk.Entry(seedframe, textvariable=self.seed_var, width=12).pack(side="left")
        ttk.Button(seedframe, text="\U0001F3B2", width=3, command=self.on_dice).pack(side="left", padx=(4, 0))
        self.random_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="new seed each run", variable=self.random_var).grid(
            row=3, column=0, columnspan=2, sticky="w", **pad
        )
        self.codes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            box, text="LM audio-code planning (generate_audio_codes)", variable=self.codes_var
        ).grid(row=3, column=2, columnspan=4, sticky="w", **pad)
        ttk.Button(
            box, text="Sampler defaults for this workflow", command=self.on_apply_preset
        ).grid(row=4, column=0, columnspan=3, sticky="w", **pad)

        # prompt source: description text + prompt file + captioner
        box = ttk.LabelFrame(outer, text="Prompt source (description, prompt file, captioner)", padding=6)
        box.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Prompt file").grid(row=0, column=0, sticky="w", **pad)
        self.promptfile_var = tk.StringVar()
        ttk.Entry(box, textvariable=self.promptfile_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(box, text="Browse...", command=self.on_pick_prompt_file).grid(row=0, column=2, **pad)
        ttk.Button(box, text="Find next to audio", command=self.on_find_prompt_file).grid(
            row=0, column=3, **pad
        )
        self.append_prompt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            box, text="append the prompt file to the description (unticked: replace it)",
            variable=self.append_prompt_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", **pad)
        self.auto_prompt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            box, text="look for one automatically", variable=self.auto_prompt_var
        ).grid(row=1, column=2, columnspan=2, sticky="w", **pad)

        ttk.Label(box, text="Captioner").grid(row=2, column=0, sticky="w", **pad)
        self.captioner_var = tk.StringVar(value="auto")
        self.captioner_box = ttk.Combobox(
            box, textvariable=self.captioner_var, width=28, state="readonly",
            values=["auto", "off"],
        )
        self.captioner_box.grid(row=2, column=1, sticky="w", **pad)
        ttk.Button(box, text="Analyse source", command=self.on_caption_now).grid(row=2, column=2, **pad)
        ttk.Button(box, text="Refresh", command=self.on_refresh_captioners).grid(row=2, column=3, **pad)
        self.auto_caption_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            box, text="run the captioner before Generate", variable=self.auto_caption_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", **pad)
        self.fill_lyrics_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            box, text="fill empty lyrics from it", variable=self.fill_lyrics_var
        ).grid(row=3, column=2, columnspan=2, sticky="w", **pad)
        ttk.Label(box, text="MOSS-Music server").grid(row=4, column=0, sticky="w", **pad)
        self.captioner_url_var = tk.StringVar(value=MOSS_DEFAULT_URL)
        ttk.Entry(box, textvariable=self.captioner_url_var).grid(
            row=4, column=1, columnspan=3, sticky="ew", **pad
        )
        self.caption_info = ttk.Label(
            box, text="captioner not checked yet", foreground="#666", wraplength=700
        )
        self.caption_info.grid(row=5, column=0, columnspan=4, sticky="w", padx=6, pady=(2, 0))

        # description
        box = ttk.LabelFrame(outer, text="Music description (what the continuation should sound like)", padding=6)
        box.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.tags_text = tk.Text(box, height=4, wrap="word")
        self.tags_text.grid(row=0, column=0, sticky="nsew", **pad)
        ttk.Label(box, text="Lyrics (leave empty for instrumental continuation)").grid(
            row=1, column=0, sticky="w", padx=6
        )
        self.lyrics_text = tk.Text(box, height=3, wrap="word")
        self.lyrics_text.grid(row=2, column=0, sticky="nsew", **pad)

        # output
        box = ttk.LabelFrame(outer, text="Output", padding=6)
        box.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="ComfyUI prefix").grid(row=0, column=0, sticky="w", **pad)
        self.prefix_var = tk.StringVar(value="audio-continue/audio")
        ttk.Entry(box, textvariable=self.prefix_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(
            box,
            text="relative to ComfyUI/output  (output/audio-continue/)",
            foreground="#666",
        ).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(box, text="Save to folder").grid(row=2, column=0, sticky="w", **pad)
        self.outdir_var = tk.StringVar()
        ttk.Entry(box, textvariable=self.outdir_var).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(box, text="...", width=3, command=self.on_pick_outdir).grid(row=2, column=2, **pad)
        self.open_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="open the result when done", variable=self.open_var).grid(
            row=3, column=1, sticky="w", padx=6
        )

        # actions
        actions = ttk.Frame(outer)
        actions.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.generate_btn = ttk.Button(actions, text="Generate", command=self.on_generate)
        self.generate_btn.pack(side="left")
        ttk.Button(actions, text="Cancel / interrupt", command=self.on_cancel).pack(side="left", padx=6)
        ttk.Button(actions, text="Open output folder", command=self.on_open_folder).pack(side="left")
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=180)
        self.progress.pack(side="right")
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(actions, textvariable=self.status_var).pack(side="right", padx=8)

        # log
        logbox = ttk.LabelFrame(outer, text="Log", padding=4)
        logbox.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        outer.rowconfigure(7, weight=1)
        self.log_text = tk.Text(logbox, height=10, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

    # -- settings glue -----------------------------------------------------------

    def _apply_settings(self):
        s = self.settings
        self.url_var.set(s.comfy_url)
        self.workflow_var.set(s.workflow)
        self.audio_var.set(s.audio_file)
        self.promptfile_var.set(s.prompt_file)
        self.append_prompt_var.set(bool(s.prompt_append))
        self.auto_prompt_var.set(bool(s.auto_prompt_file))
        self.captioner_var.set(s.captioner or "auto")
        self.captioner_url_var.set(s.captioner_url or MOSS_DEFAULT_URL)
        self.auto_caption_var.set(bool(s.auto_caption))
        self.fill_lyrics_var.set(bool(s.caption_fill_lyrics))
        self.add_var.set(str(s.add_seconds))
        self.crossfade_var.set(str(s.crossfade))
        self.mask_var.set(s.mask_mode)
        self.bpm_var.set(str(s.bpm))
        self.key_var.set(s.keyscale)
        self.lang_var.set(s.language)
        self.steps_var.set(str(s.steps))
        self.cfg_var.set(str(s.cfg))
        self.seed_var.set(str(s.seed))
        self.random_var.set(s.randomize_seed)
        self.codes_var.set(s.generate_audio_codes)
        self.prefix_var.set(s.save_prefix)
        self.outdir_var.set(s.output_dir)
        self.open_var.set(s.open_when_done)
        self.tags_text.delete("1.0", "end")
        self.tags_text.insert("1.0", s.tags)
        self.lyrics_text.delete("1.0", "end")
        self.lyrics_text.insert("1.0", s.lyrics)
        if s.audio_file:
            self._update_audio_info()

    def _collect_settings(self):
        s = self.settings
        s.comfy_url = self.url_var.get().strip() or "http://127.0.0.1:8188"
        s.workflow = self.workflow_var.get().strip() or DEFAULT_WORKFLOW
        s.audio_file = self.audio_var.get().strip()
        s.prompt_file = self.promptfile_var.get().strip()
        s.prompt_append = bool(self.append_prompt_var.get())
        s.auto_prompt_file = bool(self.auto_prompt_var.get())
        s.captioner = self.captioner_var.get().strip() or "auto"
        s.captioner_url = self.captioner_url_var.get().strip() or MOSS_DEFAULT_URL
        s.auto_caption = bool(self.auto_caption_var.get())
        s.caption_fill_lyrics = bool(self.fill_lyrics_var.get())
        s.add_seconds = _as_float(self.add_var.get(), 30.0)
        s.crossfade = _as_float(self.crossfade_var.get(), 1.0)
        s.mask_mode = self.mask_var.get()
        s.bpm = int(_as_float(self.bpm_var.get(), 120))
        s.keyscale = self.key_var.get()
        s.language = self.lang_var.get()
        s.steps = int(_as_float(self.steps_var.get(), 50))
        s.cfg = _as_float(self.cfg_var.get(), 7.0)
        s.seed = int(_as_float(self.seed_var.get(), 31))
        s.randomize_seed = bool(self.random_var.get())
        s.generate_audio_codes = bool(self.codes_var.get())
        s.save_prefix = self.prefix_var.get().strip() or "audio-continue/audio"
        s.output_dir = self.outdir_var.get().strip()
        s.open_when_done = bool(self.open_var.get())
        s.tags = self.tags_text.get("1.0", "end").strip()
        s.lyrics = self.lyrics_text.get("1.0", "end").strip()
        try:
            s.window_geometry = self.root.winfo_geometry()
        except tk.TclError:
            pass
        if s.randomize_seed:
            s.seed = random.randint(0, 2**32 - 1)
            self.seed_var.set(str(s.seed))
        return s

    def _update_audio_info(self):
        path = self.audio_var.get().strip()
        if path and os.path.isfile(path):
            size = os.path.getsize(path) / 1e6
            info = f"{path}  ({size:.1f} MB)"
            if self.auto_prompt_var.get():
                sidecar = find_prompt_file(path)
                if sidecar:
                    info += f"   prompt: {os.path.basename(sidecar)}"
                    self.promptfile_var.set(sidecar)
            self.audio_info.configure(text=info, foreground="#446")
            if not self.outdir_var.get().strip():
                self.outdir_var.set(os.path.dirname(os.path.abspath(path)))
        else:
            self.audio_info.configure(text="no file selected", foreground="#a00")

    # -- buttons -----------------------------------------------------------------

    def on_open_audio(self):
        path = filedialog.askopenfilename(
            title="Choose the music file to continue",
            filetypes=[("Audio", " ".join("*" + e for e in AUDIO_EXTENSIONS)), ("All files", "*.*")],
        )
        if path:
            self.audio_var.set(path)
            self.outdir_var.set(os.path.dirname(path))
            self._update_audio_info()

    def on_pick_workflow(self):
        path = filedialog.askopenfilename(
            title="API workflow json", initialdir=os.path.join(HERE, "workflows"),
            filetypes=[("JSON", "*.json")],
        )
        if path:
            self.workflow_var.set(path)
            self._apply_model_defaults(path)

    def on_apply_preset(self):
        self._apply_model_defaults(self.workflow_var.get().strip() or DEFAULT_WORKFLOW)

    def _apply_model_defaults(self, path):
        """Copy the model's recommended sampler numbers into the Steps / CFG fields."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                workflow = json.load(fh)
        except (OSError, ValueError) as exc:
            self._log(f"could not read workflow ({exc})")
            return
        preset = model_preset(workflow)
        if not preset:
            self._log("model variant not recognised - steps / CFG left as they are")
            return
        self.steps_var.set(str(preset["steps"]))
        self.cfg_var.set(str(preset["cfg"]))
        self.codes_var.set(bool(preset["generate_audio_codes"]))
        self._log(
            f"{preset['label']}: {preset['steps']} steps, cfg {preset['cfg']} "
            "(change this only if you know what you are doing)"
        )

    def on_pick_outdir(self):
        path = filedialog.askdirectory(title="Where to save the continued file")
        if path:
            self.outdir_var.set(path)

    def on_dice(self):
        self.seed_var.set(str(random.randint(0, 2**32 - 1)))

    # -- prompt file / captioner -------------------------------------------------

    def on_pick_prompt_file(self):
        initial = os.path.dirname(self.promptfile_var.get() or self.audio_var.get() or "")
        path = filedialog.askopenfilename(
            title="Prompt / description file",
            initialdir=initial or None,
            filetypes=[("Prompt files", "*.txt *.md *.json"), ("All files", "*.*")],
        )
        if path:
            self.promptfile_var.set(path)
            preview = read_prompt_file(path)
            if preview:
                self._log(f"prompt file: {os.path.basename(path)} ({len(preview)} chars)")
            else:
                self._log(f"prompt file {path} has no usable text")

    def on_find_prompt_file(self):
        audio = self.audio_var.get().strip()
        if not audio:
            messagebox.showinfo(APP_NAME, "Pick the music file first.")
            return
        sidecar = find_prompt_file(audio)
        if sidecar:
            self.promptfile_var.set(sidecar)
            self._log(f"found prompt file next to the audio: {sidecar}")
        else:
            self.promptfile_var.set("")
            self._log("no prompt sidecar found (*.prompt.txt, *.caption.txt, *.tags.txt, *.txt, *.json)")

    def on_refresh_captioners(self):
        settings = self._collect_settings()
        save_settings(settings)

        def job():
            client = ComfyClient(settings.comfy_url)
            info = client.object_info()
            plan = captioner_plan(info, settings.captioner)
            choices = ["auto", "off"]
            choices += [name for name in CAPTIONER_PREFERENCE if name in info]
            choices += [item["class"] for item in captioner_candidates(info)
                        if item["class"] not in choices]
            self._post("captioners", (choices, plan))
            for line in describe_captioners(info, plan):
                self._log(line)

        self._start(job, "looking for captioners")

    def on_caption_now(self):
        if self.worker and self.worker.is_alive():
            return
        settings = self._collect_settings()
        save_settings(settings)
        if not settings.audio_file or not os.path.isfile(settings.audio_file):
            messagebox.showerror(APP_NAME, "Pick a music file first.")
            return
        self.cancel_flag.clear()

        def job():
            client = ComfyClient(settings.comfy_url)
            info = client.object_info()
            plan = captioner_plan(info, settings.captioner)
            if plan["mode"] == "none" or settings.captioner == "off":
                self._log("captioner: " + (plan.get("reason") or "disabled"))
                self._status("no captioner")
                return
            self._log(f"captioner: {plan['label']} ({plan['mode']}, {plan['reason']})")
            self._log("uploading " + os.path.basename(settings.audio_file))
            uploaded = client.upload_audio(settings.audio_file)
            result = self._caption_run(client, settings, uploaded, info, plan)
            if result:
                self._apply_caption_result(result, settings)

        self._start(job, "captioning")

    def _caption_run(self, client, settings, uploaded, object_info, plan):
        """Queue the caption workflow, wait for it and return the parsed result."""
        prompt, meta = build_caption_prompt(
            object_info,
            plan,
            uploaded,
            moss_url=settings.captioner_url,
            prefer_numeric=settings.caption_prefer_numeric,
        )
        prompt_id = client.queue_prompt(prompt)
        self._status("captioning")
        entry = client.wait(prompt_id)
        result = caption_result_from_history(entry, meta)
        if not (result.get("bpm") or result.get("keyscale") or result.get("caption")
                or result.get("tags")):
            self._log("captioner returned nothing usable: " + json.dumps(result.get("raw"))[:400])
            return None
        return result

    def _apply_caption_result(self, result, settings):
        changes = merge_caption_into_settings(
            settings, result, fill_lyrics=bool(settings.caption_fill_lyrics)
        )
        self._post("caption", (result, changes))
        notes = ", ".join(result.get("notes") or [])
        self._log("caption: " + (notes or "(only free text, no bpm/key/vocals)"))
        for change in changes:
            self._log("  " + change)
        caption_tags = caption_tags_from_result(result)
        if caption_tags:
            self._log("  tags from captioner: " + caption_tags[:300])

    def on_check(self):
        settings = self._collect_settings()
        save_settings(settings)

        def job():
            node_classes = assets = preset = None
            try:
                with open(settings.workflow, "r", encoding="utf-8") as fh:
                    workflow = json.load(fh)
                node_classes = workflow_node_classes(workflow)
                assets = workflow_assets(workflow)
                preset = model_preset(workflow)
            except (OSError, ValueError) as exc:
                self._log(f"could not read workflow ({exc}); checking the default node set")
            client = ComfyClient(settings.comfy_url)
            lines, ok = check_setup(client, node_classes, assets)
            if preset:
                lines.append(
                    f"model preset: {preset['label']} -> {preset['steps']} steps, cfg {preset['cfg']}"
                )
            try:
                info = client.object_info()
                lines += describe_captioners(info, captioner_plan(info, settings.captioner))
            except ComfyError:
                pass
            for line in lines:
                self._log(line)
            self._status("ready" if ok else "setup incomplete")

        self._log("checking " + settings.comfy_url)
        self._start(job, "checking")

    def on_cancel(self):
        self.cancel_flag.set()
        try:
            ComfyClient(self.url_var.get()).interrupt()
            self._log("sent /interrupt")
        except ComfyError as exc:
            self._log(str(exc))
        self._status("cancelling")

    def on_open_folder(self):
        folder = self.outdir_var.get().strip() or os.path.dirname(self.audio_var.get() or "")
        open_in_file_manager(folder)

    def on_generate(self):
        if self.worker and self.worker.is_alive():
            return
        settings = self._collect_settings()
        save_settings(settings)

        if not settings.audio_file or not os.path.isfile(settings.audio_file):
            messagebox.showerror(APP_NAME, "Pick a music file first.")
            return
        if not os.path.isfile(settings.workflow):
            messagebox.showerror(APP_NAME, f"Workflow not found:\n{settings.workflow}")
            return
        if settings.add_seconds <= 0:
            messagebox.showerror(APP_NAME, "Add seconds must be > 0.")
            return
        if not settings.output_dir:
            settings.output_dir = os.path.dirname(os.path.abspath(settings.audio_file))

        self.cancel_flag.clear()
        self.last_output = None
        self._start(lambda: self._generate_job(settings), "generate")

    # -- worker plumbing ---------------------------------------------------------

    def _start(self, job, label):
        if self.worker and self.worker.is_alive():
            self._log("a job is already running")
            return
        self.generate_btn.configure(state="disabled")
        self.progress.start(15)
        self._status(label)

        def run():
            try:
                job()
            except ComfyError as exc:
                self._log("ERROR: " + str(exc))
                self._status("failed")
            except Exception as exc:  # noqa: BLE001 - keep the GUI alive
                self._log(f"ERROR ({type(exc).__name__}): {exc}")
                self._status("failed")
            finally:
                self._status_done()

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _status_done(self):
        self.queue.append(("done", None))

    def _log(self, message):
        self.queue.append(("log", message))

    def _status(self, message):
        self.queue.append(("status", message))

    def _drain_queue(self):
        while self.queue:
            kind, value = self.queue.pop(0)
            if kind == "log":
                stamp = time.strftime("%H:%M:%S")
                self.log_text.configure(state="normal")
                self.log_text.insert("end", f"[{stamp}] {value}\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
            elif kind == "status":
                self.status_var.set(value)
            elif kind == "done":
                self.progress.stop()
                self.generate_btn.configure(state="normal")
            elif kind == "saved":
                self.last_output = value
                self._log("saved: " + value)
                if self.open_var.get():
                    open_in_file_manager(os.path.dirname(value))
            elif kind == "caption":
                self._show_caption(*value)
            elif kind == "captioners":
                choices, plan = value
                self.captioner_box.configure(values=choices)
                self._show_caption_plan(plan)
        self.root.after(120, self._drain_queue)

    def _post(self, kind, value):
        self.queue.append((kind, value))

    def _show_caption(self, result, changes):
        """Put a caption result on screen (bpm / key fields + one info line)."""
        if result.get("bpm"):
            self.bpm_var.set(str(int(result["bpm"])))
        if result.get("keyscale"):
            self.key_var.set(result["keyscale"])
        bits = list(result.get("notes") or [])
        if result.get("structure"):
            bits.append("structure: " + result["structure"])
        if changes:
            bits.append("(" + "; ".join(changes) + ")")
        text = " · ".join(bits) if bits else "captioner returned no bpm / key / vocal info"
        self.caption_info.configure(text=text, foreground="#264" if bits else "#a00")

    def _show_caption_plan(self, plan):
        if plan.get("mode") == "none":
            self.caption_info.configure(text=plan.get("reason", "no captioner"), foreground="#a00")
        else:
            self.caption_info.configure(
                text=f"{plan['label']} available ({plan['mode']}) - press Analyse source",
                foreground="#264",
            )

    # -- the actual job ----------------------------------------------------------

    def _generate_job(self, settings):
        client = ComfyClient(settings.comfy_url)
        self._log(f"ComfyUI: {client.base_url}")

        with open(settings.workflow, "r", encoding="utf-8") as fh:
            workflow = json.load(fh)

        self._log(f"checking setup ({len(workflow)} nodes in workflow)")
        lines, ok = check_setup(
            client, workflow_node_classes(workflow), workflow_assets(workflow)
        )
        for line in lines:
            self._log("  " + line)
        if not ok:
            raise ComfyError("setup check failed, see log")

        self._log(f"uploading {os.path.basename(settings.audio_file)}")
        uploaded = client.upload_audio(settings.audio_file)
        self._log(f"  -> {uploaded}")

        # prompt file + captioner change the tag prompt (and sometimes bpm/key)
        prompt_text = ""
        if settings.prompt_file:
            prompt_text = read_prompt_file(settings.prompt_file)
            if prompt_text:
                mode = "append" if settings.prompt_append else "replace"
                self._log(
                    f"prompt file ({mode}): {os.path.basename(settings.prompt_file)} "
                    f"({len(prompt_text)} chars)"
                )
            else:
                self._log(f"prompt file {settings.prompt_file} is empty/unreadable - ignored")

        caption_tags = ""
        if settings.auto_caption and settings.captioner != "off":
            try:
                info = client.object_info()
                plan = captioner_plan(info, settings.captioner)
                if plan["mode"] == "none":
                    self._log("captioner: " + (plan.get("reason") or "not available"))
                else:
                    self._log(f"captioner: {plan['label']} ({plan['mode']})")
                    result = self._caption_run(client, settings, uploaded, info, plan)
                    if result:
                        self._apply_caption_result(result, settings)
                        caption_tags = caption_tags_from_result(result)
            except ComfyError as exc:
                self._log("captioner failed, continuing without it: " + str(exc))

        settings.tags = effective_tags(settings, prompt_text, caption_tags)
        self._log("prompt: " + settings.tags[:400])

        prompt = patch_workflow(workflow, settings, uploaded_name=uploaded)
        self._log(
            f"queueing: +{settings.add_seconds:.1f}s, bpm {settings.bpm}, {settings.keyscale}, "
            f"seed {settings.seed}, {settings.steps} steps, cfg {settings.cfg}"
        )
        prompt_id = client.queue_prompt(prompt)
        self._log(f"prompt id {prompt_id}")
        self._status("running")

        last = {"value": None}

        def progress(state):
            if state != last["value"]:
                last["value"] = state
                self._status(state)

        entry = client.wait(prompt_id, on_progress=progress)
        outputs = find_audio_outputs(entry)
        if not outputs:
            raise ComfyError("the run finished but produced no audio file:\n" + json.dumps(entry)[:2000])
        info = outputs[0]
        self._log(f"ComfyUI wrote output/{info.get('subfolder', '')}/{info['filename']}")

        out_dir = settings.output_dir or os.path.dirname(os.path.abspath(settings.audio_file))
        destination = os.path.join(out_dir, make_output_name(settings.audio_file, settings.add_seconds, info["filename"]))
        if os.path.exists(destination):
            base, ext = os.path.splitext(destination)
            destination = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
        client.download_output(info, destination)
        self._post("saved", destination)
        self._status("done")

    # -- shutdown ----------------------------------------------------------------

    def on_close(self):
        settings = self._collect_settings()
        save_settings(settings)
        self.root.destroy()


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------


def _as_float(value, default):
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default


def open_in_file_manager(path):
    if not path or not os.path.exists(path):
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except OSError as exc:
        print(f"could not open {path}: {exc}", file=sys.stderr)


def run_gui():
    if tk is None:
        print(
            "tkinter is not available in this Python. Install it (python3-tk on Debian/Ubuntu, "
            "the python.org installer on Windows/macOS) or use --check / --dry-run.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


def main(argv=None):
    result = main_cli(argv)
    if result is not None:
        return result
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        print(
            "No DISPLAY found. Use --check to test the ComfyUI setup without the GUI.",
            file=sys.stderr,
        )
    run_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
