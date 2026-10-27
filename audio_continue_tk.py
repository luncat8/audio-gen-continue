#!/usr/bin/env python3
"""
audio_continue_tk.py - small Tk GUI that continues an existing music file with a
ComfyUI server (ACE-Step 1.5 by default).

    python3 audio_continue_tk.py

Workflow
    1. pick a music file
    2. set how many seconds to add
    3. press Generate

The app uploads the file to ComfyUI, patches workflows/ace15_audio_continue_api.json,
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
DEFAULT_WORKFLOW = os.path.join(HERE, "workflows", "ace15_audio_continue_api.json")
CONFIG_PATH = os.environ.get(
    "AUDIO_CONTINUE_CONFIG", os.path.join(os.path.expanduser("~"), ".audio_continue.json")
)

REQUIRED_NODES = [
    "CheckpointLoaderSimple",
    "ModelSamplingAuraFlow",
    "LoadAudio",
    "AudioContinueExtend",
    "AudioContinueStitch",
    "TextEncodeAceStepAudio1.5",
    "ConditioningZeroOut",
    "KSampler",
    "VAEDecodeAudio",
    "SaveAudio",
]

AUDIO_EXTENSIONS = (
    ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aif", ".aiff", ".wma",
)

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
    add_seconds: float = 30.0
    tags: str = DEFAULT_TAGS
    lyrics: str = ""
    bpm: int = 120
    keyscale: str = "C major"
    language: str = "en"
    seed: int = 31
    randomize_seed: bool = True
    steps: int = 8
    cfg: float = 1.0
    generate_audio_codes: bool = True
    mask_mode: str = "generate_new_only"
    repaint_seconds: float = 0.0
    crossfade: float = 1.0
    save_prefix: str = "audio-continue/audio"
    output_dir: str = ""
    open_when_done: bool = True
    window_geometry: str = ""


def load_settings(path=CONFIG_PATH):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return Settings()
    if not isinstance(data, dict):
        return Settings()
    known = {f for f in Settings().__dataclass_fields__}
    return Settings(**{k: v for k, v in data.items() if k in known})


def save_settings(settings, path=CONFIG_PATH):
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


def workflow_checkpoint(workflow):
    for node in workflow.values():
        if node.get("class_type") == "CheckpointLoaderSimple":
            return (node.get("inputs") or {}).get("ckpt_name")
    return None


def check_setup(client, node_classes=None, checkpoint=None):
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

    ckpts = []
    try:
        entry = info.get("CheckpointLoaderSimple") or client.object_info("CheckpointLoaderSimple")
        ckpts = ((entry or {}).get("input", {}).get("required", {}).get("ckpt_name") or [[]])[0]
    except (ComfyError, IndexError, AttributeError):
        pass
    if ckpts:
        lines.append("checkpoints: " + ", ".join(map(str, ckpts[:12])))
        if checkpoint:
            if checkpoint not in ckpts:
                lines.append(f"  ! workflow checkpoint '{checkpoint}' is not installed")
                ok = False
        elif not any("ace_step" in str(c).lower() for c in ckpts):
            lines.append("  ! no ACE-Step checkpoint found (expected ace_step_1.5_turbo_aio.safetensors)")
    return lines, ok


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description="Continue a music file with ComfyUI (ACE-Step 1.5).")
    parser.add_argument("--url", default=None, help="ComfyUI base url")
    parser.add_argument("--audio", default=None, help="music file to continue")
    parser.add_argument("--add", type=float, default=None, help="seconds to add")
    parser.add_argument("--workflow", default=None, help="API workflow json")
    parser.add_argument("--check", action="store_true", help="only test the ComfyUI setup, no GUI")
    parser.add_argument("--dry-run", action="store_true", help="write the patched workflow, do not run it")
    parser.add_argument("--out", default=None, help="output path for --dry-run")
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

    if args.check:
        node_classes = checkpoint = None
        try:
            with open(settings.workflow, "r", encoding="utf-8") as fh:
                workflow = json.load(fh)
            node_classes = workflow_node_classes(workflow)
            checkpoint = workflow_checkpoint(workflow)
        except (OSError, ValueError) as exc:
            print(f"(could not read workflow: {exc}; checking the default node set)")
        lines, ok = check_setup(ComfyClient(settings.comfy_url), node_classes, checkpoint)
        print("\n".join(lines))
        return 0 if ok else 1

    if args.dry_run:
        if not os.path.isfile(settings.workflow):
            print(f"workflow not found: {settings.workflow}", file=sys.stderr)
            return 2
        with open(settings.workflow, "r", encoding="utf-8") as fh:
            workflow = json.load(fh)
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
        root.minsize(760, 720)
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
        self.steps_var = tk.StringVar(value="8")
        ttk.Spinbox(box, from_=1, to=100, width=8, textvariable=self.steps_var).grid(
            row=2, column=1, sticky="w", **pad
        )
        ttk.Label(box, text="CFG").grid(row=2, column=2, sticky="w", **pad)
        self.cfg_var = tk.StringVar(value="1.0")
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

        # description
        box = ttk.LabelFrame(outer, text="Music description (what the continuation should sound like)", padding=6)
        box.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.tags_text = tk.Text(box, height=4, wrap="word")
        self.tags_text.grid(row=0, column=0, sticky="nsew", **pad)
        ttk.Label(box, text="Lyrics (leave empty for instrumental continuation)").grid(
            row=1, column=0, sticky="w", padx=6
        )
        self.lyrics_text = tk.Text(box, height=4, wrap="word")
        self.lyrics_text.grid(row=2, column=0, sticky="nsew", **pad)

        # output
        box = ttk.LabelFrame(outer, text="Output", padding=6)
        box.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
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
        actions.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))
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
        logbox.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        outer.rowconfigure(6, weight=1)
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
        s.add_seconds = _as_float(self.add_var.get(), 30.0)
        s.crossfade = _as_float(self.crossfade_var.get(), 1.0)
        s.mask_mode = self.mask_var.get()
        s.bpm = int(_as_float(self.bpm_var.get(), 120))
        s.keyscale = self.key_var.get()
        s.language = self.lang_var.get()
        s.steps = int(_as_float(self.steps_var.get(), 8))
        s.cfg = _as_float(self.cfg_var.get(), 1.0)
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
            self.audio_info.configure(text=f"{path}  ({size:.1f} MB)", foreground="#446")
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

    def on_pick_outdir(self):
        path = filedialog.askdirectory(title="Where to save the continued file")
        if path:
            self.outdir_var.set(path)

    def on_dice(self):
        self.seed_var.set(str(random.randint(0, 2**32 - 1)))

    def on_check(self):
        settings = self._collect_settings()
        save_settings(settings)

        def job():
            node_classes = checkpoint = None
            try:
                with open(settings.workflow, "r", encoding="utf-8") as fh:
                    workflow = json.load(fh)
                node_classes = workflow_node_classes(workflow)
                checkpoint = workflow_checkpoint(workflow)
            except (OSError, ValueError) as exc:
                self._log(f"could not read workflow ({exc}); checking the default node set")
            lines, ok = check_setup(ComfyClient(settings.comfy_url), node_classes, checkpoint)
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
        self.root.after(120, self._drain_queue)

    def _post(self, kind, value):
        self.queue.append((kind, value))

    # -- the actual job ----------------------------------------------------------

    def _generate_job(self, settings):
        client = ComfyClient(settings.comfy_url)
        self._log(f"ComfyUI: {client.base_url}")

        with open(settings.workflow, "r", encoding="utf-8") as fh:
            workflow = json.load(fh)

        self._log(f"checking setup ({len(workflow)} nodes in workflow)")
        lines, ok = check_setup(
            client, workflow_node_classes(workflow), workflow_checkpoint(workflow)
        )
        for line in lines:
            self._log("  " + line)
        if not ok:
            raise ComfyError("setup check failed, see log")

        self._log(f"uploading {os.path.basename(settings.audio_file)}")
        uploaded = client.upload_audio(settings.audio_file)
        self._log(f"  -> {uploaded}")

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
