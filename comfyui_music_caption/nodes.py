"""
Music captioner nodes for ComfyUI - measure the source track and turn it into a
prompt a music model can actually use, or inspect all musical attributes in ComfyUI.

    MusicCaptionMOSSLoader  -> moss_model + status (loads OpenMOSS-Team/MOSS-Music-8B-Instruct
                               from ComfyUI/models/moss_music/, or falls back to DSP / SGLang)
    MusicCaptionAnalyze     AUDIO -> bpm / key / vocal presence+type / duration (numpy DSP)
    MusicCaptionMOSS        AUDIO (+ moss_model) -> caption / lyrics / vocal details / key / bpm
    MusicCaptionMerge       everything -> tags / lyrics / bpm / keyscale for TextEncode*
    MusicCaptionView        any caption JSON -> readable report + individual fields on canvas
    MusicCaptionTap         * -> text, for captioners from other packs

Why a separate pack: none of the existing ComfyUI captioners report the four
things a continuation prompt needs (key, BPM, vocal presence, vocal type) in a
form you can wire into an ACE-Step / MiniMax text encoder or read directly in a
ComfyUI workflow. See CAPTIONER.md in the repository for the comparison.

All sink/analysis nodes set `OUTPUT_NODE = True` and write their result into the
history `ui.text` block so ComfyUI executes them standalone (no `prompt_no_outputs`
error) and `audio_continue_tk.py` can read them back without extra preview nodes.
"""

import json
import os
import tempfile
import wave

import numpy as np

try:  # ComfyUI provides this; the plain unit tests do not
    import folder_paths
except ImportError:  # pragma: no cover
    folder_paths = None

from . import dsp, moss, moss_local, textutil

moss_local.register_folder_path()

CATEGORY = "audio/caption"

ANALYZE_OUTPUTS = (
    "tags",
    "bpm",
    "keyscale",
    "key_confidence",
    "bpm_confidence",
    "vocal_presence",
    "vocal_type",
    "duration_seconds",
    "analysis_json",
)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _to_numpy(waveform):
    """torch [B, C, T] (or anything array-like) -> numpy [C, T] float32."""
    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    if hasattr(waveform, "float"):
        try:
            waveform = waveform.float()
        except (TypeError, RuntimeError):
            pass
    array = waveform.numpy() if hasattr(waveform, "numpy") else np.asarray(waveform)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    return np.nan_to_num(array)


def _audio_payload(audio):
    waveform = _to_numpy(audio["waveform"])
    return waveform, int(audio["sample_rate"])


def _write_wav(path, waveform, sample_rate):
    """Write [C, T] float32 to a 16-bit PCM wav using only the standard library."""
    samples = np.clip(waveform, -1.0, 1.0)
    if samples.shape[0] > 2:  # mono/stereo only, MOSS does not need more
        samples = samples[:2]
    interleaved = (samples.T * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(samples.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(interleaved.tobytes())
    return path


def _segment(waveform, sample_rate, segment, max_seconds):
    if not max_seconds or max_seconds <= 0:
        return waveform
    limit = int(max_seconds * sample_rate)
    if waveform.shape[-1] <= limit:
        return waveform
    if segment == "head":
        return waveform[:, :limit]
    if segment == "full":
        # evenly spaced chunks would be nicer, but a hard trim keeps timing readable
        return waveform[:, :limit]
    return waveform[:, -limit:]


def _ui_json(payload):
    return {"ui": {"text": [json.dumps(payload, ensure_ascii=False)]}}


def _temp_audio(waveform, sample_rate, label):
    folder = None
    if folder_paths is not None:
        try:
            folder = folder_paths.get_temp_directory()
        except Exception:  # noqa: BLE001 - older ComfyUI, or no temp dir support
            folder = None
    folder = folder or tempfile.gettempdir()
    os.makedirs(folder, exist_ok=True)
    name = f"music_caption_{label}_{os.getpid()}.wav"
    return _write_wav(os.path.join(folder, name), waveform, sample_rate)


def format_caption_report(summary=None, measured=None, model=None, include_sections=True):
    """Build a human-readable report of every measured and MOSS-Music field."""
    summary = summary if isinstance(summary, dict) else {}
    measured = measured if isinstance(measured, dict) else (summary.get("measured") or {})
    model = model if isinstance(model, dict) else (summary.get("model") or {})

    lines = ["=== Music Caption & Analysis ==="]

    duration = measured.get("duration_seconds")
    analysed = measured.get("analysed_seconds")
    sr = measured.get("sample_rate")
    if duration:
        dur_line = f"Duration:     {float(duration):.1f} s"
        if analysed and float(analysed) < float(duration) - 0.5:
            dur_line += f" (analysed tail: {float(analysed):.1f} s)"
        if sr:
            dur_line += f"  |  {int(sr)} Hz"
        lines.append(dur_line)

    bpm = summary.get("bpm") or textutil.normalize_bpm(measured.get("bpm")) or textutil.normalize_bpm(model.get("bpm"))
    dsp_bpm = measured.get("bpm")
    bpm_conf = measured.get("bpm_confidence")
    candidates = measured.get("bpm_candidates") or []
    feel = measured.get("tempo_feel") or (dsp.tempo_feel(bpm) if bpm else "")
    bpm_details = []
    if dsp_bpm:
        bpm_details.append(f"DSP {dsp_bpm}" + (f" [conf {bpm_conf:.2f}]" if isinstance(bpm_conf, (int, float)) else ""))
    if model.get("bpm"):
        bpm_details.append(f"MOSS {model['bpm']}")
    if candidates:
        bpm_details.append("octave alt: " + ", ".join(str(c) for c in candidates[:3]))
    if feel and feel != "unknown":
        bpm_details.append(feel)
    lines.append(
        f"BPM:          {bpm or '-'} BPM"
        + (f"  ({'; '.join(bpm_details)})" if bpm_details else "")
    )

    keyscale = (
        summary.get("keyscale")
        or textutil.normalize_key(measured.get("key"))
        or textutil.normalize_key(model.get("keyscale") or model.get("key"))
    )
    camelot = measured.get("camelot") or model.get("camelot") or ""
    key_conf = measured.get("key_confidence")
    key_cands = measured.get("key_candidates") or []
    key_details = []
    if camelot:
        key_details.append(f"Camelot {camelot}")
    if isinstance(key_conf, (int, float)) and measured.get("key"):
        key_details.append(f"DSP {measured['key']} [conf {key_conf:.2f}]")
    if model.get("keyscale") or model.get("key"):
        key_details.append(f"MOSS {model.get('keyscale') or model.get('key')}")
    if len(key_cands) > 1:
        runners = [c.get("key") for c in key_cands[1:3] if isinstance(c, dict) and c.get("key")]
        if runners:
            key_details.append("runners-up: " + ", ".join(runners))
    lines.append(
        f"Key / Scale:  {keyscale or '-'}"
        + (f"  ({'; '.join(key_details)})" if key_details else "")
    )

    dsp_vocal = measured.get("vocal") if isinstance(measured.get("vocal"), dict) else {}
    presence = dsp_vocal.get("presence") or ("instrumental" if summary.get("instrumental") else "")
    vocal_type = (
        summary.get("vocal_type")
        or model.get("vocal_type")
        or dsp_vocal.get("type")
        or ""
    )
    vocal_desc = model.get("vocal") or ""
    vocal_bits = [b for b in (presence, vocal_type) if b and b != "unknown"]
    if dsp_vocal.get("median_f0_hz"):
        vocal_bits.append(f"median f0 {dsp_vocal['median_f0_hz']} Hz")
    if isinstance(dsp_vocal.get("confidence"), (int, float)) and presence:
        vocal_bits.append(f"DSP conf {dsp_vocal['confidence']:.2f}")
    vocal_line = ", ".join(dict.fromkeys(vocal_bits)) or "unknown"
    if vocal_desc and vocal_desc.lower() not in vocal_line.lower():
        vocal_line += f" — {vocal_desc}"
    lines.append(f"Vocals:       {vocal_line}")

    instrumental = summary.get("instrumental")
    if instrumental is None:
        instrumental = model.get("instrumental")
    language = summary.get("language") or model.get("language") or ""
    if instrumental is not None or language:
        inst_str = "yes" if instrumental else ("no" if instrumental is False else "unknown")
        extra_lang = f"  |  Language: {language}" if language else ""
        lines.append(f"Instrumental: {inst_str}{extra_lang}")

    structure = summary.get("structure") or model.get("structure") or ""
    if structure:
        lines.append(f"Structure:    {structure}")

    chords = summary.get("chords") or model.get("chords") or ""
    if chords:
        lines.append(f"Chords:       {chords}")

    loudness = measured.get("loudness") if isinstance(measured.get("loudness"), dict) else {}
    bright = measured.get("brightness_hz")
    if loudness:
        loud_line = (
            f"Loudness:     RMS {loudness.get('rms_db', '?')} dB  |  "
            f"Peak {loudness.get('peak_db', '?')} dB  |  "
            f"Crest {loudness.get('crest_db', '?')} dB"
        )
        if bright:
            loud_line += f"  |  Brightness {bright} Hz"
        lines.append(loud_line)

    sections = measured.get("sections") or []
    if include_sections and sections:
        preview = [
            f"{int(round(s.get('start', 0)))}-{int(round(s.get('end', 0)))}s "
            f"{s.get('level', '?')} ({s.get('rms_db', '?')} dB)"
            for s in sections[:10]
        ]
        if len(sections) > 10:
            preview.append(f"... ({len(sections)} blocks)")
        lines.append("Sections:     " + ", ".join(preview))

    warnings = measured.get("warnings") or []
    if warnings:
        lines.append("DSP notes:    " + "; ".join(str(w) for w in warnings))

    lines.append("")
    if model.get("ok"):
        via = model.get("backend") or "server"
        mname = model.get("model_name") or model.get("server") or "MOSS-Music"
        lines.append(f"MOSS-Music:   active ({via}: {mname})")
    elif model.get("error"):
        lines.append("MOSS-Music:   not active — using DSP measurements only")
        for err_line in str(model["error"]).splitlines():
            lines.append("              " + err_line)
    else:
        lines.append(
            "MOSS-Music:   not run (connect MOSS-Music Loader -> Music Caption (MOSS-Music) "
            f"with weights from {moss_local.HF_URL_INSTRUCT})"
        )

    caption = model.get("caption") or ""
    if caption:
        lines.extend(["", "Caption (MOSS-Music):", caption])

    tags = summary.get("tags") or measured.get("tags") or caption
    if tags:
        lines.extend(["", "Prompt tags:", tags])

    lyrics = summary.get("lyrics") or model.get("lyrics") or ""
    if lyrics:
        lines.extend(["", "Lyrics:", lyrics])

    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# MusicCaptionMOSSLoader - standard ComfyUI loader for MOSS-Music-8B
# --------------------------------------------------------------------------------------


class MusicCaptionMOSSLoader:
    """Select and configure a local MOSS-Music 8B checkpoint (or SGLang fallback)."""

    @classmethod
    def INPUT_TYPES(cls):
        installed = moss_local.installed_models()
        choices = moss_local.loader_choices()
        if installed:
            install_hint = "Installed in models/moss_music: " + ", ".join(installed) + "."
        else:
            install_hint = (
                "No local MOSS-Music weights found yet in ComfyUI/models/moss_music/. "
                f"Download {moss_local.HF_URL_INSTRUCT} into "
                "ComfyUI/models/moss_music/MOSS-Music-8B-Instruct (or tick auto_download)."
            )
        return {
            "required": {
                "model_name": (
                    choices,
                    {
                        "default": "MOSS-Music-8B-Instruct",
                        "tooltip": install_hint,
                    },
                ),
                "backend": (
                    moss_local.BACKENDS,
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto: use local weights in ComfyUI/models/moss_music/ if installed, "
                            "else an SGLang server if running, else follow when_missing. "
                            "transformers: local HuggingFace weights only (no server needed). "
                            "sglang_server: HTTP server at server_url. "
                            "disabled: skip MOSS-Music and run DSP measurements only."
                        ),
                    },
                ),
            },
            "optional": {
                "device": (
                    moss_local.DEVICES,
                    {"default": "auto", "tooltip": "GPU/CPU placement for local transformers loading."},
                ),
                "dtype": (
                    moss_local.DTYPES,
                    {"default": "auto", "tooltip": "auto picks bfloat16 on modern GPUs (~17 GB VRAM)."},
                ),
                "keep_loaded": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Keep MOSS-Music in VRAM after captioning. Leave False when ACE-Step "
                            "runs in the same workflow so VRAM is freed before generation."
                        ),
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            f"Automatically download {moss_local.HF_REPO_INSTRUCT} (~17 GB) "
                            "into ComfyUI/models/moss_music/ when missing."
                        ),
                    },
                ),
                "when_missing": (
                    moss_local.MISSING_POLICIES,
                    {
                        "default": "skip (DSP only)",
                        "tooltip": (
                            "skip (DSP only): do not fail when MOSS weights are not downloaded yet — "
                            "let the workflow finish with DSP BPM/key/vocals and show the download link. "
                            "error: stop execution with download instructions."
                        ),
                    },
                ),
                "server_url": (
                    "STRING",
                    {
                        "default": moss.DEFAULT_SERVER,
                        "tooltip": "Optional SGLang endpoint (only used for sglang_server or auto fallback).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MOSS_MODEL", "STRING")
    RETURN_NAMES = ("moss_model", "status")
    FUNCTION = "load_model"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        f"Load MOSS-Music 8B ({moss_local.HF_URL_INSTRUCT}) from "
        "ComfyUI/models/moss_music/MOSS-Music-8B-Instruct for music captioning, "
        "vocal/instrument recognition, key/BPM/chords and lyrics ASR."
    )

    def load_model(
        self,
        model_name="MOSS-Music-8B-Instruct",
        backend="auto",
        device="auto",
        dtype="auto",
        keep_loaded=False,
        auto_download=False,
        when_missing="skip (DSP only)",
        server_url=moss.DEFAULT_SERVER,
    ):
        if backend == "disabled" or model_name == moss_local.DISABLED_CHOICE:
            status = "MOSS-Music disabled on loader (running DSP measurements only)."
            return (
                {
                    "backend": "disabled",
                    "model_name": model_name,
                    "ok": False,
                    "reason": status,
                },
                status,
            )

        if backend == "sglang_server":
            if moss.server_alive(server_url):
                status = f"MOSS-Music ready via SGLang server ({server_url})"
                return (
                    {
                        "backend": "sglang_server",
                        "server_url": server_url,
                        "model_name": model_name,
                        "ok": True,
                        "reason": status,
                    },
                    status,
                )
            msg = (
                f"No MOSS-Music SGLang server reachable at {server_url}.\n"
                "Either switch backend to `auto`/`transformers` and place weights in "
                "ComfyUI/models/moss_music/MOSS-Music-8B-Instruct, or start SGLang:\n"
                "  sglang serve --model-path MOSS-Music-8B-Instruct --trust-remote-code"
            )
            if when_missing == "error":
                raise RuntimeError(msg)
            print("[music-caption] " + msg.splitlines()[0])
            return ({"backend": "missing", "model_name": model_name, "ok": False, "reason": msg}, msg)

        # `auto` or `transformers`: check local ComfyUI/models/moss_music/<model_name> first
        info = moss_local.resolve_model_dir(model_name)
        if not info["ok"] and auto_download:
            info = moss_local.maybe_download_model(model_name)

        if info["ok"]:
            status = f"MOSS-Music local weights ready: {info['path']}"
            return (
                {
                    "backend": "transformers",
                    "model_dir": info["path"],
                    "model_name": os.path.basename(info["path"]),
                    "device": device,
                    "dtype": dtype,
                    "keep_loaded": bool(keep_loaded),
                    "ok": True,
                    "reason": status,
                },
                status,
            )

        # `auto` falls back to a live SGLang server if someone runs one
        if backend == "auto" and server_url and moss.server_alive(server_url):
            status = f"MOSS-Music ready via SGLang server ({server_url})"
            return (
                {
                    "backend": "sglang_server",
                    "server_url": server_url,
                    "model_name": model_name,
                    "ok": True,
                    "reason": status,
                },
                status,
            )

        if info["state"] == "incomplete":
            msg = info["reason"] + "\n" + moss_local.download_instructions(model_name)
        else:
            msg = "MOSS-Music model is not downloaded yet.\n" + moss_local.download_instructions(model_name)

        if when_missing == "error":
            raise RuntimeError(msg)
        print("[music-caption] " + msg.replace("\n", " | "))
        return (
            {
                "backend": "missing",
                "model_name": model_name,
                "ok": False,
                "reason": msg,
            },
            msg,
        )


# --------------------------------------------------------------------------------------
# MusicCaptionAnalyze - always available, no model needed
# --------------------------------------------------------------------------------------


class MusicCaptionAnalyze:
    """Measure BPM, key, vocal presence/type and duration from the audio itself."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "max_seconds": (
                    "FLOAT",
                    {
                        "default": 90.0,
                        "min": 5.0,
                        "max": 900.0,
                        "step": 5.0,
                        "tooltip": "Analyse only this much of the track (the tail), 0 = everything.",
                    },
                ),
            },
            "optional": {
                "sections": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Also return a coarse loudness/energy section map.",
                    },
                ),
                "extra_tags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Text appended to the measured tags (your own style words).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "FLOAT", "FLOAT", "STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ANALYZE_OUTPUTS
    FUNCTION = "analyze"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Measure BPM, key, vocal presence and vocal register from the audio (numpy DSP, no model). "
        "Returns a JSON analysis as well."
    )

    def analyze(self, audio, max_seconds=90.0, sections=True, extra_tags=""):
        waveform, sample_rate = _audio_payload(audio)
        analysis = dsp.analyse(waveform, sample_rate, max_seconds=max_seconds, sections=bool(sections))
        vocal = analysis.get("vocal") or {}
        measured = dsp.tags_from_analysis(analysis)
        tags = textutil.compose_tags(measured=measured, extra=extra_tags)
        analysis["tags"] = tags

        if analysis["duration_seconds"] and max_seconds and analysis["duration_seconds"] > max_seconds:
            analysis.setdefault("warnings", []).append(
                f"only the last {max_seconds:.0f}s were analysed (max_seconds)"
            )

        payload = json.dumps(analysis, ensure_ascii=False)
        result = (
            tags,
            int(round(analysis.get("bpm") or 0)),
            analysis.get("key") or "",
            float(analysis.get("key_confidence") or 0.0),
            float(analysis.get("bpm_confidence") or 0.0),
            vocal.get("presence", "unknown"),
            vocal.get("type", "unknown"),
            float(analysis.get("duration_seconds") or 0.0),
            payload,
        )
        return {"ui": {"text": [payload]}, "result": result}


# --------------------------------------------------------------------------------------
# MusicCaptionMOSS - the audio-language model (local Loader or SGLang server)
# --------------------------------------------------------------------------------------


class MusicCaptionMOSS:
    """Caption and analyse a track with MOSS-Music 8B (local Loader or SGLang server)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "server_url": (
                    "STRING",
                    {
                        "default": moss.DEFAULT_SERVER,
                        "tooltip": (
                            "Connect MOSS-Music Loader to `moss_model` for local HuggingFace "
                            "weights, or point server_url at an SGLang server (:30000)."
                        ),
                    },
                ),
                "mode": (
                    moss.MODES,
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto = attributes + lyrics when vocals are present. The others ask "
                            "one specific question."
                        ),
                    },
                ),
            },
            "optional": {
                "moss_model": (
                    "MOSS_MODEL",
                    {
                        "tooltip": (
                            "Optional link from MOSS-Music Loader (loads "
                            "ComfyUI/models/moss_music/MOSS-Music-8B-Instruct locally)."
                        ),
                    },
                ),
                "model": ("STRING", {"default": "default"}),
                "transport": (
                    ["path", "base64"],
                    {
                        "default": "path",
                        "tooltip": (
                            "For SGLang HTTP mode only — path: server reads the temp wav itself; "
                            "base64: inline the audio."
                        ),
                    },
                ),
                "segment": (
                    ["tail", "head", "full"],
                    {
                        "default": "tail",
                        "tooltip": "Which part of the track to send - the tail is what gets continued.",
                    },
                ),
                "max_seconds": ("FLOAT", {"default": 120.0, "min": 0.0, "max": 1800.0, "step": 10.0}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "timeout": ("FLOAT", {"default": 300.0, "min": 5.0, "max": 3600.0, "step": 5.0}),
                "extra_tags": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = (
        "caption",
        "lyrics",
        "vocal",
        "bpm",
        "keyscale",
        "instrumental",
        "structure",
        "moss_json",
    )
    FUNCTION = "describe"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        f"Describe the track with MOSS-Music 8B ({moss_local.HF_URL_INSTRUCT}): "
        "caption, vocal type, key, BPM, chords, structure and lyrics. Connect MOSS-Music Loader "
        "for local ComfyUI weights, or use an SGLang server; degrades cleanly when missing."
    )

    def describe(
        self,
        audio,
        server_url=moss.DEFAULT_SERVER,
        mode="auto",
        model="default",
        transport="path",
        segment="tail",
        max_seconds=120.0,
        max_tokens=1024,
        temperature=0.0,
        timeout=300.0,
        extra_tags="",
        moss_model=None,
    ):
        waveform, sample_rate = _audio_payload(audio)
        waveform = _segment(waveform, sample_rate, segment, max_seconds)
        payload = {"service": "moss-music", "server": server_url, "mode": mode, "ok": False}

        def failed(message):
            payload["error"] = message
            print("[music-caption] " + message.splitlines()[0])
            blob = json.dumps(payload, ensure_ascii=False)
            return {"ui": {"text": [blob]}, "result": ("", "", "", 0, "", False, "", blob)}

        result = None
        used_backend = "sglang_server"
        used_model_name = model

        if isinstance(moss_model, dict):
            backend = moss_model.get("backend")
            if not moss_model.get("ok") or backend in ("disabled", "missing"):
                payload["backend"] = backend or "missing"
                return failed(moss_model.get("reason") or "MOSS-Music is not loaded")
            if backend == "transformers":
                used_backend = "transformers"
                used_model_name = moss_model.get("model_name") or "MOSS-Music-8B-Instruct"
                try:
                    result = moss_local.analyse_waveform(
                        waveform,
                        sample_rate,
                        model_dir=moss_model["model_dir"],
                        device=moss_model.get("device", "auto"),
                        dtype=moss_model.get("dtype", "auto"),
                        keep_loaded=bool(moss_model.get("keep_loaded", False)),
                        mode=mode,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                        want_lyrics=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    return failed(f"MOSS-Music local inference failed: {exc}")
            elif backend == "sglang_server":
                server_url = moss_model.get("server_url") or server_url

        if result is None:
            # No loader wired (or loader selected sglang_server). Try SGLang first; when
            # server_url is left at default and local weights exist in models/moss_music/,
            # use them automatically so standalone runs work too.
            if moss.server_alive(server_url):
                path = _temp_audio(waveform, sample_rate, "moss")
                try:
                    result = moss.analyse_audio(
                        path,
                        server=server_url,
                        model=model,
                        transport=transport,
                        mode=mode,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                        timeout=float(timeout),
                        want_lyrics=True,
                    )
                except moss.MossError as exc:
                    return failed(f"MOSS-Music failed: {exc}")
                finally:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            elif server_url == moss.DEFAULT_SERVER:
                local_info = moss_local.resolve_model_dir("MOSS-Music-8B-Instruct")
                if local_info["ok"]:
                    used_backend = "transformers"
                    used_model_name = os.path.basename(local_info["path"])
                    try:
                        result = moss_local.analyse_waveform(
                            waveform,
                            sample_rate,
                            model_dir=local_info["path"],
                            mode=mode,
                            max_tokens=int(max_tokens),
                            temperature=float(temperature),
                            want_lyrics=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        return failed(f"MOSS-Music local inference failed: {exc}")

            if result is None:
                return failed(
                    f"no MOSS-Music server at {server_url} and no local weights in "
                    "ComfyUI/models/moss_music/MOSS-Music-8B-Instruct — download "
                    f"{moss_local.HF_URL_INSTRUCT} (connect MOSS-Music Loader) or use "
                    "MusicCaptionAnalyze for DSP BPM/key/vocals"
                )

        caption = textutil.clean(result.get("caption", ""))
        vocal_description = textutil.clean(
            result.get("vocal_description") or result.get("vocal") or ""
        )
        vocal_type = textutil.clean(result.get("vocal_type", ""))
        instrumental = result.get("instrumental")
        if instrumental is None:
            instrumental = not bool(result.get("vocal_present", True))
        vocal = textutil.compose_tags(vocal=vocal_type, extra=vocal_description, limit=300)
        bpm = textutil.normalize_bpm(result.get("bpm"))
        keyscale = textutil.normalize_key(result.get("key") or result.get("keyscale"))
        lyrics = (result.get("lyrics") or "").strip()
        if textutil.looks_instrumental(lyrics):
            lyrics = ""
            instrumental = True
        structure = textutil.clean(result.get("structure", ""))
        chords = textutil.clean(result.get("chords", ""))
        language = textutil.clean(result.get("language", ""))

        if extra_tags:
            caption = textutil.compose_tags(caption=caption, extra=extra_tags)

        payload.update(
            {
                "ok": True,
                "backend": used_backend,
                "model_name": used_model_name,
                "caption": caption,
                "lyrics": lyrics,
                "vocal": vocal,
                "vocal_type": vocal_type,
                "bpm": bpm,
                "keyscale": keyscale,
                "instrumental": bool(instrumental),
                "structure": structure,
                "chords": chords,
                "language": language,
                "camelot": "",
                "answers": result.get("raw", {}),
            }
        )
        blob = json.dumps(payload, ensure_ascii=False)
        return {
            "ui": {"text": [blob]},
            "result": (
                caption,
                lyrics,
                vocal,
                int(bpm or 0),
                keyscale,
                bool(instrumental),
                structure,
                blob,
            ),
        }


# --------------------------------------------------------------------------------------
# MusicCaptionMerge - one prompt string out of everything above
# --------------------------------------------------------------------------------------


class MusicCaptionMerge:
    """Combine measured (DSP) and model (MOSS) findings into one continuation prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefer_numeric": (
                    ["dsp", "model"],
                    {
                        "default": "dsp",
                        "tooltip": "Where BPM/key come from when both are available.",
                    },
                ),
            },
            "optional": {
                "analysis_json": ("STRING", {"default": "", "forceInput": True}),
                "moss_json": ("STRING", {"default": "", "forceInput": True}),
                "caption": ("STRING", {"default": "", "forceInput": True}),
                "vocal": ("STRING", {"default": "", "forceInput": True}),
                "lyrics": ("STRING", {"default": "", "forceInput": True}),
                "extra_tags": ("STRING", {"default": "", "multiline": True}),
                "keep_lyrics": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("tags", "lyrics", "bpm", "keyscale", "instrumental", "vocal_type", "summary_json")
    FUNCTION = "merge"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Merge DSP measurements and MOSS-Music findings into tags/lyrics/bpm/keyscale."

    def merge(
        self,
        prefer_numeric="dsp",
        analysis_json="",
        moss_json="",
        caption="",
        vocal="",
        lyrics="",
        extra_tags="",
        keep_lyrics=False,
    ):
        measured = {}
        if analysis_json:
            try:
                measured = json.loads(analysis_json)
            except ValueError:
                measured = {}
        model = {}
        if moss_json:
            try:
                model = json.loads(moss_json)
            except ValueError:
                model = {}

        model_caption = textutil.clean(model.get("caption") or caption)
        model_vocal = textutil.clean(model.get("vocal") or vocal)
        model_structure = textutil.clean(model.get("structure") or "")
        model_chords = textutil.clean(model.get("chords") or "")
        model_language = textutil.clean(model.get("language") or "")
        dsp_vocal = measured.get("vocal") or {}

        measured_tags = ""
        if measured:
            measured_tags = dsp.tags_from_analysis(measured)

        measured_bpm = textutil.normalize_bpm(measured.get("bpm"))
        model_bpm = textutil.normalize_bpm(model.get("bpm"))
        measured_key = textutil.normalize_key(measured.get("key"))
        model_key = textutil.normalize_key(model.get("keyscale") or model.get("key"))

        if prefer_numeric == "model":
            bpm = model_bpm or measured_bpm
            keyscale = model_key or measured_key
        else:
            bpm = measured_bpm or model_bpm
            keyscale = measured_key or model_key

        instrumental = model.get("instrumental")
        if instrumental is None:
            presence = dsp_vocal.get("presence")
            instrumental = presence == "instrumental"
        vocal_type = textutil.clean(
            model.get("vocal_type") or dsp_vocal.get("type") or ""
        )

        tags = textutil.compose_tags(
            caption=model_caption,
            vocal=model_vocal,
            structure=model_structure,
            measured=measured_tags,
            extra=extra_tags,
        )
        raw_lyrics = (lyrics or model.get("lyrics") or "").strip()
        if textutil.looks_instrumental(raw_lyrics):
            raw_lyrics = ""
        out_lyrics = raw_lyrics if keep_lyrics else ""

        if prefer_numeric == "model":
            key_source = "model" if model_bpm or model_key else "dsp"
        else:
            key_source = "dsp" if measured_bpm or measured_key else "model"
        summary = {
            "analyzer": "merge",
            "bpm": bpm,
            "keyscale": keyscale,
            "vocal_type": vocal_type,
            "instrumental": bool(instrumental),
            "key_source": key_source,
            "structure": model_structure,
            "chords": model_chords,
            "language": model_language,
            "tags": tags,
            "lyrics": out_lyrics,
            "detected_lyrics": raw_lyrics,
            "measured": measured,
            "model": {k: v for k, v in model.items() if k != "answers"},
        }
        blob = json.dumps(summary, ensure_ascii=False, default=str)
        return {
            "ui": {"text": [blob]},
            "result": (tags, out_lyrics, int(bpm or 0), keyscale, bool(instrumental), vocal_type, blob),
        }


# --------------------------------------------------------------------------------------
# MusicCaptionView - formatted inspector for ComfyUI workflows
# --------------------------------------------------------------------------------------


class MusicCaptionView:
    """Format DSP + MOSS-Music outputs into a clear report for viewing in ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "summary_json": ("STRING", {"default": "", "forceInput": True}),
                "analysis_json": ("STRING", {"default": "", "forceInput": True}),
                "moss_json": ("STRING", {"default": "", "forceInput": True}),
                "include_sections": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Include the 5s energy section map in the report."},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "report",
        "tags",
        "caption",
        "lyrics",
        "bpm",
        "keyscale",
        "vocal_type",
        "chords",
        "structure",
    )
    FUNCTION = "view"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Display all measured and MOSS-Music data (BPM, key, Camelot, vocal type, loudness, "
        "sections, chords, structure, tags, caption, lyrics) as a formatted text report."
    )

    def view(self, summary_json="", analysis_json="", moss_json="", include_sections=True):
        def _parse(raw):
            if not raw:
                return {}
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except ValueError:
                return {}

        summary = _parse(summary_json)
        measured = _parse(analysis_json) or (
            summary.get("measured") if isinstance(summary.get("measured"), dict) else {}
        )
        model = _parse(moss_json) or (
            summary.get("model") if isinstance(summary.get("model"), dict) else {}
        )

        report = format_caption_report(
            summary=summary,
            measured=measured,
            model=model,
            include_sections=bool(include_sections),
        )
        tags = textutil.clean(summary.get("tags") or measured.get("tags") or model.get("caption") or "")
        caption = textutil.clean(model.get("caption") or "")
        lyrics = (
            summary.get("detected_lyrics")
            or summary.get("lyrics")
            or model.get("lyrics")
            or ""
        ).strip()
        raw_bpm = (
            textutil.normalize_bpm(summary.get("bpm"))
            or textutil.normalize_bpm(measured.get("bpm"))
            or textutil.normalize_bpm(model.get("bpm"))
        )
        safe_bpm = int(min(300, max(10, raw_bpm))) if raw_bpm else 120
        keyscale = (
            textutil.normalize_key(summary.get("keyscale"))
            or textutil.normalize_key(measured.get("key"))
            or textutil.normalize_key(model.get("keyscale") or model.get("key"))
        )
        dsp_vocal = measured.get("vocal") if isinstance(measured.get("vocal"), dict) else {}
        vocal_type = textutil.clean(
            summary.get("vocal_type") or model.get("vocal_type") or dsp_vocal.get("type") or ""
        )
        chords = textutil.clean(summary.get("chords") or model.get("chords") or "")
        structure = textutil.clean(summary.get("structure") or model.get("structure") or "")

        return {
            "ui": {"text": [report]},
            "result": (report, tags, caption, lyrics, safe_bpm, keyscale, vocal_type, chords, structure),
        }


# --------------------------------------------------------------------------------------
# MusicCaptionTap - read strings out of captioners from other packs
# --------------------------------------------------------------------------------------


class MusicCaptionTap:
    """Collect STRING outputs of any node (SongScribe, ACE-Step 'Get Music Infos', ...).

    Its only job is to put the values into the `ui` block of the history entry,
    where audio_continue_tk.py can read them. Inputs are positional on purpose:
    the app links the captioner's STRING outputs in order, so `text_3` must stay
    at index 2 even when the captioner returned an empty string for it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"text_{index}": ("STRING", {"default": "", "forceInput": True}) for index in range(1, 13)}
        optional["label"] = ("STRING", {"default": ""})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "collect"
    CATEGORY = CATEGORY
    DESCRIPTION = "Collect STRING outputs of a captioner into a JSON array (readable by the app)."
    OUTPUT_NODE = True

    def collect(self, label="", **texts):
        # Positional on purpose: empty values are kept so that index i always maps
        # back to the captioner's i-th STRING output. Stop at the first input the
        # caller did not provide.
        values = []
        for index in range(1, 13):
            key = f"text_{index}"
            if key not in texts:
                break
            values.append("" if texts[key] is None else str(texts[key]))
        blob = json.dumps(values, ensure_ascii=False)
        return {"ui": {"text": [blob]}, "result": (blob,)}


NODE_CLASS_MAPPINGS = {
    "MusicCaptionMOSSLoader": MusicCaptionMOSSLoader,
    "MusicCaptionAnalyze": MusicCaptionAnalyze,
    "MusicCaptionMOSS": MusicCaptionMOSS,
    "MusicCaptionMerge": MusicCaptionMerge,
    "MusicCaptionView": MusicCaptionView,
    "MusicCaptionTap": MusicCaptionTap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MusicCaptionMOSSLoader": "MOSS-Music Loader",
    "MusicCaptionAnalyze": "Music Caption (measure: bpm / key / vocals)",
    "MusicCaptionMOSS": "Music Caption (MOSS-Music)",
    "MusicCaptionMerge": "Music Caption (build prompt)",
    "MusicCaptionView": "Music Caption Viewer",
    "MusicCaptionTap": "Music Caption (collect text)",
}
