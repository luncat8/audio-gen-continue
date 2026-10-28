"""
Music captioner nodes for ComfyUI - measure the source track and turn it into a
prompt a music model can actually use.

    MusicCaptionAnalyze   AUDIO -> bpm / key / vocal presence+type / duration (numpy DSP)
    MusicCaptionMOSS      AUDIO -> caption / lyrics / vocal details / key / bpm (MOSS-Music API)
    MusicCaptionMerge     everything -> tags / lyrics / bpm / keyscale for TextEncode*
    MusicCaptionTap       *       -> text, for captioners from other packs

Why a separate pack: none of the existing ComfyUI captioners report the four
things a continuation prompt needs (key, BPM, vocal presence, vocal type) in a
form you can wire into an ACE-Step / MiniMax text encoder. See CAPTIONER.md in
the repository for the comparison and the reasoning.

All nodes also return their result as a JSON `ui` block, which is how
audio_continue_tk.py reads them back without needing a preview node.
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

from . import dsp, moss, textutil

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
# MusicCaptionMOSS - the audio-language model, optional
# --------------------------------------------------------------------------------------


class MusicCaptionMOSS:
    """Ask a MOSS-Music server (SGLang, OpenAI-compatible) about the track."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "server_url": (
                    "STRING",
                    {
                        "default": moss.DEFAULT_SERVER,
                        "tooltip": "sglang serve --model-path MOSS-Music-8B-Instruct (port 30000).",
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
                "model": ("STRING", {"default": "default"}),
                "transport": (
                    ["path", "base64"],
                    {
                        "default": "path",
                        "tooltip": (
                            "path: the server reads the temp wav itself (same machine). "
                            "base64: inline the audio (server on another machine)."
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
    DESCRIPTION = (
        "Describe the track with MOSS-Music (caption, vocal type, key, BPM, structure, lyrics). "
        "Needs a running SGLang server; without one it returns empty strings instead of failing."
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
    ):
        waveform, sample_rate = _audio_payload(audio)
        waveform = _segment(waveform, sample_rate, segment, max_seconds)
        payload = {"service": "moss-music", "server": server_url, "mode": mode, "ok": False}

        def failed(message):
            payload["error"] = message
            print("[music-caption] " + message)
            blob = json.dumps(payload, ensure_ascii=False)
            return {"ui": {"text": [blob]}, "result": ("", "", "", 0, "", False, "", blob)}

        if not moss.server_alive(server_url):
            return failed(
                f"no MOSS-Music server at {server_url} - start `sglang serve --model-path "
                "MOSS-Music-8B-Instruct --trust-remote-code` or use MusicCaptionAnalyze instead"
            )

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

        caption = textutil.clean(result.get("caption", ""))
        vocal_description = textutil.clean(result.get("vocal_description", ""))
        vocal_type = textutil.clean(result.get("vocal_type", ""))
        instrumental = result.get("instrumental")
        if instrumental is None:
            instrumental = not bool(result.get("vocal_present", True))
        vocal = textutil.compose_tags(vocal=vocal_type, extra=vocal_description, limit=300)
        bpm = textutil.normalize_bpm(result.get("bpm"))
        keyscale = textutil.normalize_key(result.get("key"))
        lyrics = (result.get("lyrics") or "").strip()
        if textutil.looks_instrumental(lyrics):
            lyrics = ""
            instrumental = True
        structure = textutil.clean(result.get("structure", ""))

        if extra_tags:
            caption = textutil.compose_tags(caption=caption, extra=extra_tags)

        payload.update(
            {
                "ok": True,
                "caption": caption,
                "vocal": vocal,
                "vocal_type": vocal_type,
                "bpm": bpm,
                "keyscale": keyscale,
                "instrumental": bool(instrumental),
                "structure": structure,
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
        out_lyrics = (lyrics or "").strip()
        if not keep_lyrics or textutil.looks_instrumental(out_lyrics):
            out_lyrics = "" if not keep_lyrics else out_lyrics

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
            "tags": tags,
            "lyrics": out_lyrics,
            "measured": measured,
            "model": {k: v for k, v in model.items() if k != "answers"},
        }
        blob = json.dumps(summary, ensure_ascii=False, default=str)
        return {
            "ui": {"text": [blob]},
            "result": (tags, out_lyrics, int(bpm or 0), keyscale, bool(instrumental), vocal_type, blob),
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
    "MusicCaptionAnalyze": MusicCaptionAnalyze,
    "MusicCaptionMOSS": MusicCaptionMOSS,
    "MusicCaptionMerge": MusicCaptionMerge,
    "MusicCaptionTap": MusicCaptionTap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MusicCaptionAnalyze": "Music Caption (measure: bpm / key / vocals)",
    "MusicCaptionMOSS": "Music Caption (MOSS-Music)",
    "MusicCaptionMerge": "Music Caption (build prompt)",
    "MusicCaptionTap": "Music Caption (collect text)",
}
