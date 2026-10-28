"""
MOSS-Music bridge - the "real" captioner, optional.

MOSS-Music (https://github.com/OpenMOSS/MOSS-Music, Apache-2.0 weights) is an
8B music-understanding model built on MOSS-Audio + Qwen3-8B. It is the only
open model we found that answers *all* of the questions a continuation prompt
needs in one go:

    caption / genre / mood / instrumentation / production
    key + scale, tempo, beat grid, chord progression
    vocal detection incl. gender, register, solo vs chorus
    section map (intro / verse / chorus / bridge / outro)
    lyrics transcription with timestamps

It is served with SGLang behind an OpenAI-compatible endpoint
(`sglang serve --model-path MOSS-Music-8B-Instruct --trust-remote-code`,
default port 30000, see moss_music_usage_guide.md in their repo). This module
talks to that endpoint with the standard library only, so the ComfyUI node pack
keeps zero extra dependencies.

Two transports:
  * `path`   - send the absolute file path. Default, and the fast one. The file
               must be readable *by the MOSS server*, which is the normal case
               when both run on the same box.
  * `base64` - inline the audio as a data: URL. Works when the server is on a
               different machine, costs ~33% more bytes.

If no server is running, everything here degrades to `None` and the DSP
analysis in dsp.py carries the result. Nothing raises just because MOSS is
missing.
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request

DEFAULT_SERVER = "http://127.0.0.1:30000"

# --------------------------------------------------------------------------------------
# prompts - written against MOSS-Music's instruction style, and each one asks for
# something a machine can use, not a poem
# --------------------------------------------------------------------------------------

PROMPT_ATTRIBUTES = (
    "Analyse this song and answer with one JSON object and nothing else - no prose, "
    "no markdown fence. Use exactly these keys: "
    '{"caption": "one dense paragraph: genre, sub-genre, mood, tempo feel, instrumentation, '
    'production and mix", '
    '"bpm": <integer or null>, '
    '"key": "<tonic and mode, e.g. \\"A minor\\" or \\"F# major\\", or null if atonal>", '
    '"vocal_present": <true or false>, '
    '"vocal_type": "<solo male | solo female | duet | choir | mixed | instrumental | unknown>", '
    '"vocal_description": "timbre, delivery, register, effects, or empty string when instrumental", '
    '"instrumental": <true or false>, '
    '"language": "<ISO-639-1 code of the singing language, or \\"zxx\\" if instrumental>", '
    '"structure": "short section map, e.g. intro-verse-chorus-verse-chorus-bridge-outro", '
    '"chords": "the main chord progression if you can hear it, else empty string"}'
)

PROMPT_CAPTION = (
    "Describe this music in one dense paragraph a producer could use as a prompt: "
    "genre and sub-genre, mood and emotional arc, groove and rhythm, the instruments "
    "you actually hear (lead, bass, drums, keys, pads, fx), the vocal character if any, "
    "and the production/mix character. Be specific and factual, no marketing words."
)

PROMPT_VOCAL = (
    "Listen to this song and answer with one JSON object and nothing else: "
    '{"vocal_present": <true or false>, "vocal_type": "<solo male | solo female | duet | '
    'choir | mixed | unknown>", "register": "<low | mid | high>", '
    '"delivery": "short description of the singing style and effects", '
    '"language": "<ISO-639-1 code or zxx>"}'
)

PROMPT_LYRICS = (
    "Transcribe the sung lyrics of this song. Mark sections as [intro], [verse], "
    "[pre-chorus], [chorus], [bridge] and [outro]. Output only the lyrics with these "
    "section tags - no commentary. If the song has no singing, output exactly: [instrumental]"
)

PROMPT_KEY_BPM = (
    "What are the tempo and the key of this song? Answer with one JSON object and nothing "
    'else: {"bpm": <integer>, "key": "<tonic and mode>", "time_signature": "<e.g. 4/4>", '
    '"confidence": <0.0-1.0>}'
)

PROMPTS = {
    "attributes": PROMPT_ATTRIBUTES,
    "caption": PROMPT_CAPTION,
    "vocal": PROMPT_VOCAL,
    "lyrics": PROMPT_LYRICS,
    "key_bpm": PROMPT_KEY_BPM,
}

# "auto" runs `attributes` and adds `lyrics` only when the answer says vocals exist.
MODES = ["attributes", "caption", "vocal", "lyrics", "key_bpm", "auto"]


# --------------------------------------------------------------------------------------
# tiny HTTP helpers
# --------------------------------------------------------------------------------------


class MossError(RuntimeError):
    pass


def _post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        raise MossError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise MossError(f"cannot reach MOSS-Music at {url} ({exc.reason})") from exc
    except TimeoutError as exc:
        raise MossError(f"timeout after {timeout}s talking to {url}") from exc
    except ValueError as exc:
        raise MossError(f"MOSS-Music returned invalid JSON: {exc}") from exc


def server_alive(base_url=DEFAULT_SERVER, timeout=3.0):
    """True when something that looks like a MOSS/SGLang server answers."""
    base = (base_url or DEFAULT_SERVER).rstrip("/")
    for path in ("/v1/models", "/get_model_info", "/health"):
        try:
            with urllib.request.urlopen(base + path, timeout=timeout) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            continue
    return False


def _audio_reference(path, transport="path"):
    path = os.path.abspath(path)
    if transport != "base64":
        return path
    with open(path, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode("ascii")
    return f"data:audio/wav;base64,{payload}"


def _strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return text.replace("<think>", "").replace("</think>", "").strip()


# --------------------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------------------


def caption(
    audio_path,
    prompt=PROMPT_CAPTION,
    server=DEFAULT_SERVER,
    model="default",
    transport="path",
    max_tokens=1024,
    temperature=0.0,
    timeout=300.0,
):
    """Ask one question about `audio_path`; returns the model's text answer."""
    base = (server or DEFAULT_SERVER).rstrip("/")
    audio = _audio_reference(audio_path, transport)

    payload = {
        "model": model or "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    try:
        response = _post_json(base + "/v1/chat/completions", payload, timeout)
    except MossError as exc:
        if "404" not in str(exc):
            raise
        # older/leaner builds only expose the native /generate route
        response = _post_json(
            base + "/generate",
            {
                "text": prompt,
                "audio_data": audio,
                "sampling_params": {
                    "max_new_tokens": int(max_tokens),
                    "temperature": float(temperature),
                },
            },
            timeout,
        )

    if isinstance(response, dict) and response.get("choices"):
        message = (response["choices"][0] or {}).get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, list):  # some builds return content parts
            text = " ".join(
                part.get("text", "") for part in text if isinstance(part, dict)
            )
        return _strip_think(text)
    for key in ("text", "output_text", "generated_text", "output"):
        value = response.get(key) if isinstance(response, dict) else None
        if isinstance(value, str):
            return _strip_think(value)
        if isinstance(value, list) and value and isinstance(value[0], str):
            return _strip_think(value[0])
    raise MossError(f"unexpected answer from {base}: {str(response)[:600]}")


def extract_json(text):
    """Pull the first JSON object out of a chat answer, fences and chatter included."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(cleaned):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                chunk = cleaned[start:index + 1]
                try:
                    return json.loads(chunk)
                except ValueError:
                    start = -1
    return None


def analyse_audio(
    audio_path,
    server=DEFAULT_SERVER,
    model="default",
    transport="path",
    mode="auto",
    max_tokens=1024,
    temperature=0.0,
    timeout=300.0,
    want_lyrics=True,
    prompts_overrides=None,
):
    """Convenience wrapper: run the questions the GUI needs, merged into one dict.

    Returns {"caption", "bpm", "key", "vocal_present", "vocal_type",
             "vocal_description", "instrumental", "language", "structure",
             "lyrics", "raw": {...}}
    """
    overrides = prompts_overrides or {}
    answers = {}
    result = {}

    if mode in ("attributes", "auto", "caption"):
        ask = overrides.get("attributes") or (
            PROMPT_ATTRIBUTES if mode != "caption" else PROMPT_CAPTION
        )
        text = caption(audio_path, ask, server, model, transport, max_tokens, temperature, timeout)
        answers["attributes"] = text
        parsed = extract_json(text)
        if isinstance(parsed, dict):
            result.update({k: v for k, v in parsed.items() if v not in (None, "")})
        elif text:
            result["caption"] = text.strip()

        if mode == "auto" and want_lyrics:
            vocal_present = result.get("vocal_present")
            if vocal_present is None:
                vocal_present = not bool(result.get("instrumental", False))
            if vocal_present:
                lyrics = caption(
                    audio_path, overrides.get("lyrics") or PROMPT_LYRICS,
                    server, model, transport, max_tokens, temperature, timeout,
                )
                answers["lyrics"] = lyrics
                result["lyrics"] = lyrics

    elif mode == "vocal":
        text = caption(audio_path, overrides.get("vocal") or PROMPT_VOCAL,
                       server, model, transport, max_tokens, temperature, timeout)
        answers["vocal"] = text
        parsed = extract_json(text) or {}
        result.update(parsed)
        result.setdefault("caption", text.strip())

    elif mode == "lyrics":
        text = caption(audio_path, overrides.get("lyrics") or PROMPT_LYRICS,
                       server, model, transport, max_tokens, temperature, timeout)
        answers["lyrics"] = text
        result["lyrics"] = text
        result["instrumental"] = text.strip().lower().startswith("[instrumental]")

    elif mode == "key_bpm":
        text = caption(audio_path, overrides.get("key_bpm") or PROMPT_KEY_BPM,
                       server, model, transport, max_tokens, temperature, timeout)
        answers["key_bpm"] = text
        parsed = extract_json(text) or {}
        result.update(parsed)
        result.setdefault("caption", text.strip())

    result["raw"] = answers
    result["model"] = model
    result["server"] = server
    return result
