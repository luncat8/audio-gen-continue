"""A tiny fake ComfyUI server used by the tests (and handy for offline UI work)."""

import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

AUDIO_BYTES = b"fLaC\x00\x00\x00\x22fake-flac-payload"

XL_UNET = "acestep_v1.5_xl_sft_bf16.safetensors"
XL_CLIPS = ["qwen_0.6b_ace15.safetensors", "qwen_4b_ace15.safetensors"]
XL_VAE = "ace_1.5_vae.safetensors"

NODES = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["ace_step_1.5_turbo_aio.safetensors"]]}}},
    "UNETLoader": {
        "input": {
            "required": {
                "unet_name": [["acestep_v1.5_xl_sft_bf16.safetensors", "acestep_v1.5_xl_turbo_bf16.safetensors"]],
                "weight_dtype": [["default", "fp8_e4m3fn"]],
            }
        }
    },
    "DualCLIPLoader": {
        "input": {
            "required": {
                "clip_name1": [["qwen_0.6b_ace15.safetensors", "qwen_1.7b_ace15.safetensors"]],
                "clip_name2": [["qwen_4b_ace15.safetensors", "qwen_0.6b_ace15.safetensors"]],
                "type": [["ace", "sdxl"]],
            }
        }
    },
    "VAELoader": {"input": {"required": {"vae_name": [["ace_1.5_vae.safetensors"]]}}},
    "LoadAudio": {"input": {"required": {"audio": [["input.wav"]]}}},
    "SaveAudio": {
        "input": {"required": {"audio": ["AUDIO"], "filename_prefix": ["STRING", {}]}},
        "output_node": True,
    },
    "SaveAudioAdvanced": {
        "input": {"required": {"audio": ["AUDIO"], "filename_prefix": ["STRING", {}]}},
        "output_node": True,
    },
    "PreviewAny": {
        "input": {"required": {"source": ["*", {}]}},
        "output": [],
        "output_name": [],
        "display_name": "Preview as Text",
        "category": "utils",
        "output_node": True,
    },
    "KSampler": {"input": {"required": {}}},
    "VAEDecodeAudio": {"input": {"required": {}}},
    "ModelSamplingAuraFlow": {"input": {"required": {}}},
    "ConditioningZeroOut": {"input": {"required": {}}},
    "TextEncodeAceStepAudio1.5": {"input": {"required": {}}},
    "AudioContinueExtend": {"input": {"required": {}}},
    "AudioContinueStitch": {"input": {"required": {}}},
    # --- comfyui_music_caption (this repo) -------------------------------------------
    "MusicCaptionMOSSLoader": {
        "input": {
            "required": {
                "model_name": [
                    ["MOSS-Music-8B-Instruct", "MOSS-Music-8B-Thinking", "disabled (DSP only)"],
                    {
                        "default": "MOSS-Music-8B-Instruct",
                        "tooltip": (
                            "No local MOSS-Music weights found yet in ComfyUI/models/moss_music/. "
                            "Download https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct "
                            "into ComfyUI/models/moss_music/MOSS-Music-8B-Instruct (or tick auto_download)."
                        ),
                    },
                ],
                "backend": [["auto", "transformers", "sglang_server", "disabled"], {"default": "auto"}],
            },
            "optional": {
                "device": [["auto", "cuda", "cpu"], {"default": "auto"}],
                "dtype": [["auto", "bfloat16", "float16", "float32"], {"default": "auto"}],
                "keep_loaded": ["BOOLEAN", {"default": False}],
                "auto_download": ["BOOLEAN", {"default": False}],
                "when_missing": [["skip (DSP only)", "error"], {"default": "skip (DSP only)"}],
                "server_url": ["STRING", {"default": "http://127.0.0.1:30000"}],
            },
        },
        "output": ["MOSS_MODEL", "STRING"],
        "output_name": ["moss_model", "status"],
        "display_name": "MOSS-Music Loader",
        "description": "Load local HuggingFace MOSS-Music 8B weights from ComfyUI/models/moss_music/.",
        "category": "audio/caption",
    },
    "MusicCaptionAnalyze": {
        "input": {
            "required": {"audio": ["AUDIO"], "max_seconds": ["FLOAT", {"default": 90.0}]},
            "optional": {
                "sections": ["BOOLEAN", {"default": True}],
                "extra_tags": ["STRING", {"default": "", "multiline": True}],
            },
        },
        "output": ["STRING", "INT", "STRING", "FLOAT", "FLOAT", "STRING", "STRING", "FLOAT", "STRING"],
        "output_name": ["tags", "bpm", "keyscale", "key_confidence", "bpm_confidence",
                        "vocal_presence", "vocal_type", "duration_seconds", "analysis_json"],
        "display_name": "Music Caption (measure: bpm / key / vocals)",
        "description": "Measure bpm, key and vocals with numpy DSP.",
        "category": "audio/caption",
        "output_node": True,
    },
    "MusicCaptionMOSS": {
        "input": {
            "required": {
                "audio": ["AUDIO"],
                "server_url": ["STRING", {"default": "http://127.0.0.1:30000"}],
                "mode": [["attributes", "caption", "vocal", "lyrics", "key_bpm", "auto"], {}],
            },
            "optional": {
                "moss_model": ["MOSS_MODEL"],
                "model": ["STRING", {"default": "default"}],
                "transport": [["path", "base64"], {"default": "path"}],
                "segment": [["tail", "head", "full"], {"default": "tail"}],
                "max_seconds": ["FLOAT", {"default": 120.0}],
            },
        },
        "output": ["STRING", "STRING", "STRING", "INT", "STRING", "BOOLEAN", "STRING", "STRING"],
        "output_name": ["caption", "lyrics", "vocal", "bpm", "keyscale", "instrumental",
                        "structure", "moss_json"],
        "display_name": "Music Caption (MOSS-Music)",
        "description": "MOSS-Music 8B captioning via MOSS-Music Loader or SGLang server.",
        "category": "audio/caption",
        "output_node": True,
    },
    "MusicCaptionMerge": {
        "input": {
            "required": {"prefer_numeric": [["dsp", "model"], {"default": "dsp"}]},
            "optional": {
                "analysis_json": ["STRING", {"default": "", "forceInput": True}],
                "moss_json": ["STRING", {"default": "", "forceInput": True}],
                "caption": ["STRING", {"default": "", "forceInput": True}],
                "vocal": ["STRING", {"default": "", "forceInput": True}],
                "lyrics": ["STRING", {"default": "", "forceInput": True}],
                "extra_tags": ["STRING", {"default": "", "multiline": True}],
                "keep_lyrics": ["BOOLEAN", {"default": False}],
            },
        },
        "output": ["STRING", "STRING", "INT", "STRING", "BOOLEAN", "STRING", "STRING"],
        "output_name": ["tags", "lyrics", "bpm", "keyscale", "instrumental", "vocal_type",
                        "summary_json"],
        "display_name": "Music Caption (build prompt)",
        "description": "Merge DSP and MOSS findings into one prompt.",
        "category": "audio/caption",
        "output_node": True,
    },
    "MusicCaptionView": {
        "input": {
            "required": {},
            "optional": {
                "summary_json": ["STRING", {"default": "", "forceInput": True}],
                "analysis_json": ["STRING", {"default": "", "forceInput": True}],
                "moss_json": ["STRING", {"default": "", "forceInput": True}],
                "include_sections": ["BOOLEAN", {"default": True}],
            },
        },
        "output": ["STRING", "STRING", "STRING", "STRING", "INT", "STRING", "STRING", "STRING", "STRING"],
        "output_name": ["report", "tags", "caption", "lyrics", "bpm", "keyscale", "vocal_type", "chords", "structure"],
        "display_name": "Music Caption Viewer",
        "description": "Display all measured and MOSS-Music caption fields inside ComfyUI.",
        "category": "audio/caption",
        "output_node": True,
    },
    "MusicCaptionTap": {
        "input": {
            "required": {},
            "optional": dict(
                {"label": ["STRING", {"default": ""}]},
                **{f"text_{i}": ["STRING", {"default": "", "forceInput": True}] for i in range(1, 13)}
            ),
        },
        "output": ["STRING"],
        "output_name": ["text"],
        "display_name": "Music Caption (collect text)",
        "description": "Collect STRING outputs of a captioner.",
        "category": "audio/caption",
        "output_node": True,
    },
    # --- a third-party captioner, to exercise the generic tap path --------------------
    "SongScribeAnalyzer": {
        "input": {
            "required": {
                "audio": ["AUDIO"],
                "describe": [["off", "clap"], {"default": "off"}],
                "style": [["verbatim", "balanced", "loose"], {"default": "balanced"}],
                "transcribe_lyrics": ["BOOLEAN", {"default": False}],
                "use_cache": ["BOOLEAN", {"default": True}],
            },
            "optional": {"audio_file": ["STRING", {"default": ""}]},
        },
        "output": ["STRING", "STRING", "FLOAT", "INT", "STRING", "SONGSCRIBE_ANALYSIS"],
        "output_name": ["caption", "lyrics", "duration", "duration_int", "duration_str", "analysis"],
        "display_name": "Song Analyzer (SongScribe)",
        "description": "Extract a structured music caption, lyrics and duration from audio.",
        "category": "audio/analysis",
    },
}

DEFAULT_CAPTION_TEXT = json.dumps(
    {
        "analyzer": "merge",
        "bpm": 96,
        "keyscale": "A minor",
        "vocal_type": "female",
        "instrumental": False,
        "tags": "dark synthwave, 96 bpm, A minor, female lead vocals",
        "lyrics": "",
        "measured": {"bpm": 96, "key": "A minor",
                     "vocal": {"presence": "vocals", "type": "female"}},
        "model": {"caption": "dark synthwave with analog bass and gated drums"},
    }
)


class FakeState:
    def __init__(self, missing_nodes=(), fail_prompt=False, prompt_delay=1, caption_text=None):
        self.uploads = {}
        self.prompts = {}
        self.missing_nodes = set(missing_nodes)
        self.fail_prompt = fail_prompt
        self.prompt_delay = prompt_delay
        self.caption_text = caption_text if caption_text is not None else DEFAULT_CAPTION_TEXT
        self.history_polls = {}


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the test output clean
            pass

        # -- helpers ----------------------------------------------------------
        def _send(self, payload, status=200, ctype="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        # -- routes -----------------------------------------------------------
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/system_stats":
                self._send(
                    {
                        "system": {"comfyui_version": "0.4.0-fake", "python_version": "3.12"},
                        "devices": [{"name": "cuda:0 Fake GPU", "vram_total": 24 * 1024**3}],
                    }
                )
            elif path == "/object_info":
                self._send({k: v for k, v in NODES.items() if k not in state.missing_nodes})
            elif path.startswith("/object_info/"):
                name = path.split("/", 2)[2]
                if name in NODES and name not in state.missing_nodes:
                    self._send({name: NODES[name]})
                else:
                    self._send({"error": "not found"}, status=404)
            elif path == "/queue":
                self._send({"queue_running": [], "queue_pending": []})
            elif path.startswith("/history/"):
                prompt_id = path.rsplit("/", 1)[-1]
                polls = state.history_polls.get(prompt_id, 0) + 1
                state.history_polls[prompt_id] = polls
                if prompt_id not in state.prompts:
                    self._send({})
                elif polls * state.prompt_delay < 1:
                    self._send({})
                elif state.fail_prompt:
                    self._send(
                        {
                            prompt_id: {
                                "outputs": {},
                                "status": {
                                    "status_str": "error",
                                    "completed": False,
                                    "messages": [["execution_error", {"exception_message": "boom"}]],
                                },
                            }
                        }
                    )
                else:
                    prompt = state.prompts[prompt_id]
                    outputs = {}
                    for node_id, node in prompt.items():
                        class_type = node.get("class_type", "")
                        if class_type in (
                            "MusicCaptionAnalyze", "MusicCaptionMOSS",
                            "MusicCaptionMerge", "MusicCaptionView",
                            "MusicCaptionTap", "PreviewAny",
                        ):
                            outputs[node_id] = {"text": [state.caption_text]}
                        if class_type.startswith("SaveAudio"):
                            prefix = node.get("inputs", {}).get("filename_prefix", "audio")
                            outputs[node_id] = {
                                "audio": [
                                    {
                                        "filename": "audio_00001_.flac",
                                        "subfolder": prefix.rsplit("/", 1)[0] if "/" in prefix else "",
                                        "type": "output",
                                    }
                                ]
                            }
                    self._send(
                        {
                            prompt_id: {
                                "outputs": outputs,
                                "status": {"status_str": "success", "completed": True, "messages": []},
                            }
                        }
                    )
            elif path == "/view":
                query = parse_qs(parsed.query)
                self.last_view = query
                self._send(AUDIO_BYTES, ctype="audio/flac")
            else:
                self._send({"error": "unknown path " + path}, status=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            body = self._read_body()
            if parsed.path == "/upload/image":
                head = body[:8192].decode("utf-8", "replace")
                name = "uploaded.wav"
                if 'filename="' in head:
                    name = head.split('filename="', 1)[1].split('"', 1)[0]
                state.uploads[name] = body
                self._send({"name": name, "subfolder": "", "type": "input"})
            elif parsed.path == "/prompt":
                payload = json.loads(body.decode())
                prompt = payload.get("prompt") or {}
                if state.fail_prompt == "node_errors":
                    self._send(
                        {
                            "error": {"type": "prompt_outputs_failed_validation"},
                            "node_errors": {"4": {"errors": ["bad add_seconds"]}},
                        },
                        status=400,
                    )
                    return
                has_output = any(
                    NODES.get((node or {}).get("class_type", ""), {}).get("output_node") is True
                    and (node or {}).get("class_type", "") not in state.missing_nodes
                    for node in prompt.values()
                    if isinstance(node, dict)
                )
                if not has_output:
                    self._send(
                        {
                            "error": {
                                "type": "prompt_no_outputs",
                                "message": "Prompt has no outputs",
                                "details": "",
                                "extra_info": {},
                            },
                            "node_errors": {},
                        },
                        status=400,
                    )
                    return
                prompt_id = uuid.uuid4().hex
                state.prompts[prompt_id] = prompt
                state.history_polls[prompt_id] = 0
                self._send({"prompt_id": prompt_id, "number": 1, "node_errors": {}})
            elif parsed.path == "/interrupt":
                self._send({})
            else:
                self._send({"error": "unknown path " + parsed.path}, status=404)

    return Handler


def start_server(state):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


if __name__ == "__main__":
    state = FakeState()
    server, url = start_server(state)
    print("fake ComfyUI on", url)
    print("press ctrl-c to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
