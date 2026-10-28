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
    "SaveAudio": {"input": {"required": {"audio": ["AUDIO"], "filename_prefix": ["STRING", {}]}}},
    "SaveAudioAdvanced": {"input": {"required": {"audio": ["AUDIO"], "filename_prefix": ["STRING", {}]}}},
    "KSampler": {"input": {"required": {}}},
    "VAEDecodeAudio": {"input": {"required": {}}},
    "ModelSamplingAuraFlow": {"input": {"required": {}}},
    "ConditioningZeroOut": {"input": {"required": {}}},
    "TextEncodeAceStepAudio1.5": {"input": {"required": {}}},
    "AudioContinueExtend": {"input": {"required": {}}},
    "AudioContinueStitch": {"input": {"required": {}}},
}


class FakeState:
    def __init__(self, missing_nodes=(), fail_prompt=False, prompt_delay=1):
        self.uploads = {}
        self.prompts = {}
        self.missing_nodes = set(missing_nodes)
        self.fail_prompt = fail_prompt
        self.prompt_delay = prompt_delay
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
                    prefix = prompt["10"]["inputs"]["filename_prefix"]
                    self._send(
                        {
                            prompt_id: {
                                "outputs": {
                                    "10": {
                                        "audio": [
                                            {
                                                "filename": "audio_00001_.flac",
                                                "subfolder": prefix.rsplit("/", 1)[0] if "/" in prefix else "",
                                                "type": "output",
                                            }
                                        ]
                                    }
                                },
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
