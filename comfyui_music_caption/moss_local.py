"""
Local (in-process) MOSS-Music loader and inference helper for ComfyUI.

Model weights on HuggingFace (Apache-2.0, ~17 GB in 4 safetensors shards):
  * https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct   (recommended)
  * https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking   (chain-of-thought variant)

Expected ComfyUI folder layout:
    ComfyUI/models/moss_music/MOSS-Music-8B-Instruct/
        config.json
        configuration_moss_music.py
        modeling_moss_music.py
        processing_moss_music.py
        preprocessor_config.json
        generation_config.json
        tokenizer_config.json
        vocab.json
        merges.txt
        model.safetensors.index.json
        model-00001-of-00004.safetensors ... model-00004-of-00004.safetensors

Download command:
    hf download OpenMOSS-Team/MOSS-Music-8B-Instruct \\
      --local-dir ComfyUI/models/moss_music/MOSS-Music-8B-Instruct

Why this module exists:
  1. Users expect a standard ComfyUI Loader node (`MOSS-Music Loader`) instead of
     having to install and run an external SGLang server on port 30000.
  2. The HuggingFace repo ships its own `configuration_moss_music.py`,
     `modeling_moss_music.py` and `processing_moss_music.py` built on `Qwen3Model`
     + `WhisperEncoderLayer` (both already in ComfyUI's `transformers`).
  3. Small runtime shims here keep those three files working across:
     - hyphenated folder names (`MOSS-Music-8B-Instruct`)
     - ComfyUI installs without `accelerate` or `torchaudio`
     - both `transformers` 4.x (tuple-returning `WhisperEncoderLayer`) and
       `transformers` 5.x (tensor-returning `WhisperEncoderLayer`, dict `_tied_weights_keys`).
"""

import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import types

import numpy as np

from . import dsp, moss

try:  # ComfyUI provides this at runtime; unit tests do not
    import folder_paths
except ImportError:  # pragma: no cover
    folder_paths = None

HF_REPO_INSTRUCT = "OpenMOSS-Team/MOSS-Music-8B-Instruct"
HF_REPO_THINKING = "OpenMOSS-Team/MOSS-Music-8B-Thinking"
HF_URL_INSTRUCT = f"https://huggingface.co/{HF_REPO_INSTRUCT}"
HF_URL_THINKING = f"https://huggingface.co/{HF_REPO_THINKING}"

KNOWN_MODELS = {
    "MOSS-Music-8B-Instruct": HF_REPO_INSTRUCT,
    "MOSS-Music-8B-Thinking": HF_REPO_THINKING,
}
DISABLED_CHOICE = "disabled (DSP only)"

BACKENDS = ["auto", "transformers", "sglang_server", "disabled"]
DEVICES = ["auto", "cuda", "cpu"]
DTYPES = ["auto", "bfloat16", "float16", "float32"]
MISSING_POLICIES = ["skip (DSP only)", "error"]

REQUIRED_CODE_FILES = (
    "config.json",
    "configuration_moss_music.py",
    "modeling_moss_music.py",
    "processing_moss_music.py",
)

# One cached runner at a time so `keep_loaded=True` reuses the 8B model across runs
# without piling up two 17 GB copies in VRAM.
_CACHE = {"key": None, "runner": None}


class MossLocalError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# model folders & discovery
# --------------------------------------------------------------------------------------


def models_root(comfy_models_dir=None, create=False):
    """Primary ComfyUI folder: `<ComfyUI>/models/moss_music`."""
    base = comfy_models_dir
    if not base and folder_paths is not None:
        base = getattr(folder_paths, "models_dir", None)
    if not base:
        env = os.environ.get("MOSS_MUSIC_DIR")
        if env:
            return os.path.abspath(env)
        return ""
    target = os.path.join(os.path.abspath(base), "moss_music")
    if create:
        try:
            os.makedirs(target, exist_ok=True)
        except OSError:
            pass
    return target


def register_folder_path():
    """Register `models/moss_music` with ComfyUI and ensure the empty folder exists."""
    if folder_paths is None:
        return
    root = models_root(create=True)
    if root and hasattr(folder_paths, "add_model_folder_path"):
        try:
            folder_paths.add_model_folder_path("moss_music", root)
        except Exception:  # noqa: BLE001
            pass


def search_roots(comfy_models_dir=None):
    """All parent folders scanned for MOSS-Music model directories."""
    roots = []
    env_dir = os.environ.get("MOSS_MUSIC_DIR")
    if env_dir:
        roots.append(os.path.abspath(env_dir))

    base = comfy_models_dir
    if not base and folder_paths is not None:
        base = getattr(folder_paths, "models_dir", None)
        if hasattr(folder_paths, "get_folder_paths"):
            try:
                for extra in folder_paths.get_folder_paths("moss_music"):
                    if extra and extra not in roots:
                        roots.append(os.path.abspath(extra))
            except Exception:  # noqa: BLE001
                pass

    if base:
        base = os.path.abspath(base)
        for sub in ("moss_music", "moss-music", "LLM"):
            candidate = os.path.join(base, sub)
            if candidate not in roots:
                roots.append(candidate)
    return roots


def inspect_model_dir(path):
    """Inspect a candidate model folder; report missing/incomplete shards clearly."""
    if not path or not os.path.isdir(path):
        return {
            "ok": False,
            "state": "missing",
            "path": path or "",
            "missing": list(REQUIRED_CODE_FILES),
            "reason": f"directory not found: {path or '(empty)'}",
        }

    missing = [name for name in REQUIRED_CODE_FILES if not os.path.isfile(os.path.join(path, name))]
    for tok_file in ("tokenizer_config.json", "vocab.json", "merges.txt"):
        if not os.path.isfile(os.path.join(path, tok_file)):
            missing.append(tok_file)

    partials = [
        name for name in os.listdir(path)
        if name.endswith((".incomplete", ".part", ".tmp"))
    ]

    index_path = os.path.join(path, "model.safetensors.index.json")
    single_weights = os.path.join(path, "model.safetensors")
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                index_data = json.load(handle)
            shards = sorted(set((index_data.get("weight_map") or {}).values()))
        except (OSError, ValueError):
            shards = []
            missing.append("model.safetensors.index.json (corrupt)")
        for shard in shards:
            shard_path = os.path.join(path, shard)
            if not os.path.isfile(shard_path) or os.path.getsize(shard_path) == 0:
                missing.append(shard)
    elif os.path.isfile(single_weights) and os.path.getsize(single_weights) > 0:
        pass
    else:
        missing.append("model.safetensors.index.json (and safetensors shards)")

    if missing or partials:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing[:6]) + ("..." if len(missing) > 6 else ""))
        if partials:
            details.append("download still in progress (" + ", ".join(partials[:3]) + ")")
        return {
            "ok": False,
            "state": "incomplete",
            "path": os.path.abspath(path),
            "missing": missing,
            "reason": f"incomplete model folder at {path}: {'; '.join(details)}",
        }

    return {
        "ok": True,
        "state": "ready",
        "path": os.path.abspath(path),
        "missing": [],
        "reason": "ready",
    }


def scan_models(comfy_models_dir=None):
    """Return `{name: inspection_dict}` for candidate MOSS-Music folders on disk."""
    found = {}
    direct_env = os.environ.get("MOSS_MUSIC_MODEL_PATH")
    if direct_env and os.path.isdir(direct_env):
        info = inspect_model_dir(direct_env)
        found[os.path.basename(os.path.abspath(direct_env))] = info

    for root in search_roots(comfy_models_dir):
        if not os.path.isdir(root):
            continue
        is_generic_llm_dir = os.path.basename(root).lower() == "llm"
        for entry in sorted(os.listdir(root)):
            if entry.startswith("."):
                continue
            full = os.path.join(root, entry)
            if not os.path.isdir(full):
                continue
            if is_generic_llm_dir and "moss" not in entry.lower() and not os.path.isfile(
                os.path.join(full, "configuration_moss_music.py")
            ):
                continue
            if entry not in found or (not found[entry]["ok"] and inspect_model_dir(full)["ok"]):
                found[entry] = inspect_model_dir(full)
    return found


def installed_models(comfy_models_dir=None):
    """Names of complete, ready-to-load MOSS-Music folders."""
    return [name for name, info in scan_models(comfy_models_dir).items() if info["ok"]]


def loader_choices(comfy_models_dir=None):
    """COMBO choices for `MusicCaptionMOSSLoader`.

    Always keeps the two official HuggingFace names in the list even before the
    weights are downloaded, so workflows open cleanly without ComfyUI rejecting
    `model_name` as "value not in list".
    """
    choices = list(KNOWN_MODELS.keys())
    for name in scan_models(comfy_models_dir):
        if name not in choices and name != DISABLED_CHOICE:
            choices.append(name)
    choices.append(DISABLED_CHOICE)
    return choices


def resolve_model_dir(model_name, comfy_models_dir=None):
    """Find `model_name` on disk (or return where it *should* be placed)."""
    name = (model_name or "MOSS-Music-8B-Instruct").strip()
    if "/" in name and not os.path.exists(name):
        # Accept "OpenMOSS-Team/MOSS-Music-8B-Instruct" too
        tail = name.rsplit("/", 1)[-1]
        if tail in KNOWN_MODELS:
            name = tail

    if os.path.isdir(name):
        return inspect_model_dir(name)

    scanned = scan_models(comfy_models_dir)
    if name in scanned:
        return scanned[name]

    root = models_root(comfy_models_dir)
    expected = os.path.join(root, name) if root else os.path.join("ComfyUI", "models", "moss_music", name)
    return inspect_model_dir(expected)


def hf_repo_for(model_name):
    name = (model_name or "").strip()
    if name in KNOWN_MODELS:
        return KNOWN_MODELS[name]
    if "/" in name and not os.path.sep in name:
        return name
    return HF_REPO_INSTRUCT


def download_instructions(model_name="MOSS-Music-8B-Instruct", comfy_models_dir=None):
    repo = hf_repo_for(model_name)
    short = repo.rsplit("/", 1)[-1]
    root = models_root(comfy_models_dir)
    folder = os.path.join(root, short) if root else f"ComfyUI/models/moss_music/{short}"
    return (
        f"Download https://huggingface.co/{repo} (~17 GB, Apache-2.0) into:\n"
        f"  {folder}\n"
        f"Command:\n"
        f"  hf download {repo} --local-dir \"{folder}\"\n"
        "(or tick `auto_download` on the MOSS-Music Loader node)."
    )


def maybe_download_model(model_name, comfy_models_dir=None):
    """Download `OpenMOSS-Team/<model_name>` into `models/moss_music/<model_name>`."""
    repo = hf_repo_for(model_name)
    short = repo.rsplit("/", 1)[-1]
    root = models_root(comfy_models_dir, create=True)
    if not root:
        raise MossLocalError(
            "Cannot determine ComfyUI/models directory for auto_download. "
            + download_instructions(model_name, comfy_models_dir)
        )
    target = os.path.join(root, short)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise MossLocalError(
            "huggingface_hub is not installed in this Python environment, so auto_download "
            "cannot run.\n" + download_instructions(model_name, comfy_models_dir)
        ) from exc

    os.makedirs(target, exist_ok=True)
    print(f"[music-caption] downloading {repo} -> {target} (~17 GB)...")
    try:
        snapshot_download(repo_id=repo, local_dir=target)
    except Exception as exc:
        raise MossLocalError(f"HuggingFace download failed for {repo}: {exc}") from exc

    info = inspect_model_dir(target)
    if not info["ok"]:
        raise MossLocalError(info["reason"])
    return info


# --------------------------------------------------------------------------------------
# dynamic module loading & transformers 4.x / 5.x compatibility
# --------------------------------------------------------------------------------------


def _ensure_compat_shims():
    """Provide harmless stubs for optional imports inside MOSS's remote-code files."""
    # `processing_moss_music.py` has an unused top-level `import torchaudio  # noqa: F401`.
    if "torchaudio" not in sys.modules and importlib.util.find_spec("torchaudio") is None:
        sys.modules["torchaudio"] = types.ModuleType("torchaudio")

    # Older transformers (<4.52) did not expose `transformers.utils.auto_docstring`.
    if importlib.util.find_spec("transformers") is not None:
        try:
            import transformers.utils.auto_docstring  # noqa: F401
        except ImportError:
            stub = types.ModuleType("transformers.utils.auto_docstring")

            def auto_docstring(*args, **kwargs):
                if args and callable(args[0]) and len(args) == 1 and not kwargs:
                    return args[0]
                return lambda obj: obj

            stub.auto_docstring = auto_docstring
            sys.modules["transformers.utils.auto_docstring"] = stub


def _load_moss_modules(model_dir):
    """Import configuration/modeling/processing from `model_dir` regardless of folder name."""
    _ensure_compat_shims()
    model_dir = os.path.abspath(model_dir)
    digest = hashlib.sha1(model_dir.encode("utf-8")).hexdigest()[:10]
    pkg_name = f"_comfy_moss_music_{digest}"

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [model_dir]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

    # Also satisfy `from src.configuration_moss_music import ...` if someone copied
    # the files out of the GitHub repo rather than the HuggingFace repo.
    created_src = False
    if "src" not in sys.modules:
        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = [model_dir]
        sys.modules["src"] = src_pkg
        created_src = True

    loaded = {}
    try:
        for mod_name in ("configuration_moss_music", "modeling_moss_music", "processing_moss_music"):
            file_path = os.path.join(model_dir, f"{mod_name}.py")
            full_name = f"{pkg_name}.{mod_name}"
            spec = importlib.util.spec_from_file_location(full_name, file_path)
            if spec is None or spec.loader is None:
                raise MossLocalError(f"cannot load {file_path}")
            module = importlib.util.module_from_spec(spec)
            module.__package__ = pkg_name
            sys.modules[full_name] = module
            if created_src or f"src.{mod_name}" not in sys.modules:
                sys.modules[f"src.{mod_name}"] = module
            spec.loader.exec_module(module)
            loaded[mod_name] = module
    finally:
        if created_src:
            sys.modules.pop("src", None)

    modeling = loaded["modeling_moss_music"]
    # transformers 5.x expects `_tied_weights_keys` to be dict | None (4.x allowed []).
    # MOSS-Music sets `tie_word_embeddings = False`, so `None` works in both 4.x and 5.x.
    if hasattr(modeling, "MossMusicModel"):
        modeling.MossMusicModel._tied_weights_keys = None
    return loaded["configuration_moss_music"], modeling, loaded["processing_moss_music"]


def _patch_encoder_layers_for_transformers5(model):
    """In transformers 4.x `WhisperEncoderLayer` returns `(hidden_states,)`; in 5.x it
    returns `hidden_states` directly, so `layer(...)[0]` in `MossMusicEncoder` would
    slice the batch axis. Wrap only this model instance's layers to return a 1-tuple
    whenever a bare Tensor comes back.
    """
    encoder = getattr(model, "audio_encoder", None)
    layers = getattr(encoder, "layers", None)
    if not layers:
        return
    for layer in layers:
        if getattr(layer, "_moss_tuple_wrapped", False):
            continue
        orig_forward = layer.forward

        def _make_wrapper(fn):
            def _wrapped(*args, **kwargs):
                out = fn(*args, **kwargs)
                return out if isinstance(out, (tuple, list)) else (out,)

            return _wrapped

        layer.forward = _make_wrapper(orig_forward)
        layer._moss_tuple_wrapped = True


def _resolve_device_and_dtype(torch_mod, device="auto", dtype="auto"):
    if device == "auto" or not device:
        resolved_device = "cuda:0" if torch_mod.cuda.is_available() else "cpu"
    elif device == "cuda":
        resolved_device = "cuda:0" if torch_mod.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    if dtype == "bfloat16":
        resolved_dtype = torch_mod.bfloat16
    elif dtype == "float16":
        resolved_dtype = torch_mod.float16
    elif dtype == "float32":
        resolved_dtype = torch_mod.float32
    else:
        if resolved_device.startswith("cuda"):
            bf16_ok = bool(getattr(torch_mod.cuda, "is_bf16_supported", lambda: True)())
            resolved_dtype = torch_mod.bfloat16 if bf16_ok else torch_mod.float16
        else:
            resolved_dtype = torch_mod.float32
    return resolved_device, resolved_dtype


def _prepare_mono_audio(waveform, sample_rate, target_rate=16000):
    """`[C, T]` or `[T]` float32 -> 1D float32 numpy array at `target_rate`."""
    mono = dsp.to_mono(waveform)
    src_rate = int(sample_rate or target_rate)
    target_rate = int(target_rate or 16000)
    if src_rate == target_rate or mono.size == 0:
        return mono
    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(src_rate, target_rate)
        resampled = resample_poly(mono, target_rate // divisor, src_rate // divisor)
        return np.asarray(resampled, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return dsp.resample_linear(mono, src_rate, target_rate)


# --------------------------------------------------------------------------------------
# in-process runner + VRAM lifecycle
# --------------------------------------------------------------------------------------


class LocalMossRunner:
    """Holds a loaded `MossMusicModel` + `MossMusicProcessor` pair."""

    def __init__(self, model_dir, device="auto", dtype="auto"):
        import torch

        self.model_dir = os.path.abspath(model_dir)
        self.model_name = os.path.basename(self.model_dir)
        self.device, self.dtype = _resolve_device_and_dtype(torch, device, dtype)

        cfg_mod, modeling_mod, proc_mod = _load_moss_modules(self.model_dir)
        config = cfg_mod.MossMusicConfig.from_pretrained(self.model_dir, local_files_only=True)
        if isinstance(getattr(config, "audio_config", None), dict):
            config.audio_config = cfg_mod.MossMusicEncoderConfig.from_dict(config.audio_config)

        has_accelerate = importlib.util.find_spec("accelerate") is not None
        load_kwargs = {"config": config, "local_files_only": True}
        if has_accelerate and self.device.startswith("cuda"):
            load_kwargs["device_map"] = self.device
            load_kwargs["low_cpu_mem_usage"] = True

        try:
            model = modeling_mod.MossMusicModel.from_pretrained(
                self.model_dir, dtype=self.dtype, **load_kwargs
            )
        except TypeError:
            model = modeling_mod.MossMusicModel.from_pretrained(
                self.model_dir, torch_dtype=self.dtype, **load_kwargs
            )

        if "device_map" not in load_kwargs:
            model = model.to(device=self.device, dtype=self.dtype)

        _patch_encoder_layers_for_transformers5(model)
        model.eval()
        self.model = model
        self.processor = proc_mod.MossMusicProcessor.from_pretrained(
            self.model_dir, local_files_only=True, enable_time_marker=True
        )

    def ask(self, waveform, sample_rate, prompt, max_tokens=1024, temperature=0.0):
        import torch

        mel_sr = int(getattr(getattr(self.processor, "config", None), "mel_sr", 16000) or 16000)
        raw_audio = _prepare_mono_audio(waveform, sample_rate, target_rate=mel_sr)

        inputs = self.processor(text=prompt, audios=[raw_audio], return_tensors="pt")
        try:
            param = next(self.model.parameters())
            model_device, model_dtype = param.device, param.dtype
        except StopIteration:
            model_device, model_dtype = torch.device(self.device), self.dtype

        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved[key] = value.to(model_device)
            else:
                moved[key] = value
        if moved.get("audio_data") is not None:
            moved["audio_data"] = moved["audio_data"].to(model_dtype)
        moved["audio_input_mask"] = moved["input_ids"] == self.processor.audio_token_id

        do_sample = float(temperature) > 0.0
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "do_sample": do_sample,
            "num_beams": 1,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs.update({"temperature": float(temperature), "top_p": 0.9, "top_k": 50})

        with torch.no_grad():
            generated = self.model.generate(**moved, **gen_kwargs)

        input_len = int(moved["input_ids"].shape[1])
        decoded = self.processor.decode(generated[0, input_len:], skip_special_tokens=True)
        return moss._strip_think(decoded)


def get_runner(model_dir, device="auto", dtype="auto"):
    key = (os.path.abspath(model_dir), str(device), str(dtype))
    if _CACHE["key"] == key and _CACHE["runner"] is not None:
        return _CACHE["runner"]
    unload_cached()
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:  # noqa: BLE001
        pass
    runner = LocalMossRunner(model_dir, device=device, dtype=dtype)
    _CACHE["key"] = key
    _CACHE["runner"] = runner
    return runner


def unload_cached():
    runner = _CACHE.get("runner")
    _CACHE["key"] = None
    _CACHE["runner"] = None
    if runner is not None:
        try:
            del runner.model
            del runner.processor
        except Exception:  # noqa: BLE001
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:  # noqa: BLE001
        pass


def analyse_waveform(
    waveform,
    sample_rate,
    model_dir,
    device="auto",
    dtype="auto",
    keep_loaded=False,
    mode="auto",
    max_tokens=1024,
    temperature=0.0,
    want_lyrics=True,
    prompts_overrides=None,
):
    """Run the same structured prompts as `moss.analyse_audio` on in-memory audio."""
    overrides = prompts_overrides or {}
    answers = {}
    result = {}

    runner = get_runner(model_dir, device=device, dtype=dtype)
    try:
        if mode in ("attributes", "auto", "caption"):
            ask_prompt = overrides.get("attributes") or (
                moss.PROMPT_ATTRIBUTES if mode != "caption" else moss.PROMPT_CAPTION
            )
            text = runner.ask(waveform, sample_rate, ask_prompt, max_tokens, temperature)
            answers["attributes"] = text
            parsed = moss.extract_json(text)
            if isinstance(parsed, dict):
                result.update({k: v for k, v in parsed.items() if v not in (None, "")})
            elif text:
                result["caption"] = text.strip()

            if mode == "auto" and want_lyrics:
                vocal_present = result.get("vocal_present")
                if vocal_present is None:
                    vocal_present = not bool(result.get("instrumental", False))
                if vocal_present:
                    lyrics = runner.ask(
                        waveform,
                        sample_rate,
                        overrides.get("lyrics") or moss.PROMPT_LYRICS,
                        max_tokens,
                        temperature,
                    )
                    answers["lyrics"] = lyrics
                    result["lyrics"] = lyrics

        elif mode == "vocal":
            text = runner.ask(
                waveform, sample_rate, overrides.get("vocal") or moss.PROMPT_VOCAL,
                max_tokens, temperature,
            )
            answers["vocal"] = text
            result.update(moss.extract_json(text) or {})
            result.setdefault("caption", text.strip())

        elif mode == "lyrics":
            text = runner.ask(
                waveform, sample_rate, overrides.get("lyrics") or moss.PROMPT_LYRICS,
                max_tokens, temperature,
            )
            answers["lyrics"] = text
            result["lyrics"] = text
            result["instrumental"] = text.strip().lower().startswith("[instrumental]")

        elif mode == "key_bpm":
            text = runner.ask(
                waveform, sample_rate, overrides.get("key_bpm") or moss.PROMPT_KEY_BPM,
                max_tokens, temperature,
            )
            answers["key_bpm"] = text
            result.update(moss.extract_json(text) or {})
            result.setdefault("caption", text.strip())
    finally:
        if not keep_loaded:
            unload_cached()

    result["raw"] = answers
    result["model"] = os.path.basename(os.path.abspath(model_dir))
    result["backend"] = "transformers"
    return result
