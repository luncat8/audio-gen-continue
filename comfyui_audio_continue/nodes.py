"""
Audio Continue - two small ComfyUI nodes that make "continue this track" possible
with a plain KSampler, without any model-specific support in ComfyUI core.

Why these nodes exist
---------------------
ComfyUI core has no node that appends empty latent frames to an *audio* latent and
masks the new region. Without that you cannot express "keep everything up to T,
generate everything after T", which is exactly what a continuation is.

AudioContinueExtend:
    AUDIO + VAE -> LATENT (original frames + N seconds of empty frames, noise_mask
    set to 1.0 on the new frames only) + the exact source/total length in seconds.
    The total length is meant to be plugged into the text-encode "duration" input
    and/or the KSampler, see RESEARCH.md.

AudioContinueStitch:
    source AUDIO + decoded AUDIO -> AUDIO where the original recording is kept
    sample-exact and only the generated tail is appended, joined by a crossfade.
    This hides the VAE round-trip of the source part and any DC/level jump at the
    seam.

Both nodes use only ComfyUI core APIs (torch + comfy.audio), no extra dependencies.
"""

import torch

import comfy.audio
import comfy.model_management

CATEGORY = "audio/continue"


def _as_waveform(audio):
    """Return (waveform[B, C, T], sample_rate) from a ComfyUI AUDIO dict."""
    waveform = audio["waveform"]
    if hasattr(waveform, "movedim") and waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, int(audio["sample_rate"])


def _match_channels(waveform, channels):
    """Repeat/drop channels so the last-but-one dim equals `channels`."""
    have = waveform.shape[-2]
    if have == channels:
        return waveform
    if have == 1:
        return waveform.repeat(1, channels, 1)
    return waveform[:, :channels]


class AudioContinueExtend:
    """Encode audio, append empty latent frames and mask them for generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "vae": ("VAE",),
                "add_seconds": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.1,
                        "max": 900.0,
                        "step": 0.1,
                        "tooltip": "How many seconds of new music to generate after the end of the file.",
                    },
                ),
            },
            "optional": {
                "mask_mode": (
                    ["generate_new_only", "repaint_tail"],
                    {
                        "default": "generate_new_only",
                        "tooltip": (
                            "generate_new_only: only the appended frames are generated. "
                            "repaint_tail: additionally regenerate the last `repaint_seconds` "
                            "of the source so the model can re-do the join, then the stitch "
                            "node crossfades back into the original."
                        ),
                    },
                ),
                "repaint_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1, "advanced": True},
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("latent", "source_seconds", "added_seconds", "total_seconds", "source_frames")
    FUNCTION = "extend"
    CATEGORY = CATEGORY
    DESCRIPTION = "Append empty latent frames to an encoded audio track and mask them for the sampler."

    def extend(self, audio, vae, add_seconds, mask_mode="generate_new_only", repaint_seconds=0.0):
        waveform, sample_rate = _as_waveform(audio)
        vae_rate = int(getattr(vae, "audio_sample_rate", sample_rate) or sample_rate)
        if vae_rate != sample_rate:
            waveform = comfy.audio.resample(waveform, sample_rate, vae_rate)

        source_seconds = float(waveform.shape[-1]) / float(vae_rate)

        # Same call the core VAEEncodeAudio node makes: [B, C, T] -> [B, T, C].
        samples = vae.encode(waveform.movedim(1, -1))
        if isinstance(samples, dict):  # older/odd VAE wrappers
            samples = samples["samples"]

        frames = int(samples.shape[-1])
        if frames < 1 or source_seconds <= 0.0:
            raise ValueError("AudioContinueExtend: could not encode the input audio.")

        frames_per_second = frames / source_seconds
        add_frames = max(1, int(round(add_seconds * frames_per_second)))

        empty = torch.zeros(
            *samples.shape[:-1],
            add_frames,
            dtype=samples.dtype,
            device=samples.device,
        )
        extended = torch.cat((samples, empty), dim=-1)

        total_frames = int(extended.shape[-1])
        noise_mask = torch.zeros(
            (extended.shape[0], 1, total_frames), dtype=torch.float32, device=extended.device
        )
        mask_start = frames
        if mask_mode == "repaint_tail" and repaint_seconds > 0.0:
            mask_start = max(0, frames - int(round(repaint_seconds * frames_per_second)))
        noise_mask[..., mask_start:] = 1.0

        latent = {
            "samples": extended,
            "noise_mask": noise_mask,
            "type": "audio",
        }
        added_seconds = add_frames / frames_per_second
        total_seconds = frames / frames_per_second + added_seconds

        # the masked prefix is still attended to at every step, so very long sources are
        # expensive. Say something instead of silently taking 10 minutes.
        if total_seconds > 600.0:
            print(
                f"[audio-continue] warning: {total_seconds:.0f}s total. ACE-Step 1.5 is trained "
                "for songs up to ~10 minutes and the whole latent is sampled at once - "
                "expect long runtimes / high VRAM on long files."
            )
        elif source_seconds > 300.0:
            print(
                f"[audio-continue] note: source is {source_seconds / 60:.1f} min. If VRAM or speed "
                "is a problem, continue a shorter tail (cut the last ~60s with any editor, "
                "continue that, then append the result)."
            )
        return (latent, source_seconds, added_seconds, total_seconds, frames)


class AudioContinueStitch:
    """Keep the original recording, append the generated tail with a crossfade."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio": ("AUDIO",),
                "continued_audio": ("AUDIO",),
                "crossfade_seconds": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.1,
                        "tooltip": "Blend window around the seam. Only the original tail is replaced.",
                    },
                ),
            },
            "optional": {
                "output_sample_rate": (
                    ["highest", "source", "continued"],
                    {
                        "default": "highest",
                        "tooltip": "Sample rate of the result when the two inputs differ.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "stitch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Join the untouched source with the generated continuation using a crossfade."

    def stitch(self, source_audio, continued_audio, crossfade_seconds, output_sample_rate="highest"):
        src, src_rate = _as_waveform(source_audio)
        gen, gen_rate = _as_waveform(continued_audio)

        if output_sample_rate == "source":
            rate = src_rate
        elif output_sample_rate == "continued":
            rate = gen_rate
        else:
            rate = max(src_rate, gen_rate)

        if src_rate != rate:
            src = comfy.audio.resample(src, src_rate, rate)
        if gen_rate != rate:
            gen = comfy.audio.resample(gen, gen_rate, rate)

        channels = max(src.shape[-2], gen.shape[-2])
        src = _match_channels(src, channels)
        gen = _match_channels(gen, channels)

        src_len = int(src.shape[-1])
        gen_len = int(gen.shape[-1])

        if gen_len <= src_len:
            # Nothing was added (or the decode came back short) - do not truncate the song.
            print(
                "[audio-continue] warning: generated audio "
                f"({gen_len / rate:.2f}s) is not longer than the source ({src_len / rate:.2f}s); "
                "returning the generated audio unchanged."
            )
            out = gen
        else:
            out = gen.clone()
            fade = max(0, min(int(round(crossfade_seconds * rate)), src_len, gen_len))
            head = src_len - fade
            if head > 0:
                out[..., :head] = src[..., :head].to(out.dtype)
            if fade > 0:
                ramp = torch.linspace(0.0, 1.0, fade, dtype=out.dtype, device=out.device)
                ramp = ramp.view(1, 1, -1)
                original = src[..., head:src_len].to(out.dtype)
                generated = out[..., head:src_len]
                out[..., head:src_len] = original * (1.0 - ramp) + generated * ramp

        return ({"waveform": out, "sample_rate": rate},)


NODE_CLASS_MAPPINGS = {
    "AudioContinueExtend": AudioContinueExtend,
    "AudioContinueStitch": AudioContinueStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioContinueExtend": "Audio Continue (extend latent)",
    "AudioContinueStitch": "Audio Continue (stitch result)",
}
