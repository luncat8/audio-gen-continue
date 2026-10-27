"""
Minimal numpy-backed stand-ins for `torch` and the tiny slice of `comfy` that
comfyui_audio_continue/nodes.py touches. Only exists so the node math can be tested
without a GPU / ComfyUI install.
"""

import sys
import types

import numpy as np


class Tensor:
    def __init__(self, array):
        self._a = np.asarray(array)

    # -- attributes ----------------------------------------------------------
    @property
    def shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def device(self):
        return "cpu"

    def dim(self):
        return self._a.ndim

    def numpy(self):
        return self._a

    # -- ops -----------------------------------------------------------------
    def clone(self):
        return Tensor(self._a.copy())

    def to(self, dtype=None, device=None, **kwargs):
        return Tensor(self._a.astype(dtype) if dtype is not None else self._a.copy())

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self._a, dim))

    def movedim(self, source, destination):
        return Tensor(np.moveaxis(self._a, source, destination))

    def repeat(self, *reps):
        return Tensor(np.tile(self._a, reps))

    def view(self, *shape):
        return Tensor(self._a.reshape(shape))

    def __getitem__(self, key):
        if isinstance(key, Tensor):
            key = key._a
        return Tensor(self._a[key])

    def __setitem__(self, key, value):
        self._a[key] = value._a if isinstance(value, Tensor) else value

    # -- arithmetic (only what the nodes use) --------------------------------
    def _value(self, other):
        return other._a if isinstance(other, Tensor) else other

    def __add__(self, other):
        return Tensor(self._a + self._value(other))

    def __radd__(self, other):
        return Tensor(self._value(other) + self._a)

    def __sub__(self, other):
        return Tensor(self._a - self._value(other))

    def __rsub__(self, other):
        return Tensor(self._value(other) - self._a)

    def __mul__(self, other):
        return Tensor(self._a * self._value(other))

    def __rmul__(self, other):
        return Tensor(self._value(other) * self._a)

    def __repr__(self):
        return f"Tensor(shape={self._a.shape}, dtype={self._a.dtype})"


def _dtype(dtype):
    return dtype if dtype is not None else np.float32


def make_torch():
    torch = types.ModuleType("torch")
    torch.float32 = np.float32
    torch.float16 = np.float16
    torch.bfloat16 = np.float32  # good enough for the tests
    torch.Tensor = Tensor

    def zeros(*shape, dtype=None, device=None, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.zeros(shape, dtype=_dtype(dtype)))

    def cat(tensors, dim=0):
        arrays = [t.numpy() if isinstance(t, Tensor) else np.asarray(t) for t in tensors]
        return Tensor(np.concatenate(arrays, axis=dim))

    def linspace(start, end, steps, dtype=None, device=None, **kwargs):
        return Tensor(np.linspace(start, end, int(steps), dtype=_dtype(dtype)))

    torch.zeros = zeros
    torch.cat = cat
    torch.linspace = linspace
    return torch


def _resample(waveform, orig_freq, new_freq):
    if orig_freq == new_freq:
        return waveform
    a = waveform.numpy() if isinstance(waveform, Tensor) else np.asarray(waveform)
    n_out = int(round(a.shape[-1] * new_freq / orig_freq))
    x_old = np.arange(a.shape[-1], dtype=np.float64)
    x_new = np.linspace(0, a.shape[-1] - 1, n_out, dtype=np.float64)
    flat = a.reshape(-1, a.shape[-1])
    out = np.stack([np.interp(x_new, x_old, row) for row in flat], axis=0)
    return Tensor(out.reshape(a.shape[:-1] + (n_out,)).astype(a.dtype))


def install():
    """Install the stub modules; call before importing the node pack."""
    torch = make_torch()
    sys.modules["torch"] = torch

    comfy = types.ModuleType("comfy")
    audio = types.ModuleType("comfy.audio")
    audio.resample = _resample
    mm = types.ModuleType("comfy.model_management")
    mm.intermediate_device = lambda: "cpu"
    mm.intermediate_dtype = lambda: np.float32
    comfy.audio = audio
    comfy.model_management = mm
    sys.modules["comfy"] = comfy
    sys.modules["comfy.audio"] = audio
    sys.modules["comfy.model_management"] = mm
    return torch


class FakeVAE:
    """25 latent frames per second of 48 kHz audio, like the ACE-Step 1.5 VAE."""

    audio_sample_rate = 48000
    hop = 1920
    channels = 64

    def encode(self, waveform):
        a = waveform.numpy() if isinstance(waveform, Tensor) else np.asarray(waveform)
        # [B, T, C] -> [B, C, T] -> mean pool by hop
        a = np.moveaxis(a, -1, 1).astype(np.float32)
        frames = a.shape[-1] // self.hop
        pooled = a[..., : frames * self.hop].reshape(a.shape[0], a.shape[1], frames, self.hop).mean(-1)
        # pretend to be a 64 channel latent
        latent = np.repeat(pooled[:, :1], self.channels, axis=1)
        return Tensor(latent)
