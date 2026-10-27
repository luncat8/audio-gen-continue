#!/usr/bin/env python3
"""Tests for comfyui_audio_continue/nodes.py (numpy stub, no GPU needed).

    python3 tests/test_nodes.py

Requires numpy (the ComfyUI pack itself only needs torch).
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import torch_stub  # noqa: E402

torch = torch_stub.install()
Tensor = torch_stub.Tensor
FakeVAE = torch_stub.FakeVAE

# load nodes.py as a standalone module (its package __init__ only re-exports)
spec = importlib.util.spec_from_file_location(
    "audio_continue_nodes", os.path.join(ROOT, "comfyui_audio_continue", "nodes.py")
)
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


def sine(seconds, rate, freq=440.0, channels=2, amplitude=0.25):
    t = np.arange(int(seconds * rate)) / rate
    mono = amplitude * np.sin(2 * np.pi * freq * t)
    data = np.stack([mono] * channels, axis=0)[None, ...]  # [1, C, T]
    return {"waveform": Tensor(data.astype(np.float32)), "sample_rate": rate}


class TestExtend(unittest.TestCase):
    def setUp(self):
        self.node = nodes.AudioContinueExtend()
        self.vae = FakeVAE()
        self.audio = sine(5.0, 44100)

    def test_lengths_and_mask(self):
        latent, src, added, total, frames = self.node.extend(self.audio, self.vae, 30.0)
        self.assertAlmostEqual(src, 5.0, places=2)
        self.assertAlmostEqual(added, 30.0, places=2)
        self.assertAlmostEqual(total, 35.0, places=2)
        self.assertEqual(frames, 125)  # 5 s * 25 fps
        self.assertEqual(latent["samples"].shape, (1, 64, 875))

        mask = latent["noise_mask"].numpy()
        self.assertEqual(mask.shape, (1, 1, 875))
        self.assertTrue(np.all(mask[..., :125] == 0.0), "source region must stay untouched")
        self.assertTrue(np.all(mask[..., 125:] == 1.0), "new region must be generated")
        self.assertEqual(latent["type"], "audio")

    def test_empty_new_region(self):
        latent, *_ = self.node.extend(self.audio, self.vae, 30.0)
        samples = latent["samples"].numpy()
        self.assertTrue(np.all(samples[..., 125:] == 0.0), "appended frames must be empty")
        self.assertFalse(np.all(samples[..., :125] == 0.0), "source frames must be encoded audio")

    def test_repaint_tail(self):
        latent, _src, _added, _total, frames = self.node.extend(
            self.audio, self.vae, 10.0, mask_mode="repaint_tail", repaint_seconds=2.0
        )
        mask = latent["noise_mask"].numpy()
        # 2 s * 25 fps = 50 frames of the source are also regenerated
        self.assertTrue(np.all(mask[..., :75] == 0.0))
        self.assertTrue(np.all(mask[..., 75:] == 1.0))
        self.assertEqual(latent["samples"].shape[-1], frames + int(round(10.0 * 25)))

    def test_zero_repaint_keeps_new_only(self):
        latent, *_ = self.node.extend(self.audio, self.vae, 10.0, mask_mode="repaint_tail", repaint_seconds=0.0)
        mask = latent["noise_mask"].numpy()
        self.assertTrue(np.all(mask[..., :125] == 0.0))
        self.assertTrue(np.all(mask[..., 125:] == 1.0))

    def test_mask_is_fp32_for_sampler(self):
        latent, *_ = self.node.extend(self.audio, self.vae, 5.0)
        self.assertEqual(latent["noise_mask"].numpy().dtype, np.float32)


class TestStitch(unittest.TestCase):
    def setUp(self):
        self.node = nodes.AudioContinueStitch()

    def test_keeps_original_and_appends(self):
        src = sine(4.0, 44100, freq=220.0, channels=2)
        gen = sine(4.0 + 20.0, 48000, freq=220.0, channels=2)
        out = self.node.stitch(src, gen, 1.0)[0]
        self.assertEqual(out["sample_rate"], 48000)
        out_a = out["waveform"].numpy()

        src_resampled = torch_stub._resample(src["waveform"], 44100, 48000).numpy()
        head = src_resampled.shape[-1] - 48000
        np.testing.assert_allclose(out_a[..., :head], src_resampled[..., :head], atol=1e-6)
        # the generated tail is untouched after the crossfade
        tail_start = src_resampled.shape[-1]
        gen_a = gen["waveform"].numpy()
        np.testing.assert_allclose(out_a[..., tail_start:], gen_a[..., tail_start:], atol=1e-6)
        # and the crossfade ends exactly on the generated audio
        np.testing.assert_allclose(out_a[..., src_resampled.shape[-1] - 1], gen_a[..., src_resampled.shape[-1] - 1], atol=1e-6)

    def test_generated_shorter_than_source_is_not_truncated(self):
        src = sine(4.0, 44100)
        gen = sine(2.0, 48000)
        out = self.node.stitch(src, gen, 1.0)[0]
        self.assertEqual(out["waveform"].shape[-1], gen["waveform"].shape[-1])

    def test_mono_source_stereo_result(self):
        src = sine(2.0, 44100, channels=1)
        gen = sine(10.0, 44100, channels=2)
        out = self.node.stitch(src, gen, 0.5)[0]
        self.assertEqual(out["waveform"].shape[1], 2)

    def test_zero_crossfade_is_a_hard_cut(self):
        src = sine(2.0, 48000, channels=2)
        gen = sine(12.0, 48000, channels=2, amplitude=0.1)
        out = self.node.stitch(src, gen, 0.0)[0]
        a = out["waveform"].numpy()
        src_len = src["waveform"].shape[-1]
        np.testing.assert_allclose(a[..., src_len - 1], src["waveform"].numpy()[..., src_len - 1], atol=1e-6)
        np.testing.assert_allclose(a[..., src_len], gen["waveform"].numpy()[..., src_len], atol=1e-6)

    def test_output_rate_modes(self):
        src = sine(2.0, 44100)
        gen = sine(6.0, 48000)
        self.assertEqual(self.node.stitch(src, gen, 0.5, output_sample_rate="source")[0]["sample_rate"], 44100)
        self.assertEqual(self.node.stitch(src, gen, 0.5, output_sample_rate="continued")[0]["sample_rate"], 48000)
        self.assertEqual(self.node.stitch(src, gen, 0.5, output_sample_rate="highest")[0]["sample_rate"], 48000)


class TestMapping(unittest.TestCase):
    def test_node_mapping(self):
        self.assertIn("AudioContinueExtend", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("AudioContinueStitch", nodes.NODE_CLASS_MAPPINGS)
        for cls in nodes.NODE_CLASS_MAPPINGS.values():
            spec = cls.INPUT_TYPES()
            self.assertIn("required", spec)
            self.assertTrue(spec["required"])
            self.assertIsInstance(cls.RETURN_TYPES, tuple)
            self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
