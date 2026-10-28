#!/usr/bin/env python3
"""Tests for comfyui_music_caption (DSP, merge, tap, MOSS client).

    python3 tests/test_caption_nodes.py

Needs numpy; torch and comfy are stubbed (tests/torch_stub.py), so no GPU.
"""

import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import torch_stub  # noqa: E402

torch_stub.install()

from comfyui_music_caption import dsp, moss, nodes, textutil  # noqa: E402

RATE = 22050


# --------------------------------------------------------------------------------------
# synthetic material
# --------------------------------------------------------------------------------------


def beat_track(bpm=120.0, seconds=8.0, notes=(57, 60, 64, 69), kick=True):
    """Arpeggio + kick drum, i.e. something with an unambiguous tempo and key."""
    count = int(RATE * seconds)
    t = np.arange(count) / RATE
    out = np.zeros(count)
    beat = 60.0 / bpm
    step = 0.0
    index = 0
    while step < seconds:
        freq = 440.0 * 2 ** ((notes[index % len(notes)] - 69) / 12.0)
        start = int(step * RATE)
        end = min(count, start + int(beat * 0.45 * RATE))
        segment = t[start:end] - step
        out[start:end] += 0.25 * np.sin(2 * np.pi * freq * segment) * np.exp(-2 * segment)
        index += 1
        step += beat / 2
    if kick:
        step = 0.0
        while step < seconds:
            start = int(step * RATE)
            end = min(count, start + int(0.12 * RATE))
            segment = t[start:end] - step
            out[start:end] += 0.6 * np.sin(
                2 * np.pi * (60 * np.exp(-8 * segment)) * segment
            ) * np.exp(-18 * segment)
            step += beat
    return out


def pad(seconds=6.0, freqs=(220.0, 277.18, 329.63)):
    t = np.arange(int(RATE * seconds)) / RATE
    return sum(0.2 * np.sin(2 * np.pi * f * t) for f in freqs)


def sung_line(seconds=6.0, f0=220.0):
    """Sawtooth with vibrato and syllable-rate amplitude modulation."""
    t = np.arange(int(RATE * seconds)) / RATE
    gliding = f0 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
    saw = 2 * ((gliding.cumsum() / RATE) % 1) - 1
    envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3.1 * t))
    noise = 0.02 * np.random.default_rng(0).standard_normal(t.size)
    return 0.35 * saw * envelope + noise


def audio_dict(samples, rate=RATE, channels=2):
    data = np.stack([samples] * channels, axis=0)[None, ...].astype(np.float32)
    return {"waveform": torch_stub.Tensor(data), "sample_rate": rate}


# --------------------------------------------------------------------------------------
# dsp
# --------------------------------------------------------------------------------------


class TestAnalysis(unittest.TestCase):
    def test_tempo_is_found_up_to_an_octave(self):
        for expected in (96.0, 120.0, 128.0):
            result = dsp.analyse(beat_track(bpm=expected)[None, :], RATE, sections=False)
            detected = result["bpm"]
            self.assertTrue(detected > 0, expected)
            # autocorrelation tempo is inherently octave-ambiguous, so accept the
            # half/double reading - but it must be one of them
            close = min(
                abs(detected - expected) / expected,
                abs(detected - 2 * expected) / expected,
                abs(detected - expected / 2) / expected,
            )
            self.assertLess(close, 0.06, f"{expected} -> {detected}")
            self.assertTrue(result["bpm_candidates"])

    def test_key_of_a_minor_arpeggio(self):
        result = dsp.analyse(beat_track(notes=(57, 60, 64, 69))[None, :], RATE, sections=False)
        self.assertEqual(result["key"], "A minor")
        self.assertEqual(result["camelot"], "8A")
        self.assertGreater(result["key_confidence"], 0.3)

    def test_sustained_pad_is_not_called_a_vocal(self):
        result = dsp.analyse(pad()[None, :], RATE, sections=False)
        self.assertEqual(result["vocal"]["presence"], "instrumental")

    def test_syllabic_line_is_flagged_as_vocals(self):
        result = dsp.analyse(sung_line()[None, :], RATE, sections=False)
        self.assertEqual(result["vocal"]["presence"], "vocals")
        self.assertIn(result["vocal"]["type"], ("male", "female", "mixed / choir", "male (low)"))
        self.assertGreater(result["vocal"]["median_f0_hz"], 100)

    def test_short_and_silent_input_is_safe(self):
        for samples in (np.zeros(10, dtype=np.float32), np.zeros(RATE, dtype=np.float32)):
            result = dsp.analyse(samples[None, :], RATE, sections=False)
            self.assertIn("bpm", result)
            self.assertIn("vocal", result)
        tiny = dsp.analyse(np.zeros(100, dtype=np.float32), RATE)
        self.assertTrue(tiny["warnings"])

    def test_sections_and_tags(self):
        result = dsp.analyse(beat_track()[None, :], RATE, sections=True)
        self.assertTrue(result["sections"])
        tags = dsp.tags_from_analysis(result)
        self.assertIn("bpm", tags)
        self.assertIn("A minor", tags)

    def test_stereo_and_odd_shapes(self):
        mono = beat_track()
        stereo = np.stack([mono, mono * 0.5])
        a = dsp.analyse(mono[None, :], RATE, sections=False)
        b = dsp.analyse(stereo, RATE, sections=False)
        self.assertEqual(a["bpm"], b["bpm"])


# --------------------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------------------


class TestTextUtil(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(textutil.normalize_key("am"), "A minor")
        self.assertEqual(textutil.normalize_key("A minor"), "A minor")
        self.assertEqual(textutil.normalize_key("f#m"), "F# minor")
        self.assertEqual(textutil.normalize_key("Bb"), "A# major")
        self.assertEqual(textutil.normalize_key("ambient"), "")
        self.assertEqual(textutil.normalize_key(None), "")

    def test_bpm(self):
        self.assertEqual(textutil.normalize_bpm("96 bpm"), 96)
        self.assertEqual(textutil.normalize_bpm(0), 0)
        self.assertEqual(textutil.normalize_bpm("--"), 0)
        self.assertEqual(textutil.normalize_bpm(1e9), 0)

    def test_tags(self):
        self.assertEqual(textutil.dedupe_tags("a, b, A ,, b"), "a, b")
        merged = textutil.compose_tags(caption="dark synthwave", vocal="female lead",
                                       measured="96 bpm", extra="same mix")
        self.assertEqual(merged, "dark synthwave, female lead, 96 bpm, same mix")
        self.assertTrue(textutil.looks_instrumental(""))
        self.assertTrue(textutil.looks_instrumental("[instrumental]"))
        self.assertFalse(textutil.looks_instrumental("[verse] hello"))


class TestMossHelpers(unittest.TestCase):
    def test_extract_json_from_chat_noise(self):
        text = 'Sure! ```json\n{"bpm": 96, "key": "A minor"}\n``` hope that helps'
        self.assertEqual(moss.extract_json(text), {"bpm": 96, "key": "A minor"})
        self.assertIsNone(moss.extract_json("no json here"))
        self.assertIsNone(moss.extract_json(""))

    def test_think_block_is_stripped(self):
        self.assertEqual(moss._strip_think("<think>x</think> answer"), "answer")

    def test_server_alive_is_false_for_a_dead_port(self):
        self.assertFalse(moss.server_alive("http://127.0.0.1:9", timeout=0.5))


# --------------------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------------------


class TestAnalyzeNode(unittest.TestCase):
    def test_analyze_returns_everything(self):
        node = nodes.MusicCaptionAnalyze()
        audio = audio_dict(beat_track())
        out = node.analyze(audio, max_seconds=30.0, sections=False, extra_tags="vinyl crackle")
        self.assertEqual(len(out["result"]), len(nodes.ANALYZE_OUTPUTS))
        tags, bpm, keyscale, key_conf, bpm_conf, presence, vocal_type, duration, payload = out["result"]
        self.assertIsInstance(bpm, int)
        self.assertGreater(bpm, 0)
        self.assertEqual(keyscale, "A minor")
        self.assertIn("vinyl crackle", tags)
        self.assertAlmostEqual(duration, 8.0, places=1)
        self.assertGreater(key_conf, 0.3)
        self.assertIn(presence, ("vocals", "instrumental", "uncertain"))
        data = json.loads(out["ui"]["text"][0])
        self.assertEqual(data["key"], "A minor")

    def test_analyze_does_not_mutate_input(self):
        node = nodes.MusicCaptionAnalyze()
        mono = beat_track()
        audio = audio_dict(mono)
        before = audio["waveform"].numpy().copy()
        node.analyze(audio, max_seconds=5.0)
        np.testing.assert_array_equal(audio["waveform"].numpy(), before)


class TestMergeNode(unittest.TestCase):
    def test_merge_prefers_dsp_numbers(self):
        measured = json.dumps(
            {"bpm": 96.0, "key": "A minor",
             "vocal": {"presence": "vocals", "type": "female"}}
        )
        model = json.dumps(
            {"caption": "dark synthwave with analog bass", "bpm": 120,
             "keyscale": "C major", "vocal_type": "female", "structure": "verse-chorus"}
        )
        node = nodes.MusicCaptionMerge()
        out = node.merge(
            prefer_numeric="dsp", analysis_json=measured, moss_json=model,
            extra_tags="same mix",
        )
        tags, lyrics, bpm, keyscale, instrumental, vocal_type, summary = out["result"]
        self.assertEqual(bpm, 96)
        self.assertEqual(keyscale, "A minor")
        self.assertEqual(vocal_type, "female")
        self.assertFalse(instrumental)
        self.assertEqual(lyrics, "")
        self.assertIn("dark synthwave", tags)
        self.assertIn("same mix", tags)
        self.assertEqual(json.loads(summary)["bpm"], 96)

    def test_merge_prefers_model_numbers_when_asked(self):
        node = nodes.MusicCaptionMerge()
        out = node.merge(
            prefer_numeric="model",
            analysis_json=json.dumps({"bpm": 96.0, "key": "A minor"}),
            moss_json=json.dumps({"bpm": 100, "keyscale": "D minor"}),
        )
        self.assertEqual(out["result"][2], 100)
        self.assertEqual(out["result"][3], "D minor")

    def test_merge_survives_broken_json(self):
        node = nodes.MusicCaptionMerge()
        out = node.merge(prefer_numeric="dsp", analysis_json="not json", moss_json="{",
                         extra_tags="lo-fi")
        self.assertEqual(out["result"][4], False)
        self.assertIn("lo-fi", out["result"][0])


class TestTapNode(unittest.TestCase):
    def test_collects_in_order_and_keeps_placeholders(self):
        node = nodes.MusicCaptionTap()
        out = node.collect(label="x", text_1="caption", text_2="", text_3="96")
        self.assertEqual(json.loads(out["ui"]["text"][0]), ["caption", "", "96"])
        self.assertEqual(json.loads(out["result"][0]), ["caption", "", "96"])

    def test_stops_at_the_first_unconnected_input(self):
        out = nodes.MusicCaptionTap().collect(text_2="second")
        self.assertEqual(json.loads(out["ui"]["text"][0]), [])

    def test_input_types_are_optional(self):
        spec = nodes.MusicCaptionTap.INPUT_TYPES()
        self.assertEqual(spec["required"], {})
        self.assertIn("text_12", spec["optional"])


class TestMossNode(unittest.TestCase):
    def test_without_a_server_it_degrades(self):
        node = nodes.MusicCaptionMOSS()
        out = node.describe(audio_dict(beat_track(seconds=2.0)),
                            server_url="http://127.0.0.1:9", mode="auto", timeout=1.0)
        caption, lyrics, vocal, bpm, keyscale, instrumental, structure, blob = out["result"]
        self.assertEqual(caption, "")
        self.assertEqual(bpm, 0)
        self.assertFalse(instrumental)
        payload = json.loads(blob)
        self.assertFalse(payload["ok"])
        self.assertIn("no MOSS-Music server", payload["error"])
        self.assertEqual(json.loads(out["ui"]["text"][0])["ok"], False)

    def test_audio_is_written_windowed(self):
        waveform = np.zeros((2, int(RATE * 10)), dtype=np.float32)
        cut = nodes._segment(waveform, RATE, "tail", 4.0)
        self.assertEqual(cut.shape[-1], 4 * RATE)
        head = nodes._segment(waveform, RATE, "head", 4.0)
        self.assertEqual(head.shape[-1], 4 * RATE)
        everything = nodes._segment(waveform, RATE, "full", 0)
        self.assertEqual(everything.shape[-1], waveform.shape[-1])

    def test_wav_roundtrip(self):
        import tempfile
        import wave

        samples = np.stack([beat_track(seconds=1.0)] * 2)
        path = os.path.join(tempfile.mkdtemp(), "x.wav")
        nodes._write_wav(path, samples, RATE)
        with wave.open(path, "rb") as handle:
            self.assertEqual(handle.getnchannels(), 2)
            self.assertEqual(handle.getframerate(), RATE)
            self.assertAlmostEqual(
                handle.getnframes(), samples.shape[-1], delta=2
            )
        os.remove(path)


class TestNodeRegistration(unittest.TestCase):
    def test_every_node_is_mapped(self):
        self.assertEqual(
            set(nodes.NODE_CLASS_MAPPINGS),
            {
                "MusicCaptionMOSSLoader",
                "MusicCaptionAnalyze",
                "MusicCaptionMOSS",
                "MusicCaptionMerge",
                "MusicCaptionView",
                "MusicCaptionTap",
            },
        )
        for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
            self.assertTrue(hasattr(cls, "INPUT_TYPES"), name)
            self.assertTrue(hasattr(cls, "RETURN_TYPES"), name)
            self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES), name)
            self.assertTrue(callable(getattr(cls, cls.FUNCTION)), name)

    def test_sink_nodes_set_output_node_true_and_match_fake_comfyui(self):
        """Regression guard for ComfyUI's prompt_no_outputs HTTP 400 error."""
        import comfyui_music_caption as pkg
        from fake_comfyui import NODES as FAKE_NODES

        for sink_name in (
            "MusicCaptionAnalyze",
            "MusicCaptionMOSS",
            "MusicCaptionMerge",
            "MusicCaptionView",
            "MusicCaptionTap",
        ):
            cls = nodes.NODE_CLASS_MAPPINGS[sink_name]
            self.assertTrue(getattr(cls, "OUTPUT_NODE", False), f"{sink_name} must set OUTPUT_NODE = True")
            self.assertTrue(FAKE_NODES[sink_name].get("output_node"), f"fake_comfyui.{sink_name}")
        self.assertFalse(getattr(nodes.MusicCaptionMOSSLoader, "OUTPUT_NODE", False))
        for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
            self.assertEqual(list(cls.RETURN_TYPES), FAKE_NODES[name]["output"], name)
            self.assertEqual(list(cls.RETURN_NAMES), FAKE_NODES[name]["output_name"], name)
        self.assertEqual(pkg.WEB_DIRECTORY, "./web")
        self.assertTrue(
            os.path.isfile(os.path.join(ROOT, "comfyui_music_caption", "web", "music_caption.js"))
        )

    def test_links_declared_in_object_info_match_return_types(self):
        """The app links these by index - keep the order stable."""
        self.assertEqual(nodes.MusicCaptionAnalyze.RETURN_NAMES[8], "analysis_json")
        self.assertEqual(nodes.MusicCaptionMOSS.RETURN_NAMES[7], "moss_json")
        self.assertEqual(nodes.MusicCaptionMerge.RETURN_NAMES[6], "summary_json")


class TestMossLoaderAndLocal(unittest.TestCase):
    def _make_fake_model_dir(self, base_dir, name="MOSS-Music-8B-Instruct", complete=True):
        from comfyui_music_caption import moss_local

        model_dir = os.path.join(base_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        for fname in (*moss_local.REQUIRED_CODE_FILES, "tokenizer_config.json", "vocab.json", "merges.txt"):
            with open(os.path.join(model_dir, fname), "w", encoding="utf-8") as fh:
                fh.write("{}" if fname.endswith(".json") else "# stub\n")
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "weight_map": {
                        "layer.0": "model-00001-of-00002.safetensors",
                        "layer.1": "model-00002-of-00002.safetensors",
                    }
                },
                fh,
            )
        with open(os.path.join(model_dir, "model-00001-of-00002.safetensors"), "wb") as fh:
            fh.write(b"0")
        if complete:
            with open(os.path.join(model_dir, "model-00002-of-00002.safetensors"), "wb") as fh:
                fh.write(b"0")
        return model_dir

    def test_inspect_model_dir_checks_all_indexed_shards(self):
        import tempfile
        from comfyui_music_caption import moss_local

        with tempfile.TemporaryDirectory() as tmp:
            missing = moss_local.inspect_model_dir(os.path.join(tmp, "nope"))
            self.assertFalse(missing["ok"])
            self.assertEqual(missing["state"], "missing")

            partial_dir = self._make_fake_model_dir(tmp, name="partial", complete=False)
            partial = moss_local.inspect_model_dir(partial_dir)
            self.assertFalse(partial["ok"])
            self.assertEqual(partial["state"], "incomplete")
            self.assertIn("model-00002-of-00002.safetensors", partial["missing"])

            ready_dir = self._make_fake_model_dir(tmp, name="ready", complete=True)
            ready = moss_local.inspect_model_dir(ready_dir)
            self.assertTrue(ready["ok"])
            self.assertEqual(ready["state"], "ready")
            self.assertEqual(ready["missing"], [])

    def test_loader_skips_cleanly_or_errors_with_hf_link_when_missing(self):
        import tempfile

        loader = nodes.MusicCaptionMOSSLoader()
        with tempfile.TemporaryDirectory() as tmp:
            old_env = os.environ.get("MOSS_MUSIC_DIR")
            os.environ["MOSS_MUSIC_DIR"] = tmp
            try:
                handle, status = loader.load_model(
                    model_name="MOSS-Music-8B-Instruct",
                    backend="auto",
                    when_missing="skip (DSP only)",
                    server_url="http://127.0.0.1:9",
                )
                self.assertFalse(handle["ok"])
                self.assertEqual(handle["backend"], "missing")
                self.assertIn("huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct", status)

                # Downstream MusicCaptionMOSS skips without hitting HTTP when loader says missing
                moss_node = nodes.MusicCaptionMOSS()
                out = moss_node.describe(
                    audio_dict(beat_track(seconds=1.5)),
                    moss_model=handle,
                    server_url="http://127.0.0.1:9",
                )
                payload = json.loads(out["result"][7])
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["backend"], "missing")

                with self.assertRaises(RuntimeError) as ctx:
                    loader.load_model(
                        model_name="MOSS-Music-8B-Instruct",
                        backend="transformers",
                        when_missing="error",
                    )
                self.assertIn("https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct", str(ctx.exception))
            finally:
                if old_env is None:
                    os.environ.pop("MOSS_MUSIC_DIR", None)
                else:
                    os.environ["MOSS_MUSIC_DIR"] = old_env

    def test_loader_discovers_local_weights_and_moss_node_runs_in_process(self):
        import tempfile
        from comfyui_music_caption import moss_local

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._make_fake_model_dir(tmp)
            old_env = os.environ.get("MOSS_MUSIC_DIR")
            os.environ["MOSS_MUSIC_DIR"] = tmp
            orig_analyse = moss_local.analyse_waveform
            try:
                schema = nodes.MusicCaptionMOSSLoader.INPUT_TYPES()
                tooltip = schema["required"]["model_name"][1]["tooltip"]
                self.assertIn("Installed in models/moss_music: MOSS-Music-8B-Instruct", tooltip)

                loader = nodes.MusicCaptionMOSSLoader()
                handle, status = loader.load_model(model_name="MOSS-Music-8B-Instruct", backend="auto")
                self.assertEqual(handle["backend"], "transformers")
                self.assertTrue(handle["ok"])
                self.assertEqual(handle["model_dir"], model_dir)
                self.assertIn(model_dir, status)

                def fake_local_analyse(waveform, sample_rate, **kwargs):
                    self.assertEqual(kwargs["model_dir"], model_dir)
                    self.assertEqual(sample_rate, RATE)
                    return {
                        "ok": True,
                        "backend": "transformers",
                        "caption": "dreamy synth-pop with shimmering arpeggios and warm female vocals",
                        "tags": "synth-pop, shimmering arpeggios, warm bass",
                        "bpm": 118,
                        "keyscale": "F# minor",
                        "instrumental": False,
                        "vocal": "warm breathy female lead",
                        "vocal_type": "female",
                        "structure": "intro -> verse -> chorus",
                        "chords": "F#m - D - A - E",
                        "language": "English",
                        "lyrics": "[Verse]\nNeon lights across the rain",
                        "raw": "{}",
                    }

                moss_local.analyse_waveform = fake_local_analyse
                moss_node = nodes.MusicCaptionMOSS()
                out = moss_node.describe(
                    audio_dict(beat_track(seconds=2.0)),
                    moss_model=handle,
                    server_url="http://127.0.0.1:9",
                    mode="auto",
                )
                caption, lyrics, vocal, bpm, keyscale, instrumental, structure, blob = out["result"]
                self.assertIn("synth-pop", caption)
                self.assertIn("Neon lights", lyrics)
                self.assertEqual(bpm, 118)
                self.assertEqual(keyscale, "F# minor")
                self.assertFalse(instrumental)
                self.assertEqual(structure, "intro -> verse -> chorus")
                self.assertEqual(json.loads(blob)["chords"], "F#m - D - A - E")
            finally:
                moss_local.analyse_waveform = orig_analyse
                if old_env is None:
                    os.environ.pop("MOSS_MUSIC_DIR", None)
                else:
                    os.environ["MOSS_MUSIC_DIR"] = old_env

    def test_analyse_waveform_runs_attributes_plus_lyrics_and_unloads_when_requested(self):
        import tempfile
        from comfyui_music_caption import moss_local

        calls = []

        class FakeRunner:
            def __init__(self):
                self.model = object()
                self.processor = object()

            def ask(self, waveform, sample_rate, prompt, max_tokens, temperature):
                calls.append(prompt)
                if len(calls) == 1:
                    return (
                        "<think>Listening to harmony and vocals...</think>\n"
                        + json.dumps(
                            {
                                "caption": "lush indie pop with jangly guitars",
                                "bpm": 112,
                                "key": "G major",
                                "vocal_present": True,
                                "vocal_type": "female",
                                "structure": "verse -> chorus",
                                "chords": "G - Em - C - D",
                            }
                        )
                    )
                return "[Verse]\nSunlight through the windowpane"

        fake_runner = FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            moss_local._CACHE["key"] = (os.path.abspath(tmp), "auto", "auto")
            moss_local._CACHE["runner"] = fake_runner
            try:
                res = moss_local.analyse_waveform(
                    beat_track(seconds=1.5),
                    RATE,
                    model_dir=tmp,
                    keep_loaded=False,
                    mode="auto",
                )
                self.assertEqual(len(calls), 2)
                self.assertEqual(res["bpm"], 112)
                self.assertEqual(res["key"], "G major")
                self.assertEqual(res["chords"], "G - Em - C - D")
                self.assertIn("Sunlight through the windowpane", res["lyrics"])
                self.assertFalse(hasattr(fake_runner, "model"))
                self.assertIsNone(moss_local._CACHE["runner"])
            finally:
                moss_local.unload_cached()


class TestViewNode(unittest.TestCase):
    def test_viewer_formats_all_fields_and_clamps_bpm(self):
        analyze_out = nodes.MusicCaptionAnalyze().analyze(audio_dict(beat_track(seconds=8.0)))
        analysis_json = analyze_out["result"][8]
        moss_json = json.dumps(
            {
                "ok": True,
                "backend": "transformers",
                "caption": "intimate acoustic piece with fingerpicked guitar",
                "tags": "acoustic, fingerpicked guitar, warm",
                "bpm": 0,
                "keyscale": "A minor",
                "instrumental": False,
                "vocal_type": "female",
                "structure": "intro -> verse",
                "chords": "Am - F - C - G",
                "language": "English",
                "lyrics": "Soft morning light",
            }
        )
        merged = nodes.MusicCaptionMerge().merge(
            prefer_numeric="dsp",
            analysis_json=analysis_json,
            moss_json=moss_json,
            keep_lyrics=False,
        )["result"]
        summary_json = merged[6]

        viewed = nodes.MusicCaptionView().view(
            summary_json=summary_json,
            analysis_json=analysis_json,
            moss_json=moss_json,
            include_sections=True,
        )
        report, tags, caption, lyrics, bpm, keyscale, vocal_type, chords, structure = viewed["result"]
        self.assertIn("BPM:", report)
        self.assertIn("Key / Scale:  A minor", report)
        self.assertIn("Chords:       Am - F - C - G", report)
        self.assertIn("Structure:    intro -> verse", report)
        self.assertIn("Soft morning light", report)
        self.assertIn("Sections:", report)
        self.assertGreaterEqual(bpm, 10)
        self.assertLessEqual(bpm, 300)
        self.assertEqual(keyscale, "A minor")
        self.assertEqual(lyrics, "Soft morning light")
        self.assertEqual(chords, "Am - F - C - G")
        self.assertEqual(viewed["ui"]["text"][0], report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
