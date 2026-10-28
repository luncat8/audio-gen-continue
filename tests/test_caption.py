#!/usr/bin/env python3
"""Tests for the prompt-file and captioner parts of audio_continue_tk.py.

    python3 tests/test_caption.py

Uses the fake ComfyUI server, so no GPU and no model are needed.
"""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import audio_continue_tk as app  # noqa: E402
import fake_comfyui  # noqa: E402
from fake_comfyui import FakeState, start_server  # noqa: E402


def info_without(*classes):
    return {name: entry for name, entry in fake_comfyui.NODES.items() if name not in classes}


class TestPromptFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.audio = os.path.join(self.dir, "song.flac")
        with open(self.audio, "wb") as fh:
            fh.write(b"fLaC")

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_no_sidecar(self):
        self.assertEqual(app.find_prompt_file(self.audio), "")
        self.assertEqual(app.find_prompt_file(""), "")

    def test_prefers_prompt_suffix_over_plain_txt(self):
        self.write("song.txt", "wrong one")
        expected = self.write("song.prompt.txt", "dark synthwave, 96 bpm")
        self.assertEqual(app.find_prompt_file(self.audio), expected)
        self.assertEqual(app.read_prompt_file(expected), "dark synthwave, 96 bpm")

    def test_plain_txt_and_markdown_work(self):
        path = self.write("song.txt", "  lo-fi hip hop, vinyl noise\n")
        self.assertEqual(app.find_prompt_file(self.audio), path)
        self.assertEqual(app.read_prompt_file(path), "lo-fi hip hop, vinyl noise")

    def test_json_sidecar_with_tags(self):
        path = self.write("song.json", json.dumps({"tags": ["ambient", "drone"]}))
        self.assertEqual(app.read_prompt_file(path), "ambient, drone")

    def test_json_sidecar_with_caption_and_tempo(self):
        path = self.write("song.json", json.dumps({"caption": "neo-soul", "bpm": 88}))
        self.assertEqual(app.read_prompt_file(path), "neo-soul")

    def test_empty_and_missing_files(self):
        empty = self.write("song.tags.txt", "")
        self.assertEqual(app.read_prompt_file(empty), "")
        self.assertEqual(app.read_prompt_file(os.path.join(self.dir, "nope.txt")), "")

    def test_merge_append_and_replace(self):
        base = "instrumental, same tempo"
        self.assertEqual(
            app.merge_prompt_text(base, "same tempo, vinyl crackle", append=True),
            "instrumental, same tempo, vinyl crackle",
        )
        self.assertEqual(
            app.merge_prompt_text(base, "dark ambient", append=False), "dark ambient"
        )
        self.assertEqual(app.merge_prompt_text(base, "", append=True), base)
        self.assertEqual(app.merge_prompt_text("", "solo", append=True), "solo")


class TestNormalisation(unittest.TestCase):
    def test_keyscale(self):
        cases = {
            "am": "A minor",
            "A Min": "A minor",
            "A minor": "A minor",
            "f#m": "F# minor",
            "Bb": "A# major",
            "G major": "G major",
            "c#min": "C# minor",
            "atonal": "",
            "": "",
            "nonsense": "",
        }
        for raw, expected in cases.items():
            self.assertEqual(app.normalize_keyscale(raw), expected, raw)

    def test_bpm(self):
        self.assertEqual(app.normalize_bpm("96 bpm"), 96)
        self.assertEqual(app.normalize_bpm(128.4), 128)
        self.assertEqual(app.normalize_bpm("fast"), 0)
        self.assertEqual(app.normalize_bpm(None), 0)
        self.assertEqual(app.normalize_bpm(900), 0)


class TestCaptionerDiscovery(unittest.TestCase):
    def test_none_installed(self):
        info = info_without(*app.CAPTIONER_PREFERENCE, app.CAPTIONER_TAP, "SongScribeAnalyzer")
        plan = app.captioner_plan(info)
        self.assertEqual(plan["mode"], "none")
        self.assertIn("comfyui_music_caption", plan["reason"])

    def test_native_when_our_pack_is_there(self):
        plan = app.captioner_plan(fake_comfyui.NODES)
        self.assertEqual(plan["mode"], "native")
        self.assertEqual(plan["class"], "MusicCaptionMerge")

    def test_third_party_needs_the_tap(self):
        info = info_without(*app.CAPTIONER_PREFERENCE)
        plan = app.captioner_plan(info)
        self.assertEqual(plan["mode"], "tap")
        self.assertEqual(plan["class"], "SongScribeAnalyzer")

        info_no_tap = info_without(*app.CAPTIONER_PREFERENCE, app.CAPTIONER_TAP)
        plan = app.captioner_plan(info_no_tap)
        self.assertEqual(plan["mode"], "tap")
        self.assertIn("missing", plan["reason"])

    def test_explicit_off(self):
        self.assertEqual(app.captioner_plan(fake_comfyui.NODES, "off")["mode"], "none")

    def test_candidates_are_ranked_and_ignore_unrelated_nodes(self):
        info = dict(fake_comfyui.NODES)
        info["SomeUpscaler"] = {
            "input": {"required": {"image": ["IMAGE"]}},
            "output": ["IMAGE"],
            "output_name": ["image"],
        }
        info["TotallyUnrelatedAudio"] = {
            "input": {"required": {"audio": ["AUDIO"]}},
            "output": ["AUDIO"],
            "output_name": ["audio"],
            "display_name": "just passing audio through",
        }
        names = [item["class"] for item in app.captioner_candidates(info)]
        self.assertIn("SongScribeAnalyzer", names)
        self.assertNotIn("SomeUpscaler", names)
        self.assertNotIn("TotallyUnrelatedAudio", names)


class TestBuildCaptionPrompt(unittest.TestCase):
    def setUp(self):
        self.info = fake_comfyui.NODES

    def test_native_graph_wires_analyze_moss_and_merge(self):
        plan = app.captioner_plan(self.info)
        prompt, meta = app.build_caption_prompt(
            self.info, plan, "song.wav", moss_url="http://moss:30000"
        )
        self.assertEqual(prompt["1"]["class_type"], "LoadAudio")
        self.assertEqual(prompt["1"]["inputs"]["audio"], "song.wav")
        self.assertEqual(prompt["2"]["class_type"], "MusicCaptionAnalyze")
        self.assertEqual(prompt["2"]["inputs"]["audio"], ["1", 0])
        self.assertEqual(prompt["3"]["class_type"], "MusicCaptionMOSS")
        self.assertEqual(prompt["3"]["inputs"]["server_url"], "http://moss:30000")
        self.assertEqual(prompt["4"]["class_type"], "MusicCaptionMerge")
        self.assertEqual(prompt["4"]["inputs"]["analysis_json"], ["2", 8])
        self.assertEqual(prompt["4"]["inputs"]["moss_json"], ["3", 7])
        self.assertEqual(meta["read_node"], "4")
        # every link has to point at a node that exists
        for node in prompt.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2:
                    self.assertIn(value[0], prompt)

    def test_analyze_only_pack(self):
        info = info_without("MusicCaptionMOSS")
        plan = app.captioner_plan(info)
        prompt, meta = app.build_caption_prompt(info, plan, "song.wav")
        self.assertNotIn("3", prompt)
        self.assertEqual(meta["read_node"], "2")

    def test_explicit_moss_choice_skips_dsp(self):
        plan = app.captioner_plan(self.info, "MusicCaptionMOSS")
        prompt, meta = app.build_caption_prompt(self.info, plan, "song.wav")
        self.assertNotIn("2", prompt)
        self.assertEqual(meta["read_node"], "3")

    def test_third_party_goes_through_the_tap(self):
        info = info_without(*app.CAPTIONER_PREFERENCE)
        plan = app.captioner_plan(info)
        prompt, meta = app.build_caption_prompt(info, plan, "song.wav")
        self.assertEqual(prompt["2"]["class_type"], "SongScribeAnalyzer")
        self.assertEqual(prompt["2"]["inputs"]["audio"], ["1", 0])
        self.assertEqual(prompt["2"]["inputs"]["describe"], "off")
        self.assertEqual(prompt["3"]["class_type"], "MusicCaptionTap")
        self.assertEqual(prompt["3"]["inputs"]["label"], "SongScribeAnalyzer")
        # the STRING outputs of SongScribe, linked in order through the tap
        self.assertEqual(prompt["3"]["inputs"]["text_1"], ["2", 0])
        self.assertEqual(prompt["3"]["inputs"]["text_2"], ["2", 1])
        self.assertEqual(prompt["3"]["inputs"]["text_3"], ["2", 4])
        self.assertNotIn("text_4", prompt["3"]["inputs"])
        self.assertEqual(
            meta["output_names"], ["caption", "lyrics", "duration_str"]
        )
        self.assertEqual(meta["read_node"], "3")

    def test_no_captioner_raises(self):
        plan = {"mode": "none", "class": "", "label": "", "reason": "nothing there"}
        with self.assertRaises(app.ComfyError):
            app.build_caption_prompt({}, plan, "song.wav")


class TestInterpret(unittest.TestCase):
    def test_native_merge_payload(self):
        payload = json.loads(fake_comfyui.DEFAULT_CAPTION_TEXT)
        result = app.interpret_caption_payload(payload)
        self.assertEqual(result["bpm"], 96)
        self.assertEqual(result["keyscale"], "A minor")
        self.assertEqual(result["vocal_type"], "female")
        self.assertFalse(result["instrumental"])
        self.assertIn("dark synthwave", result["caption"])
        self.assertIn("96 bpm", result["notes"])

    def test_songscribe_field_names(self):
        fields = {
            "caption": "Global Metadata: synthwave, 96 BPM, A minor",
            "lyrics": "",
            "duration": 201.5,
            "duration_int": 201,
            "duration_str": "3:21",
        }
        result = app.interpret_caption_payload(fields)
        self.assertEqual(result["bpm"], 0)  # not guessed out of a caption
        self.assertIn("synthwave", result["caption"])

    def test_loose_text_payload(self):
        result = app.interpret_caption_payload("Just a plain description")
        self.assertEqual(result["caption"], "Just a plain description")

    def test_caption_tags_include_measurements(self):
        result = app.interpret_caption_payload(json.loads(fake_caption_json()))
        tags = app.caption_tags_from_result(result)
        self.assertIn("dark synthwave", tags)
        self.assertIn("A minor", tags)
        self.assertIn("96 bpm", tags)


def fake_caption_json():
    return json.dumps(
        {
            "bpm": 96,
            "keyscale": "A minor",
            "vocal_type": "female",
            "instrumental": False,
            "caption": "dark synthwave with analog bass",
        }
    )


class TestCaptionOverHttp(unittest.TestCase):
    def setUp(self):
        self.state = FakeState()
        self.server, self.url = start_server(self.state)
        self.client = app.ComfyClient(self.url, timeout=5)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_caption_run_returns_fields(self):
        info = self.client.object_info()
        plan = app.captioner_plan(info)
        uploaded = self.client.upload_audio(__file__)  # any existing file
        prompt, meta = app.build_caption_prompt(info, plan, uploaded)
        entry = self.client.wait(self.client.queue_prompt(prompt))
        result = app.caption_result_from_history(entry, meta)
        self.assertEqual(result["bpm"], 96)
        self.assertEqual(result["keyscale"], "A minor")
        self.assertEqual(result["vocal_type"], "female")
        self.assertTrue(result["caption"])

    def test_third_party_payload_is_mapped_by_output_name(self):
        self.state.caption_text = json.dumps(
            ["Global Metadata: 96 BPM, A minor", "", 201.5, 201, "3:21"]
        )
        info = info_without(*app.CAPTIONER_PREFERENCE)
        plan = app.captioner_plan(info)
        uploaded = self.client.upload_audio(__file__)
        prompt, meta = app.build_caption_prompt(info, plan, uploaded)
        entry = self.client.wait(self.client.queue_prompt(prompt))
        result = app.caption_result_from_history(entry, meta)
        self.assertIn("96 BPM", result["caption"])
        self.assertEqual(result["instrumental"], None)

    def test_merge_into_settings(self):
        settings = app.Settings()
        settings.tags = "same instruments"
        settings.bpm = 120
        settings.keyscale = "C major"
        result = app.interpret_caption_payload(json.loads(fake_caption_json()))
        changes = app.merge_caption_into_settings(settings, result)
        self.assertEqual(settings.bpm, 96)
        self.assertEqual(settings.keyscale, "A minor")
        self.assertEqual(len(changes), 2)

    def test_effective_tags_order_and_dedupe(self):
        settings = app.Settings()
        settings.tags = "instrumental, same tempo"
        settings.prompt_append = True
        merged = app.effective_tags(settings, "same tempo, vinyl crackle", "96 bpm, A minor")
        self.assertEqual(merged, "instrumental, same tempo, vinyl crackle, 96 bpm, A minor")
        settings.prompt_append = False
        self.assertEqual(app.effective_tags(settings, "vinyl crackle", ""), "vinyl crackle")

    def test_check_setup_reports_captioner(self):
        lines, ok = app.check_setup(self.client, None, None)
        info = self.client.object_info()
        lines += app.describe_captioners(info, app.captioner_plan(info))
        text = "\n".join(lines)
        self.assertIn("comfyui_music_caption", text)
        self.assertIn("MusicCaptionMerge", text)

    def test_describe_without_pack(self):
        info = info_without(*app.CAPTIONER_PREFERENCE, app.CAPTIONER_TAP, "SongScribeAnalyzer")
        lines = app.describe_captioners(info, app.captioner_plan(info))
        self.assertIn("NOT installed", "\n".join(lines))


if __name__ == "__main__":
    unittest.main(verbosity=2)
