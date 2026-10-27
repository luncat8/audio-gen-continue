#!/usr/bin/env python3
"""Headless tests for audio_continue_tk.py against a fake ComfyUI server.

    python3 tests/test_client.py
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
from fake_comfyui import FakeState, start_server  # noqa: E402

API_WORKFLOW = os.path.join(ROOT, "workflows", "ace15_audio_continue_api.json")
UI_WORKFLOW = os.path.join(ROOT, "workflows", "ace15_audio_continue_ui.json")


class FakeServerCase(unittest.TestCase):
    state_kwargs = {}

    def setUp(self):
        self.state = FakeState(**self.state_kwargs)
        self.server, self.url = start_server(self.state)
        self.client = app.ComfyClient(self.url, timeout=5)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()


class TestWorkflowFiles(unittest.TestCase):
    def test_api_workflow_is_valid_prompt_graph(self):
        with open(API_WORKFLOW, encoding="utf-8") as fh:
            wf = json.load(fh)
        for nid, node in wf.items():
            self.assertIn("class_type", node, nid)
            self.assertIn("inputs", node, nid)
            for key, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                    self.assertIn(value[0], wf, f"{nid}.{key} points at a missing node")

    def test_ui_workflow_links_are_consistent(self):
        with open(UI_WORKFLOW, encoding="utf-8") as fh:
            ui = json.load(fh)
        nodes = {n["id"]: n for n in ui["nodes"]}
        self.assertEqual(ui["last_node_id"], max(nodes))
        targets = set()
        for link_id, origin, slot, target, target_slot, link_type in ui["links"]:
            self.assertIn(origin, nodes, f"link {link_id} has a bad origin")
            self.assertIn(target, nodes, f"link {link_id} has a bad target")
            self.assertLess(slot, len(nodes[origin]["outputs"]) or 1)
            self.assertLess(target_slot, len(nodes[target]["inputs"]))
            self.assertEqual(nodes[origin]["outputs"][slot]["name"], link_type)
            self.assertEqual(nodes[target]["inputs"][target_slot]["link"], link_id)
            self.assertEqual(nodes[target]["inputs"][target_slot]["type"], link_type)
            self.assertIn(link_id, nodes[origin]["outputs"][slot]["links"] or [])
            targets.add((target, target_slot))
        self.assertIn(ui["last_link_id"], [link[0] for link in ui["links"]])

    def test_extend_output_feeds_text_encode_duration(self):
        with open(API_WORKFLOW, encoding="utf-8") as fh:
            wf = json.load(fh)
        by_class = {}
        for nid, node in wf.items():
            by_class.setdefault(node["class_type"], []).append(nid)
        extend = by_class["AudioContinueExtend"][0]
        encode = by_class["TextEncodeAceStepAudio1.5"][0]
        self.assertEqual(wf[encode]["inputs"]["duration"], [extend, 3])


class TestPatchWorkflow(unittest.TestCase):
    def setUp(self):
        with open(API_WORKFLOW, encoding="utf-8") as fh:
            self.workflow = json.load(fh)
        self.settings = app.Settings(add_seconds=42.5, seed=7, steps=12, cfg=2.0, bpm=96,
                                     keyscale="A minor", tags="test tags", lyrics="la la",
                                     save_prefix="audio-continue/mine", crossfade=2.5)

    def test_values_are_applied(self):
        patched = app.patch_workflow(self.workflow, self.settings, uploaded_name="song.wav")
        values = json.dumps(patched)
        self.assertIn('"add_seconds": 42.5', values)
        self.assertIn('"song.wav"', values)
        self.assertIn('"test tags"', values)
        self.assertIn('"la la"', values)
        self.assertIn('"audio-continue/mine"', values)
        self.assertIn('"crossfade_seconds": 2.5', values)
        self.assertIn('"steps": 12', values)
        self.assertIn('"bpm": 96', values)
        # links must survive patching
        self.assertEqual(patched["7"]["inputs"]["latent_image"], ["4", 0])
        self.assertEqual(patched["5"]["inputs"]["duration"], ["4", 3])

    def test_original_workflow_is_not_mutated(self):
        before = json.dumps(self.workflow)
        app.patch_workflow(self.workflow, self.settings, uploaded_name="x.wav")
        self.assertEqual(before, json.dumps(self.workflow))

    def test_missing_nodes_raise(self):
        partial = {k: v for k, v in self.workflow.items() if v["class_type"] != "AudioContinueExtend"}
        with self.assertRaises(app.ComfyError):
            app.patch_workflow(partial, self.settings, uploaded_name="x.wav")

    def test_accepts_alternative_extend_node(self):
        wf = json.loads(json.dumps(self.workflow))
        wf["4"] = {"class_type": "ACEStep15XLExtendAudio",
                   "inputs": {"right_extend_seconds": 0.0, "left_extend_seconds": 0.0}}
        patched = app.patch_workflow(wf, self.settings, uploaded_name="x.wav")
        self.assertEqual(patched["4"]["inputs"]["right_extend_seconds"], 42.5)


class TestClientHappyPath(FakeServerCase):
    state_kwargs = {"prompt_delay": 1}

    def test_check_setup_ok(self):
        lines, ok = app.check_setup(self.client)
        self.assertTrue(ok, lines)
        self.assertTrue(any("ComfyUI OK" in line for line in lines))

    def test_full_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = os.path.join(tmp, "song with spaces & stuff.wav")
            with open(audio, "wb") as fh:
                fh.write(b"RIFF0000WAVE")

            uploaded = self.client.upload_audio(audio)
            self.assertIn("song", uploaded)
            self.assertTrue(self.state.uploads, "server saw no upload")

            with open(API_WORKFLOW, encoding="utf-8") as fh:
                workflow = json.load(fh)
            settings = app.Settings(seed=5)
            prompt = app.patch_workflow(workflow, settings, uploaded_name=uploaded)
            self.assertEqual(prompt["3"]["inputs"]["audio"], uploaded)

            prompt_id = self.client.queue_prompt(prompt)
            entry = self.client.wait(prompt_id, poll=0.1, max_wait=10)
            outputs = app.find_audio_outputs(entry)
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0]["subfolder"], "audio-continue")

            destination = os.path.join(tmp, "out.flac")
            self.client.download_output(outputs[0], destination)
            with open(destination, "rb") as fh:
                self.assertEqual(fh.read(), b"fLaC\x00\x00\x00\x22fake-flac-payload")

    def test_output_name(self):
        name = app.make_output_name("/tmp/My Song.mp3", 30.4, "audio_00001_.flac")
        self.assertTrue(name.startswith("My Song_continue+30s_"))
        self.assertTrue(name.endswith(".flac"))


class TestClientErrors(FakeServerCase):
    state_kwargs = {"missing_nodes": ("AudioContinueExtend",)}

    def test_missing_node_reported(self):
        lines, ok = app.check_setup(self.client)
        self.assertFalse(ok)
        self.assertTrue(any("AudioContinueExtend" in line for line in lines))

    def test_unreachable_server(self):
        client = app.ComfyClient("http://127.0.0.1:9", timeout=1)
        with self.assertRaises(app.ComfyError):
            client.system_stats()


class TestPromptRejected(FakeServerCase):
    state_kwargs = {"fail_prompt": "node_errors"}

    def test_node_errors(self):
        with self.assertRaises(app.ComfyError):
            self.client.queue_prompt({"1": {"class_type": "LoadAudio", "inputs": {}}})


class TestRunFails(FakeServerCase):
    state_kwargs = {"fail_prompt": True}

    def test_error_status(self):
        prompt_id = self.client.queue_prompt({"1": {"class_type": "LoadAudio", "inputs": {}}})
        with self.assertRaises(app.ComfyError):
            self.client.wait(prompt_id, poll=0.05, max_wait=10)


class TestWorkflowPatchingCli(unittest.TestCase):
    def test_dry_run_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "patched.json")
            code = app.main_cli(["--dry-run", "--audio", "/tmp/x.wav", "--add", "12",
                                 "--workflow", API_WORKFLOW, "--out", out])
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as fh:
                patched = json.load(fh)
            self.assertEqual(patched["4"]["inputs"]["add_seconds"], 12.0)


class TestSettings(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.json")
            settings = app.Settings(audio_file="/music/a.flac", add_seconds=12.5,
                                    save_prefix="audio-continue/test", bpm=88)
            self.assertTrue(app.save_settings(settings, path))
            loaded = app.load_settings(path)
            self.assertEqual(loaded.audio_file, "/music/a.flac")
            self.assertEqual(loaded.add_seconds, 12.5)
            self.assertEqual(loaded.save_prefix, "audio-continue/test")
            self.assertEqual(loaded.bpm, 88)

    def test_missing_and_broken_config(self):
        self.assertEqual(app.load_settings("/nonexistent/nope.json").comfy_url, "http://127.0.0.1:8188")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            self.assertEqual(app.load_settings(path).add_seconds, 30.0)

    def test_unknown_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"audio_file": "/a.wav", "legacy_option": True}, fh)
            loaded = app.load_settings(path)
            self.assertEqual(loaded.audio_file, "/a.wav")


if __name__ == "__main__":
    unittest.main(verbosity=2)
