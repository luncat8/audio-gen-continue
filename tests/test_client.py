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

API_WORKFLOW = os.path.join(ROOT, "workflows", "ace15xl_audio_continue_api.json")
UI_WORKFLOW = os.path.join(ROOT, "workflows", "ace15xl_audio_continue_ui.json")
ARCHIVE_API = os.path.join(ROOT, "workflows", "archive", "ace15_audio_continue_turbo_aio_api.json")
ARCHIVE_UI = os.path.join(ROOT, "workflows", "archive", "ace15_audio_continue_turbo_aio_ui.json")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def by_class(workflow, class_type):
    return {nid: n for nid, n in workflow.items() if n.get("class_type") == class_type}


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
        wf = load(API_WORKFLOW)
        for nid, node in wf.items():
            self.assertIn("class_type", node, nid)
            self.assertIn("inputs", node, nid)
            for key, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                    self.assertIn(value[0], wf, f"{nid}.{key} points at a missing node")

    def test_archived_aio_workflow_is_valid_too(self):
        wf = load(ARCHIVE_API)
        for nid, node in wf.items():
            self.assertIn("class_type", node, nid)
            self.assertIn("inputs", node, nid)
            for key, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                    self.assertIn(value[0], wf, f"{nid}.{key} points at a missing node")
        self.assertTrue(by_class(wf, "CheckpointLoaderSimple"))
        self.assertTrue(os.path.isfile(ARCHIVE_UI))

    def test_default_workflow_is_the_xl_split_graph(self):
        wf = load(API_WORKFLOW)
        self.assertEqual(app.DEFAULT_WORKFLOW, API_WORKFLOW)
        unet = list(by_class(wf, "UNETLoader").values())[0]["inputs"]
        self.assertEqual(unet["unet_name"], "acestep_v1.5_xl_sft_bf16.safetensors")
        clip = list(by_class(wf, "DualCLIPLoader").values())[0]["inputs"]
        self.assertEqual(clip["clip_name1"], "qwen_0.6b_ace15.safetensors")
        self.assertEqual(clip["clip_name2"], "qwen_4b_ace15.safetensors")
        self.assertEqual(clip["type"], "ace")
        self.assertEqual(list(by_class(wf, "VAELoader").values())[0]["inputs"]["vae_name"],
                         "ace_1.5_vae.safetensors")
        self.assertEqual(by_class(wf, "CheckpointLoaderSimple"), {})
        sampler = list(by_class(wf, "KSampler").values())[0]["inputs"]
        self.assertEqual((sampler["steps"], sampler["cfg"]), (50, 7.0))
        self.assertEqual(sampler["sampler_name"], "euler")
        self.assertEqual(sampler["scheduler"], "simple")
        self.assertEqual(sampler["denoise"], 1.0)

    def test_extend_node_gets_vae_and_clip_from_the_split_loaders(self):
        wf = load(API_WORKFLOW)
        vae_id = next(iter(by_class(wf, "VAELoader")))
        clip_id = next(iter(by_class(wf, "DualCLIPLoader")))
        extend = list(by_class(wf, "AudioContinueExtend").values())[0]["inputs"]
        encode = list(by_class(wf, "TextEncodeAceStepAudio1.5").values())[0]["inputs"]
        self.assertEqual(extend["vae"], [vae_id, 0])
        self.assertEqual(encode["clip"], [clip_id, 0])

    def test_ui_workflow_links_are_consistent(self):
        ui = load(UI_WORKFLOW)
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
        for path in (API_WORKFLOW, ARCHIVE_API):
            wf = load(path)
            extend = next(iter(by_class(wf, "AudioContinueExtend")))
            encode = next(iter(by_class(wf, "TextEncodeAceStepAudio1.5")))
            self.assertEqual(wf[encode]["inputs"]["duration"], [extend, 3], path)


class TestModelPreset(unittest.TestCase):
    def test_xl_sft_workflow(self):
        preset = app.model_preset(load(API_WORKFLOW))
        self.assertEqual(preset["name"], "xl_sft")
        self.assertEqual((preset["steps"], preset["cfg"]), (50, 7.0))

    def test_archived_aio_workflow(self):
        preset = app.model_preset(load(ARCHIVE_API))
        self.assertEqual(preset["name"], "turbo_aio")
        self.assertEqual((preset["steps"], preset["cfg"]), (8, 1.0))

    def test_xl_turbo_is_recognised(self):
        wf = load(API_WORKFLOW)
        next(iter(by_class(wf, "UNETLoader").values()))["inputs"]["unet_name"] = \
            "acestep_v1.5_xl_turbo_bf16.safetensors"
        self.assertEqual(app.model_preset(wf)["name"], "xl_turbo")

    def test_unknown_workflow(self):
        self.assertIsNone(app.model_preset({"1": {"class_type": "LoadAudio", "inputs": {}}}))

    def test_assets_are_collected(self):
        assets = app.workflow_assets(load(API_WORKFLOW))
        self.assertEqual(assets["UNETLoader"], ["acestep_v1.5_xl_sft_bf16.safetensors"])
        self.assertEqual(assets["DualCLIPLoader"],
                         ["qwen_0.6b_ace15.safetensors", "qwen_4b_ace15.safetensors"])
        self.assertEqual(assets["VAELoader"], ["ace_1.5_vae.safetensors"])
        self.assertNotIn("CheckpointLoaderSimple", assets)


class TestPatchWorkflow(unittest.TestCase):
    def setUp(self):
        self.workflow = load(API_WORKFLOW)
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

    def test_split_loaders_are_left_alone(self):
        patched = app.patch_workflow(self.workflow, self.settings, uploaded_name="song.wav")
        self.assertEqual(patched["1"]["inputs"]["unet_name"],
                         "acestep_v1.5_xl_sft_bf16.safetensors")
        self.assertEqual(patched["12"]["inputs"]["clip_name2"], "qwen_4b_ace15.safetensors")
        self.assertEqual(patched["13"]["inputs"]["vae_name"], "ace_1.5_vae.safetensors")

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

    def test_check_setup_with_xl_assets(self):
        wf = load(API_WORKFLOW)
        lines, ok = app.check_setup(
            self.client, app.workflow_node_classes(wf), app.workflow_assets(wf)
        )
        self.assertTrue(ok, lines)
        self.assertTrue(any("acestep_v1.5_xl_sft_bf16.safetensors" in line for line in lines))

    def test_check_setup_reports_missing_diffusion_model(self):
        wf = load(API_WORKFLOW)
        assets = app.workflow_assets(wf)
        assets["UNETLoader"] = ["acestep_v1.5_xl_nope_bf16.safetensors"]
        lines, ok = app.check_setup(self.client, app.workflow_node_classes(wf), assets)
        self.assertFalse(ok)
        self.assertTrue(any("models/diffusion_models" in line for line in lines), lines)

    def test_full_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = os.path.join(tmp, "song with spaces & stuff.wav")
            with open(audio, "wb") as fh:
                fh.write(b"RIFF0000WAVE")

            uploaded = self.client.upload_audio(audio)
            self.assertIn("song", uploaded)
            self.assertTrue(self.state.uploads, "server saw no upload")

            workflow = load(API_WORKFLOW)
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
