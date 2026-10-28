#!/usr/bin/env python3
"""Build the whole GUI against a fake tkinter and poke at the new controls.

    python3 tests/test_gui_smoke.py

Real tkinter is not needed (and the widgets are no-ops), so this runs on a
headless box. It is a construction/typo test, not a visual one.
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

import fake_comfyui  # noqa: E402
import fake_tkinter  # noqa: E402

fake = fake_tkinter.install()

import audio_continue_tk as app  # noqa: E402


class GuiCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.config = os.path.join(self.dir, "config.json")
        self._old_config = app.CONFIG_PATH
        app.CONFIG_PATH = self.config
        self.root = fake.Tk()
        self.gui = app.App(self.root)

    def tearDown(self):
        app.CONFIG_PATH = self._old_config
        for name in os.listdir(self.dir):
            try:
                os.remove(os.path.join(self.dir, name))
            except OSError:
                pass
        try:
            os.rmdir(self.dir)
        except OSError:
            pass

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path


class TestConstruction(GuiCase):
    def test_window_and_widgets_exist(self):
        self.assertIn("Audio Continue", self.root.title_value)
        self.assertGreaterEqual(self.root.minsize_value[0], 760)
        for name in (
            "url_var", "workflow_var", "audio_var", "promptfile_var", "append_prompt_var",
            "auto_prompt_var", "captioner_var", "captioner_box", "captioner_url_var",
            "auto_caption_var", "fill_lyrics_var", "caption_info", "tags_text",
            "lyrics_text", "log_text", "generate_btn", "progress", "status_var",
        ):
            self.assertTrue(hasattr(self.gui, name), name)

    def test_defaults_are_sane(self):
        self.assertEqual(self.gui.promptfile_var.get(), "")
        self.assertTrue(self.gui.append_prompt_var.get())
        self.assertEqual(self.gui.captioner_var.get(), "auto")
        self.assertEqual(self.gui.captioner_url_var.get(), app.MOSS_DEFAULT_URL)
        self.assertFalse(self.gui.auto_caption_var.get())
        self.assertIn("instrumental", self.gui.tags_text.get("1.0", "end"))


class TestSettingsRoundTrip(GuiCase):
    def test_ui_values_reach_the_settings(self):
        self.gui.audio_var.set(os.path.join(self.dir, "song.wav"))
        self.gui.promptfile_var.set("/tmp/prompt.txt")
        self.gui.append_prompt_var.set(False)
        self.gui.captioner_var.set("MusicCaptionAnalyze")
        self.gui.captioner_url_var.set("http://moss:30000")
        self.gui.auto_caption_var.set(True)
        self.gui.random_var.set(False)
        self.gui.bpm_var.set("96")
        self.gui.key_var.set("A minor")
        settings = self.gui._collect_settings()
        self.assertEqual(settings.prompt_file, "/tmp/prompt.txt")
        self.assertFalse(settings.prompt_append)
        self.assertEqual(settings.captioner, "MusicCaptionAnalyze")
        self.assertEqual(settings.captioner_url, "http://moss:30000")
        self.assertTrue(settings.auto_caption)
        self.assertEqual(settings.bpm, 96)
        self.assertEqual(settings.keyscale, "A minor")

        # and back into a fresh window through the saved config
        app.save_settings(settings, self.config)
        second = app.App(fake.Tk())
        self.assertEqual(second.promptfile_var.get(), "/tmp/prompt.txt")
        self.assertEqual(second.captioner_var.get(), "MusicCaptionAnalyze")
        self.assertTrue(second.auto_caption_var.get())


class TestPromptFileControls(GuiCase):
    def test_sidecar_is_picked_up_with_the_audio(self):
        audio = self.write("song.wav", "RIFF")
        self.write("song.prompt.txt", "dark synthwave, vinyl crackle")
        self.gui.audio_var.set(audio)
        self.gui.auto_prompt_var.set(True)
        self.gui._update_audio_info()
        self.assertTrue(self.gui.promptfile_var.get().endswith("song.prompt.txt"))

    def test_auto_lookup_can_be_turned_off(self):
        audio = self.write("song.wav", "RIFF")
        self.write("song.prompt.txt", "dark synthwave")
        self.gui.audio_var.set(audio)
        self.gui.auto_prompt_var.set(False)
        self.gui._update_audio_info()
        self.assertEqual(self.gui.promptfile_var.get(), "")

    def test_find_button_reports_missing_sidecar(self):
        audio = self.write("song.wav", "RIFF")
        self.gui.audio_var.set(audio)
        self.gui.on_find_prompt_file()
        self.assertEqual(self.gui.promptfile_var.get(), "")
        self.gui._drain_queue()
        log = self.gui.log_text.get("1.0", "end")
        self.assertIn("no prompt sidecar", log)

    def test_browse_button_sets_the_field(self):
        path = self.write("x.caption.txt", "ambient drone")
        fake.filedialog.answer = path
        self.gui.on_pick_prompt_file()
        self.assertEqual(self.gui.promptfile_var.get(), path)


class TestCaptionDisplay(GuiCase):
    def test_caption_result_updates_bpm_key_and_info(self):
        result = app.interpret_caption_payload(
            json.loads(
                json.dumps(
                    {"bpm": 96, "keyscale": "A minor", "vocal_type": "female",
                     "instrumental": False, "caption": "dark synthwave"}
                )
            )
        )
        self.gui._show_caption(result, ["bpm 120 -> 96"])
        self.assertEqual(self.gui.bpm_var.get(), "96")
        self.assertEqual(self.gui.key_var.get(), "A minor")
        self.assertIn("96 bpm", self.gui.caption_info._options.get("text", ""))
        self.assertIn("female", self.gui.caption_info._options["text"])

    def test_plan_display(self):
        self.gui._show_caption_plan({"mode": "none", "reason": "nothing installed"})
        self.assertIn("nothing installed", self.gui.caption_info._options["text"])
        self.gui._show_caption_plan({"mode": "native", "label": "MusicCaptionMerge"})
        text = self.gui.caption_info._options["text"]
        self.assertIn("MusicCaptionMerge", text)
        self.assertNotIn("(native)", text)
        self.assertIn("MOSS-Music", text)

    def test_captioner_list_arrives_from_the_worker(self):
        self.gui._post("captioners", (["auto", "off", "MusicCaptionAnalyze"], {"mode": "none", "reason": "x"}))
        self.gui._drain_queue()
        values = self.gui.captioner_box._options.get("values")
        self.assertIn("MusicCaptionAnalyze", values)

    def test_generate_requires_a_file(self):
        try:
            self.gui.on_generate()
        finally:
            # no worker is started without a file, so this only needs to not explode
            self.assertIsNone(self.gui.last_output)


class TestGenerateWithCaptioner(GuiCase):
    """The whole generate job against the fake ComfyUI server, captioner included."""

    def setUp(self):
        super().setUp()
        self.state = fake_comfyui.FakeState()
        self.server, self.url = fake_comfyui.start_server(self.state)
        self.audio = self.write("song.wav", "RIFF....")
        self.sidecar = self.write("song.prompt.txt", "vinyl crackle, same mix")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def _settings(self, **overrides):
        settings = app.Settings()
        settings.comfy_url = self.url
        settings.workflow = app.DEFAULT_WORKFLOW
        settings.audio_file = self.audio
        settings.prompt_file = self.sidecar
        settings.prompt_append = True
        settings.auto_caption = True
        settings.output_dir = self.dir
        settings.open_when_done = False
        settings.add_seconds = 10
        settings.randomize_seed = False
        for key, value in overrides.items():
            setattr(settings, key, value)
        return settings

    def _prompt(self):
        return list(self.state.prompts.values())[-1]

    def test_caption_and_prompt_file_reach_the_text_encoder(self):
        settings = self._settings()
        self.gui._generate_job(settings)
        self.gui._drain_queue()
        prompt = self._prompt()
        encoders = [n for n in prompt.values() if n["class_type"] == "TextEncodeAceStepAudio1.5"]
        self.assertTrue(encoders)
        inputs = encoders[0]["inputs"]
        self.assertIn("vinyl crackle", inputs["tags"])
        self.assertIn("dark synthwave", inputs["tags"])
        self.assertEqual(inputs["bpm"], 96)
        self.assertEqual(inputs["keyscale"], "A minor")
        # bpm/key from the captioner went back into the settings too
        self.assertEqual(settings.bpm, 96)
        self.assertEqual(settings.keyscale, "A minor")
        # and a file landed on disk
        saved = [name for name in os.listdir(self.dir) if "continue" in name]
        self.assertTrue(saved)
        self.assertIn("saved:", self.gui.log_text.get("1.0", "end"))

    def test_analyse_source_button_populates_bpm_key_tags_and_lyrics(self):
        self.state.caption_text = json.dumps(
            {
                "analyzer": "merge",
                "bpm": 108,
                "keyscale": "D minor",
                "vocal_type": "male",
                "instrumental": False,
                "tags": "indie rock, crisp drums, 108 bpm, D minor, male lead vocals",
                "detected_lyrics": "[Verse]\nWalking down the boulevard",
                "measured": {"bpm": 108, "key": "D minor"},
                "model": {"ok": True, "caption": "indie rock with crisp drums"},
            }
        )
        self.gui.url_var.set(self.url)
        self.gui.audio_var.set(self.audio)
        self.gui.fill_lyrics_var.set(True)
        self.gui.tags_text.delete("1.0", "end")
        self.gui.tags_text.insert("1.0", "same room reverb")
        self.gui.lyrics_text.delete("1.0", "end")

        self.gui.on_caption_now()
        self.gui.worker.join(timeout=5)
        self.gui._drain_queue()

        self.assertEqual(self.gui.bpm_var.get(), "108")
        self.assertEqual(self.gui.key_var.get(), "D minor")
        self.assertIn("same room reverb", self.gui.tags_text.get("1.0", "end"))
        self.assertIn("indie rock", self.gui.tags_text.get("1.0", "end"))
        self.assertIn("Walking down the boulevard", self.gui.lyrics_text.get("1.0", "end"))

    def test_without_the_captioner_checkbox_nothing_is_captioned(self):
        settings = self._settings(auto_caption=False)
        self.gui._generate_job(settings)
        prompt = self._prompt()
        classes = {n["class_type"] for n in prompt.values()}
        self.assertNotIn("MusicCaptionMerge", classes)
        self.assertEqual(settings.bpm, 120)
        inputs = [n for n in prompt.values()
                  if n["class_type"] == "TextEncodeAceStepAudio1.5"][0]["inputs"]
        self.assertIn("vinyl crackle", inputs["tags"])

    def test_captioner_off_falls_back_to_the_prompt_file(self):
        settings = self._settings(captioner="off")
        self.gui._generate_job(settings)
        prompt = self._prompt()
        classes = {n["class_type"] for n in prompt.values()}
        self.assertNotIn("MusicCaptionMerge", classes)
        inputs = [n for n in prompt.values()
                  if n["class_type"] == "TextEncodeAceStepAudio1.5"][0]["inputs"]
        self.assertIn("vinyl crackle", inputs["tags"])

    def test_plain_text_answer_is_used_as_caption(self):
        self.state.caption_text = "not json at all"
        settings = self._settings()
        self.gui._generate_job(settings)
        self.gui._drain_queue()
        log = self.gui.log_text.get("1.0", "end")
        self.assertIn("not json at all", log)
        self.assertIn("saved:", log)

    def test_captioner_failure_does_not_stop_the_generation(self):
        self.state.caption_text = "[]"
        settings = self._settings()
        self.gui._generate_job(settings)
        self.gui._drain_queue()
        log = self.gui.log_text.get("1.0", "end")
        self.assertIn("captioner returned nothing usable", log)
        self.assertIn("saved:", log)


if __name__ == "__main__":
    unittest.main(verbosity=2)
