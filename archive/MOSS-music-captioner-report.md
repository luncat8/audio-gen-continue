I fixed the `prompt_no_outputs` error in `comfyui_music_caption` and `audio_continue_tk.py`, added native ComfyUI loader and viewer custom nodes for MOSS-Music 8B, and generated ComfyUI workflows with on-canvas download notes and full data viewers.

### 1. Root Cause & Fix for `prompt_no_outputs` + Tk `"native"` UI
- **`prompt_no_outputs` in ComfyUI:** Previously, only `MusicCaptionTap` declared `OUTPUT_NODE = True`, while `MusicCaptionAnalyze`, `MusicCaptionMOSS`, and `MusicCaptionMerge` omitted it. When `audio_continue_tk.py` submitted its standalone caption graph (`LoadAudio → MusicCaptionAnalyze + MusicCaptionMOSS → MusicCaptionMerge`), ComfyUI rejected `/prompt` with HTTP 400 (`{'type': 'prompt_no_outputs', 'message': 'Prompt has no outputs'}`).
  - **Fix:** Set `OUTPUT_NODE = True` on `MusicCaptionAnalyze`, `MusicCaptionMOSS`, `MusicCaptionMerge`, `MusicCaptionView`, and `MusicCaptionTap` in `comfyui_music_caption/nodes.py` (and enforced `prompt_no_outputs` validation in `tests/fake_comfyui.py` so tests guard against regressions).
- **Tk GUI (`audio_continue_tk.py`):**
  - Wired `MusicCaptionMOSSLoader` automatically into `build_caption_prompt` so **Analyse source** uses local `ComfyUI/models/moss_music/` weights when installed and falls back cleanly to DSP BPM/key/vocals (`when_missing = "skip (DSP only)"`) before you download the 8B model.
  - Replaced the confusing `MOSS-Music server` + `available (native)` text with `MOSS URL (opt.)` and explicit DSP / local MOSS-Music weight status.
  - Clicking **Analyse source** now populates **BPM**, **Key**, **Music description** (de-duplicated tags), and **Lyrics** (when *fill empty lyrics from it* is ticked).

---

### 2. New Custom Nodes in `comfyui_music_caption`
Copy (or re-sync) `comfyui_music_caption/` into `ComfyUI/custom_nodes/` and restart ComfyUI:

- **`MOSS-Music Loader` (`MusicCaptionMOSSLoader`):**
  - Discovers local HuggingFace MOSS-Music weights in `ComfyUI/models/moss_music/` (also checks `models/moss-music/`, `models/LLM/`, and `$MOSS_MUSIC_DIR`), validates safetensors shards, and runs MOSS-Music 8B in-process via `transformers` (`comfyui_music_caption/moss_local.py`) with shims for `transformers` 4.x/5.x and installs without `accelerate`/`torchaudio`.
  - Defaults to `keep_loaded = False` (frees the ~17 GB model from VRAM right after captioning so ACE-Step 1.5 XL can run in the same workflow) and `when_missing = "skip (DSP only)"` (never breaks the graph if you haven't downloaded the 8B weights yet).
- **`Music Caption Viewer` (`MusicCaptionView`) + `comfyui_music_caption/web/music_caption.js`:**
  - Displays all captioner data directly inside the node on the ComfyUI canvas (BPM + confidence/octave candidates, Key/Scale + Camelot + runners-up, Vocal presence/register/description, Loudness/Brightness, Chords, Structure, Prompt Tags, Full Caption, Lyrics, and per-section DSP breakdown) and exposes typed outputs (`report`, `tags`, `caption`, `lyrics`, clamped `bpm`, `keyscale`, `vocal_type`, `chords`, `structure`).

---

### 3. MOSS-Music 8B HuggingFace Download & Folder Placement
For rich captions, instrument/genre tags, chord progressions, song structure, and lyrics (optional — DSP BPM/key/vocals work immediately with zero downloads):

- **Recommended model (Apache-2.0, ~17 GB):** <https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct>
- **Chain-of-thought variant:** <https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking>

Download into `ComfyUI/models/moss_music/MOSS-Music-8B-Instruct`:
```bash
hf download OpenMOSS-Team/MOSS-Music-8B-Instruct \
  --local-dir /path/to/ComfyUI/models/moss_music/MOSS-Music-8B-Instruct
```
*(Or tick `auto_download = True` on the **MOSS-Music Loader** node if `huggingface_hub` is installed in ComfyUI's Python environment.)*

---

### 4. New ComfyUI Workflows
Both UI workflows can be dragged directly onto ComfyUI and include an on-canvas **`MarkdownNote`** with the HuggingFace links, folder layout, and usage notes:
1. **`workflows/music_caption_ui.json`** *(opened in viewer; API pair: `workflows/music_caption_api.json`)* — Standalone music analyzer & captioner (`LoadAudio` + `MOSS-Music Loader` → `MusicCaptionAnalyze` + `MusicCaptionMOSS` → `MusicCaptionMerge` → `MusicCaptionView` + three `PreviewAny` panels showing the full report, raw DSP JSON, and raw MOSS JSON).
2. **`workflows/ace15xl_audio_continue_caption_ui.json`** *(API pair: `workflows/ace15xl_audio_continue_caption_api.json`)* — Full ACE-Step 1.5 XL audio continuation graph with the captioner + loader + on-canvas `MusicCaptionView` wired directly into `TextEncodeAceStepAudio1.5`.