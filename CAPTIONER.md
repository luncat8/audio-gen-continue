# A captioner for continuation prompts: key, BPM and vocal type

Continuing a track only works if the model is told what it is continuing. ACE-Step 1.5
takes `tags` (the arrangement), `lyrics`, **`bpm`**, **`keyscale`** and `duration`, and the
README has said since day one that the real BPM and key of the source is the biggest
continuity win — typed in by hand. This document is the investigation into automating that,
and the description of what now ships in this repo (`comfyui_music_caption/` + the app).

## 1. What the captioner has to answer

| # | Question | Why the generation needs it |
| --- | --- | --- |
| 1 | musical description (genre, instrumentation, mix) | `tags` — describes the *existing* arrangement |
| 2 | **BPM** | `bpm` in `TextEncodeAceStepAudio1.5`; wrong tempo = the tail drifts |
| 3 | **key + scale** | `keyscale`; wrong key = audible modulation at the seam |
| 4 | **vocal presence + vocal type** | decides `lyrics` (empty for instrumental) and whether the tail should keep a lead voice |
| 5 | structure / energy at the end | whether the tail should build, drop or fade |
| 6 | lyrics (optional) | only if you want the model to keep singing words |

Point 4 is where most taggers stop: they say "pop, upbeat, guitar" and never say whether a
female lead is singing. Points 2–4 are *measurable*, so a good captioner should treat them
as measurements, not as a language-model guess.

## 2. What exists (checked October 2026)

| Thing | Key / BPM | Vocal presence / **type** | ComfyUI node? | Verdict |
| --- | --- | --- | --- | --- |
| **MOSS-Music 8B** ([OpenMOSS](https://github.com/OpenMOSS/MOSS-Music), Apache-2.0 weights) | yes — *"Key / tempo / chord reasoning: identifies musical key, beats, downbeats, and chord progressions"* | yes — *"Instrument & voice recognition: identifies prominent instruments and singing voices (solo / chorus, gender, register)"* | **no dedicated node**; only buried inside [`filliptm/ComfyUI-FL-MiniMaxMusic3`](https://github.com/filliptm/ComfyUI-FL-MiniMaxMusic3) as a folder-oriented dataset preprocessor (pins the model, expects a MiniMax dataset layout, ~16.9 GiB) | **the model to use** — covers 1–6 in one model, SGLang serves it behind an OpenAI-compatible API |
| [`TheLocalLab/ComfyUI-SongScribe`](https://github.com/TheLocalLab/ComfyUI-SongScribe) — *Song Analyzer* | yes — librosa beat tracking + chroma | presence yes, **timbre explicitly disabled** ("vocal-timbre detection … returned nearly the same answer for every song") | yes, CPU | good free win, but no vocal *type*; also emits a MiniMax/YuE2-shaped caption, not ACE-Step tags |
| [`jeankassio/ComfyUI-AceStep_SFT`](https://github.com/jeankassio/ComfyUI-AceStep_SFT) — *AceStep 1.5 SFT Get Music Infos* | yes — librosa BPM + key/scale | no (tags only) | yes, but tied to that pack's model loader | fine inside that pack, not reusable as a generic captioner |
| [`DrunkTramp/comfyui-moss-audio`](https://github.com/DrunkTramp/comfyui-moss-audio) | no | no | yes (MOSS-**Audio**, not Music) | useful pattern (separate venv via subprocess) but caption text only |
| [`lum3on/ComfyUI_AudioTools`](https://github.com/lum3on/ComfyUI_AudioTools), [`christian-byrne/audio-separation-nodes-comfyui`](https://github.com/christian-byrne/audio-separation-nodes-comfyui), `FL_Audio_BPM_Analyzer`, `Audio Info` | single-purpose: BPM, sometimes key | no | yes | building blocks, nothing that produces a prompt |
| ACE-Step's own *Understand* / *Create Sample* (`ComfyUI-AceMusic`) | placeholders | placeholders | yes | the README of that pack says caption/BPM/key are placeholders — not an analysis node |
| MiniMax Music 3 caption format (3 sections) | n/a | n/a | ComfyUI core | a *format* to imitate, not a captioner |

So: **the user's instinct was right — MOSS-Music is the best fit** — and *no ComfyUI pack
implements a single-track MOSS-Music captioner that returns key/BPM/vocal type as wireable
values*. The two closest options each miss one of the four requirements.

### Why MOSS-Music specifically

From its model card and README:

* 8B total (MOSS-Audio encoder at 12.5 Hz + Qwen3-8B), **Apache-2.0 weights**;
* captioning is what it was SFT-ed for (GPT-5.4-as-judge 4.58 on SDD, 4.53 on MusicCaps in
  their own evaluation), including `vocals` and `structure` dimensions;
* key / tempo / beat / **chord** reasoning and timestamped chord transcription;
* singing-voice ASR with timestamps (best average lyrics error in their comparison);
* voice recognition: solo/chorus, **gender**, register;
* served by SGLang: `sglang serve --model-path MOSS-Music-8B-Instruct --trust-remote-code`
  on port 30000, both `/generate` and `/v1/chat/completions`, audio passed as a server-side
  path or a data URL.

That is exactly the four questions in §1, from one model, over one HTTP port.

## 3. What this repo now does

Two independent halves, because a captioner that is down must not block generation:

```
                    ┌───────────────────────────┐
   your track ─────▶│ MusicCaptionAnalyze (DSP) │  bpm, key, vocal presence+register, sections
                    └───────────┬───────────────┘        (numpy only, always available)
                                │ analysis_json
                    ┌───────────▼───────────────┐
   your track ─────▶│ MusicCaptionMOSS (8B)     │  caption, vocal details, key, bpm, structure,
                    └───────────┬───────────────┘  lyrics  (optional: needs a MOSS server)
                                │ moss_json
                    ┌───────────▼───────────────┐
                    │ MusicCaptionMerge         │  tags + lyrics + bpm + keyscale
                    └───────────────────────────┘
```

`audio_continue_tk.py` builds exactly this graph on demand, posts it to ComfyUI, reads the
JSON back out of the `/history` entry, and writes the numbers into the BPM / key fields and
the description box. Any third-party captioner (SongScribe, ACE-Step *Get Music Infos*, …)
can be used instead: the app discovers nodes that take `AUDIO`, return `STRING`s and say
"caption"/"analyze"/"scribe"/"music info" somewhere in their name, wires their STRING
outputs through `MusicCaptionTap`, and maps the values back by output name.

### Measured, not guessed

`MusicCaptionAnalyze` is plain numpy DSP (no librosa, no model download):

* **BPM** — spectral-flux onset envelope → autocorrelation → harmonic reinforcement →
  octave correction toward 60–180. Ambiguity is reported: `bpm_candidates` lists the
  half/double readings instead of hiding them.
* **Key** — chroma (55 Hz–4.2 kHz) → Krumhansl-Schmuckler correlation over 24 keys →
  `key_confidence` = margin over the runner-up. Camelot code included for DJ taggers.
* **Vocal presence / type** — three cues: energy in 250–4000 Hz, *syllabic modulation*
  (1.5–8 Hz) of that band, and its tonality. Register from the dominant f0 (HPS) of the
  vocal-ish frames: male / female / mixed-choir / unknown.
* loudness, crest factor, brightness, coarse section map, duration.

Honest limits of the DSP half:

* drum-only or rubato material → tempo is a guess; check `bpm_candidates`.
* atonal / modal / drum-only → key is a guess; `key_confidence` says so.
* the vocal heuristic cannot tell a plucked synth arpeggio from a syllabic vocal line; it
  says `uncertain` there. **This is precisely the case MOSS-Music exists for** — the LLM
  knows the difference, the DSP does not.
* the register estimate is coarse (a median f0), not "male pop tenor with airy falsetto".

The MOSS half is the opposite trade: slow, needs ~17 GB of weights and a second GPU
process, but it *understands* the track, so use it when the DSP numbers look ambiguous.

### Prompt files

Next to `song.flac` the app now looks for (in this order) `song.prompt.txt`,
`song.caption.txt`, `song.description.txt`, `song.tags.txt`, `song.txt`, `song.json` /
`song.prompt.json` (keys `prompt`, `caption`, `tags`, `description`, …). A checkbox decides
whether the file is **appended** to the description in the GUI or **replaces** it; both are
de-duplicated tag-by-tag, so `instrumental, same tempo` + `same tempo, vinyl crackle` becomes
`instrumental, same tempo, vinyl crackle`. Captioner tags are appended last.

## 4. Install

```bash
cp -r comfyui_audio_continue   /path/to/ComfyUI/custom_nodes/
cp -r comfyui_music_caption    /path/to/ComfyUI/custom_nodes/
# restart ComfyUI; nodes appear under "audio/caption"
```

Nothing to pip install: numpy ships with ComfyUI, the MOSS client is `urllib`. Without a
MOSS server the pack still works (DSP node only) — `MusicCaptionMOSS` logs the reason and
returns empty strings instead of failing the graph.

Optional, for better captions:

```bash
hf download OpenMOSS-Team/MOSS-Music-8B-Instruct --local-dir ./weights/MOSS-Music-8B-Instruct
sglang serve --model-path ./weights/MOSS-Music-8B-Instruct --trust-remote-code   # :30000
```

Notes:

* `transport = path` (default) hands the server an absolute path to the temp wav it wrote —
  fastest, and correct when ComfyUI and SGLang share a filesystem. Otherwise use `base64`.
* `segment = tail` (default) sends the last `max_seconds` of the track, i.e. the part being
  continued; `head`/`full` analyse the beginning/whole song.
* point the app at the server with the **MOSS-Music server** field (default
  `http://127.0.0.1:30000`); it is passed to the node as `server_url`.

## 5. Node reference

| Node | Inputs that matter | Outputs |
| --- | --- | --- |
| `MusicCaptionAnalyze` | `audio`, `max_seconds` (default 90, tail), `sections`, `extra_tags` | `tags`, `bpm`, `keyscale`, `key_confidence`, `bpm_confidence`, `vocal_presence`, `vocal_type`, `duration_seconds`, `analysis_json` |
| `MusicCaptionMOSS` | `audio`, `server_url`, `mode` (`auto`/`attributes`/`caption`/`vocal`/`lyrics`/`key_bpm`), `transport`, `segment`, `max_seconds`, `max_tokens`, `temperature` | `caption`, `lyrics`, `vocal`, `bpm`, `keyscale`, `instrumental`, `structure`, `moss_json` |
| `MusicCaptionMerge` | `analysis_json`, `moss_json`, `caption`, `vocal`, `lyrics` (all optional links), `prefer_numeric` (`dsp`/`model`), `extra_tags`, `keep_lyrics` | `tags`, `lyrics`, `bpm`, `keyscale`, `instrumental`, `vocal_type`, `summary_json` |
| `MusicCaptionTap` | 12 optional `STRING` inputs, `label` | the values as a JSON array in its `ui.text` (that is what the app reads) |

Every node also writes its result into the `ui.text` block of the history entry, which is
how `audio_continue_tk.py` picks the answer up without a preview node. `MusicCaptionTap`
exists for the opposite direction: it makes *other packs'* captioners readable the same way.

## 6. In the app

```
Prompt source
  Prompt file         [ song.prompt.txt        ] [Browse...] [Find next to audio]
  [x] append the prompt file to the description (unticked: replace it)   [x] look for one automatically
  Captioner           [ auto — MusicCaptionMerge ▾ ] [Analyse source] [Refresh]
  [ ] run the captioner before Generate      [ ] fill empty lyrics from it
  MOSS-Music server   [ http://127.0.0.1:30000 ]
```

* **Check setup** lists the captioners it found and what it will use.
* **Analyse source** runs the captioner now, prints the results in the log and fills the BPM
  and key fields, so you can eyeball them before spending GPU time.
* **run the captioner before Generate** does the same as part of the job. If the captioner is
  missing or fails, generation continues with your own description (the log says why) —
  captioning never blocks a render.
* Lyrics are **not** filled unless you tick the box: ACE-Step will happily sing the *source*
  lyrics in the tail, which is rarely what you want.

CLI:

```bash
python3 audio_continue_tk.py --check --url http://127.0.0.1:8188
python3 audio_continue_tk.py --caption --url http://127.0.0.1:8188 --audio song.flac
python3 audio_continue_tk.py --dry-run --audio song.flac --prompt-file song.prompt.txt --out patched.json
python3 audio_continue_tk.py --dry-run --audio song.flac --replace-prompt --captioner off
```

## 7. Known gaps / next steps

1. **MOSS-Music in-process.** Right now the model must be served by SGLang (recommended by
   OpenMOSS anyway). An in-process `transformers` path would need the `MossMusicProcessor`
   API and a second venv — `DrunkTramp/comfyui-moss-audio` shows the subprocess pattern if
   someone wants it.
2. **Vocal separation as a cue.** Demucs stems (already available via
   `audio-separation-nodes-comfyui` / `FL_Audio_Separation`) would make vocal presence and
   register far more reliable than a single-band heuristic, at the cost of a model download.
3. **Chord-aware continuation.** MOSS returns chords; ACE-Step cannot take them yet. The
   obvious use is to extend the *last* chord's pads instead of letting the model invent a
   new progression — worth trying via `tags` ("sustained Am–F–C–G progression").
4. **Cache per file.** Captioning an 8B model per run is wasteful; hash the file, cache the
   JSON next to it (`.caption.json`) and reuse it. SongScribe already does this (`use_cache`).
5. **Vocal type is still coarse.** Register + gender from a median f0 is honest but blunt;
   MOSS's `vocal_description` is the field to trust when both are available — which is why
   `MusicCaptionMerge` keeps the model's vocal text and only borrows the DSP numbers.

## 8. Sources

* MOSS-Music model card / README / SGLang usage guide — <https://github.com/OpenMOSS/MOSS-Music>,
  weights <https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct>
* ComfyUI-SongScribe (Song Analyzer, DSP + clap/maest) — <https://github.com/TheLocalLab/ComfyUI-SongScribe>
* ComfyUI-AceStep_SFT (*Get Music Infos*) — <https://github.com/jeankassio/ComfyUI-AceStep_SFT>
* comfyui-moss-audio (subprocess + separate venv pattern) — <https://github.com/DrunkTramp/comfyui-moss-audio>
* ComfyUI-FL-MiniMaxMusic3 (MOSS-Music dataset preprocessor) — <https://github.com/filliptm/ComfyUI-FL-MiniMaxMusic3>
* ComfyUI-AceMusic (what "Understand" actually does) — <https://github.com/hiroki-abe-58/ComfyUI-AceMusic>
* Krumhansl-Kessler key profiles; spectral-flux onset detection; harmonic product spectrum —
  standard MIR, implemented from scratch in `comfyui_music_caption/dsp.py`.
