"""
Music information retrieval for the captioner, numpy only.

Everything here is measured from the waveform - no neural network, no librosa
dependency (librosa is used if it happens to be installed, see `analyse`).
The point is to give the prompt real numbers instead of guesses:

    bpm, key + confidence (Krumhansl-Schmuckler chroma correlation)
    vocal presence + vocal register/type (heuristic, see below)
    loudness / crest / brightness, coarse section map

What is trustworthy and what is not:
  * BPM is robust for anything with a beat. Octave errors (70 vs 140) are
    reported through `bpm_candidates`, so the caller can pick the other one.
  * Key is reliable on tonal music, shaky on drums-only or atonal material -
    `key_confidence` is the correlation margin, not a calibrated probability.
  * `vocal_presence` is a heuristic (syllabic modulation of the 200-4000 Hz
    band + harmonic salience there). It is right on "clearly sung" vs
    "clearly instrumental" and it says `uncertain` in between. `vocal_type` is
    a register estimate from the dominant f0 in vocal-ish frames - coarse on
    purpose (male / female / mixed), and it says `unknown` when unsure.
    Any audio-language model (MOSS-Music, see moss.py) beats this; the DSP
    numbers are the fallback that always works and never hallucinates.
"""

import math

import numpy as np

ANALYSIS_RATE = 22050  # plenty for chroma/onsets/flux, 4x faster than 48 kHz
FRAME = 1024
HOP = 512

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (probe-tone ratings, the classic pair)
KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Camelot wheel, used by DJs and understood by many taggers; harmless extra info.
_CAMELOT_MAJOR = {0: "8B", 1: "3B", 2: "10B", 3: "5B", 4: "12B", 5: "7B",
                  6: "2B", 7: "9B", 8: "4B", 9: "11B", 10: "6B", 11: "1B"}
_CAMELOT_MINOR = {0: "5A", 1: "12A", 2: "7A", 3: "2A", 4: "9A", 5: "4A",
                  6: "11A", 7: "6A", 8: "1A", 9: "8A", 10: "3A", 11: "10A"}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def to_mono(samples):
    """[channels, time] or [time] -> float32 mono numpy array."""
    data = np.asarray(samples)
    if data.ndim > 1:
        # ComfyUI AUDIO is [B, C, T]; collapse everything but the last axis
        axes = tuple(range(data.ndim - 1))
        data = data.mean(axis=axes)
    return np.nan_to_num(data.astype(np.float32, copy=False))


def resample_linear(samples, source_rate, target_rate=ANALYSIS_RATE):
    if source_rate == target_rate or samples.size == 0:
        return samples
    count = max(1, int(round(samples.size * target_rate / float(source_rate))))
    old = np.arange(samples.size, dtype=np.float64)
    new = np.linspace(0.0, samples.size - 1.0, count, dtype=np.float64)
    return np.interp(new, old, samples).astype(np.float32)


def _hann(size):
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(size) / size)).astype(np.float32)


def stft_magnitude(mono, frame=FRAME, hop=HOP):
    if mono.size < frame:
        mono = np.pad(mono, (0, frame - mono.size))
    count = 1 + (mono.size - frame) // hop
    window = _hann(frame)
    frames = np.lib.stride_tricks.as_strided(
        mono, shape=(count, frame), strides=(mono.strides[0] * hop, mono.strides[0])
    )
    spectrum = np.fft.rfft(frames * window, axis=1)
    return np.abs(spectrum).T.astype(np.float32)  # [freq, frames]


def _norm(vector):
    peak = float(np.max(vector)) if vector.size else 0.0
    return vector / peak if peak > 0 else np.zeros_like(vector)


# --------------------------------------------------------------------------------------
# tempo
# --------------------------------------------------------------------------------------


def onset_envelope(magnitude):
    """Spectral flux: half-wave rectified frame-to-frame increase."""
    if magnitude.shape[1] < 2:
        return np.zeros(1, dtype=np.float32)
    diff = np.diff(magnitude, axis=1)
    flux = np.maximum(diff, 0.0).sum(axis=0)
    return flux.astype(np.float32)


def estimate_tempo(magnitude, rate=ANALYSIS_RATE, hop=HOP, min_bpm=40.0, max_bpm=220.0):
    """Autocorrelation tempo estimate in `min_bpm..max_bpm`.

    Returns (bpm, confidence, candidates). `candidates` lists the runner-up
    tempos (typically the half/double-time octave) so a caller can offer them.
    """
    flux = onset_envelope(magnitude)
    if flux.size < 8 or float(flux.max()) <= 0.0:
        return 0.0, 0.0, []

    flux = flux - np.convolve(flux, np.ones(9) / 9.0, mode="same")  # drop the DC/trend
    energy = float(np.dot(flux, flux))
    if energy <= 0.0:
        return 0.0, 0.0, []

    size = 1 << int(math.ceil(math.log2(flux.size * 2)))
    spectrum = np.fft.rfft(flux, n=size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=size)[: flux.size]
    autocorr = autocorr / autocorr[0]

    frames_per_second = rate / float(hop)
    lag_min = max(1, int(math.floor(frames_per_second * 60.0 / max_bpm)))
    lag_max = min(flux.size - 1, int(math.ceil(frames_per_second * 60.0 / min_bpm)))
    if lag_max <= lag_min:
        return 0.0, 0.0, []

    lags = np.arange(lag_min, lag_max + 1)
    bpm_of_lag = 60.0 * frames_per_second / lags
    # small tempo prior: 90-160 BPM is where most produced music sits
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm_of_lag, 1e-6) / 120.0) / 0.9) ** 2)
    # harmonic reinforcement: a true tempo also explains its own multiples
    reinforced = autocorr[lags].copy()
    for factor, weight in ((2, 0.5), (0.5, 0.25), (3, 0.25)):
        shifted = np.round(lags * factor).astype(int)
        valid = (shifted >= 1) & (shifted < flux.size)
        reinforced[valid] += weight * autocorr[shifted[valid]]
    score = reinforced * (0.55 + 0.45 * prior)

    best = int(np.argmax(score))

    # Octave correction. Autocorrelation peaks just as strongly at half tempo, and
    # producers rarely write below 70 BPM, so a weak-but-close double-time candidate
    # wins over a very slow winner (and vice versa for 200+ BPM winners).
    def prior_of(value):
        return math.exp(-0.5 * (math.log2(max(value, 1e-6) / 120.0) / 0.55) ** 2)

    best_bpm = float(bpm_of_lag[best])
    best_score = float(score[best])
    for factor in (2.0, 0.5, 1.5, 2.0 / 3.0, 3.0, 1.0 / 3.0):
        alternative = best_bpm * factor
        if not (45.0 <= alternative <= 210.0):
            continue
        # search a small window: the lag grid is coarse at high tempos
        window = np.abs(bpm_of_lag - alternative) <= 0.06 * alternative
        if not window.any():
            continue
        index = int(np.argmax(np.where(window, score, -np.inf)))
        if score[index] >= 0.45 * best_score and prior_of(alternative) > 1.6 * prior_of(best_bpm):
            best, best_bpm, best_score = index, float(bpm_of_lag[index]), float(score[index])
            break

    bpm = best_bpm
    confidence = float(np.clip((best_score - np.median(score)) / (abs(best_score) + 1e-9), 0.0, 1.0))

    # runner-up candidates: best score outside +-3% of the winner
    candidates = []
    mask = np.ones_like(score, dtype=bool)
    mask[np.abs(bpm_of_lag - bpm) <= 0.03 * bpm] = False
    for _ in range(3):
        if not mask.any():
            break
        index = int(np.argmax(np.where(mask, score, -np.inf)))
        if score[index] <= 0.35 * best_score:
            break
        candidates.append(round(float(bpm_of_lag[index]), 1))
        mask[np.abs(bpm_of_lag - bpm_of_lag[index]) <= 0.03 * bpm_of_lag[index]] = False

    return round(bpm, 1), confidence, candidates


# --------------------------------------------------------------------------------------
# key
# --------------------------------------------------------------------------------------


def chromagram(magnitude, rate=ANALYSIS_RATE, frame=FRAME):
    """Energy per pitch class per frame, [12, frames]."""
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    chroma = np.zeros((12, magnitude.shape[1]), dtype=np.float32)
    valid = (freqs >= 55.0) & (freqs <= 4200.0)  # A1 .. C8
    if not valid.any():
        return chroma
    midi = np.round(69.0 + 12.0 * np.log2(freqs[valid] / 440.0)).astype(int)
    pitch_class = midi % 12
    weights = magnitude[valid] ** 1.5  # favour strong partials, tame the noise floor
    for index in range(12):
        chroma[index] = weights[pitch_class == index].sum(axis=0)
    return chroma


def estimate_key(chroma):
    """Krumhansl-Schmuckler. Returns (key, confidence, camelot, rotation_scores)."""
    if chroma.size == 0:
        return "", 0.0, "", []
    profile = chroma.mean(axis=1)
    if float(profile.sum()) <= 0.0:
        return "", 0.0, "", []
    profile = profile / profile.sum()

    scores = []
    for root in range(12):
        rotated = np.roll(profile, -root)
        for mode, reference in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            scores.append((float(np.corrcoef(rotated, reference)[0, 1]), root, mode))
    scores.sort(reverse=True)
    best_corr, root, mode = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0.0
    confidence = float(np.clip((best_corr - runner_up) / 0.12, 0.0, 1.0))
    key = f"{PITCH_NAMES[root]} {mode}"
    camelot = (_CAMELOT_MAJOR if mode == "major" else _CAMELOT_MINOR)[root]
    summary = [
        {"key": f"{PITCH_NAMES[r]} {m}", "score": round(c, 3)} for c, r, m in scores[:4]
    ]
    return key, confidence, camelot, summary


# --------------------------------------------------------------------------------------
# vocals (heuristic)
# --------------------------------------------------------------------------------------


def _band(magnitude, rate, frame, low, high):
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    rows = (freqs >= low) & (freqs < high)
    return magnitude[rows]


def _modulation_strength(envelope, rate, hop):
    """Energy of the 1.5-8 Hz modulation band (syllables/notes), 0..1."""
    if envelope.size < 8:
        return 0.0
    env = np.maximum(envelope - envelope.mean(), 0.0)
    if float(env.max()) <= 0.0:
        return 0.0
    frames_per_second = rate / float(hop)
    size = 1 << int(math.ceil(math.log2(env.size * 2)))
    spectrum = np.abs(np.fft.rfft(env, n=size))
    freqs = np.fft.rfftfreq(size, 1.0 / frames_per_second)
    band = (freqs >= 1.5) & (freqs <= 8.0)
    total = float(spectrum[1:].sum())
    if total <= 0.0 or not band.any():
        return 0.0
    return float(np.clip(spectrum[band].sum() / total, 0.0, 1.0))


def _frame_f0(magnitude, rate, frame, low=70.0, high=420.0):
    """Per-frame f0 guess via harmonic product spectrum in the vocal range."""
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    rows = (freqs >= low) & (freqs <= high)
    if not rows.any():
        return np.zeros(magnitude.shape[1], dtype=np.float32)
    band = magnitude[rows]  # [freq, frames]
    band_freqs = freqs[rows]
    harmonic = np.zeros_like(band)
    for factor in (1, 2, 3, 4):
        target = np.searchsorted(band_freqs, band_freqs * factor).clip(0, band_freqs.size - 1)
        harmonic += np.log1p(band[target])
    best = np.argmax(harmonic, axis=0)
    return band_freqs[best].astype(np.float32), band.max(axis=0)


def estimate_vocals(magnitude, rate=ANALYSIS_RATE, frame=FRAME, hop=HOP):
    """Heuristic vocal presence + register.

    Three cheap, independent hints:
      1. how much of the energy sits in the 250-4000 Hz band where a voice lives
      2. syllabic modulation (1.5-8 Hz) of that band - singing breathes and
         articulates, sustained pads and riffs mostly do not
      3. how tonal that band is (harmonic energy vs spectral flatness)
    """
    if magnitude.shape[1] < 4:
        return {"presence": "unknown", "type": "unknown", "confidence": 0.0}

    mid = _band(magnitude, rate, frame, 250.0, 4000.0)
    total = magnitude.sum(axis=0) + 1e-9
    mid_ratio = float(np.median(mid.sum(axis=0) / total))

    envelope = mid.sum(axis=0)
    modulation = _modulation_strength(_norm(envelope), rate, hop)

    flatness = np.exp(np.mean(np.log(mid + 1e-6), axis=0)) / (np.mean(mid, axis=0) + 1e-6)
    tonality = float(np.clip(1.0 - np.median(flatness) * 4.0, 0.0, 1.0))

    f0, salience = _frame_f0(magnitude, rate, frame)
    usable = f0 > 0
    if usable.any():
        voiced = np.percentile(salience[usable], 60)
        strong = usable & (salience >= voiced)
        pitches = f0[strong]
    else:
        strong = usable
        pitches = np.zeros(0, dtype=np.float32)

    # Singing articulates: syllables and note changes modulate the mid band at
    # 1.5-8 Hz. A pad, a drone or a riff does not, so modulation carries the most
    # weight and also gates the whole score. Without this gate every sustained
    # chord would be called a vocal take.
    modulation_norm = float(np.clip(modulation / 0.40, 0.0, 1.0))
    gate = 0.30 + 0.70 * float(np.clip(modulation / 0.30, 0.0, 1.0))
    score = (
        0.15 * float(np.clip(mid_ratio / 0.55, 0.0, 1.0))
        + 0.60 * modulation_norm
        + 0.25 * tonality
    ) * gate

    if score >= 0.62:
        presence = "vocals"
    elif score <= 0.42:
        presence = "instrumental"
    else:
        presence = "uncertain"
    confidence = float(np.clip(abs(score - 0.52) / 0.35, 0.0, 1.0))

    vocal_type = "unknown"
    median_f0 = float(np.median(pitches)) if pitches.size else 0.0
    if presence == "vocals" and pitches.size > 20 and confidence > 0.2:
        low, high = np.percentile(pitches, [20, 80])
        spread = float(high - low)
        if spread > 90.0:
            vocal_type = "mixed / choir"
        elif median_f0 < 165.0:
            vocal_type = "male"
        else:
            vocal_type = "female"
        if median_f0 and median_f0 < 90.0:
            vocal_type = "male (low)"

    return {
        "presence": presence,
        "type": vocal_type,
        "confidence": round(confidence, 3),
        "score": round(float(score), 3),
        "median_f0_hz": round(median_f0, 1) if median_f0 else 0.0,
        "syllabic_modulation": round(modulation, 3),
    }


# --------------------------------------------------------------------------------------
# loudness / structure
# --------------------------------------------------------------------------------------


def _db(value):
    return 20.0 * math.log10(max(value, 1e-9))


def loudness_stats(mono, rate):
    if mono.size == 0:
        return {"peak_db": -120.0, "rms_db": -120.0, "crest_db": 0.0}
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    return {
        "peak_db": round(_db(peak), 1),
        "rms_db": round(_db(rms), 1),
        "crest_db": round(_db(peak) - _db(rms), 1),
    }


def brightness(magnitude, rate=ANALYSIS_RATE, frame=FRAME):
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    total = magnitude.sum() + 1e-9
    return float((magnitude.sum(axis=1) * freqs).sum() / total) if magnitude.size else 0.0


def section_map(mono, rate, block_seconds=5.0):
    """Coarse energy sections - enough to tell 'ends quietly' from 'ends big'."""
    if mono.size == 0:
        return []
    block = max(1, int(block_seconds * rate))
    sections = []
    for start in range(0, mono.size, block):
        chunk = mono[start:start + block]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        sections.append(
            {"start": round(start / rate, 2), "end": round(min(mono.size, start + block) / rate, 2),
             "rms_db": round(_db(rms), 1)}
        )
    if not sections:
        return []
    levels = np.array([s["rms_db"] for s in sections])
    low, high = float(np.percentile(levels, 25)), float(np.percentile(levels, 90))
    for entry in sections:
        if entry["rms_db"] < low:
            entry["level"] = "quiet"
        elif entry["rms_db"] > high:
            entry["level"] = "loud"
        else:
            entry["level"] = "mid"
    return sections


def tempo_feel(bpm):
    if not bpm:
        return "unknown"
    if bpm < 70:
        return "slow / ballad"
    if bpm < 100:
        return "mid-tempo"
    if bpm < 125:
        return "steady / danceable"
    if bpm < 150:
        return "upbeat"
    return "fast"


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------


def analyse(samples, sample_rate, max_seconds=90.0, sections=True):
    """Analyse `samples` ([C,T] or [T]) and return a plain dict of findings."""
    mono = to_mono(samples)
    original_seconds = mono.size / float(sample_rate or 1)
    mono = resample_linear(mono, sample_rate or ANALYSIS_RATE)
    if max_seconds and mono.size > max_seconds * ANALYSIS_RATE:
        # analyse the end of the track: that is the part a continuation continues
        mono = mono[-int(max_seconds * ANALYSIS_RATE):]

    result = {
        "analyzer": "dsp",
        "duration_seconds": round(original_seconds, 2),
        "analysed_seconds": round(mono.size / float(ANALYSIS_RATE), 2),
        "sample_rate": int(sample_rate or 0),
        "warnings": [],
    }
    if mono.size < ANALYSIS_RATE // 4:
        result["warnings"].append("less than 0.25 s of audio - nothing to measure")
        result.update({"bpm": 0.0, "bpm_confidence": 0.0, "bpm_candidates": [],
                       "key": "", "key_confidence": 0.0, "camelot": "",
                       "vocal": {"presence": "unknown", "type": "unknown", "confidence": 0.0},
                       "sections": []})
        return result

    magnitude = stft_magnitude(mono)
    bpm, bpm_confidence, candidates = estimate_tempo(magnitude)
    chroma = chromagram(magnitude)
    key, key_confidence, camelot, key_candidates = estimate_key(chroma)
    vocals = estimate_vocals(magnitude)

    result.update(
        {
            "bpm": bpm,
            "bpm_confidence": round(bpm_confidence, 3),
            "bpm_candidates": candidates,
            "tempo_feel": tempo_feel(bpm),
            "key": key,
            "key_confidence": round(key_confidence, 3),
            "key_candidates": key_candidates,
            "camelot": camelot,
            "vocal": vocals,
            "loudness": loudness_stats(mono, ANALYSIS_RATE),
            "brightness_hz": round(brightness(magnitude)),
        }
    )
    if bpm_confidence < 0.25:
        result["warnings"].append("beat tracking is weak - BPM may be wrong (drums-only or rubato?)")
    if key_confidence < 0.2:
        result["warnings"].append("key is ambiguous (atonal, drum-only or modal material)")
    if vocals["presence"] == "uncertain":
        result["warnings"].append("could not decide instrumental vs vocals - check the tags")
    if sections:
        result["sections"] = section_map(mono, ANALYSIS_RATE)
    return result


def tags_from_analysis(analysis, extra=""):
    """A short, factual tag string built only from what was measured."""
    parts = []
    if analysis.get("bpm"):
        parts.append(f"{int(round(analysis['bpm']))} bpm")
    if analysis.get("key"):
        parts.append(analysis["key"])
    vocal = analysis.get("vocal") or {}
    if vocal.get("presence") == "vocals":
        parts.append(f"{vocal.get('type', 'unknown')} lead vocals"
                     if vocal.get("type") not in (None, "", "unknown") else "vocals")
    elif vocal.get("presence") == "instrumental":
        parts.append("instrumental")
    elif vocal.get("presence") == "uncertain":
        parts.append("vocals uncertain")
    if extra:
        parts.append(extra)
    return ", ".join(parts)
