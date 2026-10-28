"""Small text/number helpers shared by the captioner nodes (and mirrored in the Tk app)."""

import re

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {
    "CB": "B", "DB": "C#", "EB": "D#", "FB": "E", "GB": "F#", "AB": "G#", "BB": "A#",
}
_MAJOR_WORDS = ("major", "maj", "ionian", "dur", "M")
_MINOR_WORDS = ("minor", "min", "aeolian", "moll", "m")

MIN_BPM = 20
MAX_BPM = 400


def normalize_key(value):
    """'am', 'A Min', 'F# minor', 'Bb', 'Gmaj' -> 'A minor' / 'F# minor' / 'Bb major'.

    Returns "" when the text does not contain a recognisable key.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in ("auto", "none", "null", "unknown", "n/a", "atonal"):
        return ""

    # "A minor", "F# minor", "Bbmaj", "c#min", "G major"
    match = re.search(
        r"(?<![a-z#])([a-g])\s*(#|♯|b|♭)?\s*(?:-|\s)*"
        r"(major|maj|minor|min|ionian|aeolian|m|dur|moll)?\b",
        lowered,
    )
    if not match:
        return ""
    letter, accidental, quality = match.groups()
    if accidental in ("#", "♯"):
        pitch = letter.upper() + "#"
    elif accidental in ("b", "♭"):
        # ACE-Step spells keys with sharps, so Bb -> A# (the only accepted spelling)
        flat_name = (letter + "b").upper()
        if flat_name not in _FLAT_TO_SHARP:
            return ""
        pitch = _FLAT_TO_SHARP[flat_name]
    else:
        pitch = letter.upper()
    if pitch not in PITCHES:
        return ""

    mode = "major"
    if quality in ("minor", "min", "aeolian", "m", "moll"):
        mode = "minor"
    elif quality in ("major", "maj", "ionian", "dur"):
        mode = "major"
    else:
        # no quality word: "Am"/"C#m" carry it in the trailing m; "Bb" alone is major
        tail = lowered[match.end():]
        if lowered.rstrip().endswith("m") and not lowered.rstrip().endswith("b"):
            mode = "minor"
        elif re.search(r"\bm\b", tail):
            mode = "minor"
    return f"{pitch} {mode}"


def normalize_bpm(value, default=0):
    """Anything numeric-ish -> int in [MIN_BPM, MAX_BPM], else `default`."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = re.search(r"(\d+(?:[.,]\d+)?)", str(value))
        if not match:
            return default
        number = float(match.group(1).replace(",", "."))
    if number <= 0:
        return default
    number = int(round(number))
    return number if MIN_BPM <= number <= MAX_BPM else default


def clean(text):
    """Collapse whitespace and strip the separators that would double up in a tag list."""
    return re.sub(r"\s+", " ", str(text or "")).strip().strip(",;")


# backwards-compatible private alias (used by the older nodes in this repo)
_clean = clean


def dedupe_tags(text):
    """Comma-joined tag string with case-insensitive duplicates removed."""
    if not text:
        return ""
    seen = set()
    parts = []
    for raw in re.split(r"[,\n]", str(text)):
        tag = _clean(raw)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(tag)
    return ", ".join(parts)


def compose_tags(caption="", vocal="", structure="", extra="", measured="", limit=900):
    """Join the pieces of a prompt into one ACE-Step-friendly tag string."""
    pieces = []
    if _clean(caption):
        pieces.append(_clean(caption))
    if _clean(vocal):
        pieces.append(_clean(vocal))
    if _clean(structure):
        pieces.append(_clean(structure))
    if _clean(measured):
        pieces.append(_clean(measured))
    if _clean(extra):
        pieces.append(_clean(extra))
    merged = dedupe_tags(", ".join(pieces))
    if len(merged) > limit:
        merged = merged[:limit].rsplit(",", 1)[0]
    return merged


def looks_instrumental(lyrics):
    """True for empty lyrics and for the usual 'no vocals' placeholders."""
    text = _clean(lyrics).lower()
    return text in ("", "[instrumental]", "instrumental", "[inst]", "n/a", "none")
