"""Music Caption - measure and describe an audio track for music generation prompts.

Nodes (category `audio/caption`):

    MusicCaptionMOSSLoader  -> moss_model + status (loads OpenMOSS-Team/MOSS-Music-8B-Instruct
                               from ComfyUI/models/moss_music/, or falls back to DSP / SGLang)
    MusicCaptionAnalyze     AUDIO -> bpm / keyscale / vocal presence + type / duration / tags
                               (numpy DSP, always available, no model download)
    MusicCaptionMOSS        AUDIO (+ moss_model) -> caption / lyrics / vocal details / key / bpm
    MusicCaptionMerge       -> one tags + lyrics + bpm + keyscale bundle for TextEncode*
    MusicCaptionView        -> formatted human-readable report of all measured + MOSS fields
    MusicCaptionTap         -> collects STRING outputs of captioners from other packs

No extra Python packages required for DSP; local MOSS-Music uses ComfyUI's own
`torch` + `transformers`.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
