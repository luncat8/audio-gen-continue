"""Music Caption - measure and describe an audio track for music generation prompts.

Nodes (category `audio/caption`):

    MusicCaptionAnalyze   AUDIO -> bpm / keyscale / vocal presence + type / duration / tags
                          (numpy DSP, always available, no model download)
    MusicCaptionMOSS      AUDIO -> caption / lyrics / vocal details / key / bpm
                          (MOSS-Music 8B over an SGLang OpenAI-compatible endpoint)
    MusicCaptionMerge     -> one tags + lyrics + bpm + keyscale bundle for TextEncode*
    MusicCaptionTap       -> collects STRING outputs of captioners from other packs

No extra Python packages: numpy ships with ComfyUI, everything else is stdlib.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
