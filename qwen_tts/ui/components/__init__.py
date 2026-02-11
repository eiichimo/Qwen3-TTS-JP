# coding=utf-8
"""UI tab components for Qwen3-TTS."""

from .custom_voice_tab import create_custom_voice_tab
from .voice_design_tab import create_voice_design_tab
from .voice_clone_tab import create_voice_clone_tab
from .settings_tab import create_settings_tab

__all__ = [
    "create_custom_voice_tab",
    "create_voice_design_tab",
    "create_voice_clone_tab",
    "create_settings_tab",
]
