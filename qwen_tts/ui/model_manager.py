# coding=utf-8
"""
Model manager for Qwen3-TTS UI.
Supports lazy-loading multiple model types so all tabs can function
regardless of which model was initially loaded from the CLI.

Model type mapping:
  custom_voice -> *-CustomVoice
  voice_design -> *-VoiceDesign
  base         -> *-Base
"""

import re
import threading
from typing import Any, Dict, Optional, Tuple

import torch

# Suffix mapping: model_kind -> checkpoint suffix
_KIND_TO_SUFFIX = {
    "custom_voice": "CustomVoice",
    "voice_design": "VoiceDesign",
    "base": "Base",
}

_SUFFIX_TO_KIND = {v: k for k, v in _KIND_TO_SUFFIX.items()}

# Pattern to extract base prefix from checkpoint name
# e.g. "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" -> ("Qwen/Qwen3-TTS-12Hz-1.7B", "CustomVoice")
_CKPT_PATTERN = re.compile(r"^(.+?)-(CustomVoice|VoiceDesign|Base)$")


def _parse_ckpt(ckpt: str) -> Tuple[str, str]:
    """Parse checkpoint into (base_prefix, suffix).

    Returns:
        (base_prefix, suffix) e.g. ("Qwen/Qwen3-TTS-12Hz-1.7B", "CustomVoice")

    Raises:
        ValueError: If the checkpoint doesn't match the expected pattern.
    """
    m = _CKPT_PATTERN.match(ckpt.strip().rstrip("/"))
    if not m:
        raise ValueError(
            f"Cannot parse checkpoint '{ckpt}'. "
            f"Expected format: '<prefix>-CustomVoice|VoiceDesign|Base'"
        )
    return m.group(1), m.group(2)


class ModelManager:
    """Thread-safe lazy model loader for multiple Qwen3-TTS model types.

    Usage:
        manager = ModelManager(
            primary_tts=tts,
            primary_ckpt="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            load_kwargs={"device_map": "cuda:0", "dtype": torch.bfloat16},
        )

        # Get the TTS model for voice clone (loads Base model if needed)
        tts_base = manager.get_model("base")
        wavs, sr = tts_base.generate_voice_clone(...)
    """

    def __init__(
        self,
        primary_tts: Any,
        primary_ckpt: str,
        load_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._base_prefix, primary_suffix = _parse_ckpt(primary_ckpt)
        self._primary_kind = _SUFFIX_TO_KIND[primary_suffix]

        # Cache: kind -> (tts_instance, ckpt_string)
        self._models: Dict[str, Tuple[Any, str]] = {
            self._primary_kind: (primary_tts, primary_ckpt),
        }

        # kwargs for from_pretrained (device_map, dtype, attn_implementation, etc.)
        self._load_kwargs = dict(load_kwargs or {})

        self._lock = threading.Lock()
        self._loading: Dict[str, bool] = {}

    @property
    def primary_kind(self) -> str:
        """The model kind initially loaded from CLI."""
        return self._primary_kind

    def ckpt_for_kind(self, kind: str) -> str:
        """Return the checkpoint name for a given model kind."""
        suffix = _KIND_TO_SUFFIX.get(kind)
        if suffix is None:
            raise ValueError(f"Unknown model kind: {kind}")
        return f"{self._base_prefix}-{suffix}"

    def is_loaded(self, kind: str) -> bool:
        """Check if a model kind is already loaded."""
        return kind in self._models

    def get_model(self, kind: str) -> Any:
        """Get (or lazy-load) the TTS model for the given kind.

        Args:
            kind: One of "custom_voice", "voice_design", "base".

        Returns:
            Qwen3TTSModel instance ready for generation.
        """
        if kind in self._models:
            return self._models[kind][0]

        with self._lock:
            # Double-check after acquiring lock
            if kind in self._models:
                return self._models[kind][0]

            if kind in self._loading:
                raise RuntimeError(f"Model '{kind}' is already being loaded (concurrent request).")

            self._loading[kind] = True

        try:
            ckpt = self.ckpt_for_kind(kind)
            print(f"[ModelManager] Loading model '{ckpt}' for {kind}...")

            from qwen_tts import Qwen3TTSModel
            tts = Qwen3TTSModel.from_pretrained(ckpt, **self._load_kwargs)

            with self._lock:
                self._models[kind] = (tts, ckpt)
                self._loading.pop(kind, None)

            print(f"[ModelManager] Model '{ckpt}' loaded successfully.")
            return tts

        except Exception:
            with self._lock:
                self._loading.pop(kind, None)
            raise

    def get_all_supported_langs(self) -> list:
        """Get supported languages from the primary model."""
        tts = self._models[self._primary_kind][0]
        if callable(getattr(tts.model, "get_supported_languages", None)):
            return list(tts.model.get_supported_languages() or [])
        return []

    def get_all_supported_speakers(self) -> list:
        """Get supported speakers from the custom_voice model (if loaded or primary)."""
        for kind in ("custom_voice", self._primary_kind):
            if kind in self._models:
                tts = self._models[kind][0]
                if callable(getattr(tts.model, "get_supported_speakers", None)):
                    result = tts.model.get_supported_speakers()
                    if result:
                        return list(result)
        return []
