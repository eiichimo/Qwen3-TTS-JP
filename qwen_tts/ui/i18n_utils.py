# coding=utf-8
"""
Internationalization (i18n) utility for Qwen3-TTS UI.
Provides dot-separated key lookup with JSON translation files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

_i18n: Dict[str, Any] = {}
_current_lang: str = "ja"
_I18N_DIR = Path(__file__).parent / "i18n"
_LANG_PREF_FILE = Path(__file__).parent / ".lang_pref"


def save_language_pref(lang_code: str) -> None:
    """Persist language preference to a file."""
    try:
        _LANG_PREF_FILE.write_text(lang_code, encoding="utf-8")
    except OSError:
        pass


def load_language_pref() -> str:
    """Read persisted language preference. Returns 'ja' if not found."""
    try:
        code = _LANG_PREF_FILE.read_text(encoding="utf-8").strip()
        if (_I18N_DIR / f"{code}.json").exists():
            return code
    except OSError:
        pass
    return "ja"


def set_language(lang_code: str, persist: bool = False) -> None:
    """Load translations for the given language code."""
    global _i18n, _current_lang
    json_path = _I18N_DIR / f"{lang_code}.json"
    if not json_path.exists():
        json_path = _I18N_DIR / "ja.json"
        lang_code = "ja"
    with open(json_path, "r", encoding="utf-8") as f:
        _i18n = json.load(f)
    _current_lang = lang_code
    if persist:
        save_language_pref(lang_code)


def get_current_language() -> str:
    """Return current language code."""
    return _current_lang


def t(key: str, **kwargs) -> str:
    """
    Get translated string by dot-separated key.
    Supports format placeholders: t("messages.done", count=5)
    Returns the key itself as fallback if not found.
    """
    parts = key.split(".")
    value: Any = _i18n
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
            if value is None:
                return key
        else:
            return key
    if isinstance(value, str):
        if kwargs:
            try:
                return value.format(**kwargs)
            except (KeyError, IndexError, ValueError):
                return value
        return value
    return key


def t_list(key: str) -> List[str]:
    """Get translated list by dot-separated key."""
    parts = key.split(".")
    value: Any = _i18n
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
            if value is None:
                return []
        else:
            return []
    if isinstance(value, list):
        return list(value)
    return []


def t_dict(key: str) -> Dict[str, Any]:
    """Get translated dict by dot-separated key."""
    parts = key.split(".")
    value: Any = _i18n
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
            if value is None:
                return {}
        else:
            return {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def get_available_languages() -> List[tuple]:
    """Return list of (display_name, code) tuples for available languages."""
    languages = []
    for json_file in sorted(_I18N_DIR.glob("*.json")):
        code = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            display = data.get("_meta", {}).get("display_name", code)
            languages.append((display, code))
        except Exception:
            languages.append((code, code))
    return languages


# Initialize with saved preference (or default "ja")
set_language(load_language_pref())
