import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_API_PATH = REPO_ROOT / "docs" / "api.md"

API_SOURCE_FILES = [
    REPO_ROOT / "qwen_tts" / "ui" / "components" / "custom_voice_tab.py",
    REPO_ROOT / "qwen_tts" / "ui" / "components" / "voice_design_tab.py",
    REPO_ROOT / "qwen_tts" / "ui" / "components" / "voice_clone_tab.py",
]

PRIVATE_EVENT_FILES = [
    REPO_ROOT / "qwen_tts" / "ui" / "app.py",
    REPO_ROOT / "qwen_tts" / "ui" / "components" / "settings_tab.py",
    REPO_ROOT / "qwen_tts" / "ui" / "components" / "voice_design_tab.py",
]

REQUIRED_PUBLIC_API_NAMES = {
    "custom_voice_generate",
    "voice_design_generate",
    "voice_clone_transcribe",
    "voice_clone_generate",
    "voice_clone_prompt_transcribe",
    "voice_clone_prompt_save",
    "voice_clone_prompt_generate",
}


def extract_api_names(text: str) -> set[str]:
    pattern = re.compile(r'api_name\s*=\s*["\']([A-Za-z0-9_\-]+)["\']')
    return set(pattern.findall(text))


class TestApiContract(unittest.TestCase):
    def test_required_public_api_names_exist_in_ui_sources(self):
        declared: set[str] = set()
        for path in API_SOURCE_FILES:
            declared |= extract_api_names(path.read_text(encoding="utf-8"))

        missing = REQUIRED_PUBLIC_API_NAMES - declared
        self.assertFalse(
            missing,
            f"Missing required public api_name declarations: {sorted(missing)}",
        )

    def test_static_api_doc_covers_required_public_api_names(self):
        doc_text = DOC_API_PATH.read_text(encoding="utf-8")
        missing = {
            name for name in REQUIRED_PUBLIC_API_NAMES if f"/{name}" not in doc_text
        }
        self.assertFalse(
            missing,
            f"docs/api.md does not mention required endpoints: {sorted(missing)}",
        )

    def test_internal_events_are_marked_private(self):
        private_markers = 0
        for path in PRIVATE_EVENT_FILES:
            private_markers += path.read_text(encoding="utf-8").count(
                'api_visibility="private"'
            )
        self.assertGreaterEqual(
            private_markers,
            5,
            "Expected at least 5 private event markers to keep UI-only events hidden.",
        )


if __name__ == "__main__":
    unittest.main()
