# coding=utf-8
"""Custom Voice tab component for Qwen3-TTS UI."""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from ..i18n_utils import t, t_dict
from ..model_manager import ModelManager

# Emotion -> English instruction mapping for the instruct parameter
EMOTION_INSTRUCTIONS = {
    "neutral": "",
    "happy": "Speak in a happy, joyful and cheerful tone with warmth in your voice.",
    "sad": "Speak in a sad, melancholic and sorrowful tone with a heavy heart.",
    "angry": "Speak in an angry, furious and intense tone with strong emotion.",
    "surprised": "Speak in a surprised and astonished tone, as if encountering something unexpected.",
    "fearful": "Speak in a fearful, anxious and trembling tone with nervousness.",
    "disgusted": "Speak in a disgusted and repulsed tone showing displeasure.",
    "calm": "Speak in a calm, peaceful and serene tone with composure.",
    "excited": "Speak in an excited, enthusiastic and energetic tone with great passion.",
    "tender": "Speak in a tender, gentle and loving tone with softness.",
}

EMOTION_KEYS = list(EMOTION_INSTRUCTIONS.keys())


def _build_emotion_choices() -> List[str]:
    """Build localized emotion display list."""
    return [t(f"emotions.{k}") for k in EMOTION_KEYS]


def _emotion_display_to_key(display: str) -> str:
    """Map localized emotion display name back to key."""
    for key in EMOTION_KEYS:
        if t(f"emotions.{key}") == display:
            return key
    return "neutral"


def _combine_instructions(emotion_key: str, custom_instruct: str) -> Optional[str]:
    """Combine emotion instruction with custom instruction."""
    parts = []
    emotion_text = EMOTION_INSTRUCTIONS.get(emotion_key, "")
    if emotion_text:
        parts.append(emotion_text)
    if custom_instruct and custom_instruct.strip():
        parts.append(custom_instruct.strip())
    return " ".join(parts) if parts else None


def _adjust_speed(wav: np.ndarray, sr: int, speed: float) -> Tuple[np.ndarray, int]:
    """Adjust audio playback speed via resampling."""
    if abs(speed - 1.0) < 0.01:
        return wav, sr
    target_length = max(1, int(len(wav) / speed))
    indices = np.linspace(0, len(wav) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(wav)), wav)
    return resampled.astype(np.float32), sr


def create_custom_voice_tab(
    manager: ModelManager,
    lang_choices_disp: List[str],
    lang_map: Dict[str, str],
    spk_choices_disp: List[str],
    spk_map: Dict[str, str],
    gen_kwargs_fn: Callable[[], Dict[str, Any]],
) -> None:
    """Create Custom Voice tab UI components and event handlers."""

    gr.Markdown(f"### {t('custom_voice.title')}\n{t('custom_voice.description')}")

    with gr.Row():
        with gr.Column(scale=2):
            text_in = gr.Textbox(
                label=t("custom_voice.text_input.label"),
                lines=5,
                max_lines=10,
                placeholder=t("custom_voice.text_input.placeholder"),
                interactive=True,
            )
            with gr.Row():
                spk_in = gr.Dropdown(
                    label=t("custom_voice.speaker_selector.label"),
                    info=t("custom_voice.speaker_selector.info"),
                    choices=spk_choices_disp,
                    value=spk_choices_disp[0] if spk_choices_disp else None,
                    interactive=True,
                )
                lang_in = gr.Dropdown(
                    label=t("custom_voice.language_selector.label"),
                    info=t("custom_voice.language_selector.info"),
                    choices=lang_choices_disp,
                    value="Auto",
                    interactive=True,
                )
            with gr.Row():
                emotion_in = gr.Dropdown(
                    label=t("custom_voice.emotion_selector.label"),
                    info=t("custom_voice.emotion_selector.info"),
                    choices=_build_emotion_choices(),
                    value=t("emotions.neutral"),
                    interactive=True,
                )
                speed_in = gr.Slider(
                    label=t("custom_voice.speed_slider.label"),
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                )
            instruct_in = gr.Textbox(
                label=t("custom_voice.instruction_input.label"),
                lines=2,
                placeholder=t("custom_voice.instruction_input.placeholder"),
                interactive=True,
            )
            btn = gr.Button(
                t("custom_voice.generate_button"),
                variant="primary",
                elem_classes=["primary-btn"],
            )

        with gr.Column(scale=1):
            audio_out = gr.Audio(label=t("custom_voice.audio_output"), type="numpy", interactive=False)
            status_out = gr.Textbox(label=t("custom_voice.status"), lines=2)

    def run_custom_voice(
        text: str,
        spk_disp: str,
        lang_disp: str,
        emotion_disp: str,
        speed: float,
        custom_instruct: str,
    ):
        try:
            if not text or not text.strip():
                return None, f"{t('messages.error')}: {t('messages.enter_text')}"
            if not spk_disp:
                return None, f"{t('messages.error')}: {t('messages.select_speaker')}"

            # Lazy-load custom_voice model if needed
            tts = manager.get_model("custom_voice")

            language = lang_map.get(lang_disp, "Auto")
            speaker = spk_map.get(spk_disp, spk_disp)
            emotion_key = _emotion_display_to_key(emotion_disp)
            instruct = _combine_instructions(emotion_key, custom_instruct)

            kwargs = gen_kwargs_fn()
            start = time.time()
            wavs, sr = tts.generate_custom_voice(
                text=text.strip(),
                language=language,
                speaker=speaker,
                instruct=instruct,
                **kwargs,
            )
            elapsed = time.time() - start

            wav = np.asarray(wavs[0], dtype=np.float32)
            wav, sr = _adjust_speed(wav, sr, speed)
            return (sr, wav), t("messages.generated_detail", elapsed=elapsed)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"{type(e).__name__}: {e}"

    btn.click(
        run_custom_voice,
        inputs=[text_in, spk_in, lang_in, emotion_in, speed_in, instruct_in],
        outputs=[audio_out, status_out],
        api_name="custom_voice_generate",
        api_description="Generate speech from text using a preset/custom voice speaker.",
    )
