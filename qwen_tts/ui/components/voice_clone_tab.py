# coding=utf-8
"""Voice Clone tab component for Qwen3-TTS UI."""

import gc
import os
import tempfile
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from ..i18n_utils import t, t_list
from ..model_manager import ModelManager

# ============================================================================
# Whisper integration (lazy-loaded)
# ============================================================================
_whisper_model = None
_whisper_model_name = None

WHISPER_MODELS = [
    "tiny",       # 39M params
    "base",       # 74M params
    "small",      # 244M params
    "medium",     # 769M params
    "large-v3",   # 1550M params
]


def _get_whisper_model(model_name: str = "small"):
    """Lazy-load Whisper model (reloads if model name changed)."""
    global _whisper_model, _whisper_model_name

    if _whisper_model is None or _whisper_model_name != model_name:
        try:
            from faster_whisper import WhisperModel

            # Release existing model
            if _whisper_model is not None:
                del _whisper_model
                _whisper_model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"[INFO] Loading Whisper model '{model_name}'...")
            if torch.cuda.is_available():
                _whisper_model = WhisperModel(
                    model_name, device="cuda", compute_type="float16"
                )
            else:
                _whisper_model = WhisperModel(
                    model_name, device="cpu", compute_type="int8"
                )
            _whisper_model_name = model_name
            print(f"[INFO] Whisper model '{model_name}' loaded")
        except Exception as e:
            print(f"[WARNING] Failed to initialize Whisper: {e}")
            import traceback
            traceback.print_exc()
            return None
    return _whisper_model


def get_whisper_status() -> str:
    """Return current Whisper model status for Settings tab."""
    if _whisper_model is not None and _whisper_model_name:
        return _whisper_model_name
    return None


def _transcribe_audio(audio_data, model_name: str = "small") -> str:
    """Transcribe audio data using Whisper."""
    import scipy.io.wavfile as wavfile

    if audio_data is None:
        return f"{t('messages.error')}: {t('messages.audio_data_missing')}"

    model = _get_whisper_model(model_name)
    if model is None:
        return f"{t('messages.error')}: {t('messages.whisper_unavailable')}"

    temp_path = None
    try:
        # Handle various Gradio audio formats
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sr, wav = audio_data
        elif isinstance(audio_data, dict):
            sr = audio_data.get(
                "sample_rate", audio_data.get("sampling_rate", 16000)
            )
            wav = audio_data.get("data", audio_data.get("array", None))
            if wav is None:
                return f"{t('messages.error')}: {t('messages.audio_data_missing')}"
        elif isinstance(audio_data, str):
            segments, _info = model.transcribe(audio_data, language=None)
            text = "".join([seg.text for seg in segments]).strip()
            return text if text else t("messages.no_speech_detected")
        else:
            return (
                f"{t('messages.error')}: "
                f"{t('messages.audio_format_invalid', dtype=type(audio_data).__name__)}"
            )

        wav = np.asarray(wav)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)

        # Normalize to float32
        if np.issubdtype(wav.dtype, np.integer):
            max_val = np.iinfo(wav.dtype).max
            wav = wav.astype(np.float32) / max_val
        else:
            wav = wav.astype(np.float32)
            max_abs = np.max(np.abs(wav))
            if max_abs > 1.0:
                wav = wav / max_abs

        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        wav_int16 = np.clip(wav * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(temp_path, int(sr), wav_int16)

        segments, _info = model.transcribe(temp_path, language=None)
        text = "".join([seg.text for seg in segments]).strip()
        return text if text else t("messages.no_speech_detected")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"{t('messages.error')}: "
            f"{t('messages.transcription_failed', error=f'{type(e).__name__}: {e}')}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# ============================================================================
# Audio utilities
# ============================================================================
def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to [-1, 1] float32."""
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def _audio_to_tuple(audio) -> Optional[Tuple[np.ndarray, int]]:
    """Convert Gradio audio to (wav, sr) tuple."""
    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr
    return None


def _adjust_speed(wav: np.ndarray, sr: int, speed: float) -> Tuple[np.ndarray, int]:
    """Adjust audio playback speed via resampling."""
    if abs(speed - 1.0) < 0.01:
        return wav, sr
    target_length = max(1, int(len(wav) / speed))
    indices = np.linspace(0, len(wav) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(wav)), wav)
    return resampled.astype(np.float32), sr


# ============================================================================
# Tab builder
# ============================================================================
def create_voice_clone_tab(
    manager: ModelManager,
    lang_choices_disp: List[str],
    lang_map: Dict[str, str],
    gen_kwargs_fn: Callable[[], Dict[str, Any]],
) -> None:
    """Create Voice Clone tab UI components and event handlers."""
    # Import VoiceClonePromptItem for prompt save/load
    from qwen_tts import VoiceClonePromptItem

    gr.Markdown(
        f"### {t('voice_clone.title')}\n{t('voice_clone.description')}\n\n"
        f"_{t('voice_clone.notice')}_"
    )

    with gr.Tabs():
        # ---- Sub-tab 1: Voice Clone ----
        with gr.Tab(t("voice_clone.title")):
            with gr.Row():
                with gr.Column(scale=2):
                    ref_audio = gr.Audio(
                        label=t("voice_clone.reference_audio.label"),
                    )
                    ref_text = gr.Textbox(
                        label=t("voice_clone.reference_text.label"),
                        lines=2,
                        placeholder=t("voice_clone.reference_text.placeholder"),
                        interactive=True,
                    )
                    with gr.Row():
                        whisper_model = gr.Dropdown(
                            label=t("voice_clone.whisper.model_label"),
                            choices=WHISPER_MODELS,
                            value="small",
                            interactive=True,
                            scale=2,
                        )
                        transcribe_btn = gr.Button(
                            t("voice_clone.whisper.transcribe_button"),
                            variant="secondary",
                            scale=1,
                        )
                    xvec_only = gr.Checkbox(
                        label=t("voice_clone.xvector_only.label"),
                        value=False,
                        interactive=True,
                    )

                with gr.Column(scale=2):
                    text_in = gr.Textbox(
                        label=t("voice_clone.text_input.label"),
                        lines=5,
                        placeholder=t("voice_clone.text_input.placeholder"),
                        interactive=True,
                    )
                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label=t("voice_clone.language_selector.label"),
                            choices=lang_choices_disp,
                            value="Auto",
                            interactive=True,
                        )
                        speed_in = gr.Slider(
                            label=t("voice_clone.speed_slider.label"),
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            interactive=True,
                        )
                    btn = gr.Button(
                        t("voice_clone.generate_button"),
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )

                with gr.Column(scale=1):
                    audio_out = gr.Audio(
                        label=t("voice_clone.audio_output"), type="numpy", interactive=False
                    )
                    status_out = gr.Textbox(
                        label=t("voice_clone.status"), lines=2
                    )

            # Tips section
            tips_items = t_list("voice_clone.tips.items")
            if tips_items:
                tips_md = f"**{t('voice_clone.tips.title')}**\n"
                tips_md += "\n".join(f"- {item}" for item in tips_items)
                gr.Markdown(tips_md, elem_classes=["tips-section"])

            # Event: Transcribe
            def transcribe_reference(audio, model_name):
                if audio is None:
                    return (
                        gr.update(),
                        f"{t('messages.error')}: {t('messages.upload_reference_audio')}",
                    )
                result = _transcribe_audio(audio, model_name)
                if result.startswith(t("messages.error")):
                    return gr.update(), result
                return result, t(
                    "messages.transcription_complete",
                    model=model_name,
                    chars=len(result),
                )

            transcribe_btn.click(
                transcribe_reference,
                inputs=[ref_audio, whisper_model],
                outputs=[ref_text, status_out],
            )

            # Event: Generate
            def run_voice_clone(
                ref_aud, ref_txt: str, use_xvec: bool, text: str, lang_disp: str, speed: float
            ):
                try:
                    if not text or not text.strip():
                        return (
                            None,
                            f"{t('messages.error')}: {t('messages.enter_synthesis_text')}",
                        )
                    at = _audio_to_tuple(ref_aud)
                    if at is None:
                        return (
                            None,
                            f"{t('messages.error')}: {t('messages.upload_reference_audio')}",
                        )
                    if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                        return (
                            None,
                            f"{t('messages.error')}: {t('messages.reference_text_required')}",
                        )

                    # Lazy-load base model for voice clone
                    tts = manager.get_model("base")

                    language = lang_map.get(lang_disp, "Auto")
                    kwargs = gen_kwargs_fn()
                    start = time.time()
                    wavs, sr = tts.generate_voice_clone(
                        text=text.strip(),
                        language=language,
                        ref_audio=at,
                        ref_text=(ref_txt.strip() if ref_txt else None),
                        x_vector_only_mode=bool(use_xvec),
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
                run_voice_clone,
                inputs=[ref_audio, ref_text, xvec_only, text_in, lang_in, speed_in],
                outputs=[audio_out, status_out],
            )

        # ---- Sub-tab 2: Save / Load Prompt ----
        with gr.Tab(t("voice_clone.prompt_save_load.tab_title")):
            with gr.Row():
                # Save column
                with gr.Column(scale=2):
                    gr.Markdown(
                        f"### {t('voice_clone.prompt_save_load.save_title')}\n"
                        f"{t('voice_clone.prompt_save_load.save_description')}"
                    )
                    ref_audio_s = gr.Audio(
                        label=t("voice_clone.prompt_save_load.reference_audio_label"),
                        type="numpy",
                    )
                    ref_text_s = gr.Textbox(
                        label=t("voice_clone.prompt_save_load.reference_text_label"),
                        lines=2,
                        placeholder=t("voice_clone.reference_text.placeholder"),
                        interactive=True,
                    )
                    with gr.Row():
                        whisper_model_s = gr.Dropdown(
                            label=t("voice_clone.whisper.model_label"),
                            choices=WHISPER_MODELS,
                            value="small",
                            interactive=True,
                            scale=2,
                        )
                        transcribe_btn_s = gr.Button(
                            t("voice_clone.whisper.transcribe_button"),
                            variant="secondary",
                            scale=1,
                        )
                    xvec_only_s = gr.Checkbox(
                        label=t("voice_clone.xvector_only.label"),
                        value=False,
                        interactive=True,
                    )
                    save_btn = gr.Button(
                        t("voice_clone.prompt_save_load.save_button"),
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )
                    prompt_file_out = gr.File(
                        label=t("voice_clone.prompt_save_load.prompt_file_output")
                    )

                # Load column
                with gr.Column(scale=2):
                    gr.Markdown(
                        f"### {t('voice_clone.prompt_save_load.load_title')}\n"
                        f"{t('voice_clone.prompt_save_load.load_description')}"
                    )
                    prompt_file_in = gr.File(
                        label=t("voice_clone.prompt_save_load.prompt_file_input")
                    )
                    text_in2 = gr.Textbox(
                        label=t("voice_clone.text_input.label"),
                        lines=4,
                        placeholder=t("voice_clone.text_input.placeholder"),
                        interactive=True,
                    )
                    with gr.Row():
                        lang_in2 = gr.Dropdown(
                            label=t("voice_clone.language_selector.label"),
                            choices=lang_choices_disp,
                            value="Auto",
                            interactive=True,
                        )
                        speed_in2 = gr.Slider(
                            label=t("voice_clone.speed_slider.label"),
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            interactive=True,
                        )
                    gen_btn2 = gr.Button(
                        t("voice_clone.generate_button"),
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )

                with gr.Column(scale=1):
                    audio_out2 = gr.Audio(
                        label=t("voice_clone.audio_output"), type="numpy", interactive=False
                    )
                    status_out2 = gr.Textbox(
                        label=t("voice_clone.status"), lines=2
                    )

            # Event: Transcribe (save tab)
            def transcribe_reference_s(audio, model_name):
                if audio is None:
                    return (
                        gr.update(),
                        f"{t('messages.error')}: {t('messages.upload_reference_audio')}",
                    )
                result = _transcribe_audio(audio, model_name)
                if result.startswith(t("messages.error")):
                    return gr.update(), result
                return result, t(
                    "messages.transcription_complete",
                    model=model_name,
                    chars=len(result),
                )

            transcribe_btn_s.click(
                transcribe_reference_s,
                inputs=[ref_audio_s, whisper_model_s],
                outputs=[ref_text_s, status_out2],
            )

            # Event: Save prompt
            def save_prompt(ref_aud, ref_txt: str, use_xvec: bool):
                try:
                    at = _audio_to_tuple(ref_aud)
                    if at is None:
                        return (
                            None,
                            f"{t('messages.error')}: {t('messages.upload_reference_audio')}",
                        )
                    if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                        return (
                            None,
                            f"{t('messages.error')}: {t('messages.reference_text_required')}",
                        )

                    # Lazy-load base model for voice clone prompt creation
                    tts = manager.get_model("base")

                    items = tts.create_voice_clone_prompt(
                        ref_audio=at,
                        ref_text=(ref_txt.strip() if ref_txt else None),
                        x_vector_only_mode=bool(use_xvec),
                    )
                    payload = {"items": [asdict(it) for it in items]}
                    fd, out_path = tempfile.mkstemp(
                        prefix="voice_clone_prompt_", suffix=".pt"
                    )
                    os.close(fd)
                    torch.save(payload, out_path)
                    return out_path, t("messages.prompt_saved")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return None, f"{type(e).__name__}: {e}"

            save_btn.click(
                save_prompt,
                inputs=[ref_audio_s, ref_text_s, xvec_only_s],
                outputs=[prompt_file_out, status_out2],
            )

            # Event: Load prompt and generate
            def load_prompt_and_gen(
                file_obj, text: str, lang_disp: str, speed: float
            ):
                try:
                    if file_obj is None:
                        return (
                            None,
                            f"{t('messages.error')}: {t('messages.upload_prompt_file')}",
                        )
                    if not text or not text.strip():
                        return (
                            None,
                            f"{t('messages.error')}: {t('messages.enter_synthesis_text')}",
                        )

                    path = (
                        getattr(file_obj, "name", None)
                        or getattr(file_obj, "path", None)
                        or str(file_obj)
                    )
                    payload = torch.load(
                        path, map_location="cpu", weights_only=True
                    )
                    if not isinstance(payload, dict) or "items" not in payload:
                        return (
                            None,
                            f"{t('messages.error')}: {t('messages.invalid_file_format')}",
                        )

                    items_raw = payload["items"]
                    if not isinstance(items_raw, list) or len(items_raw) == 0:
                        return (
                            None,
                            f"{t('messages.error')}: {t('messages.empty_prompt_data')}",
                        )

                    items: List[VoiceClonePromptItem] = []
                    for d in items_raw:
                        if not isinstance(d, dict):
                            return (
                                None,
                                f"{t('messages.error')}: {t('messages.invalid_internal_format')}",
                            )
                        ref_code = d.get("ref_code", None)
                        if ref_code is not None and not torch.is_tensor(ref_code):
                            ref_code = torch.tensor(ref_code)
                        ref_spk = d.get("ref_spk_embedding", None)
                        if ref_spk is None:
                            return (
                                None,
                                f"{t('messages.error')}: {t('messages.missing_speaker_vector')}",
                            )
                        if not torch.is_tensor(ref_spk):
                            ref_spk = torch.tensor(ref_spk)

                        items.append(
                            VoiceClonePromptItem(
                                ref_code=ref_code,
                                ref_spk_embedding=ref_spk,
                                x_vector_only_mode=bool(
                                    d.get("x_vector_only_mode", False)
                                ),
                                icl_mode=bool(
                                    d.get(
                                        "icl_mode",
                                        not bool(d.get("x_vector_only_mode", False)),
                                    )
                                ),
                                ref_text=d.get("ref_text", None),
                            )
                        )

                    # Lazy-load base model for voice clone
                    tts = manager.get_model("base")

                    language = lang_map.get(lang_disp, "Auto")
                    kwargs = gen_kwargs_fn()
                    start = time.time()
                    wavs, sr = tts.generate_voice_clone(
                        text=text.strip(),
                        language=language,
                        voice_clone_prompt=items,
                        **kwargs,
                    )
                    elapsed = time.time() - start

                    wav = np.asarray(wavs[0], dtype=np.float32)
                    wav, sr = _adjust_speed(wav, sr, speed)
                    return (sr, wav), t("messages.generated_detail", elapsed=elapsed)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return None, t(
                        "messages.prompt_load_failed",
                        error=f"{type(e).__name__}: {e}",
                    )

            gen_btn2.click(
                load_prompt_and_gen,
                inputs=[prompt_file_in, text_in2, lang_in2, speed_in2],
                outputs=[audio_out2, status_out2],
            )
