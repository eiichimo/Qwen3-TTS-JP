# coding=utf-8
"""Settings tab component for Qwen3-TTS UI (Windows / Blackwell optimized)."""

import platform
import sys
from typing import Any, Optional

import gradio as gr
import torch

from ..i18n_utils import t


def _get_gpu_info_md() -> str:
    """Build GPU information markdown."""
    if not torch.cuda.is_available():
        return f"**{t('settings.gpu_info.no_gpu')}**"

    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    lines = [
        f"| {t('settings.gpu_info.gpu_name')} | **{props.name}** |",
        f"| {t('settings.gpu_info.vram_total')} | {total_mem / 1024**3:.1f} GB |",
        f"| {t('settings.gpu_info.cuda_version')} | {torch.version.cuda or 'N/A'} |",
        f"| {t('settings.gpu_info.compute_capability')} | sm_{props.major}{props.minor} |",
    ]

    return "| | |\n|---|---|\n" + "\n".join(lines)


def _get_vram_info_md() -> str:
    """Build VRAM usage markdown with progress bar."""
    if not torch.cuda.is_available():
        return ""

    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    props = torch.cuda.get_device_properties(0)
    total = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
    free = total - allocated
    usage_pct = (allocated / total * 100) if total > 0 else 0

    # Choose color class
    if usage_pct < 50:
        bar_class = "low"
    elif usage_pct < 80:
        bar_class = "medium"
    else:
        bar_class = "high"

    bar_html = (
        f'<div class="vram-bar-container">'
        f'<div class="vram-bar {bar_class}" style="width: {usage_pct:.0f}%">'
        f'{usage_pct:.1f}%</div></div>'
    )

    lines = [
        bar_html,
        f"| {t('settings.vram.allocated')} | {allocated:.2f} GB |",
        f"| {t('settings.vram.reserved')} | {reserved:.2f} GB |",
        f"| {t('settings.vram.total')} | {total:.1f} GB |",
        f"| {t('settings.vram.free')} | {free:.2f} GB |",
    ]

    return lines[0] + "\n\n| | |\n|---|---|\n" + "\n".join(lines[1:])


def _get_model_info_md(tts: Any, ckpt: str, model_kind: str) -> str:
    """Build model information markdown."""
    dtype_str = "N/A"
    param_count = "N/A"

    try:
        model = tts.model
        if hasattr(model, "dtype"):
            dtype_str = str(model.dtype)
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        if total_params > 1e9:
            param_count = f"{total_params / 1e9:.2f}B"
        elif total_params > 1e6:
            param_count = f"{total_params / 1e6:.1f}M"
        else:
            param_count = f"{total_params:,}"
    except Exception:
        pass

    lines = [
        f"| {t('settings.model_info.name')} | `{ckpt}` |",
        f"| {t('settings.model_info.type')} | {model_kind} |",
        f"| {t('settings.model_info.dtype')} | {dtype_str} |",
        f"| {t('settings.model_info.parameters')} | {param_count} |",
    ]

    return "| | |\n|---|---|\n" + "\n".join(lines)


def _get_attn_info_md(attn_impl: Optional[str]) -> str:
    """Build attention implementation markdown."""
    if attn_impl == "flash_attention_2":
        label = t("settings.flash_attn.flash_attention_2")
        dot = '<span class="status-dot green"></span>'
    elif attn_impl in (None, "sdpa"):
        label = t("settings.flash_attn.sdpa")
        dot = '<span class="status-dot green"></span>'
        attn_impl = "sdpa"
    else:
        label = t("settings.flash_attn.eager")
        dot = '<span class="status-dot yellow"></span>'

    return (
        f"{dot} **{t('settings.flash_attn.current')}:** {label}\n\n"
        f"_{t('settings.flash_attn.note_windows')}_"
    )


def _get_system_info_md() -> str:
    """Build system information markdown."""
    # Whisper status
    from .voice_clone_tab import get_whisper_status
    whisper_status = get_whisper_status()
    whisper_display = whisper_status if whisper_status else t("settings.system_info.not_loaded")

    cudnn_version = "N/A"
    try:
        if torch.backends.cudnn.is_available():
            cudnn_version = str(torch.backends.cudnn.version())
    except Exception:
        pass

    lines = [
        f"| {t('settings.system_info.os')} | {platform.system()} {platform.release()} ({platform.machine()}) |",
        f"| {t('settings.system_info.python')} | {sys.version.split()[0]} |",
        f"| {t('settings.system_info.pytorch')} | {torch.__version__} |",
        f"| {t('settings.system_info.cuda')} | {torch.version.cuda or 'N/A'} |",
        f"| {t('settings.system_info.cudnn')} | {cudnn_version} |",
        f"| {t('settings.system_info.whisper_status')} | {whisper_display} |",
    ]

    return "| | |\n|---|---|\n" + "\n".join(lines)


def create_settings_tab(
    manager: Any,
    ckpt: str,
    model_kind: str,
    attn_impl: Optional[str],
) -> None:
    """Create Settings tab UI with Windows/Blackwell-specific information."""

    with gr.Row():
        # Left column: GPU + VRAM
        with gr.Column(scale=1):
            gr.Markdown(f"#### {t('settings.gpu_info.title')}")
            gpu_info_display = gr.Markdown(value=_get_gpu_info_md())

            gr.Markdown(f"#### {t('settings.vram.title')}")
            vram_display = gr.Markdown(value=_get_vram_info_md())
            with gr.Row():
                vram_refresh_btn = gr.Button(
                    t("settings.vram.refresh_button"),
                    variant="secondary",
                    scale=1,
                )
                cache_clear_btn = gr.Button(
                    t("settings.vram.clear_cache_button"),
                    variant="stop",
                    scale=1,
                )
            vram_status = gr.Textbox(label="", lines=1, visible=False)

        # Right column: Attention + Model Info
        with gr.Column(scale=1):
            gr.Markdown(f"#### {t('settings.flash_attn.title')}")
            gr.Markdown(value=_get_attn_info_md(attn_impl))

            gr.Markdown(f"#### {t('settings.model_info.title')}")
            # Get primary tts from manager for model info
            _primary_tts = manager.get_model(manager.primary_kind)
            gr.Markdown(value=_get_model_info_md(_primary_tts, ckpt, model_kind))

    # System info (full width)
    gr.Markdown(f"#### {t('settings.system_info.title')}")
    sys_info_display = gr.Markdown(value=_get_system_info_md())
    sys_refresh_btn = gr.Button(
        t("settings.system_info.refresh_button"),
        variant="secondary",
    )

    # Technical info (collapsible)
    with gr.Accordion(t("settings.technical_info.title"), open=False):
        gr.Markdown(value=t("settings.technical_info.content"))

    # Events
    def refresh_vram():
        return _get_vram_info_md()

    def clear_cuda_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _get_vram_info_md()

    def refresh_system():
        return _get_system_info_md()

    vram_refresh_btn.click(refresh_vram, outputs=[vram_display])
    cache_clear_btn.click(clear_cuda_cache, outputs=[vram_display])
    sys_refresh_btn.click(refresh_system, outputs=[sys_info_display])
