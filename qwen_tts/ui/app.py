# coding=utf-8
"""
Main Gradio application builder for Qwen3-TTS UI.
Provides the build_demo() function called from qwen_tts.cli.demo.

Architecture:
  - Language selector is placed OUTSIDE @gr.render so it persists.
  - All translatable UI content is inside @gr.render, which re-builds
    the component tree whenever the language dropdown changes.
  - Tab selection state is tracked via a mutable dict so that the
    previously selected tab is restored after a language switch.
"""

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .css import CUSTOM_CSS, UPLOAD_FIX_JS
from .i18n_utils import get_available_languages, set_language, t
from .model_manager import ModelManager
from .components import (
    create_custom_voice_tab,
    create_voice_design_tab,
    create_voice_clone_tab,
    create_settings_tab,
)

# Stable tab IDs (do NOT change -- used to preserve selection across re-renders)
_TAB_IDS = ["tab-cv", "tab-vd", "tab-vc", "tab-settings"]


def _title_case_display(s: str) -> str:
    s = (s or "").strip().replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(
    items: Optional[List[str]],
) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _detect_model_kind(ckpt: str, tts: Any) -> str:
    mt = getattr(tts.model, "tts_model_type", None)
    if mt in ("custom_voice", "voice_design", "base"):
        return mt
    raise ValueError(f"Unknown Qwen-TTS model type: {mt}")


def build_demo(
    tts: Any,
    ckpt: str,
    gen_kwargs_default: Dict[str, Any],
    attn_impl: Optional[str] = None,
) -> gr.Blocks:
    """Build the Gradio Blocks demo with i18n + tab-state preservation."""
    import traceback as _tb

    model_kind = _detect_model_kind(ckpt, tts)

    # ---- Model manager ----
    load_kwargs: Dict[str, Any] = {}
    if hasattr(tts.model, "device"):
        load_kwargs["device_map"] = str(tts.model.device)
    if hasattr(tts.model, "dtype"):
        load_kwargs["dtype"] = tts.model.dtype
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    manager = ModelManager(
        primary_tts=tts,
        primary_ckpt=ckpt,
        load_kwargs=load_kwargs,
    )

    # ---- Model langs / speakers ----
    lang_choices_disp, lang_map = _build_choices_and_map(
        list(manager.get_all_supported_langs() or [])
    )
    spk_choices_disp, spk_map = _build_choices_and_map(
        list(manager.get_all_supported_speakers() or [])
    )

    def _gen_common_kwargs() -> Dict[str, Any]:
        return dict(gen_kwargs_default)

    # ---- UI language selector data ----
    available_langs = get_available_languages()
    lang_display_choices = [d for d, _ in available_langs]
    lang_display_to_code = {d: c for d, c in available_langs}

    default_ui_lang_display = next(
        (d for d, c in available_langs if c == "ja"),
        lang_display_choices[0] if lang_display_choices else "ja",
    )

    # ---- Mutable tab-state (survives @gr.render cycles) ----
    _tab_state: Dict[str, str] = {"selected": _TAB_IDS[0]}

    # ---- Theme ----
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    _launch_extras = {"theme": theme, "css": CUSTOM_CSS, "js": UPLOAD_FIX_JS}
    _gradio_major = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 4
    _blocks_kwargs = {}
    if _gradio_major < 6:
        _blocks_kwargs = {"theme": theme, "css": CUSTOM_CSS, "js": UPLOAD_FIX_JS}

    # ==================================================================
    with gr.Blocks(title="Qwen3-TTS-WinBlackwell", **_blocks_kwargs) as demo:
        demo._launch_extras = _launch_extras

        # ---- Header (persists across re-renders) ----
        with gr.Row(elem_classes=["app-header"]):
            with gr.Column(scale=3, min_width=300):
                gr.Markdown(
                    "# Qwen3-TTS-WinBlackwell\n"
                    f"`{ckpt}` &nbsp; | &nbsp; `{model_kind}`",
                )
            with gr.Column(scale=1, min_width=160):
                ui_lang_selector = gr.Dropdown(
                    choices=lang_display_choices,
                    value=default_ui_lang_display,
                    label="",
                    show_label=False,
                    interactive=True,
                    elem_classes=["lang-selector"],
                )

        # ---- Dynamic content ----
        @gr.render(inputs=[ui_lang_selector])
        def render_main(ui_lang_display: str):
            lang_code = lang_display_to_code.get(ui_lang_display, "ja")
            set_language(lang_code)

            # Tabs -- selected= restores previously active tab
            with gr.Tabs(selected=_tab_state["selected"]) as tabs:
                with gr.Tab(t("tabs.custom_voice"), id=_TAB_IDS[0]):
                    try:
                        create_custom_voice_tab(
                            manager=manager,
                            lang_choices_disp=lang_choices_disp,
                            lang_map=lang_map,
                            spk_choices_disp=spk_choices_disp,
                            spk_map=spk_map,
                            gen_kwargs_fn=_gen_common_kwargs,
                        )
                    except Exception:
                        _tb.print_exc()
                        gr.Markdown(f"**Error:** {_tb.format_exc()}")

                with gr.Tab(t("tabs.voice_design"), id=_TAB_IDS[1]):
                    try:
                        create_voice_design_tab(
                            manager=manager,
                            lang_choices_disp=lang_choices_disp,
                            lang_map=lang_map,
                            gen_kwargs_fn=_gen_common_kwargs,
                        )
                    except Exception:
                        _tb.print_exc()
                        gr.Markdown(f"**Error:** {_tb.format_exc()}")

                with gr.Tab(t("tabs.voice_clone"), id=_TAB_IDS[2]):
                    try:
                        create_voice_clone_tab(
                            manager=manager,
                            lang_choices_disp=lang_choices_disp,
                            lang_map=lang_map,
                            gen_kwargs_fn=_gen_common_kwargs,
                        )
                    except Exception:
                        _tb.print_exc()
                        gr.Markdown(f"**Error:** {_tb.format_exc()}")

                with gr.Tab(t("tabs.settings"), id=_TAB_IDS[3]):
                    try:
                        create_settings_tab(
                            manager=manager,
                            ckpt=ckpt,
                            model_kind=model_kind,
                            attn_impl=attn_impl,
                        )
                    except Exception:
                        _tb.print_exc()
                        gr.Markdown(f"**Error:** {_tb.format_exc()}")

            # Track tab selection so it survives the next re-render
            def _on_tab_select(evt: gr.SelectData):
                if 0 <= evt.index < len(_TAB_IDS):
                    _tab_state["selected"] = _TAB_IDS[evt.index]

            tabs.select(fn=_on_tab_select)

            # Disclaimer
            gr.Markdown(t("disclaimer.text"), elem_classes=["disclaimer"])

    return demo
