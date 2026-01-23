# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
# MODIFICATIONS by hiroki-abe-58 (2026):
# - Complete Japanese localization of GUI (labels, buttons, error messages)
# - Added Whisper-based automatic transcription feature for voice cloning
# - Added Whisper model selection (tiny/base/small/medium/large-v3)
# Repository: https://github.com/hiroki-abe-58/Qwen3-TTS-JP
# ============================================================================
"""
A gradio demo for Qwen3 TTS models.
Japanese localized version with Whisper transcription support.
"""

import argparse
import os
import tempfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from .. import Qwen3TTSModel, VoiceClonePromptItem

# Whisper for automatic transcription
_whisper_model = None
_whisper_model_name = None

# 利用可能なWhisperモデル
WHISPER_MODELS = [
    "tiny",      # 最速・最小（39M パラメータ）
    "base",      # 高速（74M パラメータ）
    "small",     # バランス型（244M パラメータ）
    "medium",    # 高精度（769M パラメータ）
    "large-v3",  # 最高精度（1550M パラメータ）
]

def _get_whisper_model(model_name: str = "small"):
    """Whisperモデルを遅延ロード（モデル名が変わったら再ロード）"""
    global _whisper_model, _whisper_model_name
    
    if _whisper_model is None or _whisper_model_name != model_name:
        try:
            from faster_whisper import WhisperModel
            import torch
            
            # 既存のモデルを解放
            if _whisper_model is not None:
                del _whisper_model
                _whisper_model = None
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"[INFO] Whisperモデル '{model_name}' を読み込み中...")
            
            # GPUが利用可能かチェック
            if torch.cuda.is_available():
                _whisper_model = WhisperModel(model_name, device="cuda", compute_type="float16")
            else:
                _whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
            
            _whisper_model_name = model_name
            print(f"[INFO] Whisperモデル '{model_name}' を初期化しました")
        except Exception as e:
            print(f"[WARNING] Whisperモデルの初期化に失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    return _whisper_model

def _transcribe_audio(audio_data, model_name: str = "small") -> str:
    """音声データをWhisperで文字起こし"""
    import scipy.io.wavfile as wavfile
    
    if audio_data is None:
        return "エラー: 音声データがありません"
    
    model = _get_whisper_model(model_name)
    if model is None:
        return "エラー: Whisperモデルが利用できません"
    
    temp_path = None
    try:
        # Gradioの音声データを処理
        # Gradio 4.x/5.x: (sample_rate, numpy_array) のタプル
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sr, wav = audio_data
        # Gradio 6.x: 辞書形式の場合
        elif isinstance(audio_data, dict):
            sr = audio_data.get("sample_rate", audio_data.get("sampling_rate", 16000))
            wav = audio_data.get("data", audio_data.get("array", None))
            if wav is None:
                return "エラー: 音声データが見つかりません"
        # ファイルパスの場合
        elif isinstance(audio_data, str):
            # ファイルパスが直接渡された場合
            segments, info = model.transcribe(audio_data, language=None)
            text = "".join([seg.text for seg in segments]).strip()
            return text if text else "（音声が検出されませんでした）"
        else:
            return f"エラー: 音声データの形式が不正です (type: {type(audio_data).__name__})"
        
        # numpy配列に変換
        wav = np.asarray(wav)
        
        # ステレオをモノラルに変換
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        
        # 正規化（float32に変換）
        if np.issubdtype(wav.dtype, np.integer):
            max_val = np.iinfo(wav.dtype).max
            wav = wav.astype(np.float32) / max_val
        else:
            wav = wav.astype(np.float32)
            max_abs = np.max(np.abs(wav))
            if max_abs > 1.0:
                wav = wav / max_abs
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        # 16bit PCMに変換して保存
        wav_int16 = np.clip(wav * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(temp_path, int(sr), wav_int16)
        
        # Whisperで文字起こし
        segments, info = model.transcribe(temp_path, language=None)
        text = "".join([seg.text for seg in segments]).strip()
        
        return text if text else "（音声が検出されませんでした）"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"エラー: 文字起こしに失敗しました - {type(e).__name__}: {e}"
    finally:
        # 一時ファイルを削除
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass


def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _maybe(v):
    return v if v is not None else gr.update()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-demo",
        description=(
            "Launch a Gradio demo for Qwen3 TTS models (CustomVoice / VoiceDesign / Base).\n\n"
            "Examples:\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --port 8000 --ip 127.0.0.01\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --device cuda:0\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --dtype bfloat16 --no-flash-attn\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )

    # Positional checkpoint (also supports -c/--checkpoint)
    parser.add_argument(
        "checkpoint_pos",
        nargs="?",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (positional).",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (optional if positional is provided).",
    )

    # Model loading / from_pretrained args
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for device_map, e.g. cpu, cuda, cuda:0 (default: cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model (default: bfloat16).",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 (default: enabled).",
    )

    # Gradio server args
    parser.add_argument(
        "--ip",
        default="0.0.0.0",
        help="Server bind IP for Gradio (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port for Gradio (default: 8000).",
    )
    parser.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to create a public Gradio link (default: disabled).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Gradio queue concurrency (default: 16).",
    )

    # HTTPS args
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to SSL certificate file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to SSL key file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-verify/--no-ssl-verify",
        dest="ssl_verify",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to verify SSL certificate (default: enabled).",
    )

    # Optional generation args
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for generation (optional).")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (optional).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (optional).")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling (optional).")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (optional).")
    parser.add_argument("--subtalker-top-k", type=int, default=None, help="Subtalker top-k (optional, only for tokenizer v2).")
    parser.add_argument("--subtalker-top-p", type=float, default=None, help="Subtalker top-p (optional, only for tokenizer v2).")
    parser.add_argument(
        "--subtalker-temperature", type=float, default=None, help="Subtalker temperature (optional, only for tokenizer v2)."
    )

    return parser


def _resolve_checkpoint(args: argparse.Namespace) -> str:
    ckpt = args.checkpoint or args.checkpoint_pos
    if not ckpt:
        raise SystemExit(0)  # main() prints help
    return ckpt


def _collect_gen_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _normalize_audio(wav, eps=1e-12, clip=True):
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

        if m <= 1.0 + 1e-6:
            pass
        else:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)
    
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
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


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    wav = np.asarray(wav, dtype=np.float32)
    return sr, wav


def _detect_model_kind(ckpt: str, tts: Qwen3TTSModel) -> str:
    mt = getattr(tts.model, "tts_model_type", None)
    if mt in ("custom_voice", "voice_design", "base"):
        return mt
    else:
        raise ValueError(f"Unknown Qwen-TTS model type: {mt}")


def build_demo(tts: Qwen3TTSModel, ckpt: str, gen_kwargs_default: Dict[str, Any]) -> gr.Blocks:
    model_kind = _detect_model_kind(ckpt, tts)

    supported_langs_raw = None
    if callable(getattr(tts.model, "get_supported_languages", None)):
        supported_langs_raw = tts.model.get_supported_languages()

    supported_spks_raw = None
    if callable(getattr(tts.model, "get_supported_speakers", None)):
        supported_spks_raw = tts.model.get_supported_speakers()

    lang_choices_disp, lang_map = _build_choices_and_map([x for x in (supported_langs_raw or [])])
    spk_choices_disp, spk_map = _build_choices_and_map([x for x in (supported_spks_raw or [])])

    def _gen_common_kwargs() -> Dict[str, Any]:
        return dict(gen_kwargs_default)

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = ".gradio-container {max-width: none !important;}"

    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(
            f"""
# Qwen3 TTS デモ
**モデル:** `{ckpt}`  
**タイプ:** `{model_kind}`  
"""
        )

        if model_kind == "custom_voice":
            with gr.Row():
                with gr.Column(scale=2):
                    text_in = gr.Textbox(
                        label="テキスト（合成するテキスト）",
                        lines=4,
                        placeholder="合成するテキストを入力してください",
                    )
                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label="言語",
                            choices=lang_choices_disp,
                            value="Auto",
                            interactive=True,
                        )
                        spk_in = gr.Dropdown(
                            label="話者",
                            choices=spk_choices_disp,
                            value="Vivian",
                            interactive=True,
                        )
                    instruct_in = gr.Textbox(
                        label="指示（オプション）",
                        lines=2,
                        placeholder="例: 怒った口調で話して、悲しそうに話して",
                    )
                    btn = gr.Button("音声生成", variant="primary")
                with gr.Column(scale=3):
                    audio_out = gr.Audio(label="出力音声", type="numpy")
                    err = gr.Textbox(label="ステータス", lines=2)

            def run_instruct(text: str, lang_disp: str, spk_disp: str, instruct: str):
                try:
                    if not text or not text.strip():
                        return None, "エラー: テキストを入力してください"
                    if not spk_disp:
                        return None, "エラー: 話者を選択してください"
                    language = lang_map.get(lang_disp, "Auto")
                    speaker = spk_map.get(spk_disp, spk_disp)
                    kwargs = _gen_common_kwargs()
                    wavs, sr = tts.generate_custom_voice(
                        text=text.strip(),
                        language=language,
                        speaker=speaker,
                        instruct=(instruct or "").strip() or None,
                        **kwargs,
                    )
                    return _wav_to_gradio_audio(wavs[0], sr), "完了しました"
                except Exception as e:
                    return None, f"{type(e).__name__}: {e}"

            btn.click(run_instruct, inputs=[text_in, lang_in, spk_in, instruct_in], outputs=[audio_out, err])

        elif model_kind == "voice_design":
            with gr.Row():
                with gr.Column(scale=2):
                    text_in = gr.Textbox(
                        label="テキスト（合成するテキスト）",
                        lines=4,
                        value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
                    )
                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label="言語",
                            choices=lang_choices_disp,
                            value="Auto",
                            interactive=True,
                        )
                    design_in = gr.Textbox(
                        label="音声デザイン指示（声の特徴を記述）",
                        lines=3,
                        value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
                    )
                    btn = gr.Button("音声生成", variant="primary")
                with gr.Column(scale=3):
                    audio_out = gr.Audio(label="出力音声", type="numpy")
                    err = gr.Textbox(label="ステータス", lines=2)

            def run_voice_design(text: str, lang_disp: str, design: str):
                try:
                    if not text or not text.strip():
                        return None, "エラー: テキストを入力してください"
                    if not design or not design.strip():
                        return None, "エラー: 音声デザイン指示を入力してください"
                    language = lang_map.get(lang_disp, "Auto")
                    kwargs = _gen_common_kwargs()
                    wavs, sr = tts.generate_voice_design(
                        text=text.strip(),
                        language=language,
                        instruct=design.strip(),
                        **kwargs,
                    )
                    return _wav_to_gradio_audio(wavs[0], sr), "完了しました"
                except Exception as e:
                    return None, f"{type(e).__name__}: {e}"

            btn.click(run_voice_design, inputs=[text_in, lang_in, design_in], outputs=[audio_out, err])

        else:  # voice_clone for base
            with gr.Tabs():
                with gr.Tab("ボイスクローン"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            ref_audio = gr.Audio(
                                label="参照音声（クローン元の音声）",
                            )
                            ref_text = gr.Textbox(
                                label="参照音声のテキスト（書き起こし）",
                                lines=2,
                                placeholder="x-vectorのみモードOFF時は必須",
                            )
                            with gr.Row():
                                whisper_model = gr.Dropdown(
                                    label="Whisperモデル",
                                    choices=WHISPER_MODELS,
                                    value="small",
                                    interactive=True,
                                    scale=2,
                                )
                                transcribe_btn = gr.Button("自動文字起こし", variant="secondary", scale=1)
                            xvec_only = gr.Checkbox(
                                label="x-vectorのみモード（テキスト不要だが品質低下）",
                                value=False,
                            )

                        with gr.Column(scale=2):
                            text_in = gr.Textbox(
                                label="合成するテキスト",
                                lines=4,
                                placeholder="合成するテキストを入力してください",
                            )
                            lang_in = gr.Dropdown(
                                label="言語",
                                choices=lang_choices_disp,
                                value="Auto",
                                interactive=True,
                            )
                            btn = gr.Button("音声生成", variant="primary")

                        with gr.Column(scale=3):
                            audio_out = gr.Audio(label="出力音声", type="numpy")
                            err = gr.Textbox(label="ステータス", lines=2)

                    def run_voice_clone(ref_aud, ref_txt: str, use_xvec: bool, text: str, lang_disp: str):
                        try:
                            if not text or not text.strip():
                                return None, "エラー: 合成するテキストを入力してください"
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                return None, "エラー: 参照音声をアップロードしてください"
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                return None, (
                                    "エラー: x-vectorのみモードがOFFの場合、参照音声のテキストが必要です。\n"
                                    "テキストが不明な場合は「x-vectorのみモード」をONにしてください（品質は低下します）"
                                )
                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()
                            wavs, sr = tts.generate_voice_clone(
                                text=text.strip(),
                                language=language,
                                ref_audio=at,
                                ref_text=(ref_txt.strip() if ref_txt else None),
                                x_vector_only_mode=bool(use_xvec),
                                **kwargs,
                            )
                            return _wav_to_gradio_audio(wavs[0], sr), "完了しました"
                        except Exception as e:
                            return None, f"{type(e).__name__}: {e}"

                    btn.click(
                        run_voice_clone,
                        inputs=[ref_audio, ref_text, xvec_only, text_in, lang_in],
                        outputs=[audio_out, err],
                    )

                    def transcribe_reference_audio(audio, model_name):
                        """参照音声をWhisperで文字起こし"""
                        if audio is None:
                            return gr.update(), "エラー: 参照音声をアップロードしてください"
                        result = _transcribe_audio(audio, model_name)
                        if result.startswith("エラー"):
                            return gr.update(), result
                        return result, f"文字起こし完了（{model_name}）: {len(result)}文字"

                    transcribe_btn.click(
                        transcribe_reference_audio,
                        inputs=[ref_audio, whisper_model],
                        outputs=[ref_text, err],
                    )

                with gr.Tab("音色の保存/読み込み"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### 音色を保存
参照音声とテキストをアップロードし、再利用可能な音色ファイルとして保存します。
"""
                            )
                            ref_audio_s = gr.Audio(label="参照音声", type="numpy")
                            ref_text_s = gr.Textbox(
                                label="参照音声のテキスト",
                                lines=2,
                                placeholder="x-vectorのみモードOFF時は必須",
                            )
                            with gr.Row():
                                whisper_model_s = gr.Dropdown(
                                    label="Whisperモデル",
                                    choices=WHISPER_MODELS,
                                    value="small",
                                    interactive=True,
                                    scale=2,
                                )
                                transcribe_btn_s = gr.Button("自動文字起こし", variant="secondary", scale=1)
                            xvec_only_s = gr.Checkbox(
                                label="x-vectorのみモード（テキスト不要だが品質低下）",
                                value=False,
                            )
                            save_btn = gr.Button("音色ファイルを保存", variant="primary")
                            prompt_file_out = gr.File(label="音色ファイル")

                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### 音色を読み込んで合成
保存した音色ファイルを読み込んで、新しいテキストを合成します。
"""
                            )
                            prompt_file_in = gr.File(label="音色ファイルをアップロード")
                            text_in2 = gr.Textbox(
                                label="合成するテキスト",
                                lines=4,
                                placeholder="合成するテキストを入力してください",
                            )
                            lang_in2 = gr.Dropdown(
                                label="言語",
                                choices=lang_choices_disp,
                                value="Auto",
                                interactive=True,
                            )
                            gen_btn2 = gr.Button("音声生成", variant="primary")

                        with gr.Column(scale=3):
                            audio_out2 = gr.Audio(label="出力音声", type="numpy")
                            err2 = gr.Textbox(label="ステータス", lines=2)

                    def save_prompt(ref_aud, ref_txt: str, use_xvec: bool):
                        try:
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                return None, "エラー: 参照音声をアップロードしてください"
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                return None, (
                                    "エラー: x-vectorのみモードがOFFの場合、参照音声のテキストが必要です。\n"
                                    "テキストが不明な場合は「x-vectorのみモード」をONにしてください（品質は低下します）"
                                )
                            items = tts.create_voice_clone_prompt(
                                ref_audio=at,
                                ref_text=(ref_txt.strip() if ref_txt else None),
                                x_vector_only_mode=bool(use_xvec),
                            )
                            payload = {
                                "items": [asdict(it) for it in items],
                            }
                            fd, out_path = tempfile.mkstemp(prefix="voice_clone_prompt_", suffix=".pt")
                            os.close(fd)
                            torch.save(payload, out_path)
                            return out_path, "完了しました"
                        except Exception as e:
                            return None, f"{type(e).__name__}: {e}"

                    def load_prompt_and_gen(file_obj, text: str, lang_disp: str):
                        try:
                            if file_obj is None:
                                return None, "エラー: 音色ファイルをアップロードしてください"
                            if not text or not text.strip():
                                return None, "エラー: 合成するテキストを入力してください"

                            path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or str(file_obj)
                            payload = torch.load(path, map_location="cpu", weights_only=True)
                            if not isinstance(payload, dict) or "items" not in payload:
                                return None, "エラー: ファイル形式が正しくありません"

                            items_raw = payload["items"]
                            if not isinstance(items_raw, list) or len(items_raw) == 0:
                                return None, "エラー: 音色データが空です"

                            items: List[VoiceClonePromptItem] = []
                            for d in items_raw:
                                if not isinstance(d, dict):
                                    return None, "エラー: ファイル内部の形式が正しくありません"
                                ref_code = d.get("ref_code", None)
                                if ref_code is not None and not torch.is_tensor(ref_code):
                                    ref_code = torch.tensor(ref_code)
                                ref_spk = d.get("ref_spk_embedding", None)
                                if ref_spk is None:
                                    return None, "エラー: 話者ベクトルがありません"
                                if not torch.is_tensor(ref_spk):
                                    ref_spk = torch.tensor(ref_spk)

                                items.append(
                                    VoiceClonePromptItem(
                                        ref_code=ref_code,
                                        ref_spk_embedding=ref_spk,
                                        x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                                        icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                                        ref_text=d.get("ref_text", None),
                                    )
                                )

                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()
                            wavs, sr = tts.generate_voice_clone(
                                text=text.strip(),
                                language=language,
                                voice_clone_prompt=items,
                                **kwargs,
                            )
                            return _wav_to_gradio_audio(wavs[0], sr), "完了しました"
                        except Exception as e:
                            return None, (
                                f"エラー: 音色ファイルの読み込みまたは使用に失敗しました。\n"
                                f"ファイル形式や内容を確認してください。\n"
                                f"{type(e).__name__}: {e}"
                            )

                    save_btn.click(save_prompt, inputs=[ref_audio_s, ref_text_s, xvec_only_s], outputs=[prompt_file_out, err2])
                    gen_btn2.click(load_prompt_and_gen, inputs=[prompt_file_in, text_in2, lang_in2], outputs=[audio_out2, err2])

                    def transcribe_reference_audio_s(audio, model_name):
                        """参照音声をWhisperで文字起こし（保存タブ用）"""
                        if audio is None:
                            return gr.update(), "エラー: 参照音声をアップロードしてください"
                        result = _transcribe_audio(audio, model_name)
                        if result.startswith("エラー"):
                            return gr.update(), result
                        return result, f"文字起こし完了（{model_name}）: {len(result)}文字"

                    transcribe_btn_s.click(
                        transcribe_reference_audio_s,
                        inputs=[ref_audio_s, whisper_model_s],
                        outputs=[ref_text_s, err2],
                    )

        gr.Markdown(
            """
**免責事項**  
- この音声はAIモデルによって自動生成/合成されたものであり、モデルの機能を示すためのデモンストレーション目的でのみ提供されています。不正確または不適切な内容が含まれる場合があります。この音声は開発者/運営者の見解を代表するものではなく、専門的なアドバイスを構成するものでもありません。
- ユーザーは、この音声の評価、使用、配布、または依拠に関するすべてのリスクと責任を自ら負うものとします。適用法が許容する最大限の範囲において、開発者/運営者は、この音声の使用または使用不能から生じる直接的、間接的、偶発的、または結果的な損害について責任を負いません（法律で免責が認められない場合を除く）。
- 本サービスを使用して、違法、有害、名誉毀損、詐欺、ディープフェイク、プライバシー/肖像権/著作権/商標を侵害するコンテンツを意図的に生成または複製することは禁止されています。ユーザーがプロンプト、素材、その他の手段によって違法または侵害行為を実施または促進した場合、その法的責任はすべてユーザーが負い、開発者/運営者は一切の責任を負いません。
"""
        )

    return demo


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.checkpoint and not args.checkpoint_pos:
        parser.print_help()
        return 0

    ckpt = _resolve_checkpoint(args)

    dtype = _dtype_from_str(args.dtype)
    attn_impl = "flash_attention_2" if args.flash_attn else None

    tts = Qwen3TTSModel.from_pretrained(
        ckpt,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    gen_kwargs_default = _collect_gen_kwargs(args)
    demo = build_demo(tts, ckpt, gen_kwargs_default)

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        ssl_verify=True if args.ssl_verify else False,
    )
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
