[English](../README.md) | **日本語** | [中文](README_zh.md) | [한국어](README_ko.md) | [Русский](README_ru.md) | [Español](README_es.md) | [Italiano](README_it.md) | [Deutsch](README_de.md) | [Français](README_fr.md) | [Português](README_pt.md)

# Qwen3-TTS-JP

**Windowsネイティブ対応** の Qwen3-TTS ローカライズ版フォークです。

オリジナルのQwen3-TTSはLinux環境を前提として開発されており、FlashAttention 2の使用が推奨されていますが、FlashAttention 2はWindowsでは動作しません。本フォークでは、**WSL2やDockerを使わずにWindows上で直接動作**させるための対応と、**10言語対応のモダンなWeb UI**、Whisperによる自動文字起こし機能を追加しています。

> **Mac (Apple Silicon) をお使いの方:** Mac向けに最適化された **[Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab)** をご利用ください。MLX + PyTorch デュアルエンジン、8bit/4bit量子化、10言語Web UIに対応しています。

### Custom Voice -- プリセット話者による音声合成
<p align="center">
    <img src="../assets/CustomVoice.png" width="90%"/>
</p>

### Voice Design -- テキストで声の特徴を記述して合成
<p align="center">
    <img src="../assets/VoiceDesign.png" width="90%"/>
</p>

### Voice Clone -- 参照音声からボイスクローン
<p align="center">
    <img src="../assets/VoiceClone.png" width="90%"/>
</p>

### Settings -- GPU / VRAM / モデル情報
<p align="center">
    <img src="../assets/Settings.png" width="90%"/>
</p>

## 関連プロジェクト

| プラットフォーム | リポジトリ | 説明 |
|:---:|---|---|
| Windows | **本リポジトリ** | Windowsネイティブ対応 + 多言語Web UI |
| macOS (Apple Silicon) | [Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab) | Apple Silicon Mac完全最適化版（MLX + PyTorch デュアルエンジン、10言語Web UI） |

## 特徴

### Windowsネイティブ対応

- **FlashAttention 2不要**: `--no-flash-attn`オプションによりPyTorch標準のSDPA（Scaled Dot Product Attention）を使用
- **WSL2/Docker不要**: Windows上で直接実行可能
- **RTX 50シリーズ対応**: NVIDIA Blackwellアーキテクチャ（sm_120）用PyTorch nightlyビルドの導入手順を記載
- **SoX依存の回避**: SoXがなくても動作（警告は表示されますが無視可能）

### モダンWeb UI & 多言語対応

- **10言語対応UI**: 日本語 / English / 中文 / 한국어 / Русский / Espanol / Italiano / Deutsch / Francais / Portugues をドロップダウンで即時切り替え
- **4タブ構成**: Custom Voice / Voice Design / Voice Clone / Settings -- モデルタイプに関わらず全機能にアクセス可能。未ロードのモデルは初回使用時に自動ダウンロード
- **GPU / VRAMモニタリング**: Settingsタブでリアルタイムに使用状況を確認、CUDAキャッシュクリアも可能
- **Whisper自動文字起こし**: ボイスクローン時の参照音声テキスト入力を自動化（[faster-whisper](https://github.com/SYSTRAN/faster-whisper)）
- **Whisperモデル選択**: 用途に応じて5種類から選択可能
  - `tiny` - 最速・最小（39M パラメータ）
  - `base` - 高速（74M パラメータ）
  - `small` - バランス型（244M パラメータ）※デフォルト
  - `medium` - 高精度（769M パラメータ）
  - `large-v3` - 最高精度（1550M パラメータ）

## 動作環境

- **OS**: Windows 10/11（ネイティブ環境、WSL2不要）
- **GPU**: NVIDIA GPU（CUDA対応）
  - RTX 30/40シリーズ: PyTorch安定版で動作
  - RTX 50シリーズ（Blackwell）: PyTorch nightlyビルド（cu128）が必要
- **Python**: 3.10以上
- **VRAM**: 8GB以上推奨（モデルサイズにより異なる）

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-JP.git
cd Qwen3-TTS-JP
```

### 2. 仮想環境の作成と有効化

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. 依存パッケージのインストール

```bash
pip install -e .
pip install faster-whisper
```

### 4. PyTorch（CUDA対応版）のインストール

お使いのCUDAバージョンに合わせてインストールしてください。

```bash
# CUDA 12.x の場合
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# RTX 50シリーズ（sm_120）の場合はnightlyビルドが必要
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 使用方法

### GUIの起動

#### コマンドラインから起動

```bash
# CustomVoiceモデル（プリセット話者）
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn

# Baseモデル（ボイスクローン機能付き）
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
```

ブラウザで `http://127.0.0.1:7860` を開いてください。

#### バッチファイルで簡単起動（推奨）

以下のようなバッチファイルを作成すると、ダブルクリックで起動できます：

**run-gui.bat**（CustomVoiceモデル用）:
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS GUI
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

**run-voice-clone.bat**（Baseモデル/ボイスクローン用）:
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS Voice Clone
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

#### 高機能ランチャー（ポート自動選択・ブラウザ自動起動）

より便利な起動方法として、以下のPythonランチャーを使用できます：

<details>
<summary>launch_gui.py（クリックで展開）</summary>

```python
# coding=utf-8
import socket
import subprocess
import sys
import time
import webbrowser
import threading
import urllib.request
import urllib.error

def find_free_port(start_port=7860, max_attempts=100):
    """空いているポートを見つける"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"ポート {start_port}-{start_port + max_attempts} は全て使用中です")

def wait_for_server_and_open_browser(url, timeout=180):
    """サーバー起動を待ってブラウザを開く"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, method='HEAD')
            urllib.request.urlopen(req, timeout=2)
            webbrowser.open(url)
            return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            time.sleep(2)
    return False

def main():
    port = find_free_port(7860)
    url = f"http://127.0.0.1:{port}"
    
    threading.Thread(target=wait_for_server_and_open_browser, args=(url, 180), daemon=True).start()
    
    subprocess.run([
        sys.executable, "-m", "qwen_tts.cli.demo",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # または CustomVoice
        "--ip", "127.0.0.1",
        "--port", str(port),
        "--no-flash-attn"
    ])

if __name__ == "__main__":
    main()
```

</details>

機能:
- **ポート自動選択**: 7860が使用中なら自動で空きポートを検出
- **ブラウザ自動起動**: サーバー起動完了を検知して自動でブラウザを開く
- **文字化け対策**: UTF-8エンコーディング対応

### ボイスクローンの手順

1. 「参照音声」に音声ファイルをアップロード
2. 「Whisperモデル」でモデルを選択（初回はダウンロードに時間がかかります）
3. 「自動文字起こし」ボタンをクリック
4. 文字起こし結果が「参照音声のテキスト」に自動入力される
5. 必要に応じてテキストを修正
6. 「合成するテキスト」を入力
7. 「音声生成」をクリック

### Windowsネイティブ対応のポイント

本フォークでは以下の対応により、Windowsネイティブ環境での動作を実現しています：

| 問題 | オリジナル | 本フォークの対応 |
|------|-----------|-----------------|
| FlashAttention 2 | Linux専用、Windowsでビルド不可 | `--no-flash-attn`オプションでSDPA使用 |
| SoX依存 | インストール必須の想定 | なくても動作（警告は無視可能） |
| RTX 50シリーズ | 未対応 | PyTorch nightlyビルド手順を記載 |
| 環境構築 | conda（Linux寄り） | venv（Windows標準） |

**注意**: `--no-flash-attn`オプションは必須です。これがないとFlashAttention 2のインポートエラーで起動に失敗します。

## Windowsネイティブ環境構築の詳細

### 解決した技術的課題

本フォークの開発過程で、以下のWindows固有の問題を特定し解決しました：

#### 1. RTX 50シリーズ（Blackwell/sm_120）のCUDA対応

**問題**: PyTorch安定版はRTX 5090などの最新GPU（sm_120）をサポートしていない

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

**解決策**: PyTorch nightly版（cu128）を使用

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2. FlashAttention 2のWindows非対応

**問題**: FlashAttention 2はLinux専用でWindowsではビルド・実行不可

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: 
the package flash_attn seems to be not installed.
```

**解決策**: `--no-flash-attn`オプションでPyTorch標準のSDPA（Scaled Dot Product Attention）を使用

| Attention実装 | 速度 | メモリ効率 | Windows対応 |
|--------------|------|-----------|-------------|
| flash_attention_2 | 最速 | 最高 | 非対応 |
| sdpa (PyTorch native) | 高速 | 良好 | **対応** |
| eager (標準) | 普通 | 普通 | 対応 |

#### 3. SoX依存の回避

**問題**: 一部の音声処理でSoXが必要だが、Windowsでは標準でインストールされていない

```
SoX could not be found!
```

**解決策**: Qwen3-TTSの基本機能はSoXなしで動作。警告は無視可能

#### 4. コンソール文字化け（cp932エンコーディング）

**問題**: Windows日本語環境ではcp932エンコーディングにより非ASCII文字が化ける

```
UnicodeEncodeError: 'cp932' codec can't encode character...
```

**解決策**: UTF-8エンコーディングを明示的に設定

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

または、バッチファイルで `chcp 65001` を実行

#### 5. torchao互換性警告

**問題**: PyTorch nightlyとtorchaoのバージョン不整合による警告

```
Skipping import of cpp extensions due to incompatible torch version 2.11.0.dev+cu128 for torchao version 0.15.0
```

**解決策**: 警告のみで動作に影響なし。無視可能

#### 6. Hugging Face symlink警告

**問題**: Windowsではシンボリックリンク作成に管理者権限が必要

```
huggingface_hub cache-system uses symlinks by default...
```

**解決策**: 
- Windows設定 → 開発者モードを有効化
- または警告を無視（動作に影響なし）

### 動作確認スクリプト

環境が正しくセットアップされているか確認するには：

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

期待される出力（RTX 5090の場合）：

```
PyTorch version: 2.11.0.dev20260123+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5090
GPU Memory: 31.8 GB
```

### トラブルシューティング

| 症状 | 原因 | 解決策 |
|------|------|--------|
| `no kernel image is available` | PyTorch安定版を使用 | nightly版（cu128）をインストール |
| `FlashAttention2 cannot be used` | FlashAttentionがWindows非対応 | `--no-flash-attn`オプションを追加 |
| `SoX could not be found` | SoX未インストール | 無視可能（基本機能に影響なし） |
| GPU認識されない | CUDAドライバ古い | 最新ドライバをインストール |
| 文字化け | cp932エンコーディング | `chcp 65001`またはUTF-8設定 |

## ライセンス

本プロジェクトは [Apache License 2.0](../LICENSE) の下で公開されています。

### 使用しているオープンソースソフトウェア

| ソフトウェア | ライセンス | 著作権 |
|------------|-----------|--------|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache License 2.0 | Copyright 2026 Alibaba Cloud |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT License | Copyright SYSTRAN |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT License | Copyright OpenAI |

詳細は [NOTICE](../NOTICE) ファイルを参照してください。

## 免責事項

### 音声生成に関する免責

- この音声はAIモデルによって自動生成されたものであり、不正確または不適切な内容が含まれる場合があります
- 生成された音声は開発者の見解を代表するものではなく、専門的なアドバイスを構成するものでもありません
- ユーザーは、生成音声の使用、配布、または依拠に関するすべてのリスクと責任を自ら負うものとします

### ボイスクローンに関する警告

- **他者の声を無断で複製・使用することは、肖像権・パブリシティ権の侵害となる可能性があります**
- ボイスクローン機能は、本人の同意を得た上で、合法的な目的にのみ使用してください
- 詐欺、なりすまし、名誉毀損、ディープフェイクなどの悪意ある目的での使用は固く禁じます

### 法的責任

- 本ソフトウェアの使用によって生じたいかなる損害についても、開発者は責任を負いません
- 違法な使用によって生じた法的責任は、すべてユーザーが負うものとします
- 本ソフトウェアは「現状のまま」提供され、いかなる保証も行いません

## 謝辞

- オリジナル開発元: [Alibaba Cloud Qwen Team](https://github.com/QwenLM)
- オリジナルリポジトリ: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

## 引用

オリジナルのQwen3-TTSを引用する場合：

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```
