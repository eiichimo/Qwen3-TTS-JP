# Qwen3-TTS-JP

**Windowsネイティブ対応** の Qwen3-TTS 日本語ローカライズ版フォークです。

オリジナルのQwen3-TTSはLinux環境を前提として開発されており、FlashAttention 2の使用が推奨されていますが、FlashAttention 2はWindowsでは動作しません。本フォークでは、**WSL2やDockerを使わずにWindows上で直接動作**させるための対応と、GUIの完全日本語化、Whisperによる自動文字起こし機能を追加しています。

<p align="center">
    <img src="assets/GUI.png" width="90%"/>
</p>

## 特徴

### Windowsネイティブ対応

- **FlashAttention 2不要**: `--no-flash-attn`オプションによりPyTorch標準のSDPA（Scaled Dot Product Attention）を使用
- **WSL2/Docker不要**: Windows上で直接実行可能
- **RTX 50シリーズ対応**: NVIDIA Blackwellアーキテクチャ（sm_120）用PyTorch nightlyビルドの導入手順を記載
- **SoX依存の回避**: SoXがなくても動作（警告は表示されますが無視可能）

### 日本語ローカライズ & 機能拡張

- **GUIの完全日本語化**: ラベル、ボタン、プレースホルダー、エラーメッセージ、免責事項すべてを日本語化
- **Whisper自動文字起こし機能**: ボイスクローン時の参照音声テキスト入力を自動化（[faster-whisper](https://github.com/SYSTRAN/faster-whisper) を使用）
- **Whisperモデル選択機能**: 用途に応じて5種類のモデルから選択可能
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

```bash
# CustomVoiceモデル（プリセット話者）
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn

# Baseモデル（ボイスクローン機能付き）
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
```

ブラウザで `http://127.0.0.1:7860` を開いてください。

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

## ライセンス

本プロジェクトは [Apache License 2.0](LICENSE) の下で公開されています。

### 使用しているオープンソースソフトウェア

| ソフトウェア | ライセンス | 著作権 |
|------------|-----------|--------|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache License 2.0 | Copyright 2026 Alibaba Cloud |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT License | Copyright SYSTRAN |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT License | Copyright OpenAI |

詳細は [NOTICE](NOTICE) ファイルを参照してください。

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
