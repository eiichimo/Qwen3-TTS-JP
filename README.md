# Qwen3-TTS-JP

**English** | [日本語](docs/README_ja.md) | [中文](docs/README_zh.md) | [한국어](docs/README_ko.md) | [Русский](docs/README_ru.md) | [Español](docs/README_es.md) | [Italiano](docs/README_it.md) | [Deutsch](docs/README_de.md) | [Français](docs/README_fr.md) | [Português](docs/README_pt.md)

A **Windows-native** fork of Qwen3-TTS with a modern, multilingual Web UI.

The original Qwen3-TTS was developed primarily for Linux environments, and FlashAttention 2 is recommended. However, FlashAttention 2 does not work on Windows. This fork enables **direct execution on Windows without WSL2 or Docker**, provides a **modern Web UI supporting 10 languages**, and adds automatic transcription via Whisper.

> **Mac (Apple Silicon) users:** For the best experience on Mac, please use **[Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab)** -- fully optimized for Apple Silicon with MLX + PyTorch dual engine, 8bit/4bit quantization, and 10-language Web UI.

### Custom Voice -- Speech synthesis with preset speakers
<p align="center">
    <img src="assets/CustomVoice.png" width="90%"/>
</p>

### Voice Design -- Describe voice characteristics to synthesize
<p align="center">
    <img src="assets/VoiceDesign.png" width="90%"/>
</p>

### Voice Clone -- Clone voice from reference audio
<p align="center">
    <img src="assets/VoiceClone.png" width="90%"/>
</p>

### Settings -- GPU / VRAM / Model information
<p align="center">
    <img src="assets/Settings.png" width="90%"/>
</p>

## Related Projects

| Platform | Repository | Description |
|:---:|---|---|
| Windows | **This repository** | Windows-native + multilingual Web UI |
| macOS (Apple Silicon) | [Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab) | Fully optimized for Apple Silicon Mac (MLX + PyTorch dual engine, 10-language Web UI) |

## Engineering Workflow

This repository adopts operational rules from `engineering-playbook`.

- Adoption record: [docs/playbook.md](docs/playbook.md)
- Development workflow: [docs/dev-workflow.md](docs/dev-workflow.md)
- GitHub workflow/Issue-PR linking rules: [docs/github-workflow.md](docs/github-workflow.md)

## Features

### Windows Native Support

- **No FlashAttention 2 required**: Uses PyTorch's standard SDPA (Scaled Dot Product Attention) via the `--no-flash-attn` option
- **No WSL2/Docker required**: Runs directly on Windows
- **RTX 50 series support**: Includes instructions for installing PyTorch nightly builds for NVIDIA Blackwell architecture (sm_120)
- **SoX dependency avoided**: Works without SoX (warnings are displayed but can be safely ignored)

### Modern Web UI & Multilingual Support

- **10-language UI**: Japanese / English / Chinese / Korean / Russian / Spanish / Italian / German / French / Portuguese -- switch instantly via dropdown
- **4-tab layout**: Custom Voice / Voice Design / Voice Clone / Settings -- access all features regardless of model type; unloaded models are downloaded automatically on first use
- **GPU / VRAM monitoring**: Check real-time usage in the Settings tab; CUDA cache clearing also available
- **Whisper automatic transcription**: Automates reference audio text input for voice cloning (uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper))
- **Whisper model selection**: Choose from 5 models depending on your needs
  - `tiny` - Fastest & smallest (39M parameters)
  - `base` - Fast (74M parameters)
  - `small` - Balanced (244M parameters) *Default
  - `medium` - High accuracy (769M parameters)
  - `large-v3` - Highest accuracy (1550M parameters)

## System Requirements

- **OS**: Windows 10/11 (native environment, no WSL2 required)
- **GPU**: NVIDIA GPU (CUDA compatible)
  - RTX 30/40 series: Works with stable PyTorch
  - RTX 50 series (Blackwell): Requires PyTorch nightly build (cu128)
- **Python**: 3.10 or higher
- **VRAM**: 8GB or more recommended (varies by model size)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-JP.git
cd Qwen3-TTS-JP
```

### 2. Create and Activate Virtual Environment (PowerShell)

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
```

If activation is blocked by ExecutionPolicy, run:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install PyTorch (CUDA Version) First

Install according to your GPU/CUDA requirements.

```bash
# RTX 30/40 series (CUDA 12.x)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# RTX 50 series (Blackwell / sm_120): nightly build required
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 4. Install Project Dependencies

```bash
python -m pip install -e .
python -m pip install faster-whisper
```

### 5. Poetry (Optional)

Poetry can be used, but this project still needs manual PyTorch wheel selection
for Windows GPU environments. For most users, `venv + pip` is simpler.

```bash
poetry env use 3.11
poetry run python -m pip install -U pip setuptools wheel

# Install GPU-matched PyTorch wheels first
poetry run python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# or (RTX 50 series)
# poetry run python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

poetry install
poetry run python -m pip install faster-whisper
```

## Usage

### Starting the GUI

#### From the Command Line

```bash
# CustomVoice model (preset speakers)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn

# Base model (with voice cloning)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
```

Open `http://127.0.0.1:7860` in your browser.

#### Quick Launch with Batch Files (Recommended)

Create a batch file like the following for double-click launching:

**run-gui.bat** (for CustomVoice model):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS GUI
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

**run-voice-clone.bat** (for Base model / voice cloning):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS Voice Clone
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

#### Advanced Launcher (Auto Port Selection & Auto Browser Launch)

For a more convenient launch method, you can use the following Python launcher:

<details>
<summary>launch_gui.py (click to expand)</summary>

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
    """Find an available port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"All ports {start_port}-{start_port + max_attempts} are in use")

def wait_for_server_and_open_browser(url, timeout=180):
    """Wait for server startup then open browser"""
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
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # or CustomVoice
        "--ip", "127.0.0.1",
        "--port", str(port),
        "--no-flash-attn"
    ])

if __name__ == "__main__":
    main()
```

</details>

Features:
- **Auto port selection**: Automatically detects a free port if 7860 is in use
- **Auto browser launch**: Detects server startup completion and automatically opens the browser
- **Character encoding fix**: UTF-8 encoding support

### Voice Cloning Steps

1. Upload an audio file to "Reference Audio"
2. Select a model under "Whisper Model" (first-time download may take some time)
3. Click "Auto Transcribe"
4. The transcription result is automatically entered in "Reference Audio Text"
5. Edit the text if necessary
6. Enter the "Text to Synthesize"
7. Click "Generate Audio"

### Windows Native Support Details

This fork achieves Windows-native operation through the following measures:

| Issue | Original | This Fork's Solution |
|-------|----------|---------------------|
| FlashAttention 2 | Linux-only, cannot build on Windows | Use SDPA via `--no-flash-attn` option |
| SoX dependency | Assumes installation | Works without it (warnings can be ignored) |
| RTX 50 series | Not supported | Nightly build instructions included |
| Environment setup | conda (Linux-oriented) | venv (Windows standard) |

**Note**: The `--no-flash-attn` option is required. Without it, the application will fail to start with a FlashAttention 2 import error.

## Detailed Windows Native Environment Setup

### Resolved Technical Issues

During the development of this fork, the following Windows-specific issues were identified and resolved:

#### 1. CUDA Support for RTX 50 Series (Blackwell/sm_120)

**Problem**: Stable PyTorch does not support the latest GPUs like RTX 5090 (sm_120)

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

**Solution**: Use PyTorch nightly (cu128)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2. FlashAttention 2 Not Supported on Windows

**Problem**: FlashAttention 2 is Linux-only and cannot be built or run on Windows

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: 
the package flash_attn seems to be not installed.
```

**Solution**: Use PyTorch's standard SDPA (Scaled Dot Product Attention) via the `--no-flash-attn` option

| Attention Implementation | Speed | Memory Efficiency | Windows Support |
|--------------------------|-------|-------------------|-----------------|
| flash_attention_2 | Fastest | Best | Not supported |
| sdpa (PyTorch native) | Fast | Good | **Supported** |
| eager (standard) | Normal | Normal | Supported |

#### 3. SoX Dependency Avoidance

**Problem**: Some audio processing requires SoX, but it is not installed by default on Windows

```
SoX could not be found!
```

**Solution**: Qwen3-TTS core functionality works without SoX. Warnings can be safely ignored.

#### 4. Console Character Encoding (cp932 Encoding)

**Problem**: In Japanese Windows environments, non-ASCII characters are garbled due to cp932 encoding

```
UnicodeEncodeError: 'cp932' codec can't encode character...
```

**Solution**: Explicitly set UTF-8 encoding

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

Or run `chcp 65001` in the batch file

#### 5. torchao Compatibility Warning

**Problem**: Version mismatch warning between PyTorch nightly and torchao

```
Skipping import of cpp extensions due to incompatible torch version 2.11.0.dev+cu128 for torchao version 0.15.0
```

**Solution**: Warning only, no impact on operation. Can be safely ignored.

#### 6. Hugging Face Symlink Warning

**Problem**: Creating symbolic links on Windows requires administrator privileges

```
huggingface_hub cache-system uses symlinks by default...
```

**Solution**: 
- Enable Developer Mode in Windows Settings
- Or ignore the warning (no impact on operation)

### Verification Script

To verify that the environment is set up correctly:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

Expected output (for RTX 5090):

```
PyTorch version: 2.11.0.dev20260123+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5090
GPU Memory: 31.8 GB
```

### Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| `no kernel image is available` | Using stable PyTorch | Install nightly (cu128) |
| `FlashAttention2 cannot be used` | FlashAttention not supported on Windows | Add `--no-flash-attn` option |
| `SoX could not be found` | SoX not installed | Can be ignored (no impact on core functionality) |
| GPU not recognized | CUDA driver outdated | Install latest driver |
| Character garbling | cp932 encoding | `chcp 65001` or UTF-8 setting |

## License

This project is released under the [Apache License 2.0](LICENSE).

### Open Source Software Used

| Software | License | Copyright |
|----------|---------|-----------|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache License 2.0 | Copyright 2026 Alibaba Cloud |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT License | Copyright SYSTRAN |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT License | Copyright OpenAI |

For details, see the [NOTICE](NOTICE) file.

## Disclaimer

### Disclaimer Regarding Audio Generation

- The audio generated by this system is automatically produced by an AI model and may contain inaccurate or inappropriate content
- Generated audio does not represent the views of the developers and does not constitute professional advice
- Users assume all risks and responsibilities related to the use, distribution, or reliance on generated audio

### Voice Cloning Warning

- **Cloning or using another person's voice without their consent may constitute a violation of portrait rights and publicity rights**
- Please use the voice cloning feature only for lawful purposes with the consent of the person whose voice is being cloned
- Use for malicious purposes such as fraud, impersonation, defamation, or deepfakes is strictly prohibited

### Legal Liability

- The developers assume no liability for any damages arising from the use of this software
- All legal liability arising from illegal use shall be borne by the user
- This software is provided "as is" without any warranty

## Acknowledgments

- Original developer: [Alibaba Cloud Qwen Team](https://github.com/QwenLM)
- Original repository: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

## Citation

To cite the original Qwen3-TTS:

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```
