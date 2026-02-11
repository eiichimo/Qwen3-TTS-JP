[English](../README.md) | [日本語](README_ja.md) | [中文](README_zh.md) | **한국어** | [Русский](README_ru.md) | [Español](README_es.md) | [Italiano](README_it.md) | [Deutsch](README_de.md) | [Français](README_fr.md) | [Português](README_pt.md)

# Qwen3-TTS-JP

**Windows 네이티브 지원** Qwen3-TTS 다국어 Web UI 포크입니다.

원본 Qwen3-TTS는 Linux 환경을 기반으로 개발되었으며, FlashAttention 2 사용이 권장됩니다. 하지만 FlashAttention 2는 Windows에서 동작하지 않습니다. 본 포크에서는 **WSL2나 Docker 없이 Windows에서 직접 실행**할 수 있도록 대응하였으며, GUI의 완전한 일본어화와 Whisper를 통한 자동 음성 인식 기능을 추가하였습니다.

> **Mac (Apple Silicon) 사용자:** Mac에서 최적의 경험을 원하시면 **[Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab)** 을 이용해 주세요. Apple Silicon 완전 최적화, MLX + PyTorch 듀얼 엔진, 8bit/4bit 양자화, 10개 언어 Web UI를 지원합니다.

### 커스텀 음성 -- 프리셋 화자로 음성 합성
<p align="center">
    <img src="../assets/CustomVoice.png" width="90%"/>
</p>

### 음성 디자인 -- 음성 특성을 설명하여 합성
<p align="center">
    <img src="../assets/VoiceDesign.png" width="90%"/>
</p>

### 음성 클론 -- 참조 오디오에서 음성 복제
<p align="center">
    <img src="../assets/VoiceClone.png" width="90%"/>
</p>

### 설정 -- GPU / VRAM / 모델 정보
<p align="center">
    <img src="../assets/Settings.png" width="90%"/>
</p>

## 관련 프로젝트

| 플랫폼 | 리포지토리 | 설명 |
|:---:|---|---|
| Windows | **본 리포지토리** | Windows 네이티브 지원 + 다국어 Web UI |
| macOS (Apple Silicon) | [Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab) | Apple Silicon Mac 완전 최적화 버전 (MLX + PyTorch 듀얼 엔진, 10개 언어 Web UI) |

## 특징

### Windows 네이티브 지원

- **FlashAttention 2 불필요**: `--no-flash-attn` 옵션으로 PyTorch 표준 SDPA (Scaled Dot Product Attention) 사용
- **WSL2/Docker 불필요**: Windows에서 직접 실행 가능
- **RTX 50 시리즈 지원**: NVIDIA Blackwell 아키텍처 (sm_120)용 PyTorch nightly 빌드 설치 가이드 포함
- **SoX 의존성 회피**: SoX 없이도 동작 (경고가 표시되지만 무시 가능)

### 모던 Web UI & 다국어 지원

- **10개 언어 UI**: 일본어 / English / 중문 / 한국어 / Русский / Español / Italiano / Deutsch / Français / Português -- 드롭다운으로 즉시 전환
- **4탭 구성**: Custom Voice / Voice Design / Voice Clone / Settings -- 모델 유형에 관계없이 모든 기능에 접근 가능; 미로드 모델은 처음 사용 시 자동 다운로드
- **GPU / VRAM 모니터링**: 설정 탭에서 실시간 사용량 확인, CUDA 캐시 클리어도 가능
- **Whisper 자동 음성 인식**: 보이스 클론 시 참조 오디오 텍스트 입력을 자동화 ([faster-whisper](https://github.com/SYSTRAN/faster-whisper) 사용)
- **Whisper 모델 선택**: 용도에 따라 5종류에서 선택 가능
  - `tiny` - 가장 빠름 & 최소 (39M 파라미터)
  - `base` - 빠름 (74M 파라미터)
  - `small` - 균형형 (244M 파라미터) ※기본값
  - `medium` - 고정밀 (769M 파라미터)
  - `large-v3` - 최고 정밀도 (1550M 파라미터)

## 시스템 요구사항

- **OS**: Windows 10/11 (네이티브 환경, WSL2 불필요)
- **GPU**: NVIDIA GPU (CUDA 지원)
  - RTX 30/40 시리즈: PyTorch 안정 버전으로 동작
  - RTX 50 시리즈 (Blackwell): PyTorch nightly 빌드 (cu128) 필요
- **Python**: 3.10 이상
- **VRAM**: 8GB 이상 권장 (모델 크기에 따라 다름)

## 설치

### 1. 리포지토리 클론

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-JP.git
cd Qwen3-TTS-JP
```

### 2. 가상 환경 생성 및 활성화

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. 의존 패키지 설치

```bash
pip install -e .
pip install faster-whisper
```

### 4. PyTorch (CUDA 버전) 설치

사용 중인 CUDA 버전에 맞게 설치하세요.

```bash
# CUDA 12.x의 경우
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# RTX 50 시리즈 (sm_120)의 경우 nightly 빌드 필요
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 사용법

### GUI 시작

#### 커맨드 라인에서 시작

```bash
# CustomVoice 모델 (프리셋 화자)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn

# Base 모델 (보이스 클론 기능 포함)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
```

브라우저에서 `http://127.0.0.1:7860`을 여세요.

#### 배치 파일로 간편 시작 (권장)

다음과 같은 배치 파일을 만들면 더블 클릭으로 시작할 수 있습니다:

**run-gui.bat** (CustomVoice 모델용):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS GUI
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

**run-voice-clone.bat** (Base 모델/보이스 클론용):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS Voice Clone
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

#### 고급 런처 (자동 포트 선택 및 브라우저 자동 실행)

보다 편리한 시작 방법으로 다음 Python 런처를 사용할 수 있습니다:

<details>
<summary>launch_gui.py (클릭하여 펼치기)</summary>

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
    """사용 가능한 포트 찾기"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"포트 {start_port}-{start_port + max_attempts}이(가) 모두 사용 중입니다")

def wait_for_server_and_open_browser(url, timeout=180):
    """서버 시작을 기다린 후 브라우저 열기"""
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
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # 또는 CustomVoice
        "--ip", "127.0.0.1",
        "--port", str(port),
        "--no-flash-attn"
    ])

if __name__ == "__main__":
    main()
```

</details>

기능:
- **자동 포트 선택**: 7860이 사용 중이면 자동으로 빈 포트를 감지
- **브라우저 자동 실행**: 서버 시작 완료를 감지하여 자동으로 브라우저를 열기
- **문자 깨짐 방지**: UTF-8 인코딩 지원

### 보이스 클론 절차

1. "참조 음성"에 오디오 파일 업로드
2. "Whisper 모델"에서 모델 선택 (첫 다운로드 시 시간이 소요됩니다)
3. "자동 음성 인식" 버튼 클릭
4. 음성 인식 결과가 "참조 음성 텍스트"에 자동 입력됨
5. 필요에 따라 텍스트 수정
6. "합성할 텍스트" 입력
7. "음성 생성" 클릭

### Windows 네이티브 지원 상세

본 포크에서는 다음과 같은 대응으로 Windows 네이티브 환경에서의 동작을 실현하였습니다:

| 문제 | 원본 | 본 포크의 대응 |
|------|------|---------------|
| FlashAttention 2 | Linux 전용, Windows에서 빌드 불가 | `--no-flash-attn` 옵션으로 SDPA 사용 |
| SoX 의존성 | 설치 필수 가정 | 없이도 동작 (경고 무시 가능) |
| RTX 50 시리즈 | 미지원 | PyTorch nightly 빌드 설치 가이드 포함 |
| 환경 구축 | conda (Linux 지향) | venv (Windows 표준) |

**주의**: `--no-flash-attn` 옵션은 필수입니다. 이 옵션이 없으면 FlashAttention 2 임포트 에러로 시작에 실패합니다.

## Windows 네이티브 환경 구축 상세

### 해결한 기술적 과제

본 포크 개발 과정에서 다음과 같은 Windows 고유의 문제를 식별하고 해결하였습니다:

#### 1. RTX 50 시리즈 (Blackwell/sm_120)의 CUDA 지원

**문제**: PyTorch 안정 버전은 RTX 5090 등 최신 GPU (sm_120)를 지원하지 않음

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

**해결책**: PyTorch nightly 버전 (cu128) 사용

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2. FlashAttention 2의 Windows 미지원

**문제**: FlashAttention 2는 Linux 전용으로 Windows에서 빌드 및 실행 불가

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: 
the package flash_attn seems to be not installed.
```

**해결책**: `--no-flash-attn` 옵션으로 PyTorch 표준 SDPA (Scaled Dot Product Attention) 사용

| Attention 구현 | 속도 | 메모리 효율 | Windows 지원 |
|---------------|------|-----------|-------------|
| flash_attention_2 | 가장 빠름 | 최고 | 미지원 |
| sdpa (PyTorch native) | 빠름 | 양호 | **지원** |
| eager (표준) | 보통 | 보통 | 지원 |

#### 3. SoX 의존성 회피

**문제**: 일부 오디오 처리에 SoX가 필요하지만, Windows에서는 기본 설치되어 있지 않음

```
SoX could not be found!
```

**해결책**: Qwen3-TTS의 기본 기능은 SoX 없이 동작. 경고는 무시 가능.

#### 4. 콘솔 문자 깨짐 (cp932 인코딩)

**문제**: 일본어 Windows 환경에서는 cp932 인코딩으로 인해 비ASCII 문자가 깨짐

```
UnicodeEncodeError: 'cp932' codec can't encode character...
```

**해결책**: UTF-8 인코딩을 명시적으로 설정

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

또는 배치 파일에서 `chcp 65001` 실행

#### 5. torchao 호환성 경고

**문제**: PyTorch nightly와 torchao 버전 불일치로 인한 경고

```
Skipping import of cpp extensions due to incompatible torch version 2.11.0.dev+cu128 for torchao version 0.15.0
```

**해결책**: 경고만 표시되며 동작에 영향 없음. 무시 가능.

#### 6. Hugging Face 심볼릭 링크 경고

**문제**: Windows에서 심볼릭 링크 생성에 관리자 권한이 필요

```
huggingface_hub cache-system uses symlinks by default...
```

**해결책**: 
- Windows 설정 → 개발자 모드 활성화
- 또는 경고 무시 (동작에 영향 없음)

### 동작 확인 스크립트

환경이 올바르게 설정되었는지 확인하려면:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

예상 출력 (RTX 5090의 경우):

```
PyTorch version: 2.11.0.dev20260123+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5090
GPU Memory: 31.8 GB
```

### 문제 해결

| 증상 | 원인 | 해결책 |
|------|------|--------|
| `no kernel image is available` | PyTorch 안정 버전 사용 | nightly 버전 (cu128) 설치 |
| `FlashAttention2 cannot be used` | FlashAttention이 Windows 미지원 | `--no-flash-attn` 옵션 추가 |
| `SoX could not be found` | SoX 미설치 | 무시 가능 (기본 기능에 영향 없음) |
| GPU 인식 안 됨 | CUDA 드라이버 구버전 | 최신 드라이버 설치 |
| 문자 깨짐 | cp932 인코딩 | `chcp 65001` 또는 UTF-8 설정 |

## 라이선스

본 프로젝트는 [Apache License 2.0](../LICENSE) 하에 공개되어 있습니다.

### 사용된 오픈소스 소프트웨어

| 소프트웨어 | 라이선스 | 저작권 |
|-----------|---------|--------|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache License 2.0 | Copyright 2026 Alibaba Cloud |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT License | Copyright SYSTRAN |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT License | Copyright OpenAI |

자세한 내용은 [NOTICE](../NOTICE) 파일을 참조하세요.

## 면책 조항

### 음성 생성에 관한 면책

- 본 시스템에서 생성된 음성은 AI 모델에 의해 자동 생성된 것으로, 부정확하거나 부적절한 내용이 포함될 수 있습니다
- 생성된 음성은 개발자의 견해를 대표하지 않으며, 전문적인 조언을 구성하지 않습니다
- 사용자는 생성된 음성의 사용, 배포 또는 의존에 관한 모든 위험과 책임을 스스로 부담합니다

### 보이스 클론에 관한 경고

- **타인의 목소리를 무단으로 복제·사용하는 것은 초상권 및 퍼블리시티권 침해가 될 수 있습니다**
- 보이스 클론 기능은 본인의 동의를 얻은 후 합법적인 목적으로만 사용하세요
- 사기, 사칭, 명예 훼손, 딥페이크 등 악의적 목적으로의 사용은 엄격히 금지합니다

### 법적 책임

- 본 소프트웨어의 사용으로 발생한 어떠한 손해에 대해서도 개발자는 책임을 지지 않습니다
- 불법적인 사용으로 발생한 법적 책임은 모두 사용자가 부담합니다
- 본 소프트웨어는 "있는 그대로" 제공되며 어떠한 보증도 하지 않습니다

## 감사의 말

- 원본 개발자: [Alibaba Cloud Qwen Team](https://github.com/QwenLM)
- 원본 리포지토리: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

## 인용

원본 Qwen3-TTS를 인용하는 경우:

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```
