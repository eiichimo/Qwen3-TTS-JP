[English](../README.md) | [日本語](README_ja.md) | **中文** | [한국어](README_ko.md) | [Русский](README_ru.md) | [Español](README_es.md) | [Italiano](README_it.md) | [Deutsch](README_de.md) | [Français](README_fr.md) | [Português](README_pt.md)

# Qwen3-TTS-JP

**Windows原生支持**的 Qwen3-TTS 多语言Web UI分支。

原版Qwen3-TTS主要面向Linux环境开发，推荐使用FlashAttention 2，但FlashAttention 2无法在Windows上运行。本分支实现了**无需WSL2或Docker，直接在Windows上运行**，并提供了完整的日语GUI本地化以及基于Whisper的自动语音转文字功能。

> **Mac (Apple Silicon) 用户：** 如需在Mac上获得最佳体验，请使用 **[Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab)** -- 针对Apple Silicon全面优化，支持MLX + PyTorch双引擎、8bit/4bit量化及10语言Web UI。

### 自定义语音 -- 使用预设说话人进行语音合成
<p align="center">
    <img src="../assets/CustomVoice.png" width="90%"/>
</p>

### 语音设计 -- 描述语音特征以合成
<p align="center">
    <img src="../assets/VoiceDesign.png" width="90%"/>
</p>

### 语音克隆 -- 从参考音频克隆语音
<p align="center">
    <img src="../assets/VoiceClone.png" width="90%"/>
</p>

### 设置 -- GPU / 显存 / 模型信息
<p align="center">
    <img src="../assets/Settings.png" width="90%"/>
</p>

## 相关项目

| 平台 | 仓库 | 说明 |
|:---:|---|---|
| Windows | **本仓库** | Windows原生支持 + 多语言Web UI |
| macOS (Apple Silicon) | [Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab) | Apple Silicon Mac完全优化版（MLX + PyTorch 双引擎，10语言Web UI） |

## 特性

### Windows原生支持

- **无需FlashAttention 2**：通过`--no-flash-attn`选项使用PyTorch标准的SDPA（Scaled Dot Product Attention）
- **无需WSL2/Docker**：可直接在Windows上运行
- **支持RTX 50系列**：包含NVIDIA Blackwell架构（sm_120）PyTorch nightly构建的安装说明
- **避免SoX依赖**：无需SoX即可运行（会显示警告但可安全忽略）

### 现代Web UI与多语言支持

- **10语言UI**：日语 / English / 中文 / 한국어 / Русский / Español / Italiano / Deutsch / Français / Português -- 通过下拉菜单即时切换
- **4标签页布局**：Custom Voice / Voice Design / Voice Clone / Settings -- 无论模型类型均可访问所有功能；未加载的模型在首次使用时自动下载
- **GPU / 显存监控**：在设置标签页中实时查看使用情况，也可清除CUDA缓存
- **Whisper自动转录**：自动化语音克隆时的参考音频文本输入（使用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)）
- **Whisper模型选择**：可根据需求从5种模型中选择
  - `tiny` - 最快、最小（39M参数）
  - `base` - 快速（74M参数）
  - `small` - 均衡型（244M参数）※默认
  - `medium` - 高精度（769M参数）
  - `large-v3` - 最高精度（1550M参数）

## 系统要求

- **操作系统**：Windows 10/11（原生环境，无需WSL2）
- **GPU**：NVIDIA GPU（支持CUDA）
  - RTX 30/40系列：使用PyTorch稳定版即可
  - RTX 50系列（Blackwell）：需要PyTorch nightly构建（cu128）
- **Python**：3.10及以上
- **显存**：建议8GB以上（因模型大小而异）

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-JP.git
cd Qwen3-TTS-JP
```

### 2. 创建并激活虚拟环境

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. 安装依赖包

```bash
pip install -e .
pip install faster-whisper
```

### 4. 安装PyTorch（CUDA版）

请根据您的CUDA版本进行安装。

```bash
# CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# RTX 50系列（sm_120）需要nightly构建
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 使用方法

### 启动GUI

#### 从命令行启动

```bash
# CustomVoice模型（预设说话人）
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn

# Base模型（带语音克隆功能）
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
```

在浏览器中打开 `http://127.0.0.1:7860`。

#### 使用批处理文件快速启动（推荐）

创建如下批处理文件，双击即可启动：

**run-gui.bat**（CustomVoice模型用）：
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS GUI
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

**run-voice-clone.bat**（Base模型/语音克隆用）：
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS Voice Clone
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

#### 高级启动器（自动端口选择与自动打开浏览器）

更便捷的启动方式，可使用以下Python启动器：

<details>
<summary>launch_gui.py（点击展开）</summary>

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
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"端口 {start_port}-{start_port + max_attempts} 全部被占用")

def wait_for_server_and_open_browser(url, timeout=180):
    """等待服务器启动后打开浏览器"""
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
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # 或 CustomVoice
        "--ip", "127.0.0.1",
        "--port", str(port),
        "--no-flash-attn"
    ])

if __name__ == "__main__":
    main()
```

</details>

功能：
- **自动端口选择**：若7860被占用，自动检测可用端口
- **自动打开浏览器**：检测服务器启动完成后自动打开浏览器
- **字符编码修复**：UTF-8编码支持

### 语音克隆步骤

1. 在"参考音频"中上传音频文件
2. 在"Whisper模型"中选择模型（首次下载可能需要一些时间）
3. 点击"自动转录"按钮
4. 转录结果将自动填入"参考音频文本"
5. 根据需要编辑文本
6. 输入"待合成文本"
7. 点击"生成音频"

### Windows原生支持详情

本分支通过以下措施实现了Windows原生运行：

| 问题 | 原版 | 本分支的解决方案 |
|------|------|-----------------|
| FlashAttention 2 | 仅限Linux，无法在Windows上构建 | 通过`--no-flash-attn`选项使用SDPA |
| SoX依赖 | 假定已安装 | 无需安装即可运行（警告可忽略） |
| RTX 50系列 | 不支持 | 包含nightly构建安装说明 |
| 环境搭建 | conda（面向Linux） | venv（Windows标准） |

**注意**：`--no-flash-attn`选项是必需的。缺少此选项将导致FlashAttention 2导入错误，无法启动。

## Windows原生环境搭建详情

### 已解决的技术问题

在本分支的开发过程中，以下Windows特有的问题已被识别并解决：

#### 1. RTX 50系列（Blackwell/sm_120）的CUDA支持

**问题**：PyTorch稳定版不支持RTX 5090等最新GPU（sm_120）

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

**解决方案**：使用PyTorch nightly版（cu128）

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2. FlashAttention 2不支持Windows

**问题**：FlashAttention 2仅限Linux，无法在Windows上构建或运行

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: 
the package flash_attn seems to be not installed.
```

**解决方案**：通过`--no-flash-attn`选项使用PyTorch标准的SDPA（Scaled Dot Product Attention）

| Attention实现 | 速度 | 内存效率 | Windows支持 |
|--------------|------|---------|-------------|
| flash_attention_2 | 最快 | 最优 | 不支持 |
| sdpa (PyTorch native) | 快速 | 良好 | **支持** |
| eager (标准) | 一般 | 一般 | 支持 |

#### 3. 避免SoX依赖

**问题**：部分音频处理需要SoX，但Windows默认未安装

```
SoX could not be found!
```

**解决方案**：Qwen3-TTS的基本功能无需SoX即可运行。警告可安全忽略。

#### 4. 控制台字符乱码（cp932编码）

**问题**：在日语Windows环境中，由于cp932编码，非ASCII字符会出现乱码

```
UnicodeEncodeError: 'cp932' codec can't encode character...
```

**解决方案**：显式设置UTF-8编码

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

或在批处理文件中执行 `chcp 65001`

#### 5. torchao兼容性警告

**问题**：PyTorch nightly与torchao版本不匹配导致的警告

```
Skipping import of cpp extensions due to incompatible torch version 2.11.0.dev+cu128 for torchao version 0.15.0
```

**解决方案**：仅为警告，不影响运行。可安全忽略。

#### 6. Hugging Face符号链接警告

**问题**：在Windows上创建符号链接需要管理员权限

```
huggingface_hub cache-system uses symlinks by default...
```

**解决方案**：
- 在Windows设置中启用开发者模式
- 或忽略警告（不影响运行）

### 验证脚本

验证环境是否正确设置：

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

预期输出（以RTX 5090为例）：

```
PyTorch version: 2.11.0.dev20260123+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5090
GPU Memory: 31.8 GB
```

### 故障排除

| 症状 | 原因 | 解决方案 |
|------|------|---------|
| `no kernel image is available` | 使用了PyTorch稳定版 | 安装nightly版（cu128） |
| `FlashAttention2 cannot be used` | FlashAttention不支持Windows | 添加`--no-flash-attn`选项 |
| `SoX could not be found` | 未安装SoX | 可忽略（不影响基本功能） |
| GPU未被识别 | CUDA驱动过旧 | 安装最新驱动 |
| 字符乱码 | cp932编码 | `chcp 65001`或设置UTF-8 |

## 许可证

本项目基于 [Apache License 2.0](../LICENSE) 发布。

### 使用的开源软件

| 软件 | 许可证 | 版权 |
|------|--------|------|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache License 2.0 | Copyright 2026 Alibaba Cloud |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT License | Copyright SYSTRAN |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT License | Copyright OpenAI |

详细信息请参阅 [NOTICE](../NOTICE) 文件。

## 免责声明

### 关于音频生成的免责声明

- 本系统生成的音频由AI模型自动产生，可能包含不准确或不适当的内容
- 生成的音频不代表开发者的观点，也不构成专业建议
- 用户对生成音频的使用、分发或依赖承担所有风险和责任

### 语音克隆警告

- **未经他人同意克隆或使用其声音，可能构成对肖像权和公开权的侵犯**
- 请仅在获得本人同意的前提下，将语音克隆功能用于合法目的
- 严禁将其用于欺诈、冒充、诽谤、深度伪造等恶意目的

### 法律责任

- 开发者对使用本软件所造成的任何损害不承担责任
- 因非法使用而产生的所有法律责任由用户承担
- 本软件按"现状"提供，不提供任何保证

## 致谢

- 原始开发者：[Alibaba Cloud Qwen Team](https://github.com/QwenLM)
- 原始仓库：[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

## 引用

引用原版Qwen3-TTS：

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```
