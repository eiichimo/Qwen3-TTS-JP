[English](../README.md) | [日本語](README_ja.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Русский](README_ru.md) | [Español](README_es.md) | [Italiano](README_it.md) | **Deutsch** | [Français](README_fr.md) | [Português](README_pt.md)

# Qwen3-TTS-JP

Ein Fork von Qwen3-TTS mit **mehrsprachiger Unterstützung** und **nativer Windows-Unterstützung**.

Das originale Qwen3-TTS wurde hauptsächlich für Linux-Umgebungen entwickelt, und die Verwendung von FlashAttention 2 wird empfohlen. FlashAttention 2 funktioniert jedoch nicht unter Windows. Dieser Fork ermöglicht die **direkte Ausführung unter Windows ohne WSL2 oder Docker**, bietet eine UI in 10 Sprachen und fügt eine automatische Transkription über Whisper hinzu.

> **Mac (Apple Silicon) Nutzer:** Fur das beste Erlebnis auf Mac verwenden Sie bitte **[Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab)** -- vollstandig optimiert fur Apple Silicon mit MLX + PyTorch Dual-Engine, 8bit/4bit-Quantisierung und Web-UI in 10 Sprachen.

### Benutzerdefinierte Stimme -- Sprachsynthese mit voreingestellten Sprechern
<p align="center">
    <img src="../assets/CustomVoice.png" width="90%"/>
</p>

### Sprachdesign -- Stimmmerkmale beschreiben zur Synthese
<p align="center">
    <img src="../assets/VoiceDesign.png" width="90%"/>
</p>

### Stimmklonierung -- Stimme aus Referenzaudio klonen
<p align="center">
    <img src="../assets/VoiceClone.png" width="90%"/>
</p>

### Einstellungen -- GPU / VRAM / Modellinformationen
<p align="center">
    <img src="../assets/Settings.png" width="90%"/>
</p>

## Verwandte Projekte

| Plattform | Repository | Beschreibung |
|:---:|---|---|
| Windows | **Dieses Repository** | Native Windows-Unterstützung + UI in 10 Sprachen |
| macOS (Apple Silicon) | [Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab) | Vollständig optimiert für Apple Silicon Mac (MLX + PyTorch Dual-Engine, Web-UI in 10 Sprachen) |

## Funktionen

### Native Windows-Unterstützung

- **Kein FlashAttention 2 erforderlich**: Verwendet PyTorchs Standard-SDPA (Scaled Dot Product Attention) über die Option `--no-flash-attn`
- **Kein WSL2/Docker erforderlich**: Direkte Ausführung unter Windows
- **RTX 50er-Serie Unterstützung**: Enthält Anleitungen zur Installation von PyTorch Nightly-Builds für die NVIDIA Blackwell-Architektur (sm_120)
- **Keine SoX-Abhängigkeit**: Funktioniert ohne SoX (Warnungen werden angezeigt, können aber ignoriert werden)

### Moderne Web-UI & Mehrsprachige Unterstutzung

- **UI in 10 Sprachen**: Japanisch / English / Chinesisch / Koreanisch / Russisch / Spanisch / Italienisch / Deutsch / Franzosisch / Portugiesisch -- sofortiger Wechsel uber Dropdown-Menu
- **4-Tab-Layout**: Custom Voice / Voice Design / Voice Clone / Settings -- Zugriff auf alle Funktionen unabhangig vom Modelltyp; nicht geladene Modelle werden bei der ersten Verwendung automatisch heruntergeladen
- **GPU / VRAM-Uberwachung**: Echtzeit-Nutzungsanzeige im Einstellungen-Tab, CUDA-Cache-Bereinigung verfugbar
- **Automatische Whisper-Transkription**: Automatisiert die Texteingabe des Referenzaudios beim Voice Cloning ([faster-whisper](https://github.com/SYSTRAN/faster-whisper))
- **Whisper-Modellauswahl**: Wahle aus 5 Modellen je nach Bedarf
  - `tiny` - Schnellstes & kleinstes (39M Parameter)
  - `base` - Schnell (74M Parameter)
  - `small` - Ausgewogen (244M Parameter) *Standard
  - `medium` - Hohe Genauigkeit (769M Parameter)
  - `large-v3` - Hochste Genauigkeit (1550M Parameter)

## Systemanforderungen

- **Betriebssystem**: Windows 10/11 (native Umgebung, kein WSL2 erforderlich)
- **GPU**: NVIDIA GPU (CUDA-kompatibel)
  - RTX 30/40er-Serie: Funktioniert mit der stabilen PyTorch-Version
  - RTX 50er-Serie (Blackwell): Erfordert PyTorch Nightly-Build (cu128)
- **Python**: 3.10 oder höher
- **VRAM**: 8 GB oder mehr empfohlen (variiert je nach Modellgröße)

## Installation

### 1. Repository klonen

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-JP.git
cd Qwen3-TTS-JP
```

### 2. Virtuelle Umgebung erstellen und aktivieren

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Abhängigkeiten installieren

```bash
pip install -e .
pip install faster-whisper
```

### 4. PyTorch (CUDA-Version) installieren

Installieren Sie entsprechend Ihrer CUDA-Version.

```bash
# Für CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Für RTX 50er-Serie (sm_120) ist der Nightly-Build erforderlich
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Verwendung

### GUI starten

#### Von der Kommandozeile

```bash
# CustomVoice-Modell (voreingestellte Sprecher)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn

# Base-Modell (mit Voice-Cloning)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
```

Öffnen Sie `http://127.0.0.1:7860` im Browser.

#### Schnellstart mit Batch-Dateien (empfohlen)

Erstellen Sie eine Batch-Datei wie folgt für den Start per Doppelklick:

**run-gui.bat** (für das CustomVoice-Modell):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS GUI
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

**run-voice-clone.bat** (für das Base-Modell / Voice-Cloning):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS Voice Clone
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

#### Erweiterter Launcher (automatische Port-Auswahl und automatischer Browser-Start)

Für eine bequemere Startmethode können Sie den folgenden Python-Launcher verwenden:

<details>
<summary>launch_gui.py (zum Aufklappen klicken)</summary>

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
    """Einen freien Port finden"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Alle Ports {start_port}-{start_port + max_attempts} sind belegt")

def wait_for_server_and_open_browser(url, timeout=180):
    """Auf Serverstart warten und Browser öffnen"""
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
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # oder CustomVoice
        "--ip", "127.0.0.1",
        "--port", str(port),
        "--no-flash-attn"
    ])

if __name__ == "__main__":
    main()
```

</details>

Funktionen:
- **Automatische Port-Auswahl**: Erkennt automatisch einen freien Port, wenn 7860 belegt ist
- **Automatischer Browser-Start**: Erkennt den Abschluss des Serverstarts und öffnet automatisch den Browser
- **Zeichenkodierungskorrektur**: UTF-8-Kodierungsunterstützung

### Schritte zum Voice-Cloning

1. Laden Sie eine Audiodatei unter „Referenzaudio" hoch
2. Wählen Sie ein Modell unter „Whisper-Modell" (der erste Download kann einige Zeit dauern)
3. Klicken Sie auf „Automatische Transkription"
4. Das Transkriptionsergebnis wird automatisch in „Text des Referenzaudios" eingetragen
5. Bearbeiten Sie den Text bei Bedarf
6. Geben Sie den „Zu synthetisierenden Text" ein
7. Klicken Sie auf „Audio generieren"

### Details zur nativen Windows-Unterstützung

Dieser Fork erreicht nativen Windows-Betrieb durch folgende Maßnahmen:

| Problem | Original | Lösung dieses Forks |
|---------|----------|---------------------|
| FlashAttention 2 | Nur Linux, kann unter Windows nicht kompiliert werden | Verwendung von SDPA über die Option `--no-flash-attn` |
| SoX-Abhängigkeit | Installation wird vorausgesetzt | Funktioniert ohne (Warnungen können ignoriert werden) |
| RTX 50er-Serie | Nicht unterstützt | Anleitung für Nightly-Build enthalten |
| Umgebungseinrichtung | conda (Linux-orientiert) | venv (Windows-Standard) |

**Hinweis**: Die Option `--no-flash-attn` ist zwingend erforderlich. Ohne sie schlägt der Start der Anwendung mit einem FlashAttention 2 Import-Fehler fehl.

## Detaillierte native Windows-Umgebungseinrichtung

### Gelöste technische Probleme

Während der Entwicklung dieses Forks wurden die folgenden Windows-spezifischen Probleme identifiziert und gelöst:

#### 1. CUDA-Unterstützung für RTX 50er-Serie (Blackwell/sm_120)

**Problem**: Die stabile PyTorch-Version unterstützt neueste GPUs wie die RTX 5090 (sm_120) nicht

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

**Lösung**: Verwendung der PyTorch Nightly-Version (cu128)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2. FlashAttention 2 unter Windows nicht unterstützt

**Problem**: FlashAttention 2 ist nur für Linux verfügbar und kann unter Windows weder kompiliert noch ausgeführt werden

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: 
the package flash_attn seems to be not installed.
```

**Lösung**: Verwendung von PyTorchs Standard-SDPA (Scaled Dot Product Attention) über die Option `--no-flash-attn`

| Attention-Implementierung | Geschwindigkeit | Speichereffizienz | Windows-Unterstützung |
|--------------------------|----------------|-------------------|----------------------|
| flash_attention_2 | Schnellste | Beste | Nicht unterstützt |
| sdpa (PyTorch native) | Schnell | Gut | **Unterstützt** |
| eager (Standard) | Normal | Normal | Unterstützt |

#### 3. Vermeidung der SoX-Abhängigkeit

**Problem**: Einige Audioverarbeitungen erfordern SoX, das unter Windows standardmäßig nicht installiert ist

```
SoX could not be found!
```

**Lösung**: Die Grundfunktionen von Qwen3-TTS funktionieren ohne SoX. Warnungen können sicher ignoriert werden.

#### 4. Zeichenfehler in der Konsole (cp932-Kodierung)

**Problem**: In japanischen Windows-Umgebungen werden Nicht-ASCII-Zeichen aufgrund der cp932-Kodierung falsch dargestellt

```
UnicodeEncodeError: 'cp932' codec can't encode character...
```

**Lösung**: Explizites Setzen der UTF-8-Kodierung

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

Oder `chcp 65001` in der Batch-Datei ausführen

#### 5. torchao-Kompatibilitätswarnung

**Problem**: Warnung wegen Versionsinkonsistenz zwischen PyTorch Nightly und torchao

```
Skipping import of cpp extensions due to incompatible torch version 2.11.0.dev+cu128 for torchao version 0.15.0
```

**Lösung**: Nur eine Warnung, keine Auswirkung auf den Betrieb. Kann sicher ignoriert werden.

#### 6. Hugging Face Symlink-Warnung

**Problem**: Das Erstellen symbolischer Links unter Windows erfordert Administratorrechte

```
huggingface_hub cache-system uses symlinks by default...
```

**Lösung**: 
- Entwicklermodus in den Windows-Einstellungen aktivieren
- Oder die Warnung ignorieren (keine Auswirkung auf den Betrieb)

### Überprüfungsskript

Um zu überprüfen, ob die Umgebung korrekt eingerichtet ist:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

Erwartete Ausgabe (für RTX 5090):

```
PyTorch version: 2.11.0.dev20260123+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5090
GPU Memory: 31.8 GB
```

### Fehlerbehebung

| Symptom | Ursache | Lösung |
|---------|---------|--------|
| `no kernel image is available` | Stabile PyTorch-Version wird verwendet | Nightly-Version (cu128) installieren |
| `FlashAttention2 cannot be used` | FlashAttention unter Windows nicht unterstützt | Option `--no-flash-attn` hinzufügen |
| `SoX could not be found` | SoX nicht installiert | Kann ignoriert werden (keine Auswirkung auf Grundfunktionen) |
| GPU wird nicht erkannt | CUDA-Treiber veraltet | Neuesten Treiber installieren |
| Zeichenfehler | cp932-Kodierung | `chcp 65001` oder UTF-8-Einstellung |

## Lizenz

Dieses Projekt wird unter der [Apache License 2.0](../LICENSE) veröffentlicht.

### Verwendete Open-Source-Software

| Software | Lizenz | Urheberrecht |
|----------|--------|-------------|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache License 2.0 | Copyright 2026 Alibaba Cloud |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT License | Copyright SYSTRAN |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT License | Copyright OpenAI |

Einzelheiten finden Sie in der [NOTICE](../NOTICE)-Datei.

## Haftungsausschluss

### Haftungsausschluss zur Audiogenerierung

- Das von diesem System generierte Audio wird automatisch von einem KI-Modell erzeugt und kann ungenaue oder unangemessene Inhalte enthalten
- Generiertes Audio repräsentiert nicht die Ansichten der Entwickler und stellt keine professionelle Beratung dar
- Benutzer übernehmen alle Risiken und Verantwortlichkeiten im Zusammenhang mit der Nutzung, Verbreitung oder dem Vertrauen auf generiertes Audio

### Warnung zum Voice-Cloning

- **Das Klonen oder Verwenden der Stimme einer anderen Person ohne deren Zustimmung kann eine Verletzung von Persönlichkeits- und Publizitätsrechten darstellen**
- Bitte verwenden Sie die Voice-Cloning-Funktion nur für rechtmäßige Zwecke mit Zustimmung der Person, deren Stimme geklont wird
- Die Verwendung für böswillige Zwecke wie Betrug, Identitätsdiebstahl, Verleumdung oder Deepfakes ist strengstens untersagt

### Rechtliche Haftung

- Die Entwickler übernehmen keine Haftung für Schäden, die durch die Nutzung dieser Software entstehen
- Alle rechtlichen Haftungen, die aus illegaler Nutzung entstehen, werden vom Benutzer getragen
- Diese Software wird „wie besehen" ohne jegliche Garantie bereitgestellt

## Danksagungen

- Ursprünglicher Entwickler: [Alibaba Cloud Qwen Team](https://github.com/QwenLM)
- Ursprüngliches Repository: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

## Zitierung

Zum Zitieren des originalen Qwen3-TTS:

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```
