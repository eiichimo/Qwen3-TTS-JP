[English](../README.md) | [日本語](README_ja.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Русский](README_ru.md) | **Español** | [Italiano](README_it.md) | [Deutsch](README_de.md) | [Français](README_fr.md) | [Português](README_pt.md)

# Qwen3-TTS-JP

Fork de Qwen3-TTS con interfaz Web multilingüe y **soporte nativo para Windows**.

El Qwen3-TTS original fue desarrollado principalmente para entornos Linux, y se recomienda el uso de FlashAttention 2. Sin embargo, FlashAttention 2 no funciona en Windows. Este fork permite la **ejecución directa en Windows sin WSL2 ni Docker**, proporciona una GUI completamente localizada en japonés y añade la función de transcripción automática mediante Whisper.

> **Usuarios de Mac (Apple Silicon):** Para la mejor experiencia en Mac, utilice **[Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab)** -- totalmente optimizado para Apple Silicon con motor dual MLX + PyTorch, cuantizacion 8bit/4bit e interfaz Web en 10 idiomas.

### Voz personalizada -- Síntesis de voz con hablantes predefinidos
<p align="center">
    <img src="../assets/CustomVoice.png" width="90%"/>
</p>

### Diseño de voz -- Describir características de voz para sintetizar
<p align="center">
    <img src="../assets/VoiceDesign.png" width="90%"/>
</p>

### Clonación de voz -- Clonar voz desde audio de referencia
<p align="center">
    <img src="../assets/VoiceClone.png" width="90%"/>
</p>

### Configuración -- GPU / VRAM / Información del modelo
<p align="center">
    <img src="../assets/Settings.png" width="90%"/>
</p>

## Proyectos relacionados

| Plataforma | Repositorio | Descripción |
|:---:|---|---|
| Windows | **Este repositorio** | Soporte nativo para Windows + Web UI multilingüe |
| macOS (Apple Silicon) | [Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab) | Versión totalmente optimizada para Apple Silicon Mac (motor dual MLX + PyTorch, Web UI en 10 idiomas) |

## Características

### Soporte nativo para Windows

- **FlashAttention 2 no requerido**: Usa SDPA (Scaled Dot Product Attention) estándar de PyTorch mediante la opción `--no-flash-attn`
- **WSL2/Docker no requerido**: Ejecución directa en Windows
- **Soporte para RTX serie 50**: Incluye instrucciones para instalar builds nightly de PyTorch para la arquitectura NVIDIA Blackwell (sm_120)
- **Sin dependencia de SoX**: Funciona sin SoX (se muestran advertencias pero se pueden ignorar)

### Interfaz Web moderna y soporte multilingue

- **UI en 10 idiomas**: japones / English / chino / coreano / ruso / Espanol / Italiano / Deutsch / Frances / Portugues -- cambio instantaneo desde el menu desplegable
- **4 pestanas**: Custom Voice / Voice Design / Voice Clone / Settings -- acceso a todas las funciones independientemente del tipo de modelo; los modelos no cargados se descargan automaticamente en el primer uso
- **Monitoreo de GPU / VRAM**: consulta el uso en tiempo real en la pestana de configuracion, limpieza de cache CUDA disponible
- **Transcripcion automatica con Whisper**: automatiza la entrada de texto del audio de referencia al clonar voces ([faster-whisper](https://github.com/SYSTRAN/faster-whisper))
- **Seleccion de modelo Whisper**: elige entre 5 modelos segun tus necesidades
  - `tiny` - Mas rapido y pequeno (39M parametros)
  - `base` - Rapido (74M parametros)
  - `small` - Equilibrado (244M parametros) *Por defecto
  - `medium` - Alta precision (769M parametros)
  - `large-v3` - Maxima precision (1550M parametros)

## Requisitos del sistema

- **SO**: Windows 10/11 (entorno nativo, sin necesidad de WSL2)
- **GPU**: GPU NVIDIA (compatible con CUDA)
  - RTX serie 30/40: Funciona con la versión estable de PyTorch
  - RTX serie 50 (Blackwell): Requiere build nightly de PyTorch (cu128)
- **Python**: 3.10 o superior
- **VRAM**: Se recomiendan 8 GB o más (varía según el tamaño del modelo)

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-JP.git
cd Qwen3-TTS-JP
```

### 2. Crear y activar el entorno virtual

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -e .
pip install faster-whisper
```

### 4. Instalar PyTorch (versión CUDA)

Instale según su versión de CUDA.

```bash
# Para CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Para RTX serie 50 (sm_120), se requiere el build nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Uso

### Iniciar la GUI

#### Desde la línea de comandos

```bash
# Modelo CustomVoice (voces preconfiguradas)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn

# Modelo Base (con clonación de voz)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
```

Abra `http://127.0.0.1:7860` en su navegador.

#### Inicio rápido con archivos batch (recomendado)

Cree un archivo batch como el siguiente para iniciar con doble clic:

**run-gui.bat** (para el modelo CustomVoice):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS GUI
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

**run-voice-clone.bat** (para el modelo Base / clonación de voz):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS Voice Clone
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

#### Lanzador avanzado (selección automática de puerto y apertura automática del navegador)

Para un método de inicio más cómodo, puede usar el siguiente lanzador Python:

<details>
<summary>launch_gui.py (clic para expandir)</summary>

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
    """Buscar un puerto disponible"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Todos los puertos {start_port}-{start_port + max_attempts} están en uso")

def wait_for_server_and_open_browser(url, timeout=180):
    """Esperar al inicio del servidor y abrir el navegador"""
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
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # o CustomVoice
        "--ip", "127.0.0.1",
        "--port", str(port),
        "--no-flash-attn"
    ])

if __name__ == "__main__":
    main()
```

</details>

Funcionalidades:
- **Selección automática de puerto**: Detecta automáticamente un puerto libre si el 7860 está en uso
- **Apertura automática del navegador**: Detecta la finalización del inicio del servidor y abre automáticamente el navegador
- **Corrección de codificación**: Soporte de codificación UTF-8

### Pasos para la clonación de voz

1. Suba un archivo de audio en "Audio de referencia"
2. Seleccione un modelo en "Modelo Whisper" (la primera descarga puede tardar)
3. Haga clic en "Transcripción automática"
4. El resultado de la transcripción se ingresa automáticamente en "Texto del audio de referencia"
5. Edite el texto si es necesario
6. Ingrese el "Texto a sintetizar"
7. Haga clic en "Generar audio"

### Detalles del soporte nativo de Windows

Este fork logra el funcionamiento nativo en Windows mediante las siguientes medidas:

| Problema | Original | Solución de este fork |
|----------|----------|----------------------|
| FlashAttention 2 | Solo Linux, no se puede compilar en Windows | Uso de SDPA mediante la opción `--no-flash-attn` |
| Dependencia de SoX | Se asume que está instalado | Funciona sin él (las advertencias se pueden ignorar) |
| RTX serie 50 | No soportado | Instrucciones de instalación del build nightly incluidas |
| Configuración del entorno | conda (orientado a Linux) | venv (estándar de Windows) |

**Nota**: La opción `--no-flash-attn` es obligatoria. Sin ella, la aplicación no se iniciará debido a un error de importación de FlashAttention 2.

## Configuración detallada del entorno nativo de Windows

### Problemas técnicos resueltos

Durante el desarrollo de este fork, se identificaron y resolvieron los siguientes problemas específicos de Windows:

#### 1. Soporte CUDA para RTX serie 50 (Blackwell/sm_120)

**Problema**: La versión estable de PyTorch no soporta las GPUs más recientes como la RTX 5090 (sm_120)

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

**Solución**: Usar la versión nightly de PyTorch (cu128)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2. FlashAttention 2 no compatible con Windows

**Problema**: FlashAttention 2 es exclusivo de Linux y no se puede compilar ni ejecutar en Windows

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: 
the package flash_attn seems to be not installed.
```

**Solución**: Usar SDPA (Scaled Dot Product Attention) estándar de PyTorch mediante la opción `--no-flash-attn`

| Implementación de Attention | Velocidad | Eficiencia de memoria | Soporte Windows |
|----------------------------|-----------|----------------------|-----------------|
| flash_attention_2 | Máxima | Óptima | No soportado |
| sdpa (PyTorch native) | Rápida | Buena | **Soportado** |
| eager (estándar) | Normal | Normal | Soportado |

#### 3. Eliminación de la dependencia de SoX

**Problema**: Algunos procesamientos de audio requieren SoX, pero no está instalado por defecto en Windows

```
SoX could not be found!
```

**Solución**: La funcionalidad básica de Qwen3-TTS funciona sin SoX. Las advertencias se pueden ignorar.

#### 4. Caracteres corruptos en la consola (codificación cp932)

**Problema**: En entornos japoneses de Windows, los caracteres no ASCII se corrompen debido a la codificación cp932

```
UnicodeEncodeError: 'cp932' codec can't encode character...
```

**Solución**: Establecer explícitamente la codificación UTF-8

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

O ejecutar `chcp 65001` en el archivo batch

#### 5. Advertencia de compatibilidad de torchao

**Problema**: Advertencia por incompatibilidad de versiones entre PyTorch nightly y torchao

```
Skipping import of cpp extensions due to incompatible torch version 2.11.0.dev+cu128 for torchao version 0.15.0
```

**Solución**: Solo es una advertencia, no afecta al funcionamiento. Se puede ignorar.

#### 6. Advertencia de enlaces simbólicos de Hugging Face

**Problema**: La creación de enlaces simbólicos en Windows requiere privilegios de administrador

```
huggingface_hub cache-system uses symlinks by default...
```

**Solución**: 
- Activar el Modo de desarrollador en la configuración de Windows
- O ignorar la advertencia (no afecta al funcionamiento)

### Script de verificación

Para verificar que el entorno está configurado correctamente:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

Salida esperada (para RTX 5090):

```
PyTorch version: 2.11.0.dev20260123+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5090
GPU Memory: 31.8 GB
```

### Solución de problemas

| Síntoma | Causa | Solución |
|---------|-------|----------|
| `no kernel image is available` | Uso de PyTorch estable | Instalar versión nightly (cu128) |
| `FlashAttention2 cannot be used` | FlashAttention no compatible con Windows | Añadir opción `--no-flash-attn` |
| `SoX could not be found` | SoX no instalado | Se puede ignorar (no afecta la funcionalidad básica) |
| GPU no reconocida | Controlador CUDA desactualizado | Instalar el controlador más reciente |
| Caracteres corruptos | Codificación cp932 | `chcp 65001` o configuración UTF-8 |

## Licencia

Este proyecto se publica bajo la [Apache License 2.0](../LICENSE).

### Software de código abierto utilizado

| Software | Licencia | Derechos de autor |
|----------|----------|-------------------|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache License 2.0 | Copyright 2026 Alibaba Cloud |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT License | Copyright SYSTRAN |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT License | Copyright OpenAI |

Para más detalles, consulte el archivo [NOTICE](../NOTICE).

## Descargo de responsabilidad

### Descargo sobre la generación de audio

- El audio generado por este sistema es producido automáticamente por un modelo de IA y puede contener contenido inexacto o inapropiado
- El audio generado no representa las opiniones de los desarrolladores y no constituye asesoramiento profesional
- Los usuarios asumen todos los riesgos y responsabilidades relacionados con el uso, distribución o dependencia del audio generado

### Advertencia sobre clonación de voz

- **Clonar o usar la voz de otra persona sin su consentimiento puede constituir una violación de los derechos de imagen y de publicidad**
- Por favor, utilice la función de clonación de voz solo con fines legales y con el consentimiento de la persona cuya voz se clona
- Se prohíbe estrictamente su uso con fines maliciosos como fraude, suplantación de identidad, difamación o deepfakes

### Responsabilidad legal

- Los desarrolladores no asumen responsabilidad por ningún daño derivado del uso de este software
- Toda responsabilidad legal derivada del uso ilegal será asumida por el usuario
- Este software se proporciona "tal cual" sin ninguna garantía

## Agradecimientos

- Desarrollador original: [Alibaba Cloud Qwen Team](https://github.com/QwenLM)
- Repositorio original: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

## Citación

Para citar el Qwen3-TTS original:

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```
