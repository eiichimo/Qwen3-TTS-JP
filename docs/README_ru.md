[English](../README.md) | [日本語](README_ja.md) | [中文](README_zh.md) | [한국어](README_ko.md) | **Русский** | [Español](README_es.md) | [Italiano](README_it.md) | [Deutsch](README_de.md) | [Français](README_fr.md) | [Português](README_pt.md)

# Qwen3-TTS-JP

Форк Qwen3-TTS с многоязычным Web UI и **нативной поддержкой Windows**.

Оригинальный Qwen3-TTS разработан преимущественно для Linux-сред, и рекомендуется использовать FlashAttention 2. Однако FlashAttention 2 не работает в Windows. Данный форк обеспечивает **прямой запуск в Windows без WSL2 или Docker**, полную японскую локализацию GUI, а также добавляет автоматическую транскрипцию через Whisper.

> **Пользователям Mac (Apple Silicon):** Для лучшего опыта на Mac используйте **[Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab)** -- полная оптимизация для Apple Silicon с двойным движком MLX + PyTorch, квантизацией 8bit/4bit и Web UI на 10 языках.

### Пользовательский голос -- Синтез речи с предустановленными говорящими
<p align="center">
    <img src="../assets/CustomVoice.png" width="90%"/>
</p>

### Дизайн голоса -- Описание характеристик голоса для синтеза
<p align="center">
    <img src="../assets/VoiceDesign.png" width="90%"/>
</p>

### Клонирование голоса -- Клонирование голоса из эталонного аудио
<p align="center">
    <img src="../assets/VoiceClone.png" width="90%"/>
</p>

### Настройки -- GPU / VRAM / Информация о модели
<p align="center">
    <img src="../assets/Settings.png" width="90%"/>
</p>

## Связанные проекты

| Платформа | Репозиторий | Описание |
|:---:|---|---|
| Windows | **Данный репозиторий** | Нативная поддержка Windows + многоязычный Web UI |
| macOS (Apple Silicon) | [Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab) | Полная оптимизация для Apple Silicon Mac (MLX + PyTorch двойной движок, Web UI на 10 языках) |

## Особенности

### Нативная поддержка Windows

- **FlashAttention 2 не требуется**: Использование стандартного SDPA (Scaled Dot Product Attention) PyTorch через опцию `--no-flash-attn`
- **WSL2/Docker не требуется**: Прямой запуск в Windows
- **Поддержка RTX 50 серии**: Инструкции по установке nightly-сборок PyTorch для архитектуры NVIDIA Blackwell (sm_120)
- **Без зависимости от SoX**: Работает без SoX (предупреждения отображаются, но их можно игнорировать)

### Современный Web UI и многоязычная поддержка

- **UI на 10 языках**: японский / English / 中文 / 한국어 / Русский / Español / Italiano / Deutsch / Français / Português -- мгновенное переключение через выпадающее меню
- **4 вкладки**: Custom Voice / Voice Design / Voice Clone / Settings -- доступ ко всем функциям независимо от типа модели; незагруженные модели автоматически скачиваются при первом использовании
- **Мониторинг GPU / VRAM**: просмотр использования в реальном времени на вкладке настроек, очистка кэша CUDA
- **Автоматическая транскрипция Whisper**: автоматизация ввода текста референсного аудио при клонировании голоса ([faster-whisper](https://github.com/SYSTRAN/faster-whisper))
- **Выбор модели Whisper**: выбор из 5 моделей в зависимости от потребностей
  - `tiny` - самая быстрая и маленькая (39M параметров)
  - `base` - быстрая (74M параметров)
  - `small` - сбалансированная (244M параметров) *По умолчанию
  - `medium` - высокая точность (769M параметров)
  - `large-v3` - максимальная точность (1550M параметров)

## Системные требования

- **ОС**: Windows 10/11 (нативная среда, WSL2 не требуется)
- **GPU**: NVIDIA GPU (с поддержкой CUDA)
  - RTX 30/40 серии: Работает со стабильной версией PyTorch
  - RTX 50 серии (Blackwell): Требуется nightly-сборка PyTorch (cu128)
- **Python**: 3.10 и выше
- **VRAM**: Рекомендуется 8 ГБ и более (зависит от размера модели)

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-JP.git
cd Qwen3-TTS-JP
```

### 2. Создание и активация виртуального окружения

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
pip install -e .
pip install faster-whisper
```

### 4. Установка PyTorch (версия с CUDA)

Установите в соответствии с вашей версией CUDA.

```bash
# Для CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Для RTX 50 серии (sm_120) требуется nightly-сборка
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Использование

### Запуск GUI

#### Из командной строки

```bash
# Модель CustomVoice (предустановленные голоса)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn

# Базовая модель (с функцией клонирования голоса)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
```

Откройте `http://127.0.0.1:7860` в браузере.

#### Быстрый запуск с помощью bat-файлов (рекомендуется)

Создайте bat-файл для запуска двойным кликом:

**run-gui.bat** (для модели CustomVoice):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS GUI
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

**run-voice-clone.bat** (для базовой модели / клонирования голоса):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS Voice Clone
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

#### Расширенный лаунчер (автовыбор порта и автозапуск браузера)

Для более удобного запуска можно использовать следующий Python-лаунчер:

<details>
<summary>launch_gui.py (нажмите для раскрытия)</summary>

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
    """Поиск свободного порта"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Все порты {start_port}-{start_port + max_attempts} заняты")

def wait_for_server_and_open_browser(url, timeout=180):
    """Ожидание запуска сервера и открытие браузера"""
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
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # или CustomVoice
        "--ip", "127.0.0.1",
        "--port", str(port),
        "--no-flash-attn"
    ])

if __name__ == "__main__":
    main()
```

</details>

Возможности:
- **Автовыбор порта**: Если порт 7860 занят, автоматически находит свободный порт
- **Автозапуск браузера**: Обнаруживает завершение запуска сервера и автоматически открывает браузер
- **Исправление кодировки**: Поддержка кодировки UTF-8

### Порядок клонирования голоса

1. Загрузите аудиофайл в поле «Эталонное аудио»
2. Выберите модель в «Модель Whisper» (первая загрузка может занять некоторое время)
3. Нажмите кнопку «Автоматическая транскрипция»
4. Результат транскрипции автоматически вводится в поле «Текст эталонного аудио»
5. При необходимости отредактируйте текст
6. Введите «Текст для синтеза»
7. Нажмите «Генерация аудио»

### Детали нативной поддержки Windows

Данный форк обеспечивает работу в нативной среде Windows благодаря следующим мерам:

| Проблема | Оригинал | Решение в данном форке |
|----------|----------|----------------------|
| FlashAttention 2 | Только для Linux, невозможно собрать в Windows | Использование SDPA через опцию `--no-flash-attn` |
| Зависимость от SoX | Предполагается установка | Работает без SoX (предупреждения можно игнорировать) |
| RTX 50 серии | Не поддерживается | Инструкции по установке nightly-сборки |
| Настройка окружения | conda (ориентирован на Linux) | venv (стандарт Windows) |

**Примечание**: Опция `--no-flash-attn` обязательна. Без неё приложение не запустится из-за ошибки импорта FlashAttention 2.

## Подробная настройка нативной среды Windows

### Решённые технические проблемы

В процессе разработки данного форка были выявлены и решены следующие проблемы, специфичные для Windows:

#### 1. Поддержка CUDA для RTX 50 серии (Blackwell/sm_120)

**Проблема**: Стабильная версия PyTorch не поддерживает новейшие GPU, такие как RTX 5090 (sm_120)

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

**Решение**: Использование nightly-версии PyTorch (cu128)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2. FlashAttention 2 не поддерживается в Windows

**Проблема**: FlashAttention 2 предназначен только для Linux и не может быть собран или запущен в Windows

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: 
the package flash_attn seems to be not installed.
```

**Решение**: Использование стандартного SDPA (Scaled Dot Product Attention) PyTorch через опцию `--no-flash-attn`

| Реализация Attention | Скорость | Эффективность памяти | Поддержка Windows |
|---------------------|----------|---------------------|-------------------|
| flash_attention_2 | Максимальная | Лучшая | Не поддерживается |
| sdpa (PyTorch native) | Высокая | Хорошая | **Поддерживается** |
| eager (стандартная) | Обычная | Обычная | Поддерживается |

#### 3. Устранение зависимости от SoX

**Проблема**: Некоторые операции обработки аудио требуют SoX, но он не установлен по умолчанию в Windows

```
SoX could not be found!
```

**Решение**: Основной функционал Qwen3-TTS работает без SoX. Предупреждения можно безопасно игнорировать.

#### 4. Искажение символов в консоли (кодировка cp932)

**Проблема**: В японской среде Windows символы, не входящие в ASCII, искажаются из-за кодировки cp932

```
UnicodeEncodeError: 'cp932' codec can't encode character...
```

**Решение**: Явная установка кодировки UTF-8

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

Или выполните `chcp 65001` в bat-файле

#### 5. Предупреждение о совместимости torchao

**Проблема**: Несоответствие версий PyTorch nightly и torchao

```
Skipping import of cpp extensions due to incompatible torch version 2.11.0.dev+cu128 for torchao version 0.15.0
```

**Решение**: Только предупреждение, не влияет на работу. Можно безопасно игнорировать.

#### 6. Предупреждение о символических ссылках Hugging Face

**Проблема**: Создание символических ссылок в Windows требует прав администратора

```
huggingface_hub cache-system uses symlinks by default...
```

**Решение**: 
- Включите режим разработчика в настройках Windows
- Или проигнорируйте предупреждение (не влияет на работу)

### Скрипт проверки

Для проверки корректности настройки окружения:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

Ожидаемый вывод (для RTX 5090):

```
PyTorch version: 2.11.0.dev20260123+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5090
GPU Memory: 31.8 GB
```

### Устранение неполадок

| Симптом | Причина | Решение |
|--------|---------|---------|
| `no kernel image is available` | Используется стабильная версия PyTorch | Установите nightly-версию (cu128) |
| `FlashAttention2 cannot be used` | FlashAttention не поддерживается в Windows | Добавьте опцию `--no-flash-attn` |
| `SoX could not be found` | SoX не установлен | Можно игнорировать (не влияет на основной функционал) |
| GPU не распознаётся | Устаревший драйвер CUDA | Установите последний драйвер |
| Искажение символов | Кодировка cp932 | `chcp 65001` или настройка UTF-8 |

## Лицензия

Данный проект распространяется под лицензией [Apache License 2.0](../LICENSE).

### Используемое открытое ПО

| Программное обеспечение | Лицензия | Авторские права |
|------------------------|----------|----------------|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache License 2.0 | Copyright 2026 Alibaba Cloud |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT License | Copyright SYSTRAN |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT License | Copyright OpenAI |

Подробности см. в файле [NOTICE](../NOTICE).

## Отказ от ответственности

### Отказ от ответственности в отношении генерации аудио

- Аудио, генерируемое данной системой, создаётся автоматически с помощью модели ИИ и может содержать неточный или неуместный контент
- Сгенерированное аудио не отражает мнения разработчиков и не является профессиональной консультацией
- Пользователи принимают на себя все риски и ответственность, связанные с использованием, распространением или опорой на сгенерированное аудио

### Предупреждение о клонировании голоса

- **Клонирование или использование голоса другого человека без его согласия может являться нарушением прав на изображение и публичных прав**
- Пожалуйста, используйте функцию клонирования голоса только в законных целях с согласия лица, чей голос клонируется
- Использование в злонамеренных целях, таких как мошенничество, выдача себя за другое лицо, клевета или дипфейки, строго запрещено

### Юридическая ответственность

- Разработчики не несут ответственности за любой ущерб, возникший в результате использования данного ПО
- Вся юридическая ответственность, возникающая из незаконного использования, возлагается на пользователя
- Данное ПО предоставляется «как есть» без каких-либо гарантий

## Благодарности

- Оригинальный разработчик: [Alibaba Cloud Qwen Team](https://github.com/QwenLM)
- Оригинальный репозиторий: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

## Цитирование

Для цитирования оригинального Qwen3-TTS:

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```
