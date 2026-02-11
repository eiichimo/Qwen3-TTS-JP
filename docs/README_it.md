[English](../README.md) | [日本語](README_ja.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Русский](README_ru.md) | [Español](README_es.md) | **Italiano** | [Deutsch](README_de.md) | [Français](README_fr.md) | [Português](README_pt.md)

# Qwen3-TTS-JP

Fork di Qwen3-TTS con **supporto multilingue** e **supporto nativo per Windows**.

L'originale Qwen3-TTS è stato sviluppato principalmente per ambienti Linux, e si raccomanda l'uso di FlashAttention 2. Tuttavia, FlashAttention 2 non funziona su Windows. Questo fork consente l'**esecuzione diretta su Windows senza WSL2 o Docker**, fornisce una GUI in 10 lingue e aggiunge la trascrizione automatica tramite Whisper.

> **Utenti Mac (Apple Silicon):** Per la migliore esperienza su Mac, utilizzate **[Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab)** -- completamente ottimizzato per Apple Silicon con doppio motore MLX + PyTorch, quantizzazione 8bit/4bit e Web UI in 10 lingue.

### Voce personalizzata -- Sintesi vocale con speaker predefiniti
<p align="center">
    <img src="../assets/CustomVoice.png" width="90%"/>
</p>

### Progettazione vocale -- Descrivere le caratteristiche vocali per sintetizzare
<p align="center">
    <img src="../assets/VoiceDesign.png" width="90%"/>
</p>

### Clonazione vocale -- Clonare voce da audio di riferimento
<p align="center">
    <img src="../assets/VoiceClone.png" width="90%"/>
</p>

### Impostazioni -- GPU / VRAM / Informazioni sul modello
<p align="center">
    <img src="../assets/Settings.png" width="90%"/>
</p>

## Progetti correlati

| Piattaforma | Repository | Descrizione |
|:---:|---|---|
| Windows | **Questo repository** | Supporto nativo Windows + interfaccia in 10 lingue |
| macOS (Apple Silicon) | [Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab) | Versione completamente ottimizzata per Apple Silicon Mac (doppio motore MLX + PyTorch, Web UI in 10 lingue) |

## Caratteristiche

### Supporto nativo per Windows

- **FlashAttention 2 non richiesto**: Utilizza SDPA (Scaled Dot Product Attention) standard di PyTorch tramite l'opzione `--no-flash-attn`
- **WSL2/Docker non richiesto**: Esecuzione diretta su Windows
- **Supporto RTX serie 50**: Include istruzioni per l'installazione delle build nightly di PyTorch per l'architettura NVIDIA Blackwell (sm_120)
- **Nessuna dipendenza da SoX**: Funziona senza SoX (vengono mostrati avvisi ma possono essere ignorati)

### Interfaccia Web moderna e supporto multilingue

- **UI in 10 lingue**: giapponese / English / cinese / coreano / russo / spagnolo / Italiano / Deutsch / francese / portoghese -- cambio istantaneo tramite menu a tendina
- **4 schede**: Custom Voice / Voice Design / Voice Clone / Settings -- accesso a tutte le funzionalita indipendentemente dal tipo di modello; i modelli non caricati vengono scaricati automaticamente al primo utilizzo
- **Monitoraggio GPU / VRAM**: visualizzazione dell'utilizzo in tempo reale nella scheda impostazioni, pulizia cache CUDA disponibile
- **Trascrizione automatica Whisper**: automatizza l'inserimento del testo audio di riferimento durante la clonazione vocale ([faster-whisper](https://github.com/SYSTRAN/faster-whisper))
- **Selezione modello Whisper**: scegli tra 5 modelli in base alle esigenze
  - `tiny` - Piu veloce e piccolo (39M parametri)
  - `base` - Veloce (74M parametri)
  - `small` - Bilanciato (244M parametri) *Predefinito
  - `medium` - Alta precisione (769M parametri)
  - `large-v3` - Massima precisione (1550M parametri)

## Requisiti di sistema

- **SO**: Windows 10/11 (ambiente nativo, WSL2 non richiesto)
- **GPU**: GPU NVIDIA (compatibile CUDA)
  - RTX serie 30/40: Funziona con la versione stabile di PyTorch
  - RTX serie 50 (Blackwell): Richiede la build nightly di PyTorch (cu128)
- **Python**: 3.10 o superiore
- **VRAM**: 8 GB o più raccomandati (varia in base alla dimensione del modello)

## Installazione

### 1. Clonare il repository

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-JP.git
cd Qwen3-TTS-JP
```

### 2. Creare e attivare l'ambiente virtuale

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Installare le dipendenze

```bash
pip install -e .
pip install faster-whisper
```

### 4. Installare PyTorch (versione CUDA)

Installare in base alla propria versione di CUDA.

```bash
# Per CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Per RTX serie 50 (sm_120), è necessaria la build nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Utilizzo

### Avviare la GUI

#### Dalla riga di comando

```bash
# Modello CustomVoice (voci preimpostate)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn

# Modello Base (con clonazione vocale)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
```

Aprire `http://127.0.0.1:7860` nel browser.

#### Avvio rapido con file batch (consigliato)

Creare un file batch come il seguente per l'avvio con doppio clic:

**run-gui.bat** (per il modello CustomVoice):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS GUI
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

**run-voice-clone.bat** (per il modello Base / clonazione vocale):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS Voice Clone
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

#### Launcher avanzato (selezione automatica della porta e apertura automatica del browser)

Per un metodo di avvio più comodo, è possibile utilizzare il seguente launcher Python:

<details>
<summary>launch_gui.py (clicca per espandere)</summary>

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
    """Trova una porta disponibile"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Tutte le porte {start_port}-{start_port + max_attempts} sono in uso")

def wait_for_server_and_open_browser(url, timeout=180):
    """Attende l'avvio del server e apre il browser"""
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

Funzionalità:
- **Selezione automatica della porta**: Rileva automaticamente una porta libera se la 7860 è in uso
- **Apertura automatica del browser**: Rileva il completamento dell'avvio del server e apre automaticamente il browser
- **Correzione della codifica**: Supporto codifica UTF-8

### Procedura di clonazione vocale

1. Caricare un file audio in "Audio di riferimento"
2. Selezionare un modello in "Modello Whisper" (il primo download potrebbe richiedere tempo)
3. Cliccare su "Trascrizione automatica"
4. Il risultato della trascrizione viene inserito automaticamente in "Testo dell'audio di riferimento"
5. Modificare il testo se necessario
6. Inserire il "Testo da sintetizzare"
7. Cliccare su "Genera audio"

### Dettagli del supporto nativo Windows

Questo fork ottiene il funzionamento nativo su Windows attraverso le seguenti misure:

| Problema | Originale | Soluzione di questo fork |
|----------|-----------|------------------------|
| FlashAttention 2 | Solo Linux, impossibile compilare su Windows | Utilizzo di SDPA tramite l'opzione `--no-flash-attn` |
| Dipendenza da SoX | Si assume che sia installato | Funziona senza (gli avvisi possono essere ignorati) |
| RTX serie 50 | Non supportato | Istruzioni per la build nightly incluse |
| Configurazione dell'ambiente | conda (orientato a Linux) | venv (standard Windows) |

**Nota**: L'opzione `--no-flash-attn` è obbligatoria. Senza di essa, l'applicazione non si avvierà a causa di un errore di importazione di FlashAttention 2.

## Configurazione dettagliata dell'ambiente nativo Windows

### Problemi tecnici risolti

Durante lo sviluppo di questo fork, sono stati identificati e risolti i seguenti problemi specifici di Windows:

#### 1. Supporto CUDA per RTX serie 50 (Blackwell/sm_120)

**Problema**: La versione stabile di PyTorch non supporta le GPU più recenti come la RTX 5090 (sm_120)

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

**Soluzione**: Utilizzare la versione nightly di PyTorch (cu128)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2. FlashAttention 2 non compatibile con Windows

**Problema**: FlashAttention 2 è esclusivo per Linux e non può essere compilato o eseguito su Windows

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: 
the package flash_attn seems to be not installed.
```

**Soluzione**: Utilizzare SDPA (Scaled Dot Product Attention) standard di PyTorch tramite l'opzione `--no-flash-attn`

| Implementazione Attention | Velocità | Efficienza memoria | Supporto Windows |
|--------------------------|----------|-------------------|-----------------|
| flash_attention_2 | Massima | Ottima | Non supportato |
| sdpa (PyTorch native) | Veloce | Buona | **Supportato** |
| eager (standard) | Normale | Normale | Supportato |

#### 3. Eliminazione della dipendenza da SoX

**Problema**: Alcune elaborazioni audio richiedono SoX, ma non è installato di default su Windows

```
SoX could not be found!
```

**Soluzione**: Le funzionalità di base di Qwen3-TTS funzionano senza SoX. Gli avvisi possono essere ignorati.

#### 4. Caratteri corrotti nella console (codifica cp932)

**Problema**: Negli ambienti giapponesi di Windows, i caratteri non ASCII vengono corrotti a causa della codifica cp932

```
UnicodeEncodeError: 'cp932' codec can't encode character...
```

**Soluzione**: Impostare esplicitamente la codifica UTF-8

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

Oppure eseguire `chcp 65001` nel file batch

#### 5. Avviso di compatibilità torchao

**Problema**: Avviso di incompatibilità di versione tra PyTorch nightly e torchao

```
Skipping import of cpp extensions due to incompatible torch version 2.11.0.dev+cu128 for torchao version 0.15.0
```

**Soluzione**: Solo un avviso, non influisce sul funzionamento. Può essere ignorato.

#### 6. Avviso sui link simbolici di Hugging Face

**Problema**: La creazione di link simbolici su Windows richiede privilegi di amministratore

```
huggingface_hub cache-system uses symlinks by default...
```

**Soluzione**: 
- Attivare la Modalità sviluppatore nelle impostazioni di Windows
- Oppure ignorare l'avviso (non influisce sul funzionamento)

### Script di verifica

Per verificare che l'ambiente sia configurato correttamente:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

Output atteso (per RTX 5090):

```
PyTorch version: 2.11.0.dev20260123+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5090
GPU Memory: 31.8 GB
```

### Risoluzione dei problemi

| Sintomo | Causa | Soluzione |
|---------|-------|----------|
| `no kernel image is available` | Utilizzo di PyTorch stabile | Installare la versione nightly (cu128) |
| `FlashAttention2 cannot be used` | FlashAttention non supportato su Windows | Aggiungere l'opzione `--no-flash-attn` |
| `SoX could not be found` | SoX non installato | Può essere ignorato (non influisce sulle funzionalità di base) |
| GPU non riconosciuta | Driver CUDA obsoleto | Installare il driver più recente |
| Caratteri corrotti | Codifica cp932 | `chcp 65001` o impostazione UTF-8 |

## Licenza

Questo progetto è rilasciato sotto la [Apache License 2.0](../LICENSE).

### Software open source utilizzato

| Software | Licenza | Copyright |
|----------|---------|-----------|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache License 2.0 | Copyright 2026 Alibaba Cloud |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT License | Copyright SYSTRAN |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT License | Copyright OpenAI |

Per i dettagli, consultare il file [NOTICE](../NOTICE).

## Dichiarazione di non responsabilità

### Dichiarazione sulla generazione audio

- L'audio generato da questo sistema è prodotto automaticamente da un modello di IA e potrebbe contenere contenuti imprecisi o inappropriati
- L'audio generato non rappresenta le opinioni degli sviluppatori e non costituisce consulenza professionale
- Gli utenti si assumono tutti i rischi e le responsabilità relative all'uso, alla distribuzione o all'affidamento sull'audio generato

### Avvertenza sulla clonazione vocale

- **Clonare o utilizzare la voce di un'altra persona senza il suo consenso può costituire una violazione dei diritti all'immagine e alla pubblicità**
- Si prega di utilizzare la funzione di clonazione vocale solo per scopi leciti con il consenso della persona la cui voce viene clonata
- L'uso per scopi malevoli come frode, impersonificazione, diffamazione o deepfake è severamente vietato

### Responsabilità legale

- Gli sviluppatori non si assumono alcuna responsabilità per eventuali danni derivanti dall'uso di questo software
- Tutta la responsabilità legale derivante da un uso illegale sarà a carico dell'utente
- Questo software è fornito "così com'è" senza alcuna garanzia

## Ringraziamenti

- Sviluppatore originale: [Alibaba Cloud Qwen Team](https://github.com/QwenLM)
- Repository originale: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

## Citazione

Per citare l'originale Qwen3-TTS:

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```
