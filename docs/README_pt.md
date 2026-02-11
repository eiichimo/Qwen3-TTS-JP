[English](../README.md) | [日本語](README_ja.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Русский](README_ru.md) | [Español](README_es.md) | [Italiano](README_it.md) | [Deutsch](README_de.md) | [Français](README_fr.md) | **Português**

# Qwen3-TTS-JP

Fork do Qwen3-TTS com **suporte multilíngue** e **suporte nativo para Windows**.

O Qwen3-TTS original foi desenvolvido principalmente para ambientes Linux, e o uso do FlashAttention 2 é recomendado. No entanto, o FlashAttention 2 não funciona no Windows. Este fork permite a **execução direta no Windows sem WSL2 ou Docker**, fornece uma interface em 10 idiomas e adiciona a transcrição automática via Whisper.

> **Usuarios de Mac (Apple Silicon):** Para a melhor experiencia no Mac, utilize o **[Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab)** -- totalmente otimizado para Apple Silicon com motor duplo MLX + PyTorch, quantizacao 8bit/4bit e interface Web em 10 idiomas.

### Voz personalizada -- Síntese de voz com falantes predefinidos
<p align="center">
    <img src="../assets/CustomVoice.png" width="90%"/>
</p>

### Design de voz -- Descrever características de voz para sintetizar
<p align="center">
    <img src="../assets/VoiceDesign.png" width="90%"/>
</p>

### Clonagem de voz -- Clonar voz a partir de áudio de referência
<p align="center">
    <img src="../assets/VoiceClone.png" width="90%"/>
</p>

### Configurações -- GPU / VRAM / Informações do modelo
<p align="center">
    <img src="../assets/Settings.png" width="90%"/>
</p>

## Projetos relacionados

| Plataforma | Repositório | Descrição |
|:---:|---|---|
| Windows | **Este repositório** | Suporte nativo Windows + interface em 10 idiomas |
| macOS (Apple Silicon) | [Qwen3-TTS-Mac-GeneLab](https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab) | Versão totalmente otimizada para Apple Silicon Mac (motor duplo MLX + PyTorch, Web UI em 10 idiomas) |

## Características

### Suporte nativo para Windows

- **FlashAttention 2 não necessário**: Utiliza o SDPA (Scaled Dot Product Attention) padrão do PyTorch através da opção `--no-flash-attn`
- **WSL2/Docker não necessário**: Execução direta no Windows
- **Suporte para RTX série 50**: Inclui instruções para instalação das builds nightly do PyTorch para a arquitetura NVIDIA Blackwell (sm_120)
- **Sem dependência do SoX**: Funciona sem o SoX (avisos são exibidos mas podem ser ignorados)

### Interface Web moderna e suporte multilíngue

- **UI em 10 idiomas**: japones / English / chines / coreano / russo / espanhol / italiano / alemao / Frances / Portugues -- troca instantanea pelo menu suspenso
- **4 abas**: Custom Voice / Voice Design / Voice Clone / Settings -- acesso a todas as funcionalidades independentemente do tipo de modelo; modelos nao carregados sao baixados automaticamente no primeiro uso
- **Monitoramento de GPU / VRAM**: consulta de uso em tempo real na aba de configuracoes, limpeza de cache CUDA disponivel
- **Transcricao automatica Whisper**: automatiza a entrada de texto do audio de referencia ao clonar vozes ([faster-whisper](https://github.com/SYSTRAN/faster-whisper))
- **Selecao de modelo Whisper**: escolha entre 5 modelos conforme sua necessidade
  - `tiny` - Mais rapido e menor (39M parametros)
  - `base` - Rapido (74M parametros)
  - `small` - Equilibrado (244M parametros) *Padrao
  - `medium` - Alta precisao (769M parametros)
  - `large-v3` - Precisao maxima (1550M parametros)

## Requisitos do sistema

- **SO**: Windows 10/11 (ambiente nativo, sem necessidade de WSL2)
- **GPU**: GPU NVIDIA (compatível com CUDA)
  - RTX série 30/40: Funciona com a versão estável do PyTorch
  - RTX série 50 (Blackwell): Requer build nightly do PyTorch (cu128)
- **Python**: 3.10 ou superior
- **VRAM**: 8 GB ou mais recomendados (varia conforme o tamanho do modelo)

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-JP.git
cd Qwen3-TTS-JP
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install -e .
pip install faster-whisper
```

### 4. Instalar PyTorch (versão CUDA)

Instale de acordo com sua versão do CUDA.

```bash
# Para CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Para RTX série 50 (sm_120), a build nightly é necessária
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Uso

### Iniciar a GUI

#### Pela linha de comando

```bash
# Modelo CustomVoice (vozes predefinidas)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn

# Modelo Base (com clonagem de voz)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
```

Abra `http://127.0.0.1:7860` no seu navegador.

#### Início rápido com arquivos batch (recomendado)

Crie um arquivo batch como o seguinte para iniciar com duplo clique:

**run-gui.bat** (para o modelo CustomVoice):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS GUI
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

**run-voice-clone.bat** (para o modelo Base / clonagem de voz):
```batch
@echo off
chcp 65001 >nul
title Qwen3-TTS Voice Clone
cd /d "%~dp0"
.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 127.0.0.1 --port 7860 --no-flash-attn
pause
```

#### Launcher avançado (seleção automática de porta e abertura automática do navegador)

Para um método de início mais prático, você pode usar o seguinte launcher Python:

<details>
<summary>launch_gui.py (clique para expandir)</summary>

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
    """Encontrar uma porta disponível"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Todas as portas {start_port}-{start_port + max_attempts} estão em uso")

def wait_for_server_and_open_browser(url, timeout=180):
    """Aguardar o início do servidor e abrir o navegador"""
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
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # ou CustomVoice
        "--ip", "127.0.0.1",
        "--port", str(port),
        "--no-flash-attn"
    ])

if __name__ == "__main__":
    main()
```

</details>

Funcionalidades:
- **Seleção automática de porta**: Detecta automaticamente uma porta livre se a 7860 estiver em uso
- **Abertura automática do navegador**: Detecta a conclusão do início do servidor e abre automaticamente o navegador
- **Correção de codificação**: Suporte a codificação UTF-8

### Etapas da clonagem de voz

1. Carregue um arquivo de áudio em "Áudio de referência"
2. Selecione um modelo em "Modelo Whisper" (o primeiro download pode demorar)
3. Clique em "Transcrição automática"
4. O resultado da transcrição é automaticamente inserido em "Texto do áudio de referência"
5. Edite o texto se necessário
6. Insira o "Texto a sintetizar"
7. Clique em "Gerar áudio"

### Detalhes do suporte nativo Windows

Este fork alcança operação nativa no Windows através das seguintes medidas:

| Problema | Original | Solução deste fork |
|----------|----------|-------------------|
| FlashAttention 2 | Apenas Linux, não pode ser compilado no Windows | Uso de SDPA via opção `--no-flash-attn` |
| Dependência do SoX | Pressupõe instalação | Funciona sem ele (avisos podem ser ignorados) |
| RTX série 50 | Não suportado | Instruções de instalação da build nightly incluídas |
| Configuração do ambiente | conda (orientado para Linux) | venv (padrão Windows) |

**Nota**: A opção `--no-flash-attn` é obrigatória. Sem ela, a aplicação não iniciará devido a um erro de importação do FlashAttention 2.

## Configuração detalhada do ambiente nativo Windows

### Problemas técnicos resolvidos

Durante o desenvolvimento deste fork, os seguintes problemas específicos do Windows foram identificados e resolvidos:

#### 1. Suporte CUDA para RTX série 50 (Blackwell/sm_120)

**Problema**: A versão estável do PyTorch não suporta as GPUs mais recentes como a RTX 5090 (sm_120)

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

**Solução**: Usar a versão nightly do PyTorch (cu128)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2. FlashAttention 2 não compatível com Windows

**Problema**: FlashAttention 2 é exclusivo para Linux e não pode ser compilado nem executado no Windows

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: 
the package flash_attn seems to be not installed.
```

**Solução**: Usar o SDPA (Scaled Dot Product Attention) padrão do PyTorch via opção `--no-flash-attn`

| Implementação de Attention | Velocidade | Eficiência de memória | Suporte Windows |
|---------------------------|-----------|----------------------|-----------------|
| flash_attention_2 | Máxima | Ótima | Não suportado |
| sdpa (PyTorch native) | Rápida | Boa | **Suportado** |
| eager (padrão) | Normal | Normal | Suportado |

#### 3. Eliminação da dependência do SoX

**Problema**: Alguns processamentos de áudio requerem o SoX, mas ele não está instalado por padrão no Windows

```
SoX could not be found!
```

**Solução**: As funcionalidades básicas do Qwen3-TTS funcionam sem o SoX. Os avisos podem ser ignorados com segurança.

#### 4. Caracteres corrompidos no console (codificação cp932)

**Problema**: Em ambientes japoneses do Windows, caracteres não ASCII ficam corrompidos devido à codificação cp932

```
UnicodeEncodeError: 'cp932' codec can't encode character...
```

**Solução**: Definir explicitamente a codificação UTF-8

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

Ou executar `chcp 65001` no arquivo batch

#### 5. Aviso de compatibilidade do torchao

**Problema**: Aviso de incompatibilidade de versão entre o PyTorch nightly e o torchao

```
Skipping import of cpp extensions due to incompatible torch version 2.11.0.dev+cu128 for torchao version 0.15.0
```

**Solução**: Apenas um aviso, sem impacto no funcionamento. Pode ser ignorado.

#### 6. Aviso de links simbólicos do Hugging Face

**Problema**: A criação de links simbólicos no Windows requer privilégios de administrador

```
huggingface_hub cache-system uses symlinks by default...
```

**Solução**: 
- Ativar o Modo de Desenvolvedor nas configurações do Windows
- Ou ignorar o aviso (sem impacto no funcionamento)

### Script de verificação

Para verificar se o ambiente está configurado corretamente:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

Saída esperada (para RTX 5090):

```
PyTorch version: 2.11.0.dev20260123+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5090
GPU Memory: 31.8 GB
```

### Solução de problemas

| Sintoma | Causa | Solução |
|---------|-------|---------|
| `no kernel image is available` | Uso da versão estável do PyTorch | Instalar a versão nightly (cu128) |
| `FlashAttention2 cannot be used` | FlashAttention não compatível com Windows | Adicionar opção `--no-flash-attn` |
| `SoX could not be found` | SoX não instalado | Pode ser ignorado (sem impacto nas funcionalidades básicas) |
| GPU não reconhecida | Driver CUDA desatualizado | Instalar o driver mais recente |
| Caracteres corrompidos | Codificação cp932 | `chcp 65001` ou configuração UTF-8 |

## Licença

Este projeto é publicado sob a [Apache License 2.0](../LICENSE).

### Software de código aberto utilizado

| Software | Licença | Direitos autorais |
|----------|---------|-------------------|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache License 2.0 | Copyright 2026 Alibaba Cloud |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT License | Copyright SYSTRAN |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT License | Copyright OpenAI |

Para mais detalhes, consulte o arquivo [NOTICE](../NOTICE).

## Isenção de responsabilidade

### Isenção sobre geração de áudio

- O áudio gerado por este sistema é produzido automaticamente por um modelo de IA e pode conter conteúdo impreciso ou inadequado
- O áudio gerado não representa as opiniões dos desenvolvedores e não constitui aconselhamento profissional
- Os usuários assumem todos os riscos e responsabilidades relacionados ao uso, distribuição ou dependência do áudio gerado

### Aviso sobre clonagem de voz

- **Clonar ou usar a voz de outra pessoa sem seu consentimento pode constituir uma violação dos direitos de imagem e publicidade**
- Por favor, utilize a função de clonagem de voz apenas para fins legais com o consentimento da pessoa cuja voz está sendo clonada
- O uso para fins maliciosos como fraude, personificação, difamação ou deepfakes é estritamente proibido

### Responsabilidade legal

- Os desenvolvedores não assumem responsabilidade por quaisquer danos decorrentes do uso deste software
- Toda responsabilidade legal decorrente do uso ilegal será assumida pelo usuário
- Este software é fornecido "como está" sem qualquer garantia

## Agradecimentos

- Desenvolvedor original: [Alibaba Cloud Qwen Team](https://github.com/QwenLM)
- Repositório original: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

## Citação

Para citar o Qwen3-TTS original:

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```
