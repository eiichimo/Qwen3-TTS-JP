# API Specification (Static)

This document provides a static API reference for integrating this project from other projects.
The runtime API page (`/?view=api`) is still available, but this file can be reviewed without launching the server.

## Scope

- API type: Gradio event API (`api_name` based)
- Target: local/fork usage (same operating assumption as `docs/security.md`)
- Auth: not implemented by default (local use assumption)

## Base URL

- Default: `http://127.0.0.1:7860`
- API page (runtime): `http://127.0.0.1:7860/?view=api`

## Public Endpoints

The following endpoints are explicitly named and intended for external integration.

| API Name | Purpose |
|---|---|
| `/custom_voice_generate` | Synthesize speech with preset speaker + optional instruction |
| `/voice_design_generate` | Synthesize speech from free-form voice description |
| `/voice_clone_transcribe` | Transcribe reference audio with Whisper |
| `/voice_clone_generate` | Voice clone synthesis from reference audio/text |
| `/voice_clone_prompt_transcribe` | Transcribe reference audio for prompt export |
| `/voice_clone_prompt_save` | Export voice-clone prompt file (`.pt`) |
| `/voice_clone_prompt_generate` | Synthesize using uploaded prompt file |

## Endpoint Details

### `/custom_voice_generate`

- Inputs:
  - `text` (`str`)
  - `speaker_display_name` (`str`) - UI display name from speaker dropdown
  - `language_display_name` (`str`) - UI display name (e.g. `Auto`)
  - `emotion_display_name` (`str`)
  - `speed` (`float`, expected `0.5..2.0`)
  - `custom_instruction` (`str`)
- Outputs:
  - `audio` (generated waveform)
  - `status` (`str`)
- Validation highlights:
  - text is required
  - speaker is required

### `/voice_design_generate`

- Inputs:
  - `text` (`str`)
  - `language_display_name` (`str`)
  - `voice_description` (`str`)
  - `speed` (`float`, expected `0.5..2.0`)
- Outputs:
  - `audio`
  - `status` (`str`)
- Validation highlights:
  - text is required
  - voice description is required

### `/voice_clone_transcribe`

- Inputs:
  - `reference_audio`
  - `whisper_model` (`str`): one of `tiny|base|small|medium|large-v3`
- Outputs:
  - `transcribed_text` (`str`)
  - `status` (`str`)
- Validation highlights:
  - reference audio is required

### `/voice_clone_generate`

- Inputs:
  - `reference_audio`
  - `reference_text` (`str`)
  - `x_vector_only_mode` (`bool`)
  - `text` (`str`) - synthesis target text
  - `language_display_name` (`str`)
  - `speed` (`float`, expected `0.5..2.0`)
- Outputs:
  - `audio`
  - `status` (`str`)
- Validation highlights:
  - synthesis text is required
  - reference audio is required
  - reference text is required unless `x_vector_only_mode=true`

### `/voice_clone_prompt_transcribe`

- Inputs:
  - `reference_audio`
  - `whisper_model` (`str`)
- Outputs:
  - `transcribed_text` (`str`)
  - `status` (`str`)

### `/voice_clone_prompt_save`

- Inputs:
  - `reference_audio`
  - `reference_text` (`str`)
  - `x_vector_only_mode` (`bool`)
- Outputs:
  - `prompt_file` (`.pt`)
  - `status` (`str`)
- Validation highlights:
  - reference audio is required
  - reference text is required unless `x_vector_only_mode=true`

### `/voice_clone_prompt_generate`

- Inputs:
  - `prompt_file` (`.pt`)
  - `text` (`str`)
  - `language_display_name` (`str`)
  - `speed` (`float`, expected `0.5..2.0`)
- Outputs:
  - `audio`
  - `status` (`str`)
- Validation highlights:
  - prompt file is required
  - text is required
  - prompt file size must be `<= 10 MiB`

## Recommended Client Usage

Use `gradio_client` instead of hand-crafting raw HTTP calls.

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")

# Example: voice design generation
result = client.predict(
    "Hello from static API docs.",
    "Auto",
    "A calm and warm female voice",
    1.0,
    api_name="/voice_design_generate",
)
print(result)
```

## Stability Notes

- API names listed above are intentionally fixed for integration.
- Internal UI-only events are marked private and are excluded from this spec.
- If endpoint names or required parameters change, this document should be updated in the same change set.
