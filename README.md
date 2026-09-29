# TTS-with-RVC-ONNX 0.1.10

Generate speech with Edge TTS and convert it to a selected voice using an ONNX
RVC model. Inference supports CPU, NVIDIA CUDA and Windows DirectML providers.

For PyTorch `.pth` models, use [TTS-with-RVC](https://github.com/Atm4x/tts-with-rvc/tree/releases).
Both distributions use the `tts_with_rvc` import name; install them in separate
Python environments.

## Release notes

### 0.1.10 — September 30, 2026

- **MultiGPU support:** create independent instances for `cuda:0`, `cuda:1`,
  `dml:0`, `dml:1` and other available adapters in the same process.
- The RVC, ContentVec and ONNX RMVPE sessions receive the selected provider and
  adapter ID. Unavailable providers and activation failures raise explicit errors.
- Each converter owns its sessions, predictor cache and random generator.
- Device changes build a new backend before closing the old one; failed changes
  preserve the working backend. Model and sampling settings have dedicated setters.
- Added `async_call()`, `close()` and context-manager support, with request-state
  snapshots and temporary audio cleanup after conversion.
- Added `models_dir`, configurable `vec_path` and per-instance `random_seed`.
- Imports are lazy; asyncio and parent logger configuration are left to the host.
- Migrated packaging to `pyproject.toml` and added CI and release workflows.

### Earlier releases

- **0.1.9.4 — September 17, 2025:** constrained CFFI to a compatible version.
- **September 3, 2025:** dependency adjustment for NumPy compatibility on AMD setups.
- **0.1.9 — April 10, 2025:** synchronized RVC controls, ONNX RMVPE support,
  pitch-padding fixes and runtime device switching.
- **0.1.6:** initial ONNX support.

## Requirements and installation

- Python 3.10, 3.11 or 3.12.
- An ONNX RVC voice model (`.onnx`) and optionally a compatible Faiss `.index`.
- FFmpeg accessible through `PATH`.
- For GPU inference, drivers and an ONNX Runtime build exposing the requested provider.
- Network access for Edge TTS and missing auxiliary model downloads.

Choose an installation for your backend:

```bash
pip install tts-with-rvc-onnx
```

```bash
pip install "tts-with-rvc-onnx[cuda]"
```

```bash
pip install "tts-with-rvc-onnx[dml]"
```

Use a separate environment for each runtime setup. Check the available providers:

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

CUDA requires `CUDAExecutionProvider`; DirectML requires `DmlExecutionProvider`.
PyTorch is also a declared dependency and is used in audio processing even when
voice conversion and RMVPE run through ONNX.

To install the current release branch before its PyPI publication:

```bash
pip install "git+https://github.com/Atm4x/tts-with-rvc.git@releases-onnx"
```

For local development, run `pip install -e .` from the repository root.

## Basic usage

```python
from tts_with_rvc import TTS_RVC

with TTS_RVC(
    model_path="models/voice.onnx",
    device="cpu",
    voice="ru-RU-DmitryNeural",
    output_directory="output",
) as tts:
    output_path = tts(text="Hello, world!", pitch=0)
    print(output_path)
```

For NVIDIA GPUs, select `device="cuda:0"`. For DirectML, select `device="dml:0"`.
Pass `index_path="models/voice.index"` to enable index blending.
The result is the path to the converted audio file.

## MultiGPU and adapter selection

Each instance uses one adapter. Create a separate instance per GPU and schedule
requests in your application:

```python
from concurrent.futures import ThreadPoolExecutor
from tts_with_rvc import TTS_RVC

with TTS_RVC(
    model_path="models/voice.onnx",
    device="cuda:0",
    output_directory="output/gpu0",
) as first, TTS_RVC(
    model_path="models/voice.onnx",
    device="cuda:1",
    output_directory="output/gpu1",
) as second:
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(first, "First request."),
            pool.submit(second, "Second request."),
        ]
        output_paths = [job.result() for job in jobs]
        print(output_paths)
```

For DirectML adapters, use `dml:0` and `dml:1` with the DirectML runtime.
A bare `"cuda"` or `"dml"` selects adapter zero. Device IDs are interpreted by
the selected provider. Each adapter needs memory for its own model sessions.
An individual converter locks conversion operations; use separate instances
to process independent requests in parallel.

## Async usage

```python
import asyncio
from tts_with_rvc import TTS_RVC

async def main():
    with TTS_RVC(model_path="models/voice.onnx", device="cpu") as tts:
        output_path = await tts.async_call("Hello from an async application.")
        print(output_path)

asyncio.run(main())
```

## Constructor options

| Option | Default | Purpose |
|---|---|---|
| `model_path` | required | ONNX RVC model |
| `device` | `"dml"` | `"cpu"`, `"cuda:N"` or `"dml:N"` |
| `voice` | `"ru-RU-DmitryNeural"` | Edge TTS voice |
| `index_path` | `""` | Optional Faiss index |
| `f0_method` | `"pm"` | `pm`, `harvest`, `dio` or `rmvpe` |
| `sampling_rate` | `40000` | RVC model sample rate |
| `hop_size` | `512` | RVC model hop size |
| `vec_path` | `"vec-768-layer-12.onnx"` | ContentVec model path/name |
| `models_dir` | `None` | Directory for auxiliary models |
| `random_seed` | `None` | Seed for the instance's noise generator |
| `tmp_directory` | `None` | Defaults to `tts_with_rvc_onnx` under system temp |
| `output_directory` | `None` | Defaults to `tts_with_rvc_onnx/output` under system temp |

`input_directory` is deprecated; use `tmp_directory`.
Choose the sample rate and hop size that match your exported RVC model.
Precision follows the exported ONNX model and runtime.

Missing ContentVec and RMVPE models are resolved from local paths,
`models_dir`, or Hugging Face. RMVPE is loaded when selected as the F0 method.

## Conversion controls

`tts(text, ...)` and `await tts.async_call(text, ...)` accept:

| Option | Default | Purpose |
|---|---|---|
| `pitch` | `0` | RVC pitch shift in semitones |
| `tts_rate`, `tts_volume`, `tts_pitch` | `0` | Edge TTS rate/volume percentages and pitch in Hz |
| `output_filename` | `None` | Output name; a unique name is generated by default |
| `index_rate` | `0.75` | Index blending strength |
| `f0method` | `None` | Override F0 method for this call |
| `file_index2` | `""` | Secondary index path |
| `filter_radius` | `3` | Pitch median-filter control |
| `resample_sr` | `0` | Output resampling; zero retains the model rate |
| `rms_mix_rate` | `0.5` | Output volume-envelope blending |
| `protect` | `0.33` | Unvoiced-consonant protection; `0.5` disables protection |
| `verbose` | `False` | Conversion logging |

## Runtime and model changes

- `tts.set_device("cuda:1")` changes the selected adapter.
- `tts.set_model(path)` loads a new ONNX RVC model.
- `tts.set_sampling_params(sr, hop)` updates model sampling settings.
- `tts.set_voice(voice)` changes the Edge TTS voice.
- `tts.set_index_path(path)` changes the index.
- `tts.set_output_directory(path)` changes the output directory.
- `tts.close()` releases sessions and predictor resources; `with` closes automatically.

To convert existing audio without Edge TTS:

```python
from tts_with_rvc import TTS_RVC

with TTS_RVC(model_path="models/voice.onnx", device="cpu") as tts:
    output_path = tts.voiceover_file("input.wav", output_filename="converted.wav")
```

`OnnxRVCConverter` is also exported for lower-level integrations.

## Text parameters

`process_args()` extracts `--tts-rate`, `--tts-volume`, `--tts-pitch` and
`--rvc-pitch`, returning `[rate, volume, tts_pitch, rvc_pitch]` and the cleaned text:

```python
args, message = tts.process_args("Hello --tts-rate -10 --rvc-pitch -2")
output_path = tts(
    message, tts_rate=args[0], tts_volume=args[1],
    tts_pitch=args[2], pitch=args[3],
)
```

## Troubleshooting

- Missing provider: inspect `onnxruntime.get_available_providers()` and your runtime installation.
- Provider activation failure: check drivers, device ID and runtime compatibility.
- Model/index mismatch: use the ContentVec and Faiss index expected by your voice model.
- Audio decoding failure: check FFmpeg and input paths.
- Unsupported F0 method: choose `pm`, `harvest`, `dio` or ONNX `rmvpe`.
- In an existing asyncio application, use `async_call()` to avoid blocking its event loop.

## Development and releases

CI tests Python 3.10–3.12 on Windows and Linux, then builds wheel + sdist.
The `releases-onnx` branch contains release candidates. Publishing is controlled
by `PYPI_PUBLISH_ENABLED`; see [release setup and version commands](.github/RELEASING.md).

## Acknowledgements and license

Based on the [RVC Project](https://github.com/RVC-Project/).
Distributed under the [MIT License](LICENSE).
Maintained by [Atm4x](https://github.com/Atm4x) (Artem Dikarev).
