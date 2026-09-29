# TTS-with-RVC 0.1.10

Generate speech with Edge TTS and convert it to a selected voice with a
PyTorch RVC model. Use a `.pth` voice model and an optional Faiss `.index` file.

For ONNX models, use [TTS-with-RVC-ONNX](https://github.com/Atm4x/tts-with-rvc/tree/releases-onnx).
The two packages share the `tts_with_rvc` import name; use a separate Python
environment for each variant.

## Release notes

### 0.1.10 — September 30, 2026

- **MultiGPU support:** create independent `TTS_RVC` or `RVCConverter` instances
  on `cuda:0`, `cuda:1`, and other available GPUs in the same process.
- Each converter owns its model, runtime configuration and F0 predictor state.
  Tensor allocation follows the selected device and precision.
- Device and precision changes rebuild the instance's runtime; failed
  reconfiguration preserves the previous runtime.
- Added automatic precision selection using the selected GPU's capabilities,
  memory-based chunking and explicit `is_half` / `set_precision()` control.
- Added `generate_async()`, `close()` and context-manager support.
- Model downloads can be directed to `models_dir`. Temporary TTS files are
  cleaned up after conversion, including when conversion fails.
- Package exports are lazy; importing the package does not patch asyncio or
  change the host application's logger levels.
- Migrated packaging to `pyproject.toml` and added CI and release workflows.

### Earlier releases

- **0.1.9.2 — September 17, 2025:** constrained CFFI to a compatible version.
- **0.1.9 — March 31, 2025:** fixes including fairseq installation on Linux.
- **0.1.8 — March 30, 2025:** more RVC controls, FCPE support and PyPI installation.
- **0.1.6 — March 28, 2025:** updated RVC inference and reduced dependencies.
- **0.1.5 — February 21, 2025:** removed `rvc_path` and added F0 selection.
- **0.1.4 — November 22, 2024:** added index path and blending controls.

## Requirements and installation

- Python 3.10, 3.11 or 3.12.
- PyTorch installed for your platform. NVIDIA CUDA is recommended; CPU is
  available, and the runtime also supports MPS device selection.
- FFmpeg installed and accessible through `PATH`.
- A compatible RVC `.pth` model and, optionally, its `.index` file.
- Network access for Edge TTS and for downloading missing auxiliary models.

Install PyTorch using the [official guide](https://pytorch.org/get-started/locally/), then:

```bash
pip install tts-with-rvc
```

To install the current release branch before its PyPI publication:

```bash
pip install "git+https://github.com/Atm4x/tts-with-rvc.git@releases"
```

For local development, run `pip install -e .` from the repository root.

## Basic usage

```python
from tts_with_rvc import TTS_RVC

with TTS_RVC(
    model_path="models/voice.pth",
    device="cuda:0",
    voice="ru-RU-DmitryNeural",
    output_directory="output",
) as tts:
    output_path = tts(text="Hello, world!", pitch=0)
    print(output_path)
```

Pass `index_path="models/voice.index"` to enable index blending. A missing index
is disabled with a warning. The result is the path to the converted audio file.

## MultiGPU

Each instance uses one GPU and owns its model state. Create one instance per
device and distribute requests from your application:

```python
from concurrent.futures import ThreadPoolExecutor
from tts_with_rvc import TTS_RVC

with TTS_RVC(
    model_path="models/voice.pth",
    device="cuda:0",
    output_directory="output/gpu0",
) as first, TTS_RVC(
    model_path="models/voice.pth",
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

A bare `"cuda"` selects `cuda:0`. With `device=None`, the runtime selects CUDA,
then MPS, then CPU according to availability. An explicitly requested unavailable
device raises an error. Each GPU needs enough memory for its own model state.
Use separate instances for parallel conversion; an individual converter locks
its conversion operations.

## Async usage

```python
import asyncio
from tts_with_rvc import TTS_RVC

async def main():
    with TTS_RVC(model_path="models/voice.pth", device="cuda:0") as tts:
        output_path = await tts.generate_async("Hello from an async application.")
        print(output_path)

asyncio.run(main())
```

## Constructor options

| Option | Default | Purpose |
|---|---|---|
| `model_path` | required | RVC `.pth` model |
| `device` | `None` | Automatic selection, `"cuda:N"`, `"cpu"` or `"mps"` |
| `voice` | `"ru-RU-DmitryNeural"` | Edge TTS voice |
| `index_path` | `""` | Optional Faiss index |
| `f0_method` | `"rmvpe"` | `rmvpe`, `fcpe`, `pm`, `harvest`, `dio` or `crepe` |
| `is_half` | `None` | Automatic precision; `True` for FP16, `False` for FP32 |
| `models_dir` | `None` | Location for downloaded auxiliary models |
| `tmp_directory` | `None` | Temporary Edge TTS audio directory; defaults to system temp |
| `output_directory` | `None` | Converted audio directory; defaults to `temp` relative to the working directory |

`input_directory` is deprecated; use `tmp_directory`.
Automatic precision uses FP32 on CPU/MPS and chooses precision from CUDA
capabilities on NVIDIA GPUs. CPU FP16 is rejected. FCPE uses FP32 internally.

## Conversion controls

`tts(text, ...)` and `await tts.generate_async(text, ...)` accept:

| Option | Default | Purpose |
|---|---|---|
| `pitch` | `0` | RVC pitch shift in semitones |
| `tts_rate`, `tts_volume`, `tts_pitch` | `0` | Edge TTS rate/volume percentages and pitch in Hz |
| `output_filename` | `None` | Output name; a unique name is generated by default |
| `index_rate` | `0.75` | Index blending strength |
| `is_half` | `None` | Use the instance's precision policy or override it |
| `f0method` | `None` | Override the instance's F0 method for this call |
| `file_index2` | `""` | Secondary index path |
| `filter_radius` | `3` | Pitch median-filter control |
| `resample_sr` | `0` | Output resampling; zero retains the model rate |
| `rms_mix_rate` | `0.5` | Output volume-envelope blending |
| `protect` | `0.33` | Unvoiced-consonant protection; `0.5` disables protection |
| `verbose` | `False` | Conversion logging |

## Model, device and voice changes

- `tts.set_voice(voice)` changes the Edge TTS voice.
- `tts.set_index_path(path)` changes the index.
- `tts.set_model(path)` selects the RVC model for subsequent calls.
- `tts.device = "cuda:1"` changes the device by replacing the runtime.
- `tts.set_precision(None)` restores automatic precision; pass `False` for FP32.
- `tts.set_output_directory(path)` changes the output directory.
- `tts.close()` releases converter resources. A `with` block closes automatically.

For conversion of existing audio without Edge TTS:

```python
from tts_with_rvc import TTS_RVC

with TTS_RVC(model_path="models/voice.pth", device="cuda:0") as tts:
    output_path = tts.voiceover_file("input.wav", filename="converted.wav")
```

For lower-level integrations, `RVCConverter` provides an instance-owned converter;
`rvc_convert` remains available as a convenience function.
Voice discovery is exposed as the async module-level function `get_voices()`.

## Text parameters

`process_args()` extracts `--tts-rate`, `--tts-volume`, `--tts-pitch` and
`--rvc-pitch` from text, returning `[rate, volume, tts_pitch, rvc_pitch]` and
the cleaned message:

```python
args, message = tts.process_args("Hello --tts-rate -10 --rvc-pitch -2")
output_path = tts(
    message, tts_rate=args[0], tts_volume=args[1],
    tts_pitch=args[2], pitch=args[3],
)
```

## Troubleshooting

- Unavailable CUDA device: check the device index and install a CUDA-enabled PyTorch build.
- Audio decoding errors: check FFmpeg and the input audio path.
- Model or index errors: check paths and compatibility with the selected RVC model.
- In an existing asyncio application, use `generate_async()` to avoid blocking its event loop.

## Development and releases

CI tests Python 3.10–3.12 on Windows and Linux, then builds wheel + sdist.
The `releases` branch contains release candidates. PyPI publishing is enabled
with `PYPI_PUBLISH_ENABLED=true` after configuring a token or Trusted Publisher.
See [release setup and version commands](.github/RELEASING.md).

## Acknowledgements and license

Based on the [RVC Project](https://github.com/RVC-Project/).
Distributed under the [MIT License](LICENSE).
Maintained by [Atm4x](https://github.com/Atm4x) (Artem Dikarev).
