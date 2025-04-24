# **TTS-with-RVC-ONNX** 0.1.9

***TTS-with-RVC-ONNX** (Text-to-Speech with RVC using ONNX)* is a package designed to enhance the capabilities of *text-to-speech (TTS)* systems by introducing an *RVC* module running on the ONNX Runtime. The package enables users to not only convert text into speech but also personalize and customize the voice output according to their preferences with RVC support, optimized for various hardware backends (DirectML, CUDA, CPU).

ONNX Runtime is used for RVC inference, potentially leveraging hardware acceleration (DirectML on Windows/AMD, CUDA on NVIDIA). PyTorch is required *only* for specific F0 predictors (`rmvpe`).

**It may contain bugs. Report an issue in case of error.**

## Release notes

**0.1.9** - April 10, 2025: Current ONNX Branch Sync
*   Synced RVC parameters with main branch 0.1.9 (`rms_mix_rate`, `protect`, `filter_radius`, `resample_sr`, `file_index2`, `verbose`).
*   Added support for F0 predictors: `rmvpe` (using ONNX), `pm`, `dio`, `harvest`.
*   Fixed F0 length mismatch issue and implemented correct audio padding.
*   Added `set_device` method to switch ONNX Runtime providers.
*   Updated dependencies and ONNX Runtime selection.

*(Based on main branch 0.1.9)*

**0.1.6** - Initial ONNX support.

## Prerequisites

You must have **Python >= 3.8 and <= 3.12** installed (3.12 is recommended).
You must have **ONNX Runtime** compatible hardware/drivers if using GPU acceleration (DirectML for AMD on Windows, CUDA for NVIDIA). The CPU provider works generally.
*   **PyTorch** is required *only* if using `f0_method='rmvpe'`.
*   **FFmpeg** must be installed and accessible in your system's PATH or placed in the script's directory. Download from [ffmpeg.org](https://ffmpeg.org/download.html).

## **Installation**

1.  Install the package using pip:
    CPU Version:
    ```bash
    pip install tts-with-rvc-onnx
    ```

    CUDA version:
    ```bash
    pip install tts-with-rvc-onnx[cuda]
    ```

    DML version (recommedned for AMD):
    ```bash
    pip install tts-with-rvc-onnx[dml]
    ```

2.  Ensure **FFmpeg** is installed and accessible (see Prerequisites).

## How it Works

1.  **Text-to-Speech (TTS):** Uses `edge-tts` to convert input text into speech, saved as a temporary audio file in the `tmp_directory`.
2.  **RVC (ONNX):** With the `.onnx` file provided, the RVC module (via ONNX Runtime) reads the temporary audio file, processes it (feature extraction, F0, conversion, index lookup), and generates a new audio file saved in `output_directory` with the voice replaced.

## Usage

TTS-with-RVC-ONNX has a class called `TTS_RVC`.

**Constructor Parameters:**

*   `model_path` (str): **Required.** Path to your `.onnx` RVC model file.

*And optional parameters:*

*   `voice` (str): Voice from `edge-tts` list (default: `"ru-RU-DmitryNeural"`).

*   `device` (str): ONNX Runtime provider (`"dml"`, `"cuda:0"`, `"cpu"`, etc.). Defaults to `"dml"`.

*   `tmp_directory` (str): Path to directory for temporary TTS files (default: system temp folder).

*   `output_directory` (str): Directory for saving final voiced audio (default: `"temp/"`).

*   `index_path` (str): Path to the Faiss `.index` file for voice adjustments (default: `""`).

*   `f0_method` (str): Method for calculating pitch. Available: `'rmvpe'`, `'pm'`, `'harvest'`, `'dio'`, `'crepe'`. Defaults to `"rmvpe"`.

*   `sampling_rate` (int): Target sample rate of the RVC model (default: `40000`).

*   `hop_size` (int): Hop size of the RVC model (default: `512`).

*Deprecated:*

*   `input_directory`: Use `tmp_directory` instead.

**Initialization Example:**

```python
from tts_with_rvc_onnx import TTS_RVC

tts = TTS_RVC(model_path="models/YourModel.onnx",
                index_path="logs/YourIndex.index",
                f0_method="rmvpe",
                device="dml") # Or "cuda:0", "cpu"
```

`tts.get_voices()` **is disabled indefinitely due to the problems**

Next, set the voice for TTS with `tts.set_voice()` function:

```python
tts.set_voice("ru-RU-DmitryNeural")
```

Setting the appropriate language is necessary if you are using other languages for voiceovers!

And final step is calling `tts` (the `__call__` method) to generate and replace voice:

```python
path = tts(text="Привет, мир!", pitch=6, index_rate=0.50)
```

**`__call__` Parameters:**

*   `text` (str): **Required.** Text for TTS.

*   `pitch` (int, optional): Pitch change (transpose) for RVC in semitones. Negative values compatible. Default: `0`.

*   `tts_rate` (int, optional): Extra rate of speech for Edge TTS in percentage (+/-). Default: `0`.

*   `tts_volume` (int, optional): Extra volume of speech for Edge TTS in percentage (+/-). Default: `0`.

*   `tts_pitch` (int, optional): Extra pitch of TTS-generated audio in Hz (+/-). **Not recommended**. Default: `0`.

*   `output_filename` (str, optional): Name for the output file. If `None`, a unique name is generated. Default: `None`.

*   `index_rate` (float, optional): Blending rate between original and indexed voice conversion (0 to 1). Default: `0.75`.

*   `f0method` (str, optional): F0 extraction method for this specific call, overrides the instance default: `'rmvpe'`, `'pm'`, `'harvest'`, `'dio'`. Default uses instance setting.

*   `file_index2` (str, optional): Path to secondary index file for RVC. Default: `""`.

*   `filter_radius` (int, optional): Median filter radius for pitch results. Values `>=3` reduce breathiness. Default: `3`.

*   `resample_sr` (int, optional): Sample rate to resample final audio to. `0` means use model's sample rate. Default: `0`.

*   `rms_mix_rate` (float, optional): Volume envelope scaling (0-1). Lower values mimic original volume more closely. Default: `0.25`.

*   `protect` (float, optional): Protection for voiceless consonants and breaths (0-0.5). Lower values increase protection. `0.5` disables. Default: `0.33`.

*   `verbose` (bool, optional): Enable verbose logging for RVC conversion. Default: `False`.

*(Note: `is_half` parameter is removed as precision is handled by ONNX Runtime.)*

## Example of usage

A simple example for voicing text:

```python
import os
from tts_with_rvc_onnx import TTS_RVC
# from playsound import playsound # Optional

# --- Configuration ---
model_file = "models/DenVot.onnx"
index_file = "logs/added_IVF1749_Flat_nprobe_1.index" # Optional
temp_dir = "audio_temp"
output_dir = "audio_output"

os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# --- Initialize ---
try:
    tts = TTS_RVC(
        model_path=model_file,
        index_path=index_file,
        tmp_directory=temp_dir,
        output_directory=output_dir,
        device="dml", # Or 'cuda:0', 'cpu'
        f0_method="rmvpe"
    )

    tts.set_voice("ru-RU-DmitryNeural")

    # --- Generate ---
    path = tts(text="Привет, мир!", pitch=6, index_rate=0.9)
    print(f"Audio saved to: {path}")

    # --- Play (Optional) ---
    # playsound(path)

except Exception as e:
    print(f"An error occurred: {e}")

```

## Text parameters

There are some text parameters processor for integration issues such as adding GPT module.

You can process them using `process_args` in `TTS_RVC` class:

*   `--tts-rate (value)`: TTS parameter to edit the speech rate.

*   `--tts-volume (value)`: TTS parameter to edit the speech volume. **May have limited effect due to RVC volume normalization.**

*   `--tts-pitch (value)`: TTS parameter to edit the pitch of TTS generated audio. **Not recommended.**

*   `--rvc-pitch (value)`: RVC parameter to edit the pitch of the output audio (semitones).

Now the principle of work:

```python
from tts_with_rvc_onnx import TTS_RVC

tts = TTS_RVC(model_path="models/YourModel.onnx", device="dml", tmp_directory="temp/")

message_with_args = "This is a test --rvc-pitch -2 and slower --tts-rate -10"

# This method returns arguments and original text without these text parameters
args, clean_message = tts.process_args(message_with_args)
# args = [-10, 0, 0, -2] # [tts_rate, tts_volume, tts_pitch, rvc_pitch]
# clean_message = "This is a test and slower"

# Use extracted arguments for generation:
path = tts(clean_message, tts_rate=args[0],
                        tts_volume=args[1],
                        tts_pitch=args[2],
                        pitch=args[3])
```

The `args` variable contains a list with the following structure:

`args[0]` - TTS Rate

`args[1]` - TTS Volume

`args[2]` - TTS Pitch

`args[3]` - RVC pitch

## Methods

*   `set_voice(voice)`: Changes the Edge TTS voice.

*   `set_index_path(index_path)`: Updates the path to the Faiss `.index` file.

*   `set_device(device)`: Changes the ONNX Runtime provider (e.g., 'dml', 'cuda:0', 'cpu') and reinitializes the backend.

*   `set_output_directory(directory_path)`: Sets the default directory for saving output files.

*   `process_args(text)`: Extracts text parameters (see above).

*   `voiceover_file(input_path, ...)`: Applies RVC voice conversion directly to an existing audio file (accepts same RVC parameters as `__call__`).

## Exceptions

*   **`RuntimeError: Failed to load ONNX model...`**: Check `.onnx` model path and integrity. Ensure correct `onnxruntime-*` package is installed.
*   **`RuntimeError: Failed to initialize ONNX backend...`**: Check ONNX Runtime installation, drivers (CUDA/DirectML), or model compatibility.
*   **`FileNotFoundError`**: Input audio, `.onnx` model, `.index` file, or required predictor models (`rmvpe.onnx`) not found.
*   **`ValueError: Dimension mismatch...`**: Faiss `.index` file dimension doesn't match `ContentVec` output dimension (e.g., 256 vs 768). Use a compatible index.
*   **`RuntimeError: Failed to load audio...`**: Ensure FFmpeg is installed and accessible in PATH.
*   **Errors during F0 computation**: Check if required libraries (`parselmouth`, `pyworld`, `torch` for rmvpe) are installed correctly.

## Acknowledgements

*   [RVC Project](https://github.com/RVC-Project/) - For the original RVC model and concepts.

## License

MIT License

## Authors

*   [Atm4x](https://github.com/Atm4x) (Artem Dikarev)
