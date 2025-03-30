# **TTS-with-RVC** 0.1.8

***TTS-with-RVC** (Text-to-Speech with RVC)* is a package designed to enhance the capabilities of *text-to-speech (TTS)* systems by introducing a *RVC* module. The package enables users to not only convert text into speech but also personalize and customize the voice output according to their preferences with RVC support.

Pytorch with CUDA or MPS is required to get TTS-with-RVC work.

**It may contain bugs. Report an issue in case of error.**

## Release notes

**0.1.8** - March 30, 2025: Added all RVC parameters, implemented `FCPE` support, added PyPL installation support, fixed bug with rmvpe-only f0 method.

**0.1.6** - March 28, 2025: Updated all files with the latest RVC commit - 1.5-2x times faster inference. Reduced required packages. ONNX support [here](https://github.com/Atm4x/tts-with-rvc/tree/0.1.6-onnx).

**0.1.5** - February 21, 2025: Removed all unnecessary packages, **Removed** `rvc_path`, Added `f0_method` for more control.

**0.1.4** - November 22, 2024: Added `index_path` and `index_rate` parameters for more control over index-based voice conversion.

**0.1.3** - fixed a lot problems, some optimization. 

## Prerequisites

You must have **Python<=3.12** installed (3.12 is recommended, mostly tested on 3.10).

You must have **CUDA or MPS** support for your GPU (mps is not tested yet). Otherwise it will use CPU, which is very slow.

## **Installation**
1) Install pytorch **with CUDA or MPS support** here: https://pytorch.org/get-started/locally/

2) Then, install TTS-with-RVC using pip install:
```
pip install tts-with-rvc
```
3) And finally, install [ffmpeg](https://ffmpeg.org/download.html) if you don't already have one, and add it to the folder with your script **or better yet** add ffmpeg to the `Environment variables` in `Path`. 

## How it Works
1. **Text-to-Speech (TTS):** Users enter text into the TTS module, which then processes it and generates the corresponding speech as a file saved in the temp directory
2. **RVC:** With .pth file provided, RVC module reads the generated audio file, processes it and generates an new audio saved in *output_directory* with voice replaced.

## Usage

TTS-with-RVC has a class called `TTS_RVC`. There are a few parameters that are required:

`model_path` - path to your .pth model

And optional parameters:

`voice` - voice from edge-tts list *(default is "ru-RU-DmitryNeural")*

`device` - set device ("cpu", "cuda:0", "mps:0", *default is "cuda:0"*)

`tmp_directory` - path to TTS input directory (Temp directory for saving TTS output, default is Temp folder)

`output_directory` - directory for saving voiced audio (`temp/` is default).

`index_path` - path to the file index for voice model adjustments (default is empty string `""`).

`index_rate` - blending rate between original and indexed voice conversion (default is `0.75`).

`f0_method` - method for calculating the pitch of the audio (default is `rmvpe`). Available: 'rmvpe', 'fcpe' (fp32 only), 'pm', 'harvest', 'dio', 'crepe'.

Deprecated:

`input_directory` - path to TTS input directory (Temp directory for saving TTS output, default is None)



To set the voice, firstly, make instance of TTS_RVC:

```python
from tts_with_rvc import TTS_RVC

tts = TTS_RVC(model_path="models\\YourModel.pth",
                index_path="logs\\YourIndex.index",
                f0_method="rmvpe")
```


All voices available placed in `voices.txt` file:

`tts.get_voices()` **is disabled indefinitely due to the problems**

Next, set the voice for TTS with `tts.set_voice()` function:

```python
tts.set_voice("un-Un-SelectedNeural")
```

Setting the appropriate language is necessary if you are using other languages for voiceovers!

And final step is calling `tts` to replace voice:

```python 
path = tts(text="Привет, мир!", pitch=6, index_rate=0.50)
```

Parameters:

`text` - text for TTS (required)

`pitch` - pitch change (transpose) for RVC in semitones (optional, neg. values are compatible, default is 0)

`tts_rate` - extra rate of speech for Edge TTS in percentage (+-) (optional, neg. values are compatible, default is 0)

`tts_volume` - extra volume of speech for Edge TTS in percentage (+-) (optional, neg. values are compatible, default is 0)

`tts_pitch` - extra pitch of TTS-generated audio in Hz (+-) (optional, neg. values are compatible, <b>not recommended</b>, default is 0)

`output_filename` - name for the output file (optional, default is `None`, a unique name is generated)

`index_rate` - blending rate between original and indexed voice conversion (0 to 1) (optional, default is `0.75`).

`is_half` - Determines half-precision for RVC inference. True or False. (optional, default is `True`).

`f0method` - F0 extraction method for this specific call, overrides the instance default: 'rmvpe', 'fcpe' (fp32 only), 'pm', 'harvest', 'dio', 'crepe'. (optional, default uses instance setting).

`file_index2` - Path to secondary index file for RVC. (optional, default is empty string `""`).

`filter_radius` - Median filter radius for pitch results. Values >=3 reduce breathiness. (optional, default is `3`).

`resample_sr` - Sample rate to resample audio to before RVC. 0 means no resampling. (optional, default is `0`).

`rms_mix_rate` - Volume envelope scaling (0-1). Lower values mimic original volume more closely. (optional, default is `0.5`).

`protect` - Protection for voiceless consonants and breaths (0-1). Lower values increase protection. 0.5 disables. (optional, default is `0.33`).

`verbose` - Enable verbose logging for RVC conversion. (optional, default is `False`).



## Example of usage
A simple example for voicing text:

```python
from tts_with_rvc import TTS_RVC
from playsound import playsound

tts = TTS_RVC(
    model_path="models\\DenVot13800.pth",
    index_path="logs\\added_IVF1749_Flat_nprobe_1.index"
)

tts.set_voice("ru-RU-DmitryNeural")
path = tts(text="Привет, мир!", pitch=6, index_rate=0.9)

# Normalize path for playsound if needed (example)
# path = path.replace("\\\\", "/").replace("\\","/")

playsound(path)
```
## Text parameters

There are some text parameters processor for integration issues such as adding GPT module.

You can process them using `process_args` in `TTS_RVC` class:

`--tts-rate (value)` - TTS parameter to edit the speech rate (negative value for decreasing rate and positive value for increasing rate)

`--tts-volume (value)` - TTS parameter to edit the speech volume (negative value for decreasing volume and positive value for increasing volume) <b>Seems to not work because of the RVC module conversion.</b>

`--tts-pitch (value)` - TTS parameter to edit the pitch of TTS generated audio (negative value for decreasing pitch and positive value for increasing pitch) <b>I do not recommend using this because the RVC module has its own `pitch` for output.</b>

`--rvc-pitch (value)` - RVC parameter to edit the pitch of the output audio (negative value for decreasing pitch and positive value for increasing pitch)

Now the principle of work:

```python
from tts_with_rvc import TTS_RVC

tts = TTS_RVC(model_path="models\\YourModel.pth")

# This method returns arguments and original text without these text parameters
args, message = tts.process_args(message)
```

The `args` variable contains an array with the following structure:

`args[0]` - TTS Rate

`args[1]` - TTS Volume

`args[2]` - TTS Pitch

`args[3]` - RVC pitch

And now we are ready to use it for generation:
```python
path = tts(message, tts_rate=args[0], 
                    tts_volume=args[1], 
                    tts_pitch=args[2],
                    pitch=args[3])
```

### Methods

`set_index_path(index_path)` - updates the path to the index file for voice model adjustments. 

`voiceover_file(path)` - voiceovers the file at the specified path without TTS.


## Exceptions
1) NameError:
```NameError: name 'device' is not defined```

Be sure your device supports CUDA and you installed right version of Torch.

2) RuntimeError:
```RuntimeError: Failed to load audio: {e}```

Be sure you installed `ffmpeg`.

## Acknowledgements
[RVC Project](https://github.com/RVC-Project/) - for RVC


## License
MIT License

## Authors
[Atm4x](https://github.com/Atm4x) (Artem Dikarev)



