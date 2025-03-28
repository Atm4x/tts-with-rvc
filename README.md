# **TTS-with-RVC** 0.1.6 ONNX

<hr/>

### <span style="color:red">Currently there are differences between Torch and Onnx version, soon I'll fix it</span>

<hr/>

***TTS-with-RVC** (Text-to-Speech with RVC)* is a package designed to enhance the capabilities of *text-to-speech (TTS)* systems by introducing a *RVC* module. The package enables users to not only convert text into speech but also personalize and customize the voice output according to their preferences with RVC support.

Pytorch with CUDA or MPS is required to get TTS-with-RVC work.

**It may contain bugs. Report an issue in case of error.**

## Release notes

**0.1.6.onnx** - Added onnx support.

## Prerequisites

You must have **Python<=3.12** installed (3.12 is recommended, mostly tested on 3.10).

## **Installation**

1) Then, install TTS-with-RVC using pip install:
```
python -m pip install git+https://github.com/Atm4x/tts-with-rvc.git@0.1.6-onnx#egg=tts_with_rvc
```
2) Then install [ffmpeg](https://ffmpeg.org/download.html) if you don't already have one, and add it to the folder with your script **or better yet** add ffmpeg to the `Environment variables` in `Path`. 

## How it Works
1. **Text-to-Speech (TTS):** Users enter text into the TTS module, which then processes it and generates the corresponding speech as a file saved in the entered input directory
2. **RVC:** With .pth file provided, RVC module reads the generated audio file, processes it and generates an new audio saved in *output_directory* with voice replaced.

## Usage

TTS-with-RVC has a class called `TTS_RVC`. There are a few parameters that are required:

`input_directory` - path to your input directory (Temp directory for saving TTS output)

`model_path` - path to your .onnx model

And optional parameters:

`voice` - voice from edge-tts list *(default is "ru-RU-DmitryNeural")*

`device` - device to work with (CUDA, CPU or DML, default is `dml`)

`output_directory` - directory for saving voiced audio (`temp/` is default).

`index_path` - path to the file index for voice model adjustments (default is empty string `""`).

`index_rate` - blending rate between original and indexed voice conversion (default is `0.75`).

`f0_method` - method for calculating the pitch of the audio (default is `dio`).



To set the voice, firstly, make instance of TTS_RVC:

```python
from tts_with_rvc import TTS_RVC

tts = TTS_RVC(model_path="models\\YourModel.onnx",
                device="dml", 
                input_directory="input\\", 
                index_path="logs\\YourIndex.index",
                f0_method="dio")
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

`pitch` - pitch for RVC (optional, neg. values are compatible, default is 0)

`tts_rate` - extra rate of speech (optional, neg. values are compatible, default is 0)

`tts_volume` - extra volume of speech (optional, neg. values are compatible, default is 0)

`tts_pitch` - extra pitch of TTS-generated audio (optional, neg. values are compatible, <b>not recommended</b>, default is 0)

`output_filename` - specified path for voiced audio (optional, default is `None`)

## Example of usage
A simple example for voicing text:

```python
from tts_with_rvc import TTS_RVC
from playsound import playsound

tts = TTS_RVC(
    model_path="models\\DenVot.onnx",
    device="dml", 
    input_directory="input\\",
    index_path="logs\\added_IVF1749_Flat_nprobe_1.index"
)
tts.set_voice("ru-RU-DmitryNeural")
path = tts(text="Привет, мир!", pitch=6, index_rate=0.9)

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

tts = TTS_RVC(
    device="dml",
    model_path="models\\YourModel.pth",
    input_directory="input\\")

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


## Exceptions

Nothing found yet.

## License
No license

## Authors
[Atm4x](https://github.com/Atm4x) (Artem Dikarev)


