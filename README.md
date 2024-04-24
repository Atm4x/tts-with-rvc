# **TTS-with-RVC** 0.1.2

***TTS-with-RVC** (Text-to-Speech with RVC)* is a package designed to enhance the capabilities of *text-to-speech (TTS)* systems by introducing a *RVC* module. The package enables users to not only convert text into speech but also personalize and customize the voice output according to their preferences with RVC support.

Pytorch with CUDA or MPS is required to get TTS-with-RVC work.

**It may contain bugs. Report an issue in case of error.**
## **Installation**
1) Install pytorch **with CUDA or MPS support** here: https://pytorch.org/get-started/locally/

2) Then, install TTS-with-RVC using pip install:
```
python -m pip install git+https://github.com/Atm4x/tts-with-rvc.git@dev#egg=tts_with_rvc
```
3) After, install rvc:
```
python -m pip install git+https://github.com/Atm4x/rvc-lib.git@dev#egg=rvc
```
4) And finally, install the [fixed](https://github.com/SunSual/rvc-tts-pipeline-fix) version of rvc-tts-pipeline:
```
python -m pip install git+https://github.com/Atm4x/rvc-tts-pipeline-fix.git@dev#egg=rvc_tts_pipe
```

## How it Works
1. **Text-to-Speech (TTS):** Users enter text into the TTS module, which then processes it and generates the corresponding speech as a file saved in the entered input directory
2. **RVC:** With .pth file provided, RVC module reads the generated audio file, processes it and generates an new audio saved in *output* directory with voice replaced.

## Usage

TTS-with-RVC has a class called `TTS_RVC`. There are a few parameters that are required:

`rvc_path` - path to your installed *rvc* directory (Usually in the venv/src folder) 

`input_directory` - path to your input directory (Temp directory for saving TTS output)

`model_path` - path to your .pth model

And optional parameters:

`voice` - voice from edge-tts list *(default is "ru-RU-DmitryNeural")*

To get voice list, firstly, make instance of TTS_RVC:

```python
from tts_with_rvc.inference import TTS_RVC

tts = TTS_RVC(rvc_path="src\\rvc", model_path="models\\YourModel.pth", input_directory="input\\")
```

Next step is calling `tts.get_voices()` and setting voice:

```python
voices = tts.get_voices()
print(voices)
tts.set_voice(voices[0])
```

This code will print all voices available and then you choose with any method you want (For example, `voices[0]`).

This is necessary if you are using other languages for voiceovers!

And final step is calling `tts` to replace voice:

```python 
path = tts(text="Привет, мир!", pitch=6)
```

Parameters:

`text` - text for TTS (required)

`pitch` - pitch for RVC (optional, neg. values are compatible, default is 0)

`tts_rate` - extra rate of speech (optional, neg. values are compatible, default is 0)

`tts_volume` - extra volume of speech (optional, neg. values are compatible, default is 0)

`tts_pitch` - extra pitch of TTS-generated audio (optional, neg. values are compatible, <b>not recommended</b>, default is 0)

## Example of usage
A simple example for voicing generated text:

```python
from tts_with_rvc.inference import TTS_RVC
from playsound import playsound

tts = TTS_RVC(rvc_path="src\\rvc", model_path="models\\DenVot.pth", input_directory="input\\")
tts.set_voice("ru-RU-DmitryNeural")
path = tts(text="Привет, мир!", pitch=6)

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
from inference import TTS_RVC

tts = TTS_RVC(rvc_path="src\\rvc", model_path="models\\YourModel.pth", input_directory="input\\")

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
path = tts(message, pitch=args[3],
               tts_rate=args[0],
               tts_volume=args[1],
               tts_pitch=args[2])
```

## Exceptions
1) Exception:
```NameError: name 'device' is not defined```

Be sure your device supports CUDA and you installed right version of Torch.

## License
No license

## Authors
[Atm4x](https://github.com/Atm4x) (Artem Dikarev)


