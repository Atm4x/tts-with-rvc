import os
import edge_tts as tts
from edge_tts import VoicesManager
import asyncio, concurrent.futures
from tts_with_rvc.vc_infer import rvc_convert
import hashlib
from datetime import datetime
import nest_asyncio
import tempfile
import logging
import warnings

logger = logging.getLogger(__name__)

logging.getLogger('torch').setLevel(logging.ERROR)

nest_asyncio.apply()

class TTS_RVC:
    """
    Combines Edge TTS for text-to-speech with RVC for voice conversion.

    Args:
        model_path (str): Path to the RVC .pth model file.
        voice (str): Edge TTS voice identifier (e.g., "ru-RU-DmitryNeural").
        index_path (str, optional): Path to the RVC .index file. Defaults to "".
        output_directory (str, optional): Directory to save voiceovered audios. Defaults to 'temp'.
        f0_method (str, optional): F0 extraction method for RVC ('rmvpe', 'pm', 'harvest', 'dio', 'crepe'). Defaults to "rmvpe".
        tmp_directory (str, optional): Directory for temporary TTS files. Default is Temp
    """
    def __init__(self, model_path, tmp_directory=None, voice="ru-RU-DmitryNeural", index_path="", f0_method="rmvpe", output_directory=None, input_directory=None):
        if input_directory is not None:
            warnings.warn("Parameter 'input_directory' is deprecated and will be deleted in the future"
                        "Use tmp_directory instead of it", DeprecationWarning, stacklevel=2)
            if tmp_directory is None: 
                self.tmp_directory = input_directory
        else:
            self.tmp_directory = tmp_directory
        self.pool = concurrent.futures.ThreadPoolExecutor()
        self.current_voice = voice
        self.can_speak = True
        self.current_model = model_path
        self.output_directory = output_directory
        self.f0_method = f0_method
        if(index_path != ""):
            if not os.path.exists(index_path):
                logger.info("Index path not found, skipping...")
            else:
                logger.info("Index path: " + index_path)
        self.index_path = index_path

        
        # os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

    def set_voice(self, voice):
        self.current_voice = voice
    
    def set_index_path(self, index_path):
        if not os.path.exists(index_path) and index_path != "":
            print("Index path not found, skipping...")
        else:
            print("Index path:", index_path)
        self.index_path = index_path

    #def get_voices(self):
    #    loop = asyncio.new_event_loop()
    #    voices = loop.run_until_complete(get_voices())
    #    loop.close()
    #    return voices

    def set_output_directory(self, directory_path):
        self.output_directory = directory_path
    
    def __call__(self,
                 text,
                 pitch=0,
                 tts_rate=0,
                 tts_volume=0,
                 tts_pitch=0,
                 output_filename=None,
                 index_rate=0.75) -> str:
        
        """
        Generates speech from text using Edge TTS and converts it using RVC.

        Args:
            text (str): The text to synthesize.
            pitch (int, optional): Pitch change (transpose) for RVC in semitones. Defaults to 0.
            tts_rate (int, optional): Speed adjustment for Edge TTS in percentage (+-). Defaults to 0.
            tts_volume (int, optional): Volume adjustment for Edge TTS in percentage (+-). Defaults to 0.
            tts_pitch (int, optional): Pitch adjustment for Edge TTS in Hz (+-). Defaults to 0.
            index_rate (float, optional): Contribution of the RVC index file (0 to 1). Defaults to 0.75.
            output_filename (str, optional): Name for the output file. If None, a unique name is generated. Defaults to None.

        Returns:
            str: The absolute path to the generated audio file.

        Raises:
            RuntimeError: If TTS or RVC process fails.
            ValueError: If parameters are invalid.
        """

        path = (self.pool.submit
                (asyncio.run, speech(model_path=self.current_model,
                                     tmp_directory=self.tmp_directory,
                                     text=text,
                                     pitch=pitch,
                                     voice=self.current_voice,
                                     tts_add_rate=tts_rate,
                                     tts_add_volume=tts_volume,
                                     tts_add_pitch=tts_pitch,
                                     output_directory=self.output_directory,
                                     filename=output_filename,
                                     index_path=self.index_path,
                                     index_rate=index_rate)).result())
        return path

    def voiceover_file(self, input_path, pitch=0, output_directory=None, filename=None, index_rate=0.75):
        """
        Applies RVC voice conversion directly to an existing audio file.

        Args:
            input_path (str): Path to the input audio file (WAV recommended).
            pitch (int, optional): Pitch change (transpose) for RVC in semitones. Defaults to 0.
            index_rate (float, optional): Contribution of the RVC index file (0 to 1). Defaults to 0.75.
            output_directory (str, optional): Directory to save voiceovered audios. Defaults to TTS_RVC's output directory.
            filename (str, optional): Name for the output file. If None, derived from input name + hash. Defaults to None.

        Returns:
            str: The absolute path to the converted audio file.

        Raises:
            FileNotFoundError: If the input file doesn't exist.
            RuntimeError: If RVC process fails.
            ValueError: If parameters are invalid.
        """
        global can_speak
        if not can_speak:
            print("Can't speak now")
            return
        
        if output_directory is None:
            output_directory = self.output_directory

        name = (date_to_short_hash() + ".wav") if filename is None else filename
        output_path = rvc_convert(model_path=self.current_model,
                                  input_path=input_path,
                                  f0_up_key=pitch,
                                  f0method=self.f0_method,
                                  output_filename=name,
                                  output_dir_path=output_directory,
                                  file_index=self.index_path,
                                  index_rate=index_rate)
        
        

        return os.path.abspath(output_path)

    def process_args(self, text):
        rate_param, text = process_text(text, param="--tts-rate")
        volume_param, text = process_text(text, param="--tts-volume")
        tts_pitch_param, text = process_text(text, param="--tts-pitch")
        rvc_pitch_param, text = process_text(text, param="--rvc-pitch")
        return [rate_param, volume_param, tts_pitch_param, rvc_pitch_param], text


def date_to_short_hash():
    current_date = datetime.now()
    date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
    sha256_hash = hashlib.sha256(date_str.encode()).hexdigest()
    short_hash = sha256_hash[:10]
    return short_hash


async def get_voices():
    voicesobj = await VoicesManager.create()
    return [data["ShortName"] for data in voicesobj.voices]

can_speak = True

async def tts_communicate(text,
                          tmp_directory=None,
                          filename=None,
                          voice="ru-RU-DmitryNeural",
                          tts_add_rate=0,
                          tts_add_volume=0,
                          tts_add_pitch=0):
    if not tmp_directory:
        temp_dir = os.path.join(tempfile.gettempdir(), "tts_with_rvc")
        os.makedirs(temp_dir, exist_ok=True)
        tmp_directory = temp_dir

    communicate = tts.Communicate(
        text=text,
        voice=voice,
        rate=f'{"+" if tts_add_rate >= 0 else ""}{tts_add_rate}%',
        volume=f'{"+" if tts_add_volume >= 0 else ""}{tts_add_volume}%',
        pitch=f'{"+" if tts_add_pitch >= 0 else ""}{tts_add_pitch}Hz'
    )

    file_name = date_to_short_hash()
    input_path = os.path.join(tmp_directory, file_name)
    await communicate.save(input_path)
    return input_path, file_name

async def speech(model_path,
                 text,
                 pitch=0,
                 tmp_directory = None,
                 voice="ru-RU-DmitryNeural",
                 tts_add_rate=0,
                 tts_add_volume=0,
                 tts_add_pitch=0,
                 filename=None,
                 output_directory=None,
                 index_path="",
                 index_rate=0.75,
                 f0_method="rmvpe"):
    global can_speak
    
    if tmp_directory and not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    input_path, file_name = await tts_communicate(tmp_directory=tmp_directory,
              text=text,
              voice=voice,
              tts_add_rate=tts_add_rate,
              tts_add_volume=tts_add_volume,
              tts_add_pitch=tts_add_pitch)

    while not can_speak:
        await asyncio.sleep(1)
    can_speak = False

    
    name = (file_name + ".wav") if not filename else filename

    output_path = rvc_convert(model_path=model_path,
                              input_path=input_path,
                              f0_up_key=pitch,
                              f0method=f0_method,
                              output_filename=name,
                              output_dir_path=output_directory,
                              file_index=index_path,
                              index_rate=index_rate)

    os.remove(input_path)
    can_speak = True
    return os.path.abspath(output_path)


def process_text(input_text, param, default_value=0):
    try:
        words = input_text.split()

        value = default_value

        i = 0
        while i < len(words):
            if words[i] == param:
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word.isdigit() or (next_word[0] == '-' and next_word[1:].isdigit()):
                        value = int(next_word)
                        words.pop(i)
                        words.pop(i)
                    else:
                        raise ValueError(f"Invalid type of argument in \"{param}\"")
                else:
                    raise ValueError(f"There is no value for parameter \"{param}\"")
            i += 1

        final_string = ' '.join(words)

        return value, final_string

    except Exception as e:
        print(f"Ошибка: {e}")
        return 0, input_text
