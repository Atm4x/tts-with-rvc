import os
import edge_tts as tts
from edge_tts import VoicesManager
import asyncio, concurrent.futures
import gradio as gr
from rvc_tts_pipeline.rvc_infer import rvc_convert
import hashlib
from datetime import datetime


class TTS_RVC:
    def __init__(self, rvc_path, input_directory, model_path, voice="ru-RU-DmitryNeural", index_path="", output_directory=None):
        self.pool = concurrent.futures.ThreadPoolExecutor()
        self.current_voice = voice
        self.input_directory = input_directory
        self.can_speak = True
        self.current_model = model_path
        self.rvc_path = rvc_path
        self.output_directory = output_directory
        if not os.path.exists(index_path) and index_path != "":
            print("Index path not found, skipping...")
        else:
            print("Index path:", index_path)
        self.index_path = index_path

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
                 index_rate=0.75):
        path = (self.pool.submit
                (asyncio.run, speech(model_path=self.current_model,
                                     rvc_path=self.rvc_path,
                                     input_directory=self.input_directory,
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

    def speech(self, input_path, pitch=0, output_directory=None, filename=None, index_rate=0.75):
        global can_speak
        if not can_speak:
            print("Can't speak now")
            return
        output_path = rvc_convert(model_path=self.current_model,
                                  input_path=input_path,
                                  rvc_path=self.rvc_path,
                                  f0_up_key=pitch,
                                  output_filename=filename,
                                  output_dir_path=output_directory,
                                  file_index=self.index_path,
                                  index_rate=index_rate)
        name = date_to_short_hash()
        if filename is None:
            if output_directory is None:
                output_directory = "temp"
            
            new_path = os.path.join(output_directory, name + ".wav")
            os.rename(output_path, new_path)
            output_path = new_path

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
    short_hash = sha256_hash[:8]
    return short_hash


async def get_voices():
    voicesobj = await VoicesManager.create()
    return [data["ShortName"] for data in voicesobj.voices]

can_speak = True

async def tts_comminicate(input_directory,
                 text,
                 voice="ru-RU-DmitryNeural",
                 tts_add_rate=0,
                 tts_add_volume=0,
                 tts_add_pitch=0):
    communicate = tts.Communicate(text=text,
                                  voice=voice,
                                  rate=f'{"+" if tts_add_rate >= 0 else ""}{tts_add_rate}%',
                                  volume=f'{"+" if tts_add_volume >= 0 else ""}{tts_add_volume}%',
                                  pitch=f'{"+" if tts_add_pitch >= 0 else ""}{tts_add_pitch}Hz')
    file_name = date_to_short_hash()
    input_path = os.path.join(input_directory, file_name)
    await communicate.save(input_path)
    return input_path, file_name

async def speech(model_path,
                 input_directory,
                 rvc_path,
                 text,
                 pitch=0,
                 voice="ru-RU-DmitryNeural",
                 tts_add_rate=0,
                 tts_add_volume=0,
                 tts_add_pitch=0,
                 filename=None,
                 output_directory=None,
                 index_path="",
                 index_rate=0.75):
    global can_speak

    input_path, file_name = await tts_comminicate(input_directory=input_directory,
              text=text,
              voice=voice,
              tts_add_rate=tts_add_rate,
              tts_add_volume=tts_add_volume,
              tts_add_pitch=tts_add_pitch)

    while not can_speak:
        await asyncio.sleep(1)
    can_speak = False

    output_path = rvc_convert(model_path=model_path,
                              input_path=input_path,
                              rvc_path=rvc_path,
                              f0_up_key=pitch,
                              output_filename=filename,
                              output_dir_path=output_directory,
                              file_index=index_path,
                              index_rate=index_rate)
    name = date_to_short_hash()
    if filename is None:
        if output_directory is None:
                output_directory = "temp"
        new_path = os.path.join(output_directory, name + ".wav")
        os.rename(output_path, new_path)
        output_path = new_path

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
