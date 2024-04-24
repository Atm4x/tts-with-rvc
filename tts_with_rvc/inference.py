import os
import edge_tts as tts
from edge_tts import VoicesManager
import asyncio, concurrent.futures
import gradio as gr
from rvc_tts_pipe.rvc_infer import rvc_convert
import config
import hashlib
from datetime import datetime


class TTS_RVC:
    def __init__(self, rvc_path, input_directory, model_path, voice="ru-RU-DmitryNeural"):
        if not os.path.exists('input'):
            os.mkdir('input')
        if not os.path.exists('output'):
            os.mkdir('output')

        self.pool = concurrent.futures.ThreadPoolExecutor()
        self.current_voice = voice
        self.input_directory = input_directory
        self.can_speak = True
        self.current_model = model_path
        self.rvc_path = rvc_path

    def set_voice(self, voice):
        self.current_voice = voice

    def get_voices(self):
        return get_voices()

    def __call__(self,
                 text,
                 pitch=0,
                 tts_rate=0,
                 tts_volume=0,
                 tts_pitch=0):
        path = (self.pool.submit
                (asyncio.run, speech(model_path=self.current_model,
                                     rvc_path=self.rvc_path,
                                     input_directory=self.input_directory,
                                     text=text,
                                     pitch=pitch,
                                     voice=self.current_voice,
                                     tts_add_rate=tts_rate,
                                     tts_add_volume=tts_volume,
                                     tts_add_pitch=tts_pitch)).result())
        return path

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

async def speech(model_path,
                 input_directory,
                 rvc_path,
                 text,
                 pitch=0,
                 voice="ru-RU-DmitryNeural",
                 tts_add_rate=0,
                 tts_add_volume=0,
                 tts_add_pitch=0):
    global can_speak
    communicate = tts.Communicate(text=text,
                                  voice=voice,
                                  rate=f'{"+" if tts_add_rate >= 0 else ""}{tts_add_rate}%',
                                  volume=f'{"+" if tts_add_volume >= 0 else ""}{tts_add_volume}%',
                                  pitch=f'{"+" if tts_add_pitch >= 0 else ""}{tts_add_pitch}Hz')
    file_name = date_to_short_hash()
    input_path = os.path.join(input_directory, file_name)
    while not can_speak:
        await asyncio.sleep(1)
    can_speak = False
    await communicate.save(input_path)

    output_path = rvc_convert(model_path=model_path,
                              input_path=input_path,
                              rvc_path=rvc_path,
                              f0_up_key=pitch)
    name = date_to_short_hash()
    os.rename("output\\out.wav", "output\\" + name + ".wav")
    os.remove("input\\" + file_name)
    output_path = "output\\" + name + ".wav"

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