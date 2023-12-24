from rvc_infer import rvc_convert

import os
import edge_tts as tts
from edge_tts import VoicesManager
import asyncio, concurrent.futures
import gradio as gr
from rvc_infer import rvc_convert
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

    def __call__(self, text, pitch=0):
        if not self.can_speak:
            print("TTS is busy.")
            return None
        self.can_speak = False
        path = (self.pool.submit
                (asyncio.run, speech(model_path=self.current_model,
                                     rvc_path=self.rvc_path,
                                     input_directory=self.input_directory,
                                     text=text,
                                     pitch=pitch,
                                     voice=self.current_voice)).result())
        self.can_speak = True
        return path


def date_to_short_hash():
    current_date = datetime.now()
    date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
    sha256_hash = hashlib.sha256(date_str.encode()).hexdigest()
    short_hash = sha256_hash[:8]
    return short_hash


async def get_voices():
    voicesobj = await VoicesManager.create()
    return [data["ShortName"] for data in voicesobj.voices]


async def speech(model_path, input_directory, rvc_path, text, pitch=0, voice="ru-RU-DmitryNeural"):
    rate_param, text = process_text(text, param="--add-rate")
    volume_param, text = process_text(text, param="--add-volume")
    pitch_param, text = process_text(text, param="--add-pitch")

    communicate = tts.Communicate(text=text,
                                  voice=voice,
                                  rate=f'+{rate_param}%',
                                  volume=f'+{volume_param}%',
                                  pitch=f'+{pitch_param}Hz')
    file_name = "test"
    input_path = os.path.join(input_directory, file_name)
    await communicate.save(input_path)
    output_path = rvc_convert(model_path=model_path,
                              input_path=input_path,
                              rvc_path=rvc_path,
                              f0_up_key=pitch)
    name = date_to_short_hash()
    os.rename("output\\out.wav", "output\\" + name + ".wav")
    os.remove("input\\" + file_name)
    return os.path.abspath("output\\" + name + ".wav")


def process_text(input_text, param, default_value=0):
    try:
        words = input_text.split()
        value = default_value
        i = 0
        while i < len(words):
            if words[i] == param:
                if i + 1 < len(words) and words[i + 1].isdigit():
                    value = int(words[i + 1])
                    words.pop(i)
                    words.pop(i)
                    i -= 2
                else:
                    raise ValueError("Неверный формат значения параметра")
            i += 1
        final_string = ' '.join(words)

        return value, final_string

    except Exception as e:
        print(f"Ошибка: {e}")
        return 0, input_text
