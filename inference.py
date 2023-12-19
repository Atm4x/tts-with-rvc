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
    def __init__(self, rvc_path, model_path, voice="ru-RU-DmitryNeural"):
        if not os.path.exists('input'):
            os.mkdir('input')
        if not os.path.exists('output'):
            os.mkdir('output')

        self.pool = concurrent.futures.ThreadPoolExecutor()
        self.current_voice = voice
        self.can_speak = True
        self.current_model = model_path
        self.rvc_path = rvc_path

    def set_voice(self, voice):
        self.current_voice = voice

    def get_voices(self):
        return get_voices()

    def __call__(self, text, input_path, pitch=0):
        if not self.can_speak:
            print("TTS is busy.")
            return None
        self.can_speak = False
        path = (self.pool.submit
                (asyncio.run, speech(self.current_model, input_path, self.rvc_path, text, pitch, self.current_voice)).result())
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

async def speech(model_path, input_path, rvc_path, message, pitch=0, voice="ru-RU-DmitryNeural"):
    communicate = tts.Communicate(message, voice)
    file_name = "test"
    await communicate.save("input\\" + file_name)
    output_path = rvc_convert(model_path=model_path,
                              input_path=input_path,
                              rvc_path=rvc_path,
                              f0_up_key=pitch)
    name = date_to_short_hash()
    os.rename("output\\out.wav", "output\\" + name + ".wav")
    os.remove("input\\" + file_name)
    return "output\\" + name + ".wav"

