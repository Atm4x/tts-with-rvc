import sys
import os

now_dir = os.getcwd()
sys.path.append(now_dir)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"

from scipy.io import wavfile
from fairseq import checkpoint_utils
from lib.audio import load_audio
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from vc_infer_pipeline import VC
from multiprocessing import cpu_count
import numpy as np
import torch
import glob
import argparse
import pdb
import torch
import time

now_dir = os.getcwd()
sys.path.append(now_dir)

using_cli = False
device = "cuda:0"
is_half = False

class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available() and device != "cpu":
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                # for config_file in ["32k.json", "40k.json", "48k.json"]:
                #     with open(f"configs/{config_file}", "r") as f:
                #         strr = f.read().replace("true", "false")
                #     with open(f"configs/{config_file}", "w") as f:
                #         f.write(strr)
                # with open("trainset_preprocess_pipeline_print.py", "r") as f:
                #     strr = f.read().replace("3.7", "3.0")
                # with open("trainset_preprocess_pipeline_print.py", "w") as f:
                #     f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            # if self.gpu_mem <= 4:
            #     with open("trainset_preprocess_pipeline_print.py", "r") as f:
            #         strr = f.read().replace("3.7", "3.0")
            #     with open("trainset_preprocess_pipeline_print.py", "w") as f:
            #         f.write(strr)
        elif torch.backends.mps.is_available():
            print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
        else:
            print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


config = Config(device, is_half)
now_dir = os.getcwd()
sys.path.append(now_dir)

hubert_model = None


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def vc_single(
    sid=0,
    input_audio_path=None,
    f0_up_key=0,
    f0_file=None,
    f0_method="pm",
    file_index="",  # .index file
    file_index2="",
    # file_big_npy,
    index_rate=1.0,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=1.0,
    model_path="",
    output_path="",
    protect=0.33,
):
    global tgt_sr, net_g, vc, hubert_model, version
    get_vc(model_path)
    if input_audio_path is None:
        return "You need to upload an audio file", None

    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio_path, 16000)
    audio_max = np.abs(audio).max() / 0.95

    if audio_max > 1:
        audio /= audio_max
    times = [0, 0, 0]

    if hubert_model == None:
        load_hubert()

    if_f0 = cpt.get("f0", 1)

    file_index = (
        (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )
        if file_index != ""
        else file_index2
    )

    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        # file_big_npy,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        f0_file=f0_file,
        protect=protect,
    )
    wavfile.write(output_path, tgt_sr, audio_opt)
    return "processed"


def get_vc(model_path):
    global n_spk, tgt_sr, net_g, vc, cpt, device, is_half, version
    print("loading pth %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    # return {"visible": True,"maximum": n_spk, "__type__": "update"}

import os
import random
import string
from pathlib import Path
import traceback

def generate_random_filename(length=10):
    """Генерирует случайное имя файла"""
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def ensure_output_dir():
    """Проверяет и создает директорию output если её нет"""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def process_audio(f0_up_key=0):
    """Основная функция обработки аудио"""
    try:
        print("Начинаем обработку аудио...")
        
        # Фиксированные параметры
        input_audio = "input/check.mp3"
        model_path = "weights/DenVot13800.pth"
        
        # Генерация имени выходного файла
        output_dir = ensure_output_dir()
        output_filename = f"{generate_random_filename()}.wav"
        output_path = str(output_dir / output_filename)
        
        print(f"Входной файл: {input_audio}")
        print(f"Используемая модель: {model_path}")
        print(f"Выходной файл будет сохранен как: {output_path}")
        
        # Вызов функции обработки
        vc_single(
            sid=0,
            input_audio_path=input_audio,
            f0_up_key=f0_up_key,
            f0_file=None,
            f0_method="rmvpe",
            file_index="D:/Projects/Yep/logs/added_IVF1749_Flat_nprobe_1.index",
            file_index2="",
            index_rate=1,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=0,
            model_path=model_path,
            output_path=output_path,
        )
        
        print(f"\nУспешно! Файл сохранен в: {output_path}")
        return output_path
        
    except FileNotFoundError as e:
        print(f"Ошибка: Файл не найден - {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Запуск программы обработки аудио")
    try:
        output_file = process_audio(f0_up_key=0)  # можно изменить f0_up_key по необходимости
        if output_file:
            print("Программа успешно завершена")
        else:
            print("Программа завершена с ошибкой")
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        traceback.print_exc()