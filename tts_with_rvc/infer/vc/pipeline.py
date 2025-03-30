import os
import sys
import traceback
import logging

logger = logging.getLogger(__name__)

from functools import lru_cache
from time import time as ttime

import faiss
import librosa
import numpy as np
import parselmouth
import pyworld # Оставляем импорт для harvest и dio
import torch
import torch.nn.functional as F
import torchcrepe
from scipy import signal
from huggingface_hub import hf_hub_download

from typing import Optional, Union, Dict, Any
import shutil

now_dir = os.getcwd()
sys.path.append(now_dir)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {} 

@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    if input_audio_path not in input_audio_path2wav:
         try:
             audio, _ = librosa.load(input_audio_path, sr=fs)
             input_audio_path2wav[input_audio_path] = audio.astype(np.double)
         except Exception as e:
             logger.error(f"Failed to load audio for harvest cache: {e}")
             return np.zeros(1) 

    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0

def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


import os
import shutil
import logging
from typing import Optional, Union
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

def download_hf_file(
    repo_id: str,
    repo_filename: str,
    destination_dir: str,
    hf_token: Optional[Union[str, bool]] = False,
    force_download: bool = False,
    verbose: bool = False
) -> Optional[str]:
    local_filename = os.path.basename(repo_filename)
    target_path = os.path.join(destination_dir, local_filename)

    try:
        os.makedirs(destination_dir, exist_ok=True)
    except OSError as e:
        if verbose:
            logger.info(f"Error: Failed to create destination directory '{destination_dir}': {e}")
        return None

    if not force_download and os.path.exists(target_path):
        if verbose:
            logger.info(f"File '{target_path}' already exists. Skipping download.")
        return target_path

    if verbose:
        logger.info(f"Downloading file '{repo_filename}' from repo '{repo_id}'...")

    try:
        cached_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=repo_filename,
            token=hf_token,
            cache_dir=None,
            force_download=force_download
        )
        if verbose:
            logger.info(f"File successfully downloaded to cache: {cached_file_path}")
            logger.info(f"Copying file to: {target_path}")

        shutil.copy2(cached_file_path, target_path)

        if verbose:
            logger.info(f"File successfully copied to '{target_path}'")
        return target_path

    except Exception as e:
        if verbose:
            logger.info(f"An error occurred while downloading or copying the file: {e}")
        if os.path.exists(target_path) and not force_download:
            try:
                os.remove(target_path)
                if verbose:
                    logger.info(f"Removed partially created file: {target_path}")
            except OSError:
                pass
        return None
    
class Pipeline(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half, 
        )
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.device = config.device
        self.model_rmvpe = None
        self.model_fcpe = None

    def get_f0(
        self,
        input_audio_path, 
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        inp_f0=None,
        crepe_hop_length=160,
        fcpe_threshold=0.05,
    ):
        global input_audio_path2wav 

        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0 = None

        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            logger.info("Loading Harvest model")
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, time_step)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
            # del input_audio_path2wav[input_audio_path]

        elif f0_method == "crepe":
            logger.info("Loading Crepe model")
            model = "full"
            batch_size = 512
            audio_torch = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(
                audio_torch,
                self.sr,
                crepe_hop_length,
                f0_min,
                f0_max,
                model,
                batch_size=batch_size,
                device=self.device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
        elif f0_method == "rmvpe":
            from tts_with_rvc.infer.lib.rmvpe import RMVPE

            if self.model_rmvpe is None:
                logger.info("Loading rmvpe model")
                if not os.path.exists(os.path.join(os.getcwd(), "rmvpe.pt")):
                    hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", filename="rmvpe.pt", local_dir=os.getcwd(), token=False)
                self.model_rmvpe = RMVPE(
                    "rmvpe.pt",
                    device=self.device,
                    is_half=self.is_half
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

            if hasattr(self.model_rmvpe, 'model') and "privateuseone" in str(self.device):
                try:
                   del self.model_rmvpe.model
                   del self.model_rmvpe
                   self.model_rmvpe = None
                except Exception as e:
                   logger.warning(f"Could not clean RMVPE ortruntime memory: {e}")

        elif f0_method == "fcpe":
            from tts_with_rvc.infer.lib.infer_pack.f0_modules.F0Predictor.FCPE import FCPEF0Predictor

            if self.model_fcpe is None:
                logger.info("Loading FCPE model")
                if not os.path.exists(os.path.join(os.getcwd(), "fcpe.pt")):

                    destination = os.getcwd()
                    downloaded_path = download_hf_file(
                        repo_id="IAHispano/Applio",
                        repo_filename="Resources/predictors/fcpe.pt",
                        destination_dir=destination,
                        # force_download=True 
                    )
                    if not os.path.exists(downloaded_path):
                        raise Exception("fcpe.pt wasn't downloaded successfuly.")
                self.model_fcpe = FCPEF0Predictor(
                    "fcpe.pt",
                    f0_min=f0_min,
                    f0_max=f0_max,
                    hop_length=self.window, 
                    dtype=torch.float16 if self.is_half else torch.float32, 
                    device=self.device,
                    sample_rate=self.sr,
                    threshold=fcpe_threshold
                )
            f0 = self.model_fcpe.compute_f0(x, p_len=p_len)

        elif f0_method == "dio":
            logger.info("Loading Dio model")
            _f0, t = pyworld.dio(
                 x.astype(np.double),
                 fs=self.sr,
                 f0_ceil=f0_max,
                 f0_floor=f0_min,
                 frame_period=time_step,
            )
            f0 = pyworld.stonemask(x.astype(np.double), _f0, t, self.sr)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
            if len(f0) < p_len:
                 pad_size = (p_len - len(f0) + 1) // 2
                 if pad_size < 0: pad_size = 0 # Добавлено для избежания отрицательного padding
                 pad_right = p_len - len(f0) - pad_size
                 if pad_right < 0: pad_right = 0 # Добавлено для избежания отрицательного padding
                 f0 = np.pad(f0, [[pad_size, pad_right]], mode="constant")
            elif len(f0) > p_len:
                 f0 = f0[:p_len]

        if f0 is None:
            raise ValueError(f"Unknown f0_method: {f0_method}")


        f0 *= pow(2, f0_up_key / 12)

        tf0 = self.sr // self.window
        if inp_f0 is not None:
            try:
                time_origin = inp_f0[:, 0]
                f0_origin = inp_f0[:, 1]
                time_target = np.arange(p_len) * (self.window / self.sr) + (self.t_pad / self.sr) 
                replace_f0 = np.interp(time_target, time_origin, f0_origin)
                replace_f0 *= pow(2, f0_up_key / 12)

                f0[:p_len] = replace_f0 

            except Exception as e:
                logger.error(f"Error applying external F0 file: {e}")
                traceback.print_exc()


        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel <= 0] = 0
        valid_mask = f0_mel > 0
        if np.any(valid_mask):
             f0_mel[valid_mask] = (f0_mel[valid_mask] - f0_mel_min) * 254 / (
                 f0_mel_max - f0_mel_min
             ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)

        return f0_coarse, f0bak

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()

        if (
            not isinstance(index, type(None))
            and not isinstance(big_npy, type(None))
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            arg = (feats, p_len, pitch, pitchf, sid) if hasp else (feats, p_len, sid)
            audio1 = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            del hasp, arg
        del feats, padding_mask
        if protect < 0.5 and pitch is not None and pitchf is not None:
            del feats0 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        input_audio_path, 
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        f0_file=None,
        crepe_hop_length=160,
        fcpe_threshold=0.05,
    ):
        if (
            file_index != ""
            and os.path.exists(file_index) == True
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window): 
                audio_sum += audio_pad[i : i - self.window] 
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )

        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name") and f0_file is not None:
             try:
                 with open(f0_file.name, "r") as f:
                     lines = f.read().strip("\n").split("\n")
                 inp_f0 = []
                 for line in lines:
                     inp_f0.append([float(i) for i in line.split(",")])
                 inp_f0 = np.array(inp_f0, dtype="float32")
             except:
                 traceback.print_exc()

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
            input_audio_path=input_audio_path,
            x=audio_pad,
            p_len=p_len,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            filter_radius=filter_radius,
            inp_f0=inp_f0,
            crepe_hop_length=crepe_hop_length,
            fcpe_threshold=fcpe_threshold,
            )
            
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]

            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()


        t2 = ttime()
        times[1] += t2 - t1

        for t in opt_ts:
            t = t // self.window * self.window 
            start_idx = s
            end_idx = t + self.t_pad2 + self.window
            audio_segment = audio_pad[start_idx:end_idx]

            current_pitch, current_pitchf = None, None
            if if_f0 == 1:
                 pitch_start_idx = s // self.window
                 pitch_end_idx = min(pitch.shape[1], (t + self.t_pad2) // self.window)
                 current_pitch = pitch[:, pitch_start_idx:pitch_end_idx]
                 current_pitchf = pitchf[:, pitch_start_idx:pitch_end_idx]


            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_segment,
                    current_pitch,
                    current_pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
            s = t

        start_idx = t if t is not None else 0
        audio_segment = audio_pad[start_idx:]
        current_pitch, current_pitchf = None, None
        if if_f0 == 1:
            pitch_start_idx = start_idx // self.window
            if isinstance(pitch, (np.ndarray, torch.Tensor)) and len(pitch.shape) > 1 and pitch_start_idx < pitch.shape[1]:
                current_pitch = pitch[:, pitch_start_idx:]
                current_pitchf = pitchf[:, pitch_start_idx:]
            else:
                # Здесь обработка случая, когда pitch не имеет нужной формы
                current_pitch = None
                current_pitchf = None


        audio_opt.append(
            self.vc(
                model,
                net_g,
                sid,
                audio_segment,
                current_pitch,
                current_pitchf,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )[self.t_pad_tgt : -self.t_pad_tgt]
        )
        audio_opt = np.concatenate(audio_opt)

        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if tgt_sr != resample_sr and resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr, res_type="soxr_vhq"
            )

        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32767
        if audio_max > 1:
            audio_opt /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)

        del pitch, pitchf, sid, index, big_npy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_opt