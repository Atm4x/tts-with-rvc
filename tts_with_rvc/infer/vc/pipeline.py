from __future__ import annotations

import logging
import os
from time import time as ttime

import faiss
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from tts_with_rvc.assets import ModelStore
from tts_with_rvc.infer.vc.f0 import F0Extractor
from tts_with_rvc.runtime import RuntimeConfig

logger = logging.getLogger(__name__)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )
    rms2 = librosa.feature.rms(
        y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2
    )
    rms1 = F.interpolate(
        torch.from_numpy(rms1).unsqueeze(0),
        size=data2.shape[0],
        mode="linear",
    ).squeeze()
    rms2 = F.interpolate(
        torch.from_numpy(rms2).unsqueeze(0),
        size=data2.shape[0],
        mode="linear",
    ).squeeze()
    rms2 = torch.maximum(rms2, torch.full_like(rms2, 1e-6))
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline:
    def __init__(
        self,
        tgt_sr: int,
        config: RuntimeConfig,
        *,
        model_store: ModelStore | None = None,
    ) -> None:
        self.device = config.device
        self.dtype = config.dtype
        self.is_half = config.is_half
        self.model_store = model_store or ModelStore()

        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max

        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max

        self._f0 = F0Extractor(
            device=self.device,
            rvc_dtype=self.dtype,
            model_store=self.model_store,
            sample_rate=self.sr,
            hop_length=self.window,
        )

    def close(self) -> None:
        self._f0.close()

    def get_f0(
        self,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        inp_f0=None,
        crepe_hop_length=160,
        fcpe_threshold=0.05,
    ):
        f0_min = 50
        f0_max = 1100
        f0 = self._f0.extract(
            x,
            method=f0_method,
            frame_count=p_len,
            filter_radius=filter_radius,
            crepe_hop_length=crepe_hop_length,
            fcpe_threshold=fcpe_threshold,
            f0_min=f0_min,
            f0_max=f0_max,
        )
        f0 *= pow(2, f0_up_key / 12)

        if inp_f0 is not None:
            time_origin = inp_f0[:, 0]
            f0_origin = inp_f0[:, 1]
            time_target = (
                np.arange(p_len) * (self.window / self.sr)
                + (self.t_pad / self.sr)
            )
            replace_f0 = np.interp(time_target, time_origin, f0_origin)
            replace_f0 *= pow(2, f0_up_key / 12)
            f0[:p_len] = replace_f0

        f0bak = f0.copy()
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel <= 0] = 0
        valid_mask = f0_mel > 0
        if np.any(valid_mask):
            f0_mel[valid_mask] = (
                (f0_mel[valid_mask] - f0_mel_min)
                * 254
                / (f0_mel_max - f0_mel_min)
                + 1
            )
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return np.rint(f0_mel).astype(np.int32), f0bak

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
        feats = torch.from_numpy(audio0).to(
            device=self.device,
            dtype=self.dtype,
        )
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.zeros(feats.shape, dtype=torch.bool, device=self.device)

        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()

        if index is not None and big_npy is not None and index_rate != 0:
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
                torch.from_numpy(npy).unsqueeze(0).to(device=self.device, dtype=self.dtype) * index_rate
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
        if file_index and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception:
                logger.warning("Failed to load FAISS index %s", file_index, exc_info=True)
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
        if f0_file is not None and hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r", encoding="utf-8") as handle:
                    lines = handle.read().strip("\n").split("\n")
                inp_f0 = np.array(
                    [[float(value) for value in line.split(",")] for line in lines],
                    dtype="float32",
                )
            except Exception:
                logger.warning("Failed to parse external F0 file", exc_info=True)

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
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

        audio_max = (float(np.max(np.abs(audio_opt))) if audio_opt.size else 0.0) / 0.99
        max_int16 = 32767
        if audio_max > 1:
            audio_opt /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)

        del pitch, pitchf, sid, index, big_npy

        return audio_opt