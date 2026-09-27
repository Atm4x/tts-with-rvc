from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import faiss
import librosa
import numpy as np
from scipy import signal
import torch
import torch.nn.functional as F

from tts_with_rvc.assets import ModelStore
from tts_with_rvc.runtime_onnx import (
    OnnxExecutionDevice,
    create_session,
    resolve_execution_device,
)

logger = logging.getLogger(__name__)


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )[0]
    rms2 = librosa.feature.rms(
        y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2
    )[0]
    rms1_torch = torch.from_numpy(rms1).float()[None, None]
    rms2_torch = torch.from_numpy(rms2).float()[None, None]
    target_len = data2.shape[0]
    rms1_interp = F.interpolate(
        rms1_torch, size=target_len, mode="linear", align_corners=False
    ).squeeze()
    rms2_interp = F.interpolate(
        rms2_torch, size=target_len, mode="linear", align_corners=False
    ).squeeze()
    rms2_interp = torch.maximum(rms2_interp, torch.tensor(1e-6))
    weight = torch.pow(rms1_interp, 1 - rate) * torch.pow(rms2_interp, rate - 1)
    data2 *= weight.numpy()
    return data2


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self._hann_windows: dict[tuple[int, int], torch.Tensor] = {}
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        elif not isinstance(audio, torch.Tensor):
            raise TypeError("Input audio must be a numpy array or torch tensor")
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.cpu()

        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        window_key = (win_length_new, audio.device.index or -1)
        window = self._hann_windows.get(window_key)
        if window is None:
            window = torch.hann_window(win_length_new, device=audio.device)
            self._hann_windows[window_key] = window

        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=window,
            center=center,
            return_complex=True,
        )
        magnitude = fft.abs()
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            magnitude = F.interpolate(
                magnitude.unsqueeze(1),
                size=(size, magnitude.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        mel = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel, min=self.clamp))


class RMVPEONNXPredictor:
    def __init__(
        self,
        model_path,
        runtime: OnnxExecutionDevice,
    ):
        self.runtime = runtime
        self.model = create_session(model_path, runtime)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.mel_extractor = MelSpectrogram(
            n_mel_channels=128,
            sampling_rate=16000,
            win_length=1024,
            hop_length=160,
            n_fft=1024,
            mel_fmin=30,
            mel_fmax=8000,
        ).eval()
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4)).astype(np.float32)
        logger.info("RMVPE loaded on %s", runtime)

    def to_local_average_cents(self, salience, thred=0.05):
        salience = np.asarray(salience, dtype=np.float32)
        if salience.ndim == 3:
            salience = salience.squeeze(0)
        center_indices = np.argmax(salience, axis=1)
        max_salience = np.max(salience, axis=1)
        padded = np.pad(salience, ((0, 0), (4, 4)), mode="constant")
        windows = center_indices[:, None] + np.arange(9)
        row_indices = np.arange(salience.shape[0])[:, None]
        weights = padded[row_indices, windows]
        cents = self.cents_mapping[windows]
        denominator = np.maximum(np.sum(weights, axis=1), 1e-8)
        result = np.sum(weights * cents, axis=1) / denominator
        result[max_salience < thred] = 0.0
        return result.astype(np.float32)

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[cents_pred < 1e-5] = 0
        return f0.astype(np.float32)

    def compute_f0(self, wav, p_len=None, orig_sr=None):
        if orig_sr is None:
            raise ValueError("Original sampling rate (orig_sr) must be provided")
        wav = np.asarray(wav, dtype=np.float32)
        if orig_sr != 16000:
            wav = librosa.resample(
                wav,
                orig_sr=orig_sr,
                target_sr=16000,
                res_type="soxr_vhq",
            )

        with torch.no_grad():
            mel = self.mel_extractor(torch.from_numpy(wav).float(), center=True).squeeze(0)
        mel_np = mel.numpy().astype(np.float32)
        n_frames = mel_np.shape[1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad:
            mel_np = np.pad(mel_np, ((0, 0), (0, n_pad)), mode="constant")

        hidden = self.model.run(
            [self.output_name],
            {self.input_name: np.expand_dims(mel_np, axis=0)},
        )[0]
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 360:
            raise ValueError(
                f"Unexpected RMVPE ONNX output shape: {hidden.shape}; expected [1, T, 360]"
            )
        f0 = self.decode(hidden[0, :n_frames, :])
        if p_len is not None and len(f0) != p_len:
            old_x = np.linspace(0, 1, len(f0))
            new_x = np.linspace(0, 1, p_len)
            f0 = np.interp(new_x, old_x, f0)
        return f0.astype(np.float32)

    def close(self) -> None:
        self.model = None


class ContentVec:
    def __init__(self, vec_path, runtime: OnnxExecutionDevice):
        self.runtime = runtime
        self.model = create_session(vec_path, runtime)
        self.input_name = self.model.get_inputs()[0].name
        logger.info("ContentVec loaded on %s", runtime)

    def __call__(self, wav):
        feats = np.asarray(wav, dtype=np.float32)
        if feats.ndim == 2:
            feats = feats.mean(-1)
        if feats.ndim != 1:
            raise ValueError(f"Expected mono audio, got shape {feats.shape}")
        feats = feats[None, None, :]
        logits = self.model.run(None, {self.input_name: feats})[0]
        return logits.transpose(0, 2, 1)

    def close(self) -> None:
        self.model = None


class F0Extractor:
    def __init__(
        self,
        *,
        runtime: OnnxExecutionDevice,
        model_store: ModelStore,
        sampling_rate: int,
        hop_length: int,
    ) -> None:
        self.runtime = runtime
        self.model_store = model_store
        self.sampling_rate = int(sampling_rate)
        self.hop_length = int(hop_length)
        self._predictors: dict[str, object] = {}

    def reconfigure(self, *, sampling_rate: int, hop_length: int) -> None:
        sampling_rate = int(sampling_rate)
        hop_length = int(hop_length)
        if sampling_rate == self.sampling_rate and hop_length == self.hop_length:
            return
        self.close()
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

    def get(self, method: str, *, cr_threshold: float = 0.05):
        del cr_threshold
        key = str(method).lower()
        predictor = self._predictors.get(key)
        if predictor is not None:
            return predictor

        if key == "pm":
            from tts_with_rvc.lib.infer_pack.modules.F0Predictor.PMF0Predictor import PMF0Predictor

            predictor = PMF0Predictor(
                hop_length=self.hop_length,
                sampling_rate=self.sampling_rate,
            )
        elif key == "harvest":
            from tts_with_rvc.lib.infer_pack.modules.F0Predictor.HarvestF0Predictor import HarvestF0Predictor

            predictor = HarvestF0Predictor(
                hop_length=self.hop_length,
                sampling_rate=self.sampling_rate,
            )
        elif key == "dio":
            from tts_with_rvc.lib.infer_pack.modules.F0Predictor.DioF0Predictor import DioF0Predictor

            predictor = DioF0Predictor(
                hop_length=self.hop_length,
                sampling_rate=self.sampling_rate,
            )
        elif key == "rmvpe":
            path = self.model_store.resolve(
                "rmvpe.onnx",
                repo_id="lj1995/VoiceConversionWebUI",
            )
            predictor = RMVPEONNXPredictor(path, runtime=self.runtime)
        else:
            raise ValueError(f"Unknown F0 predictor: {method}")

        self._predictors[key] = predictor
        return predictor

    def close(self) -> None:
        for predictor in self._predictors.values():
            close = getattr(predictor, "close", None)
            if callable(close):
                close()
        self._predictors.clear()


class OnnxRVC:
    def __init__(
        self,
        model_path,
        sr=40000,
        hop_size=512,
        vec_path="vec-768-layer-12.onnx",
        device="cpu",
        x_pad=3,
        models_dir=None,
        random_seed: int | None = None,
    ):
        self.runtime = resolve_execution_device(device)
        self.device = str(self.runtime)
        self._lock = threading.RLock()
        self._model_store = ModelStore(models_dir)
        self._rng = np.random.default_rng(random_seed)
        self._closed = False
        self.current_rvc_model_path: str | None = None
        self.model = None

        vec_local_path = self._model_store.resolve(
            vec_path,
            repo_id="NaruseMioShirakana/MoeSS-SUBModel",
        )
        self.vec_model = ContentVec(vec_local_path, self.runtime)

        self.sampling_rate = int(sr)
        self.hop_size = int(hop_size)
        self.x_pad = int(x_pad)
        self.sr_hubert = 16000
        self.pad_seconds = self.x_pad
        self._refresh_sampling_state()
        self.f0_extractor = F0Extractor(
            runtime=self.runtime,
            model_store=self._model_store,
            sampling_rate=self.sampling_rate,
            hop_length=self.f0_hop_size,
        )
        self.load_new_rvc_model(model_path)
        logger.info("OnnxRVC initialized on %s", self.runtime)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OnnxRVC is closed")

    def _refresh_sampling_state(self) -> None:
        self.f0_hop_size = int(160 * (self.sampling_rate / self.sr_hubert))
        self.t_pad_main_sr = int(self.sampling_rate * self.pad_seconds)
        self.t_pad_hubert_sr = int(self.sr_hubert * self.pad_seconds)

    def _build_rvc_session(self, model_path):
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RVC model file not found: {model_path}")
        return path, create_session(path, self.runtime)

    def load_new_rvc_model(self, model_path):
        with self._lock:
            self._ensure_open()
            target = str(Path(model_path).expanduser().resolve())
            if self.current_rvc_model_path == target and self.model is not None:
                return
            path, candidate = self._build_rvc_session(model_path)
            old_model = self.model
            self.model = candidate
            self.current_rvc_model_path = str(path)
            del old_model

    def set_sr_and_hop(self, sr, hop):
        with self._lock:
            self._ensure_open()
            sr = int(sr)
            hop = int(hop)
            if sr == self.sampling_rate and hop == self.hop_size:
                return
            self.sampling_rate = sr
            self.hop_size = hop
            self._refresh_sampling_state()
            self.f0_extractor.reconfigure(
                sampling_rate=self.sampling_rate,
                hop_length=self.f0_hop_size,
            )

    def load_index(self, index_path):
        if not index_path or not os.path.exists(index_path):
            return None, None
        try:
            index = faiss.read_index(index_path)
            if index.ntotal == 0:
                return None, None
            return index, index.reconstruct_n(0, index.ntotal).astype(np.float32)
        except Exception:
            logger.exception("Failed to load Faiss index %s", index_path)
            return None, None

    @staticmethod
    def apply_index(hubert_features, index, big_npy, index_rate):
        if index is None or big_npy is None or index_rate == 0:
            return hubert_features.astype(np.float32)
        hubert_dim = hubert_features.shape[1]
        if hubert_dim != index.d:
            logger.error(
                "Dimension mismatch: Hubert(%s) != Index(%s); index skipped",
                hubert_dim,
                index.d,
            )
            return hubert_features.astype(np.float32)
        try:
            hubert_np = hubert_features.squeeze(0).T.astype(np.float32)
            distances, indices = index.search(hubert_np, k=8)
            retrieved = big_npy[indices].astype(np.float32)
            weights = 1.0 / (distances.astype(np.float32) ** 2 + 1e-8)
            weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-8)
            mixed = (1 - index_rate) * hubert_np + index_rate * np.sum(
                retrieved * weights[:, :, None], axis=1
            )
            return mixed.T[None, :, :].astype(np.float32)
        except Exception:
            logger.exception("Failed to apply Faiss index")
            return hubert_features.astype(np.float32)

    @staticmethod
    def apply_protection(hubert_features, pitchf, protect_rate):
        if protect_rate >= 0.5 or pitchf is None:
            return hubert_features.astype(np.float32)
        pitchf = np.asarray(pitchf)
        if pitchf.ndim == 1:
            pitchf = pitchf[None, :]
        hubert_features = np.asarray(hubert_features)
        protect_mask = np.where(pitchf > 0, 1.0, protect_rate).astype(np.float32)[:, None, :]
        if hubert_features.shape[2] != pitchf.shape[1]:
            hubert_tensor = torch.from_numpy(hubert_features)
            hubert_features = F.interpolate(
                hubert_tensor,
                size=pitchf.shape[1],
                mode="linear",
                align_corners=False,
            ).numpy()
        return (hubert_features * protect_mask).astype(np.float32)

    def forward(self, hubert, hubert_length, pitch, pitchf, ds, rnd):
        inputs = self.model.get_inputs()
        onnx_input = {
            inputs[0].name: hubert.transpose(0, 2, 1).astype(np.float32),
            inputs[1].name: hubert_length.astype(np.int64),
            inputs[2].name: pitch.astype(np.int64),
            inputs[3].name: pitchf.astype(np.float32),
            inputs[4].name: ds.astype(np.int64),
            inputs[5].name: rnd.astype(np.float32),
        }
        return self.model.run(None, onnx_input)[0]

    def inference(
        self,
        raw_path,
        sid,
        f0_method="dio",
        f0_up_key=0,
        index_file=None,
        index_file2=None,
        index_rate=0.75,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.25,
        protect=0.33,
        cr_threshold=0.05,
        verbose=False,
    ):
        del verbose
        with self._lock:
            self._ensure_open()
            wav_orig, sr_orig = librosa.load(raw_path, sr=None, mono=True)
            if sr_orig != self.sampling_rate:
                wav_main_sr = librosa.resample(
                    wav_orig,
                    orig_sr=sr_orig,
                    target_sr=self.sampling_rate,
                    res_type="soxr_vhq",
                )
            else:
                wav_main_sr = wav_orig

            original_length = len(wav_main_sr)
            wav_padded_main = np.pad(
                wav_main_sr,
                (self.t_pad_main_sr, self.t_pad_main_sr),
                mode="reflect",
            )
            wav16k = librosa.resample(
                wav_main_sr,
                orig_sr=self.sampling_rate,
                target_sr=self.sr_hubert,
                res_type="soxr_vhq",
            )
            wav16k_padded = np.pad(
                wav16k,
                (self.t_pad_hubert_sr, self.t_pad_hubert_sr),
                mode="reflect",
            )

            pitch_length = wav_padded_main.shape[0] // self.f0_hop_size
            predictor = self.f0_extractor.get(
                f0_method,
                cr_threshold=cr_threshold,
            )
            kwargs = {"wav": wav_padded_main, "p_len": pitch_length}
            if isinstance(predictor, RMVPEONNXPredictor):
                kwargs["orig_sr"] = self.sampling_rate
            pitchf = predictor.compute_f0(**kwargs)

            if filter_radius >= 3:
                pad = int((filter_radius - 1) / 2)
                filtered = np.pad(pitchf, pad, mode="reflect")
                pitchf = signal.medfilt(filtered, filter_radius)[pad:-pad]

            pitchf = pitchf * (2 ** (f0_up_key / 12))
            f0_min, f0_max = 50, 1100
            mel_min = 1127 * np.log(1 + f0_min / 700)
            mel_max = 1127 * np.log(1 + f0_max / 700)
            f0_mel = 1127 * np.log(1 + pitchf / 700)
            positive = f0_mel > 0
            f0_mel[positive] = (
                (f0_mel[positive] - mel_min) * 254 / (mel_max - mel_min) + 1
            )
            f0_mel[f0_mel <= 1] = 1
            f0_mel[f0_mel > 255] = 255
            pitch = np.rint(f0_mel).astype(np.int64)

            hubert = np.repeat(self.vec_model(wav16k_padded), 2, axis=2)
            hubert_length = hubert.shape[2]
            if len(pitch) != hubert_length:
                pitch = _resize_1d(pitch, hubert_length, discrete=True)
                pitchf = _resize_1d(pitchf, hubert_length, discrete=False)

            index_path = index_file or index_file2
            index, big_npy = self.load_index(index_path)
            if index is not None and index_rate > 0:
                hubert = self.apply_index(hubert, index, big_npy, index_rate)
            hubert = self.apply_protection(hubert, pitchf, protect)

            pitch = pitch.reshape(1, hubert_length)
            pitchf_final = pitchf.reshape(1, hubert_length).astype(np.float32)
            ds = np.asarray([sid], dtype=np.int64)
            rnd = self._rng.standard_normal((1, 192, hubert_length)).astype(np.float32)
            length = np.asarray([hubert_length], dtype=np.int64)
            output = self.forward(hubert, length, pitch, pitchf_final, ds, rnd).squeeze()

            start = self.t_pad_main_sr
            output = output[start : min(start + original_length, len(output))]
            if len(output) < original_length:
                output = np.pad(output, (0, original_length - len(output)))
            else:
                output = output[:original_length]

            if rms_mix_rate < 1.0:
                output = change_rms(
                    wav_main_sr,
                    self.sampling_rate,
                    output,
                    self.sampling_rate,
                    rms_mix_rate,
                )
            final_sr = self.sampling_rate
            if resample_sr > 0 and resample_sr != self.sampling_rate:
                output = librosa.resample(
                    output,
                    orig_sr=self.sampling_rate,
                    target_sr=resample_sr,
                    res_type="soxr_vhq",
                )
                final_sr = resample_sr

            audio_max = np.abs(output).max() / 0.99 if output.size else 0.0
            if audio_max > 1:
                output /= audio_max
            return (output * 32767).astype(np.int16)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.f0_extractor.close()
            self.vec_model.close()
            self.model = None
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _resize_1d(values, length: int, *, discrete: bool):
    values = np.asarray(values)
    if len(values) == length:
        return values
    if len(values) == 0:
        return np.ones(length, dtype=np.int64) if discrete else np.zeros(length, dtype=np.float32)
    if discrete:
        indices = np.linspace(0, len(values) - 1, length).round().astype(np.int64)
        return values[indices]
    return np.interp(
        np.linspace(0, 1, length),
        np.linspace(0, 1, len(values)),
        values,
    ).astype(np.float32)
