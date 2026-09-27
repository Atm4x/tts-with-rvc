from __future__ import annotations

import logging

import numpy as np
import parselmouth
import pyworld
import torch
import torchcrepe
from scipy import signal

from tts_with_rvc.assets import ModelStore
from tts_with_rvc.runtime import device_type

logger = logging.getLogger(__name__)


class F0Extractor:
    def __init__(
        self,
        *,
        device,
        rvc_dtype: torch.dtype,
        model_store: ModelStore,
        sample_rate: int = 16000,
        hop_length: int = 160,
    ) -> None:
        self.device = device
        self.rvc_dtype = rvc_dtype
        self.model_store = model_store
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self._model_name: str | None = None
        self._model = None

    def close(self) -> None:
        self._release_model()

    def _release_model(self) -> None:
        self._model = None
        self._model_name = None

    def _get_rmvpe(self):
        if self._model_name != "rmvpe":
            self._release_model()
            from tts_with_rvc.infer.lib.rmvpe import RMVPE

            model_path = self.model_store.get(
                "lj1995/VoiceConversionWebUI",
                "rmvpe.pt",
            )
            logger.info("Loading RMVPE model on %s", self.device)
            self._model = RMVPE(
                str(model_path),
                device=self.device,
                is_half=self.rvc_dtype == torch.float16,
            )
            self._model_name = "rmvpe"
        return self._model

    def _get_fcpe(self, *, f0_min: float, f0_max: float, threshold: float):
        if self._model_name != "fcpe":
            self._release_model()
            from tts_with_rvc.infer.lib.infer_pack.f0_modules.F0Predictor.FCPE import (
                FCPEF0Predictor,
            )

            model_path = self.model_store.get(
                "IAHispano/Applio",
                "fcpe.pt",
                repo_filename="Resources/predictors/fcpe.pt",
            )
            logger.info("Loading FCPE model on %s", self.device)
            self._model = FCPEF0Predictor(
                str(model_path),
                f0_min=f0_min,
                f0_max=f0_max,
                hop_length=self.hop_length,
                dtype=torch.float32,
                device=self.device,
                sample_rate=self.sample_rate,
                threshold=threshold,
            )
            self._model_name = "fcpe"
        self._model.threshold = threshold
        return self._model

    def extract(
        self,
        audio: np.ndarray,
        *,
        method: str,
        frame_count: int,
        filter_radius: int,
        crepe_hop_length: int = 160,
        fcpe_threshold: float = 0.05,
        f0_min: float = 50,
        f0_max: float = 1100,
    ) -> np.ndarray:
        method = method.lower()
        if method not in {"rmvpe", "fcpe"} and self._model is not None:
            self._release_model()

        time_step_ms = self.hop_length / self.sample_rate * 1000

        if method == "pm":
            f0 = (
                parselmouth.Sound(audio, self.sample_rate)
                .to_pitch_ac(
                    time_step=time_step_ms / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
        elif method in {"harvest", "dio"}:
            audio64 = audio.astype(np.float64, copy=False)
            estimator = pyworld.harvest if method == "harvest" else pyworld.dio
            f0, timestamps = estimator(
                audio64,
                fs=self.sample_rate,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=time_step_ms,
            )
            f0 = pyworld.stonemask(audio64, f0, timestamps, self.sample_rate)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
        elif method == "crepe":
            audio_tensor = torch.from_numpy(np.array(audio, copy=True))[None].float()
            f0_tensor, periodicity = torchcrepe.predict(
                audio_tensor,
                self.sample_rate,
                crepe_hop_length,
                f0_min,
                f0_max,
                "full",
                batch_size=512,
                device=self.device,
                return_periodicity=True,
            )
            periodicity = torchcrepe.filter.median(periodicity, 3)
            f0_tensor = torchcrepe.filter.mean(f0_tensor, 3)
            f0_tensor[periodicity < 0.1] = 0
            f0 = f0_tensor[0].cpu().numpy()
        elif method == "rmvpe":
            f0 = self._get_rmvpe().infer_from_audio(audio, thred=0.03)
            if device_type(self.device) == "privateuseone":
                self._release_model()
        elif method == "fcpe":
            f0 = self._get_fcpe(
                f0_min=f0_min,
                f0_max=f0_max,
                threshold=fcpe_threshold,
            ).compute_f0(audio, p_len=frame_count)
        else:
            raise ValueError(f"Unknown f0_method: {method}")

        f0 = np.asarray(f0, dtype=np.float64)
        if f0.shape[0] < frame_count:
            left = max(0, (frame_count - f0.shape[0] + 1) // 2)
            right = max(0, frame_count - f0.shape[0] - left)
            f0 = np.pad(f0, (left, right), mode="constant")
        elif f0.shape[0] > frame_count:
            f0 = f0[:frame_count]
        return f0
