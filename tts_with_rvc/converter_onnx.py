from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import soundfile

from tts_with_rvc.lib.infer_pack.onnx_inference import OnnxRVC
from tts_with_rvc.runtime_onnx import resolve_execution_device

logger = logging.getLogger(__name__)


class OnnxRVCConverter:
    def __init__(
        self,
        *,
        model_path,
        device="cpu",
        sampling_rate=40000,
        hop_size=512,
        vec_path="vec-768-layer-12.onnx",
        models_dir=None,
        random_seed: int | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._device_spec = device
        self._sampling_rate = int(sampling_rate)
        self._hop_size = int(hop_size)
        self._vec_path = vec_path
        self._models_dir = models_dir
        self._random_seed = random_seed
        self._model_path = str(Path(model_path).expanduser().resolve())
        self._runtime = resolve_execution_device(device)
        self._backend = self._build_backend(self._runtime)
        self._closed = False

    def _build_backend(self, runtime):
        return OnnxRVC(
            model_path=self._model_path,
            sr=self._sampling_rate,
            hop_size=self._hop_size,
            vec_path=self._vec_path,
            device=runtime,
            models_dir=self._models_dir,
            random_seed=self._random_seed,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OnnxRVCConverter is closed")

    @property
    def device(self) -> str:
        return str(self._runtime)

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    def reconfigure(self, *, device) -> None:
        with self._lock:
            self._ensure_open()
            runtime = resolve_execution_device(device)
            if runtime == self._runtime:
                self._device_spec = device
                return

            candidate = self._build_backend(runtime)
            old_backend = self._backend
            self._backend = candidate
            self._runtime = runtime
            self._device_spec = device
            old_backend.close()
            logger.info("ONNX RVC runtime switched to %s", runtime)

    def set_model(self, model_path) -> None:
        with self._lock:
            self._ensure_open()
            target = str(Path(model_path).expanduser().resolve())
            self._backend.load_new_rvc_model(target)
            self._model_path = target

    def set_sampling_params(self, sampling_rate, hop_size) -> None:
        with self._lock:
            self._ensure_open()
            sampling_rate = int(sampling_rate)
            hop_size = int(hop_size)
            self._backend.set_sr_and_hop(sampling_rate, hop_size)
            self._sampling_rate = sampling_rate
            self._hop_size = hop_size

    def convert(
        self,
        *,
        input_path,
        output_path,
        pitch=0,
        index_path="",
        index_path2="",
        index_rate=0.75,
        f0_method="pm",
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.5,
        protect=0.33,
        verbose=False,
    ) -> str:
        with self._lock:
            self._ensure_open()
            audio = self._backend.inference(
                raw_path=input_path,
                sid=0,
                f0_method=f0_method,
                f0_up_key=pitch,
                index_file=index_path,
                index_file2=index_path2,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                verbose=verbose,
            )
            target_sr = self._sampling_rate if resample_sr == 0 else int(resample_sr)
            output_path = Path(output_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(str(output_path), audio, target_sr)
            return str(output_path)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._backend.close()
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
