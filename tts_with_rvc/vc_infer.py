from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from scipy.io import wavfile

from tts_with_rvc.assets import ModelStore
from tts_with_rvc.infer.vc.modules import VC
from tts_with_rvc.runtime import RuntimeConfig

logger = logging.getLogger(__name__)

_UNSET = object()



def resolve_output_path(output_dir_path, output_filename):
    if output_filename and os.path.isabs(output_filename):
        output_path = Path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return str(output_path)

    filename = output_filename or "out.wav"
    output_dir = Path(output_dir_path) if output_dir_path else Path("temp")
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


class RVCConverter:
    def __init__(
        self,
        *,
        device=None,
        is_half: bool | None = None,
        models_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._model_store = ModelStore(models_dir)
        self._device_spec = device
        self._is_half_policy = is_half
        self._config = RuntimeConfig(device=device, is_half=is_half)
        self._vc = VC(self._config, model_store=self._model_store)
        self._closed = False

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    @property
    def device(self):
        return self._config.device

    @property
    def is_half(self) -> bool:
        return self._config.is_half

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RVCConverter is closed")

    def reconfigure(self, *, device=_UNSET, is_half=_UNSET) -> None:
        with self._lock:
            self._ensure_open()
            candidate_device = self._device_spec if device is _UNSET else device
            candidate_precision = (
                self._is_half_policy if is_half is _UNSET else is_half
            )
            new_config = RuntimeConfig(
                device=candidate_device,
                is_half=candidate_precision,
                n_cpu=self._config.n_cpu,
                use_jit=self._config.use_jit,
            )

            if new_config == self._config:
                self._device_spec = candidate_device
                self._is_half_policy = candidate_precision
                return

            new_vc = VC(new_config, model_store=self._model_store)
            old_vc = self._vc
            self._device_spec = candidate_device
            self._is_half_policy = candidate_precision
            self._config = new_config
            self._vc = new_vc
            old_vc.close()

    def convert(
        self,
        model_path,
        f0_up_key=0,
        input_path=None,
        output_dir_path=None,
        is_half=_UNSET,
        f0method="rmvpe",
        file_index="",
        file_index2="",
        index_rate=1,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.5,
        protect=0.33,
        verbose=False,
        device=_UNSET,
        output_filename="out.wav",
    ) -> str:
        del verbose
        if input_path is None:
            raise ValueError("input_path is required")

        with self._lock:
            self._ensure_open()
            if device is not _UNSET or is_half is not _UNSET:
                self.reconfigure(device=device, is_half=is_half)

            self._vc.load_model(model_path)
            target_sr, output_audio = self._vc.vc_single(
                0,
                input_path,
                f0_up_key,
                None,
                f0method,
                file_index,
                file_index2,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
            )

            output_path = resolve_output_path(output_dir_path, output_filename)
            wavfile.write(output_path, target_sr, output_audio)
            saved_to = os.path.abspath(output_path)
            logger.info("Saved: %s", saved_to)
            return saved_to

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._vc.close()
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def rvc_convert(
    model_path,
    f0_up_key=0,
    input_path=None,
    output_dir_path=None,
    _is_half=None,
    f0method="rmvpe",
    file_index="",
    file_index2="",
    index_rate=1,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.5,
    protect=0.33,
    verbose=False,
    device=None,
    output_filename="out.wav",
    models_dir=None,
):
    with RVCConverter(
        device=device,
        is_half=_is_half,
        models_dir=models_dir,
    ) as converter:
        return converter.convert(
            model_path=model_path,
            f0_up_key=f0_up_key,
            input_path=input_path,
            output_dir_path=output_dir_path,
            f0method=f0method,
            file_index=file_index,
            file_index2=file_index2,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            verbose=verbose,
            output_filename=output_filename,
        )


if __name__ == "__main__":
    rvc_convert(model_path="models\\DenVot.pth", input_path="out.wav")
