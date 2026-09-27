from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import threading
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import edge_tts as tts

from tts_with_rvc.converter_onnx import OnnxRVCConverter

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RequestState:
    voice: str
    index_path: str
    f0_method: str
    output_directory: str | None
    tmp_directory: str


class TTS_RVC:
    def __init__(
        self,
        model_path,
        tmp_directory=None,
        voice="ru-RU-DmitryNeural",
        index_path="",
        f0_method="pm",
        output_directory=None,
        sampling_rate=40000,
        hop_size=512,
        device="dml",
        input_directory=None,
        models_dir=None,
        vec_path="vec-768-layer-12.onnx",
        random_seed: int | None = None,
    ):
        if input_directory is not None:
            warnings.warn(
                "Parameter 'input_directory' is deprecated; use 'tmp_directory' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if tmp_directory is None:
                tmp_directory = input_directory

        if tmp_directory is None:
            tmp_directory = os.path.join(tempfile.gettempdir(), "tts_with_rvc_onnx")
        Path(tmp_directory).mkdir(parents=True, exist_ok=True)

        self.tmp_directory = str(tmp_directory)
        self.current_voice = voice
        self.output_directory = output_directory
        self.f0_method = f0_method
        self.index_path = _validated_index_path(index_path)
        self._state_lock = threading.RLock()
        self._converter = OnnxRVCConverter(
            model_path=model_path,
            device=device,
            sampling_rate=sampling_rate,
            hop_size=hop_size,
            vec_path=vec_path,
            models_dir=models_dir,
            random_seed=random_seed,
        )
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TTS_RVC is closed")

    @property
    def device(self) -> str:
        return self._converter.device

    @device.setter
    def device(self, value) -> None:
        self.set_device(value)

    @property
    def current_model(self) -> str:
        return self._converter.model_path

    @property
    def sampling_rate(self) -> int:
        return self._converter.sampling_rate

    def _snapshot(self, *, f0method=None) -> _RequestState:
        with self._state_lock:
            self._ensure_open()
            return _RequestState(
                voice=self.current_voice,
                index_path=self.index_path,
                f0_method=f0method or self.f0_method,
                output_directory=self.output_directory,
                tmp_directory=self.tmp_directory,
            )

    def set_device(self, device):
        with self._state_lock:
            self._ensure_open()
            self._converter.reconfigure(device=device)

    def set_model(self, model_path):
        with self._state_lock:
            self._ensure_open()
            self._converter.set_model(model_path)

    def set_sampling_params(self, sr, hop):
        with self._state_lock:
            self._ensure_open()
            self._converter.set_sampling_params(sr, hop)

    def set_voice(self, voice):
        with self._state_lock:
            self._ensure_open()
            self.current_voice = voice

    def set_index_path(self, index_path):
        with self._state_lock:
            self._ensure_open()
            self.index_path = _validated_index_path(index_path)

    def set_output_directory(self, directory_path):
        with self._state_lock:
            self._ensure_open()
            self.output_directory = directory_path

    def __call__(
        self,
        text,
        pitch=0,
        tts_rate=0,
        tts_volume=0,
        tts_pitch=0,
        output_filename=None,
        index_rate=0.75,
        f0method=None,
        file_index2="",
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.5,
        protect=0.33,
        verbose=False,
    ):
        return _run_coroutine_sync(
            self.async_call(
                text=text,
                pitch=pitch,
                tts_rate=tts_rate,
                tts_volume=tts_volume,
                tts_pitch=tts_pitch,
                output_filename=output_filename,
                index_rate=index_rate,
                f0method=f0method,
                file_index2=file_index2,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                verbose=verbose,
            )
        )

    async def async_call(
        self,
        text,
        pitch=0,
        tts_rate=0,
        tts_volume=0,
        tts_pitch=0,
        output_filename=None,
        index_rate=0.75,
        f0method=None,
        file_index2="",
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.5,
        protect=0.33,
        verbose=False,
    ):
        state = self._snapshot(f0method=f0method)
        input_path, generated_name = await tts_communicate(
            tmp_directory=state.tmp_directory,
            text=text,
            voice=state.voice,
            tts_add_rate=tts_rate,
            tts_add_volume=tts_volume,
            tts_add_pitch=tts_pitch,
        )
        output_path = resolve_output_path(
            state.output_directory,
            output_filename or f"{generated_name}.wav",
        )
        try:
            return await asyncio.to_thread(
                self._converter.convert,
                input_path=input_path,
                output_path=output_path,
                pitch=pitch,
                index_path=state.index_path,
                index_path2=file_index2,
                index_rate=index_rate,
                f0_method=state.f0_method,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                verbose=verbose,
            )
        finally:
            _remove_file(input_path)

    def voiceover_file(
        self,
        input_path,
        pitch=0,
        output_filename=None,
        index_rate=0.75,
        f0method=None,
        file_index2="",
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.5,
        protect=0.33,
        verbose=False,
    ):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        state = self._snapshot(f0method=f0method)
        output_path = resolve_output_path(state.output_directory, output_filename)
        return self._converter.convert(
            input_path=input_path,
            output_path=output_path,
            pitch=pitch,
            index_path=state.index_path,
            index_path2=file_index2,
            index_rate=index_rate,
            f0_method=state.f0_method,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            verbose=verbose,
        )

    def process_args(self, text):
        rate_param, text = process_text(text, param="--tts-rate")
        volume_param, text = process_text(text, param="--tts-volume")
        tts_pitch_param, text = process_text(text, param="--tts-pitch")
        rvc_pitch_param, text = process_text(text, param="--rvc-pitch")
        return [rate_param, volume_param, tts_pitch_param, rvc_pitch_param], text

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._converter.close()
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _validated_index_path(index_path) -> str:
    if not index_path:
        return ""
    if not os.path.exists(index_path):
        logger.warning("Index path not found, index disabled: %s", index_path)
        return ""
    return str(Path(index_path).expanduser().resolve())


def _run_coroutine_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = []
    error = []

    def runner():
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=runner, name="tts-with-rvc-onnx-edge-tts")
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _remove_file(path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def date_to_short_hash():
    payload = f"{datetime.now().isoformat()}:{uuid.uuid4().hex}"
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def resolve_output_path(output_dir_path, output_filename=None):
    if output_filename and os.path.isabs(output_filename):
        output = Path(output_filename)
    else:
        filename = output_filename or f"{date_to_short_hash()}.wav"
        root = (
            Path(output_dir_path)
            if output_dir_path
            else Path(tempfile.gettempdir()) / "tts_with_rvc_onnx" / "output"
        )
        output = root / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    return str(output.resolve())


async def tts_communicate(
    tmp_directory,
    text,
    voice="ru-RU-DmitryNeural",
    tts_add_rate=0,
    tts_add_volume=0,
    tts_add_pitch=0,
):
    os.makedirs(tmp_directory, exist_ok=True)
    communicate = tts.Communicate(
        text=text,
        voice=voice,
        rate=f'{"+" if tts_add_rate >= 0 else ""}{tts_add_rate}%',
        volume=f'{"+" if tts_add_volume >= 0 else ""}{tts_add_volume}%',
        pitch=f'{"+" if tts_add_pitch >= 0 else ""}{tts_add_pitch}Hz',
    )
    file_name = date_to_short_hash()
    input_path = os.path.join(tmp_directory, f"{file_name}.wav")
    await communicate.save(input_path)
    return input_path, file_name


def process_text(input_text, param, default_value=0):
    words = input_text.split()
    value = default_value
    i = 0
    while i < len(words):
        if words[i] != param:
            i += 1
            continue
        if i + 1 >= len(words):
            logger.warning("No value provided for parameter %s; ignoring", param)
            words.pop(i)
            continue
        candidate = words[i + 1]
        try:
            value = int(candidate)
        except ValueError:
            logger.warning(
                "Invalid value %r for parameter %s; expected an integer",
                candidate,
                param,
            )
            words.pop(i)
            continue
        del words[i : i + 2]
    return value, " ".join(words)
