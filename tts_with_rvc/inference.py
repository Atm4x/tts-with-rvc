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

import edge_tts as tts
from edge_tts import VoicesManager

from tts_with_rvc.vc_infer import RVCConverter

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class _RequestState:
    model_path: str
    voice: str
    index_path: str
    f0_method: str
    output_directory: str | None
    tmp_directory: str | None
    is_half: bool | None



class TTS_RVC:
    def __init__(
        self,
        model_path,
        tmp_directory=None,
        voice="ru-RU-DmitryNeural",
        index_path="",
        f0_method="rmvpe",
        device=None,
        output_directory=None,
        input_directory=None,
        models_dir=None,
        is_half=None,
    ):
        if input_directory is not None:
            warnings.warn(
                "Parameter 'input_directory' is deprecated; use 'tmp_directory' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if tmp_directory is None:
                tmp_directory = input_directory

        self.tmp_directory = tmp_directory
        self.current_voice = voice
        self.current_model = model_path
        self.output_directory = output_directory
        self.f0_method = f0_method
        self.index_path = index_path if not index_path or os.path.exists(index_path) else ""
        self._default_is_half_policy = is_half
        self._state_lock = threading.RLock()
        self._converter = RVCConverter(
            device=device,
            is_half=is_half,
            models_dir=models_dir,
        )
        self._closed = False

        if index_path and not os.path.exists(index_path):
            logger.warning("Index path not found, index disabled: %s", index_path)

    @property
    def device(self):
        return str(self._converter.device)

    @device.setter
    def device(self, value) -> None:
        with self._state_lock:
            self._ensure_open()
            self._converter.reconfigure(
                device=value,
                is_half=self._default_is_half_policy,
            )

    @property
    def is_half(self) -> bool:
        return self._converter.is_half

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TTS_RVC is closed")

    def set_voice(self, voice):
        with self._state_lock:
            self._ensure_open()
            self.current_voice = voice

    def set_index_path(self, index_path):
        with self._state_lock:
            self._ensure_open()
            if index_path and not os.path.exists(index_path):
                logger.warning("Index path not found, index disabled: %s", index_path)
                self.index_path = ""
                return
            self.index_path = index_path

    def set_output_directory(self, directory_path):
        with self._state_lock:
            self._ensure_open()
            self.output_directory = directory_path

    def set_model(self, model_path) -> None:
        with self._state_lock:
            self._ensure_open()
            self.current_model = model_path

    def set_precision(self, is_half: bool | None) -> None:
        with self._state_lock:
            self._ensure_open()
            self._converter.reconfigure(is_half=is_half)
            self._default_is_half_policy = is_half

    def _resolve_call_precision(self, is_half):
        return self._default_is_half_policy if is_half is None else is_half

    def _snapshot(self, *, is_half=None, f0method=None):
        with self._state_lock:
            self._ensure_open()
            return _RequestState(
                model_path=self.current_model,
                voice=self.current_voice,
                index_path=self.index_path,
                f0_method=f0method or self.f0_method,
                output_directory=self.output_directory,
                tmp_directory=self.tmp_directory,
                is_half=self._resolve_call_precision(is_half),
            )

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

    def __call__(
        self,
        text,
        pitch=0,
        tts_rate=0,
        tts_volume=0,
        tts_pitch=0,
        output_filename=None,
        index_rate=0.75,
        is_half=None,
        f0method=None,
        file_index2="",
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.5,
        protect=0.33,
        verbose=False,
    ) -> str:
        state = self._snapshot(is_half=is_half, f0method=f0method)
        input_path, generated_name = _run_coroutine_sync(
            tts_communicate(
                tmp_directory=state.tmp_directory,
                text=text,
                voice=state.voice,
                tts_add_rate=tts_rate,
                tts_add_volume=tts_volume,
                tts_add_pitch=tts_pitch,
            )
        )
        output_name = output_filename or f"{generated_name}.wav"

        try:
            return self._converter.convert(
                model_path=state.model_path,
                input_path=input_path,
                f0_up_key=pitch,
                output_dir_path=state.output_directory,
                is_half=state.is_half,
                f0method=state.f0_method,
                file_index=state.index_path,
                file_index2=file_index2,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                verbose=verbose,
                output_filename=output_name,
            )
        finally:
            _remove_file(input_path)

    async def generate_async(
        self,
        text,
        pitch=0,
        tts_rate=0,
        tts_volume=0,
        tts_pitch=0,
        output_filename=None,
        index_rate=0.75,
        is_half=None,
        f0method=None,
        file_index2="",
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.5,
        protect=0.33,
        verbose=False,
    ) -> str:
        state = self._snapshot(is_half=is_half, f0method=f0method)
        input_path, generated_name = await tts_communicate(
            tmp_directory=state.tmp_directory,
            text=text,
            voice=state.voice,
            tts_add_rate=tts_rate,
            tts_add_volume=tts_volume,
            tts_add_pitch=tts_pitch,
        )
        output_name = output_filename or f"{generated_name}.wav"
        try:
            return await asyncio.to_thread(
                self._converter.convert,
                model_path=state.model_path,
                input_path=input_path,
                f0_up_key=pitch,
                output_dir_path=state.output_directory,
                is_half=state.is_half,
                f0method=state.f0_method,
                file_index=state.index_path,
                file_index2=file_index2,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                verbose=verbose,
                output_filename=output_name,
            )
        finally:
            _remove_file(input_path)

    def voiceover_file(
        self,
        input_path,
        pitch=0,
        output_directory=None,
        filename=None,
        index_rate=0.75,
        is_half=None,
        f0method=None,
        file_index2="",
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.5,
        protect=0.33,
        verbose=False,
    ) -> str:
        state = self._snapshot(is_half=is_half, f0method=f0method)
        output_directory = (
            state.output_directory if output_directory is None else output_directory
        )
        output_name = filename or f"{date_to_short_hash()}.wav"
        return self._converter.convert(
            model_path=state.model_path,
            input_path=input_path,
            f0_up_key=pitch,
            f0method=state.f0_method,
            output_filename=output_name,
            output_dir_path=output_directory,
            file_index=state.index_path,
            file_index2=file_index2,
            index_rate=index_rate,
            is_half=state.is_half,
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

    thread = threading.Thread(target=runner, name="tts-with-rvc-edge-tts")
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


async def get_voices():
    voices = await VoicesManager.create()
    return [data["ShortName"] for data in voices.voices]


async def tts_communicate(
    text,
    tmp_directory=None,
    voice="ru-RU-DmitryNeural",
    tts_add_rate=0,
    tts_add_volume=0,
    tts_add_pitch=0,
):
    if not tmp_directory:
        tmp_directory = os.path.join(tempfile.gettempdir(), "tts_with_rvc")
    os.makedirs(tmp_directory, exist_ok=True)

    communicate = tts.Communicate(
        text=text,
        voice=voice,
        rate=f'{"+" if tts_add_rate >= 0 else ""}{tts_add_rate}%',
        volume=f'{"+" if tts_add_volume >= 0 else ""}{tts_add_volume}%',
        pitch=f'{"+" if tts_add_pitch >= 0 else ""}{tts_add_pitch}Hz',
    )

    file_name = date_to_short_hash()
    input_path = os.path.join(tmp_directory, file_name)
    await communicate.save(input_path)
    return input_path, file_name


async def speech(
    model_path,
    text,
    pitch=0,
    tmp_directory=None,
    voice="ru-RU-DmitryNeural",
    tts_add_rate=0,
    tts_add_volume=0,
    tts_add_pitch=0,
    filename=None,
    output_directory=None,
    index_path="",
    index_rate=0.75,
    is_half=None,
    f0method="rmvpe",
    file_index2="",
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.5,
    protect=0.33,
    device=None,
    verbose=False,
    models_dir=None,
):
    input_path, generated_name = await tts_communicate(
        tmp_directory=tmp_directory,
        text=text,
        voice=voice,
        tts_add_rate=tts_add_rate,
        tts_add_volume=tts_add_volume,
        tts_add_pitch=tts_add_pitch,
    )
    output_name = filename or f"{generated_name}.wav"

    try:
        with RVCConverter(
            device=device,
            is_half=is_half,
            models_dir=models_dir,
        ) as converter:
            return await asyncio.to_thread(
                converter.convert,
                model_path=model_path,
                input_path=input_path,
                f0_up_key=pitch,
                output_dir_path=output_directory,
                f0method=f0method,
                file_index=index_path,
                file_index2=file_index2,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                verbose=verbose,
                output_filename=output_name,
            )
    finally:
        _remove_file(input_path)


def process_text(input_text, param, default_value=0):
    words = input_text.split()
    value = default_value
    index = 0
    while index < len(words):
        if words[index] != param:
            index += 1
            continue
        if index + 1 >= len(words):
            raise ValueError(f'There is no value for parameter "{param}"')
        try:
            value = int(words[index + 1])
        except ValueError as exc:
            raise ValueError(f'Invalid type of argument in "{param}"') from exc
        del words[index : index + 2]
    return value, " ".join(words)
