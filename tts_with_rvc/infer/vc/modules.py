from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary

from tts_with_rvc.assets import ModelStore
from tts_with_rvc.infer.lib.audio import load_audio
from tts_with_rvc.infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from tts_with_rvc.infer.vc.pipeline import Pipeline
from tts_with_rvc.runtime import RuntimeConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoadedRVCModel:
    path: Path
    target_sr: int
    if_f0: int
    version: str
    net_g: torch.nn.Module
    pipeline: Pipeline

    def close(self) -> None:
        self.pipeline.close()


class VC:
    def __init__(self, config: RuntimeConfig, model_store: ModelStore | None = None) -> None:
        self.config = config
        self.model_store = model_store or ModelStore()
        self._loaded: LoadedRVCModel | None = None
        self.hubert_model = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def net_g(self):
        return None if self._loaded is None else self._loaded.net_g

    @property
    def pipeline(self):
        return None if self._loaded is None else self._loaded.pipeline

    @property
    def tgt_sr(self):
        return None if self._loaded is None else self._loaded.target_sr

    @property
    def version(self):
        return None if self._loaded is None else self._loaded.version

    @property
    def if_f0(self):
        return None if self._loaded is None else self._loaded.if_f0

    @property
    def loaded_model_path(self) -> str | None:
        return None if self._loaded is None else str(self._loaded.path)

    def load_model(self, model_path: str | os.PathLike[str]) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RVC model not found: {path}")
        if self._loaded is not None and self._loaded.path == path:
            return

        logger.info("Loading RVC model: %s", path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model_config = list(checkpoint["config"])
        model_config[-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
        target_sr = int(model_config[-1])
        if_f0 = int(checkpoint.get("f0", 1))
        version = str(checkpoint.get("version", "v1"))

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }.get((version, if_f0))
        if synthesizer_class is None:
            raise ValueError(
                f"Unsupported RVC model combination: version={version}, f0={if_f0}"
            )

        net_g = synthesizer_class(*model_config, is_half=self.config.is_half)
        if hasattr(net_g, "enc_q"):
            del net_g.enc_q
        net_g.load_state_dict(checkpoint["weight"], strict=False)
        del checkpoint

        self.unload_model(keep_hubert=True)
        net_g.eval().to(device=self.config.device, dtype=self.config.dtype)
        pipeline = Pipeline(target_sr, self.config, model_store=self.model_store)
        self._loaded = LoadedRVCModel(
            path=path,
            target_sr=target_sr,
            if_f0=if_f0,
            version=version,
            net_g=net_g,
            pipeline=pipeline,
        )

    def get_vc(self, sid) -> None:
        if sid == "" or sid == []:
            self.close()
            return
        self.load_model(sid)

    def unload_model(self, *, keep_hubert: bool = True) -> None:
        if self._loaded is not None:
            self._loaded.close()
            self._loaded = None
        if not keep_hubert:
            self.hubert_model = None

    def close(self) -> None:
        self.unload_model(keep_hubert=False)

    def _load_hubert(self):
        hubert_path = self.model_store.get(
            "lj1995/VoiceConversionWebUI",
            "hubert_base.pt",
        )
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        context = safe_globals([Dictionary]) if safe_globals is not None else nullcontext()
        with context:
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                [str(hubert_path)],
                suffix="",
            )
        hubert = models[0]
        return hubert.to(
            device=self.config.device,
            dtype=self.config.dtype,
        ).eval()

    def _ensure_hubert(self):
        if self.hubert_model is None:
            self.hubert_model = self._load_hubert()
        return self.hubert_model

    @staticmethod
    def _resolve_index_path(file_index: str, file_index2: str) -> str:
        source = file_index or file_index2 or ""
        return (
            source.strip()
            .strip('"')
            .strip()
            .replace("trained", "added")
        )

    def _require_model(self) -> LoadedRVCModel:
        if self._loaded is None:
            raise RuntimeError("No RVC model is loaded")
        return self._loaded

    def _convert_audio(
        self,
        *,
        sid: int,
        audio: np.ndarray,
        f0_up_key: int,
        f0_file: Any,
        f0_method: str,
        file_index: str,
        file_index2: str,
        index_rate: float,
        filter_radius: int,
        resample_sr: int,
        rms_mix_rate: float,
        protect: float,
    ):
        loaded = self._require_model()
        audio = np.asarray(audio, dtype=np.float32)
        audio_max = (float(np.max(np.abs(audio))) if audio.size else 0.0) / 0.95
        if audio_max > 1:
            audio = audio / audio_max

        times = [0.0, 0.0, 0.0]
        index_path = self._resolve_index_path(file_index, file_index2)
        hubert = self._ensure_hubert()

        audio_opt = loaded.pipeline.pipeline(
            hubert,
            loaded.net_g,
            sid,
            audio,
            times,
            int(f0_up_key),
            f0_method,
            index_path,
            index_rate,
            loaded.if_f0,
            filter_radius,
            loaded.target_sr,
            resample_sr,
            rms_mix_rate,
            loaded.version,
            protect,
            f0_file,
        )
        target_sr = (
            resample_sr
            if loaded.target_sr != resample_sr >= 16000
            else loaded.target_sr
        )
        return target_sr, audio_opt

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        if input_audio_path is None:
            raise ValueError("input_audio_path is required")
        audio = load_audio(input_audio_path, 16000)
        return self._convert_audio(
            sid=sid,
            audio=audio,
            f0_up_key=f0_up_key,
            f0_file=f0_file,
            f0_method=f0_method,
            file_index=file_index,
            file_index2=file_index2,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )

    def vc_stream(
        self,
        sid,
        audio,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        return self._convert_audio(
            sid=sid,
            audio=audio,
            f0_up_key=f0_up_key,
            f0_file=f0_file,
            f0_method=f0_method,
            file_index=file_index,
            file_index2=file_index2,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )
