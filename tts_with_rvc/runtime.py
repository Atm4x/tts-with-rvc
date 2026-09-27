from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Any

import torch

logger = logging.getLogger(__name__)


class DeviceResolutionError(ValueError):
    pass


def device_type(device: Any) -> str:
    kind = getattr(device, "type", None)
    if kind is None:
        kind = torch.device(device).type
    return str(kind)


def _resolve_directml_device(raw: str):
    try:
        import torch_directml
    except ImportError as exc:
        raise DeviceResolutionError(
            "DirectML device requested, but torch-directml is not installed"
        ) from exc

    try:
        index = int(raw.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise DeviceResolutionError(
            f"Invalid DirectML device '{raw}'. Expected dml:N"
        ) from exc

    if index < 0:
        raise DeviceResolutionError(f"Invalid DirectML device index: {index}")
    return torch_directml.device(index)


def resolve_device(device: Any = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if isinstance(device, str):
        raw = device.strip().lower()
        if not raw:
            raise DeviceResolutionError("Device must not be empty")
        if raw.startswith("dml:"):
            return _resolve_directml_device(raw)
        if raw == "cuda":
            raw = "cuda:0"
        elif raw.startswith("mps:"):
            try:
                index = int(raw.split(":", 1)[1])
            except ValueError as exc:
                raise DeviceResolutionError(f"Invalid MPS device '{device}'") from exc
            if index != 0:
                raise DeviceResolutionError("MPS exposes a single logical device")
            raw = "mps"
        try:
            resolved = torch.device(raw)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise DeviceResolutionError(f"Unsupported device '{device}'") from exc
    elif isinstance(device, torch.device):
        resolved = device
    elif getattr(device, "type", None) is not None:
        resolved = device
    else:
        raise DeviceResolutionError(f"Unsupported device value: {device!r}")

    kind = device_type(resolved)
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise DeviceResolutionError("CUDA device requested, but CUDA is unavailable")
        index = 0 if resolved.index is None else int(resolved.index)
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise DeviceResolutionError(
                f"CUDA device index {index} is unavailable; detected {count} CUDA device(s)"
            )
        return torch.device("cuda", index)

    if kind == "mps":
        if not torch.backends.mps.is_available():
            raise DeviceResolutionError("MPS device requested, but MPS is unavailable")
        return torch.device("mps")

    if kind == "cpu":
        return torch.device("cpu")

    if kind == "privateuseone":
        return resolved

    raise DeviceResolutionError(f"Unsupported device type '{kind}'")


def _cuda_auto_dtype(device) -> torch.dtype:
    major, minor = torch.cuda.get_device_capability(device)
    efficient_fp16 = major >= 7 or (major == 6 and minor == 0)
    return torch.float16 if efficient_fp16 else torch.float32


def resolve_dtype(device: Any, is_half: bool | None = None) -> torch.dtype:
    kind = device_type(device)
    if is_half is None:
        return _cuda_auto_dtype(device) if kind == "cuda" else torch.float32
    if is_half and kind == "cpu":
        raise ValueError("Half precision is not supported for the CPU runtime")
    return torch.float16 if is_half else torch.float32


@dataclass(frozen=True, slots=True)
class ChunkingProfile:
    pad_seconds: int
    query_seconds: int
    center_seconds: int
    max_seconds: int
    preprocess_per: float


def _select_chunking_profile(*, dtype: torch.dtype, gpu_memory_gib: float | None) -> ChunkingProfile:
    if gpu_memory_gib is not None and gpu_memory_gib <= 4.5:
        return ChunkingProfile(1, 5, 30, 32, 3.0)
    if dtype == torch.float16:
        return ChunkingProfile(3, 10, 60, 65, 3.7)
    return ChunkingProfile(1, 6, 38, 41, 3.7)


@dataclass(frozen=True, slots=True, init=False)
class RuntimeConfig:
    device: Any
    dtype: torch.dtype
    n_cpu: int
    gpu_memory_gib: float | None
    chunking: ChunkingProfile
    use_jit: bool

    def __init__(
        self,
        device: Any = None,
        is_half: bool | None = None,
        n_cpu: int | None = None,
        use_jit: bool = False,
    ) -> None:
        resolved_device = resolve_device(device)
        dtype = resolve_dtype(resolved_device, is_half)
        workers = cpu_count() if n_cpu is None else int(n_cpu)
        if workers < 1:
            raise ValueError("n_cpu must be at least 1")

        gpu_memory_gib = None
        if device_type(resolved_device) == "cuda":
            gpu_memory_gib = (
                float(torch.cuda.get_device_properties(resolved_device).total_memory)
                / (1024**3)
            )

        chunking = _select_chunking_profile(
            dtype=dtype,
            gpu_memory_gib=gpu_memory_gib,
        )

        object.__setattr__(self, "device", resolved_device)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "n_cpu", workers)
        object.__setattr__(self, "gpu_memory_gib", gpu_memory_gib)
        object.__setattr__(self, "chunking", chunking)
        object.__setattr__(self, "use_jit", bool(use_jit))

        logger.info(
            "RVC runtime configured: device=%s, dtype=%s, gpu_memory_gib=%s",
            resolved_device,
            dtype,
            None if gpu_memory_gib is None else round(gpu_memory_gib, 2),
        )

    @property
    def is_half(self) -> bool:
        return self.dtype == torch.float16

    @property
    def gpu_mem(self) -> int | None:
        if self.gpu_memory_gib is None:
            return None
        return round(self.gpu_memory_gib)

    @property
    def preprocess_per(self) -> float:
        return self.chunking.preprocess_per

    @property
    def x_pad(self) -> int:
        return self.chunking.pad_seconds

    @property
    def x_query(self) -> int:
        return self.chunking.query_seconds

    @property
    def x_center(self) -> int:
        return self.chunking.center_seconds

    @property
    def x_max(self) -> int:
        return self.chunking.max_seconds


Config = RuntimeConfig
