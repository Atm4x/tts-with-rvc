from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ExecutionDeviceError(ValueError):
    pass


_PROVIDER_BY_BACKEND = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
}


@dataclass(frozen=True, slots=True)
class OnnxExecutionDevice:
    backend: str
    index: int | None = None

    @property
    def provider_name(self) -> str:
        return _PROVIDER_BY_BACKEND[self.backend]

    @property
    def providers(self) -> tuple[Any, ...]:
        if self.backend == "cpu":
            return ("CPUExecutionProvider",)
        return (
            (self.provider_name, {"device_id": str(self.index)}),
            "CPUExecutionProvider",
        )

    def __str__(self) -> str:
        if self.backend == "cpu":
            return "cpu"
        return f"{self.backend}:{self.index}"


def _parse_device(device: str | OnnxExecutionDevice | None) -> OnnxExecutionDevice:
    if isinstance(device, OnnxExecutionDevice):
        return device

    raw = "cpu" if device is None else str(device).strip().lower()
    if raw == "cpu":
        return OnnxExecutionDevice("cpu", None)
    if raw in {"cuda", "dml"}:
        return OnnxExecutionDevice(raw, 0)

    backend, separator, ordinal = raw.partition(":")
    if backend not in {"cuda", "dml"} or not separator:
        raise ExecutionDeviceError(
            f"Unsupported execution device '{device}'. Expected cpu, cuda:N, or dml:N"
        )
    try:
        index = int(ordinal)
    except ValueError as exc:
        raise ExecutionDeviceError(f"Invalid device index in '{device}'") from exc
    if index < 0:
        raise ExecutionDeviceError(f"Device index must be non-negative: {device}")
    return OnnxExecutionDevice(backend, index)


def resolve_execution_device(
    device: str | OnnxExecutionDevice | None,
    *,
    ort_module=None,
) -> OnnxExecutionDevice:
    resolved = _parse_device(device)
    if ort_module is None:
        try:
            import onnxruntime as ort_module
        except ImportError as exc:
            raise ExecutionDeviceError(
                "ONNX Runtime is not installed. Install the CPU, CUDA, or DirectML runtime package."
            ) from exc

    available = set(ort_module.get_available_providers())
    required = resolved.provider_name
    if required not in available:
        raise ExecutionDeviceError(
            f"{resolved} requires {required}, but this ONNX Runtime build exposes: "
            f"{', '.join(sorted(available)) or 'no providers'}"
        )
    return resolved


def create_session(model_path, runtime: OnnxExecutionDevice, *, ort_module=None):
    if ort_module is None:
        import onnxruntime as ort_module

    options = ort_module.SessionOptions()
    if runtime.backend == "dml":
        options.enable_mem_pattern = False
        options.execution_mode = ort_module.ExecutionMode.ORT_SEQUENTIAL

    session = ort_module.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=list(runtime.providers),
    )

    active = session.get_providers()
    if not active or active[0] != runtime.provider_name:
        raise RuntimeError(
            f"Requested {runtime.provider_name} for {runtime}, but ONNX Runtime activated {active}"
        )
    return session
