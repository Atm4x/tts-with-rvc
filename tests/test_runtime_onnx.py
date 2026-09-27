import unittest
from types import SimpleNamespace

from tts_with_rvc.runtime_onnx import (
    ExecutionDeviceError,
    create_session,
    resolve_execution_device,
)


class _SessionOptions:
    def __init__(self):
        self.enable_mem_pattern = True
        self.execution_mode = None


class _Session:
    def __init__(self, path, sess_options, providers, active=None):
        self.path = path
        self.options = sess_options
        self.providers_arg = providers
        self._active = active or [
            item[0] if isinstance(item, tuple) else item for item in providers
        ]

    def get_providers(self):
        return self._active


class _FakeOrt:
    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"

    SessionOptions = _SessionOptions

    def __init__(self, available, active=None):
        self.available = available
        self.active = active
        self.sessions = []

    def get_available_providers(self):
        return list(self.available)

    def InferenceSession(self, path, sess_options, providers):
        session = _Session(path, sess_options, providers, self.active)
        self.sessions.append(session)
        return session


class RuntimeTests(unittest.TestCase):
    def test_cuda_index_is_preserved_in_provider_options(self):
        ort = _FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
        runtime = resolve_execution_device("cuda:1", ort_module=ort)
        self.assertEqual(str(runtime), "cuda:1")
        self.assertEqual(runtime.index, 1)
        self.assertEqual(
            runtime.providers[0],
            ("CUDAExecutionProvider", {"device_id": "1"}),
        )

    def test_bare_cuda_and_dml_resolve_to_adapter_zero(self):
        ort = _FakeOrt(
            ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        )
        self.assertEqual(str(resolve_execution_device("cuda", ort_module=ort)), "cuda:0")
        self.assertEqual(str(resolve_execution_device("dml", ort_module=ort)), "dml:0")

    def test_missing_provider_fails_instead_of_falling_back(self):
        ort = _FakeOrt(["CPUExecutionProvider"])
        with self.assertRaises(ExecutionDeviceError):
            resolve_execution_device("cuda:1", ort_module=ort)

    def test_invalid_device_fails(self):
        ort = _FakeOrt(["CPUExecutionProvider"])
        for value in ("cuda:x", "cuda:-1", "banana", "mps:0"):
            with self.subTest(value=value):
                with self.assertRaises(ExecutionDeviceError):
                    resolve_execution_device(value, ort_module=ort)

    def test_cuda_session_receives_explicit_device_id(self):
        ort = _FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
        runtime = resolve_execution_device("cuda:2", ort_module=ort)
        session = create_session("model.onnx", runtime, ort_module=ort)
        self.assertEqual(
            session.providers_arg,
            [
                ("CUDAExecutionProvider", {"device_id": "2"}),
                "CPUExecutionProvider",
            ],
        )

    def test_dml_session_uses_required_session_options(self):
        ort = _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
        runtime = resolve_execution_device("dml:1", ort_module=ort)
        session = create_session("model.onnx", runtime, ort_module=ort)
        self.assertFalse(session.options.enable_mem_pattern)
        self.assertEqual(session.options.execution_mode, "sequential")
        self.assertEqual(
            session.providers_arg[0],
            ("DmlExecutionProvider", {"device_id": "1"}),
        )

    def test_session_rejects_silent_primary_provider_fallback(self):
        ort = _FakeOrt(
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            active=["CPUExecutionProvider"],
        )
        runtime = resolve_execution_device("cuda:1", ort_module=ort)
        with self.assertRaises(RuntimeError):
            create_session("model.onnx", runtime, ort_module=ort)


if __name__ == "__main__":
    unittest.main()
