import importlib
import sys
import types
import unittest
from dataclasses import dataclass


@dataclass(frozen=True)
class _Runtime:
    name: str

    def __str__(self):
        return self.name


class _Backend:
    created = []
    fail_devices = set()

    def __init__(self, model_path, sr, hop_size, vec_path, device, models_dir, random_seed):
        name = str(device)
        if name in self.fail_devices:
            raise RuntimeError("backend init failed")
        self.device = name
        self.model_path = str(model_path)
        self.closed = False
        self.sampling = (sr, hop_size)
        self.created.append(self)

    def close(self):
        self.closed = True

    def load_new_rvc_model(self, model_path):
        if str(model_path).endswith("bad.onnx"):
            raise RuntimeError("model load failed")
        self.model_path = str(model_path)

    def set_sr_and_hop(self, sr, hop):
        self.sampling = (sr, hop)

    def inference(self, **kwargs):
        return [0]


class ConverterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_core = types.ModuleType("tts_with_rvc.lib.infer_pack.onnx_inference")
        fake_core.OnnxRVC = _Backend
        sys.modules["tts_with_rvc.lib.infer_pack.onnx_inference"] = fake_core
        cls.module = importlib.import_module("tts_with_rvc.converter_onnx")

    def setUp(self):
        _Backend.created.clear()
        _Backend.fail_devices.clear()
        self.module.resolve_execution_device = lambda value: _Runtime(str(value))

    def test_reconfigure_builds_candidate_before_closing_old_runtime(self):
        converter = self.module.OnnxRVCConverter(model_path="voice.onnx", device="cuda:0")
        old = converter._backend
        converter.reconfigure(device="cuda:1")
        self.assertEqual(converter.device, "cuda:1")
        self.assertTrue(old.closed)
        self.assertFalse(converter._backend.closed)

    def test_failed_reconfigure_keeps_old_runtime_intact(self):
        converter = self.module.OnnxRVCConverter(model_path="voice.onnx", device="cuda:0")
        old = converter._backend
        _Backend.fail_devices.add("cuda:9")
        with self.assertRaises(RuntimeError):
            converter.reconfigure(device="cuda:9")
        self.assertIs(converter._backend, old)
        self.assertEqual(converter.device, "cuda:0")
        self.assertFalse(old.closed)

    def test_failed_model_swap_does_not_change_converter_model_path(self):
        converter = self.module.OnnxRVCConverter(model_path="voice.onnx", device="cpu")
        before = converter.model_path
        with self.assertRaises(RuntimeError):
            converter.set_model("bad.onnx")
        self.assertEqual(converter.model_path, before)


if __name__ == "__main__":
    unittest.main()
