import ast
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "tts_with_rvc/cuda_device.py"
SPEC = importlib.util.spec_from_file_location("cuda_device", MODULE_PATH)
CUDA_DEVICE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CUDA_DEVICE)
set_cuda_device_for_thread = CUDA_DEVICE.set_cuda_device_for_thread


class CudaDeviceTests(unittest.TestCase):
    def test_rvc_convert_sets_cuda_device_before_loading_and_inference(self):
        source = (ROOT / "tts_with_rvc/vc_infer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        convert = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "rvc_convert"
        )
        helper_call = next(
            node
            for node in ast.walk(convert)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set_cuda_device_for_thread"
        )

        validation = next(
            node
            for node in ast.walk(convert)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "ValueError"
        )
        self.assertLess(validation.lineno, helper_call.lineno)

    def test_selects_requested_cuda_device(self):
        selected = SimpleNamespace(type="cuda", index=1)
        torch_module = SimpleNamespace(
            device=Mock(return_value=selected),
            cuda=SimpleNamespace(set_device=Mock()),
        )

        set_cuda_device_for_thread(torch_module, "cuda:1")

        torch_module.device.assert_called_once_with("cuda:1")
        torch_module.cuda.set_device.assert_called_once_with(selected)

    def test_does_not_change_cpu_device(self):
        torch_module = SimpleNamespace(
            device=Mock(return_value=SimpleNamespace(type="cpu")),
            cuda=SimpleNamespace(set_device=Mock()),
        )

        set_cuda_device_for_thread(torch_module, "cpu")

        torch_module.cuda.set_device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
