import importlib.util
from pathlib import Path
import sys
import types
import unittest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).parents[1]


class PublicApiTests(unittest.TestCase):
    def test_package_import_is_lazy(self):
        import tts_with_rvc

        with (ROOT / "pyproject.toml").open("rb") as stream:
            version = tomllib.load(stream)["project"]["version"]
        self.assertEqual(tts_with_rvc.__version__, version)
        self.assertIn("TTS_RVC", tts_with_rvc.__all__)

    def _load_inference_with_stubs(self):
        edge = types.ModuleType("edge_tts")
        edge.Communicate = object
        sys.modules["edge_tts"] = edge
        converter = types.ModuleType("tts_with_rvc.converter_onnx")
        converter.OnnxRVCConverter = object
        sys.modules["tts_with_rvc.converter_onnx"] = converter
        path = ROOT / "tts_with_rvc/inference_onnx.py"
        spec = importlib.util.spec_from_file_location("_inference_onnx_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_process_text_keeps_invalid_value_as_text(self):
        module = self._load_inference_with_stubs()
        value, text = module.process_text("hello --tts-rate nope world", "--tts-rate")
        self.assertEqual(value, 0)
        self.assertEqual(text, "hello nope world")

    def test_process_text_extracts_integer_argument(self):
        module = self._load_inference_with_stubs()
        value, text = module.process_text("hello --rvc-pitch -3 world", "--rvc-pitch")
        self.assertEqual(value, -3)
        self.assertEqual(text, "hello world")


if __name__ == "__main__":
    unittest.main()
