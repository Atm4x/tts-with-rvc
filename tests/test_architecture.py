from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_provider_resolution_is_centralized(self):
        core = (ROOT / "tts_with_rvc/lib/infer_pack/onnx_inference.py").read_text()
        self.assertNotIn("_get_onnx_providers", core)
        self.assertNotIn("InferenceSession(", core)
        self.assertNotIn('startswith("cuda")', core)
        self.assertNotIn("os.getcwd()", core)

    def test_inference_module_has_no_process_wide_async_or_logging_mutations(self):
        source = (ROOT / "tts_with_rvc/inference_onnx.py").read_text()
        self.assertNotIn("nest_asyncio", source)
        self.assertNotIn("new_event_loop", source)
        self.assertNotIn("setLevel(", source)
        self.assertNotIn("def __del__", source)

    def test_core_has_instance_rng_and_no_parent_logger_mutation(self):
        source = (ROOT / "tts_with_rvc/lib/infer_pack/onnx_inference.py").read_text()
        self.assertIn("np.random.default_rng", source)
        self.assertNotIn("np.random.randn", source)
        self.assertNotIn("logger.parent.setLevel", source)

    def test_setup_no_longer_requires_nest_asyncio(self):
        source = (ROOT / "setup.py").read_text()
        self.assertNotIn("nest_asyncio", source)


if __name__ == "__main__":
    unittest.main()
