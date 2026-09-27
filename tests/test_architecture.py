from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "tts_with_rvc"


class ArchitectureTests(unittest.TestCase):
    def test_active_runtime_has_no_process_wide_cuda_or_import_side_effects(self):
        paths = [
            PACKAGE / "inference.py",
            PACKAGE / "vc_infer.py",
            PACKAGE / "infer/vc/modules.py",
            PACKAGE / "infer/vc/pipeline.py",
            PACKAGE / "infer/vc/f0.py",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for forbidden in (
            "torch.cuda.set_device",
            "torch.cuda.empty_cache",
            "sys.path.append",
            "nest_asyncio",
            "torch.serialization.add_safe_globals",
        ):
            self.assertNotIn(forbidden, combined)

    def test_runtime_config_does_not_match_gpu_marketing_names(self):
        source = (PACKAGE / "runtime.py").read_text(encoding="utf-8")
        for forbidden in ("gpu_name", "P40", "P10", "1060", "1070", "1080"):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("get_device_name", source)

    def test_vc_infer_has_no_module_singleton_runtime(self):
        source = (PACKAGE / "vc_infer.py").read_text(encoding="utf-8")
        self.assertNotIn("config = Config()", source)
        self.assertNotIn("vc = VC(config)", source)
        self.assertNotIn("last_model_path", source)
        self.assertIn("class RVCConverter", source)

    def test_loaded_model_state_is_owned_by_vc(self):
        source = (PACKAGE / "infer/vc/modules.py").read_text(encoding="utf-8")
        self.assertIn("class LoadedRVCModel", source)
        self.assertIn("self._loaded", source)
        self.assertNotIn("self.cpt", source)

    def test_f0_runtime_is_separate_from_audio_pipeline(self):
        pipeline = (PACKAGE / "infer/vc/pipeline.py").read_text(encoding="utf-8")
        f0 = (PACKAGE / "infer/vc/f0.py").read_text(encoding="utf-8")
        self.assertIn("F0Extractor", pipeline)
        self.assertIn("class F0Extractor", f0)
        self.assertIn("dtype=torch.float32", f0)
        self.assertNotIn('f0_method.lower() == "fcpe"', (PACKAGE / "vc_infer.py").read_text(encoding="utf-8"))

    def test_package_exports_are_lazy(self):
        source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("def __getattr__", source)
        self.assertNotIn("from .inference import *", source)

    def test_internal_predictors_do_not_autoselect_cuda(self):
        rmvpe = (PACKAGE / "infer/lib/rmvpe.py").read_text(encoding="utf-8")
        fcpe = (
            PACKAGE
            / "infer/lib/infer_pack/f0_modules/F0Predictor/FCPE.py"
        ).read_text(encoding="utf-8")
        active_rmvpe = rmvpe[: rmvpe.find('if __name__ == "__main__"')]
        self.assertNotIn("torch.cuda.is_available()", active_rmvpe)
        self.assertNotIn("torch.cuda.is_available()", fcpe)


if __name__ == "__main__":
    unittest.main()
