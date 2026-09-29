import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / "scripts/package.py"
spec = importlib.util.spec_from_file_location("release_package", path)
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class ReleaseGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.package_dir = self.root / "tts_with_rvc"
        self.package_dir.mkdir()
        self.metadata = self.root / "pyproject.toml"
        self.metadata.write_text('[project]\nname = "tts-with-rvc-onnx"\nversion = "0.1.9.5"\n', encoding="utf-8")
        self.init = self.package_dir / "__init__.py"
        self.init.write_text('__version__ = "0.1.9.5"\n', encoding="utf-8")
        root_patch = patch.object(package, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        env_patch = patch.dict(os.environ, {"GITHUB_REF_NAME": "releases-onnx", "GITHUB_REPOSITORY": "Atm4x/tts-with-rvc"})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_wrong_branch_rejected_before_pypi_lookup(self):
        with patch.dict(os.environ, {"GITHUB_REF_NAME": "releases"}), patch.object(package, "published_versions") as lookup:
            with self.assertRaisesRegex(ValueError, "only be published"):
                package.check(release=True)
            lookup.assert_not_called()

    def test_wrong_repository_rejected(self):
        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "someone/tts-with-rvc"}):
            with self.assertRaisesRegex(ValueError, "Unexpected publishing repository"):
                package.check(release=True)

    def test_existing_version_rejected(self):
        with patch.object(package, "published_versions", return_value=[package.Version("0.1.9.5")]):
            with self.assertRaisesRegex(ValueError, "already exists"):
                package.check(release=True)

    def test_lower_unpublished_version_rejected(self):
        with patch.object(package, "published_versions", return_value=[package.Version("0.1.9.6")]):
            with self.assertRaisesRegex(ValueError, "greater than"):
                package.check(release=True)

    def test_source_version_disagreement_rejected(self):
        self.init.write_text('__version__ = "0.1.9.4"\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "disagree"):
            package.check()

    def test_bump_updates_metadata_and_runtime_together(self):
        with contextlib.redirect_stdout(io.StringIO()):
            package.set_version(bump=True)
        self.assertIn('version = "0.1.9.6"', self.metadata.read_text(encoding="utf-8"))
        self.assertEqual(package.runtime_version("tts_with_rvc"), "0.1.9.6")

    def test_version_update_preserves_spacing_and_final_newline(self):
        self.init.write_text('__version__ = "0.1.9.5"\n\n\ndef example():\n    return 1\n', encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            package.set_version(bump=True)
        self.assertEqual(
            self.init.read_text(encoding="utf-8"),
            '__version__ = "0.1.9.6"\n\n\ndef example():\n    return 1\n',
        )
        self.assertTrue(self.metadata.read_text(encoding="utf-8").endswith("\n"))

    def test_invalid_version_does_not_modify_files(self):
        before = (self.metadata.read_bytes(), self.init.read_bytes())
        with self.assertRaises(ValueError):
            package.set_version(value="0.1.9.4")
        self.assertEqual(before, (self.metadata.read_bytes(), self.init.read_bytes()))

    def test_next_version_is_allowed_and_emits_correct_environment(self):
        with patch.object(package, "published_versions", return_value=[package.Version("0.1.9.4")]), contextlib.redirect_stdout(io.StringIO()) as output:
            package.check(release=True)
        self.assertIn('"environment": "pypi-onnx"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
