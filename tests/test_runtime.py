from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from tts_with_rvc.runtime import Config, DeviceResolutionError, resolve_device


class RuntimeConfigTests(unittest.TestCase):
    def test_cpu_runtime_is_explicit_fp32(self):
        config = Config(device="cpu")
        self.assertEqual(config.device, torch.device("cpu"))
        self.assertEqual(config.dtype, torch.float32)
        self.assertFalse(config.is_half)
        self.assertIsNone(config.gpu_mem)

    def test_runtime_config_is_immutable(self):
        config = Config(device="cpu")
        with self.assertRaises(AttributeError):
            config.device = torch.device("cpu")

    def test_cpu_rejects_half_precision(self):
        with self.assertRaises(ValueError):
            Config(device="cpu", is_half=True)

    def test_cuda_uses_selected_device_properties(self):
        gib = 1024**3

        def properties(device):
            self.assertEqual(device, torch.device("cuda:1"))
            return SimpleNamespace(total_memory=4 * gib)

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
            patch.object(torch.cuda, "get_device_properties", side_effect=properties),
        ):
            config = Config(device="cuda:1")

        self.assertEqual(config.device, torch.device("cuda:1"))
        self.assertEqual(config.dtype, torch.float16)
        self.assertAlmostEqual(config.gpu_memory_gib, 4.0)
        self.assertEqual(config.gpu_mem, 4)
        self.assertEqual(
            (config.x_pad, config.x_query, config.x_center, config.x_max),
            (1, 5, 30, 32),
        )

    def test_auto_precision_uses_capability_not_gpu_name(self):
        gib = 1024**3
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=1),
            patch.object(torch.cuda, "get_device_capability", return_value=(6, 1)),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(total_memory=8 * gib),
            ),
        ):
            config = Config(device="cuda:0")
        self.assertEqual(config.dtype, torch.float32)

    def test_bare_cuda_is_normalized_to_cuda_zero(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            device = resolve_device("cuda")
        self.assertEqual(device, torch.device("cuda:0"))

    def test_invalid_cuda_index_is_rejected(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=1),
        ):
            with self.assertRaises(DeviceResolutionError):
                resolve_device("cuda:1")


if __name__ == "__main__":
    unittest.main()
