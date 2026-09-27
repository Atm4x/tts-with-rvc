import importlib
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from tts_with_rvc.runtime import DeviceResolutionError


class FakeVC:
    instances = []

    def __init__(self, config, model_store=None):
        self.config = config
        self.model_store = model_store
        self.closed = False
        self.__class__.instances.append(self)

    def close(self):
        self.closed = True


class ConverterStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_modules = ModuleType("tts_with_rvc.infer.vc.modules")
        fake_modules.VC = FakeVC
        sys.modules["tts_with_rvc.infer.vc.modules"] = fake_modules
        sys.modules.pop("tts_with_rvc.vc_infer", None)
        cls.module = importlib.import_module("tts_with_rvc.vc_infer")

    def setUp(self):
        FakeVC.instances.clear()

    def test_reconfigure_replaces_runtime_instead_of_mutating_it(self):
        gib = 1024**3
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(total_memory=8 * gib),
            ),
        ):
            converter = self.module.RVCConverter(device="cuda:1")
            first = FakeVC.instances[-1]
            converter.reconfigure(device="cuda:0")
            second = FakeVC.instances[-1]

        self.assertTrue(first.closed)
        self.assertIsNot(first, second)
        self.assertEqual(converter.device, torch.device("cuda:0"))
        converter.close()
        self.assertTrue(second.closed)

    def test_equivalent_configuration_keeps_runtime(self):
        converter = self.module.RVCConverter(device="cpu")
        first = FakeVC.instances[-1]
        converter.reconfigure(device="cpu", is_half=None)
        self.assertIs(first, FakeVC.instances[-1])
        self.assertFalse(first.closed)
        converter.close()

    def test_invalid_reconfigure_is_transactional(self):
        converter = self.module.RVCConverter(device="cpu")
        first = FakeVC.instances[-1]
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(DeviceResolutionError):
                converter.reconfigure(device="cuda:1")
        self.assertEqual(converter.device, torch.device("cpu"))
        self.assertIs(first, FakeVC.instances[-1])
        self.assertFalse(first.closed)
        converter.close()

    def test_two_converters_own_distinct_runtimes(self):
        first_converter = self.module.RVCConverter(device="cpu")
        second_converter = self.module.RVCConverter(device="cpu")
        first_vc, second_vc = FakeVC.instances[-2:]
        self.assertIsNot(first_vc, second_vc)
        self.assertIsNot(first_converter._vc, second_converter._vc)
        first_converter.close()
        self.assertTrue(first_vc.closed)
        self.assertFalse(second_vc.closed)
        second_converter.close()


if __name__ == "__main__":
    unittest.main()
