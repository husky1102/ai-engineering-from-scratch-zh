import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import main as gpu_main


class CpuOnlyCuda:
    @staticmethod
    def is_available():
        return False


class CpuOnlyTorch:
    __version__ = "2.4.0"
    version = type("Version", (), {"cuda": None})()
    cuda = CpuOnlyCuda()


class FakeProps:
    total_memory = 24_000_000_000
    major = 8
    minor = 9


class CudaAvailable:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_name(index):
        assert index == 0
        return "NVIDIA Test GPU"

    @staticmethod
    def get_device_properties(index):
        assert index == 0
        return FakeProps()


class CudaTorch:
    __version__ = "2.4.0"
    version = type("Version", (), {"cuda": "12.1"})()
    cuda = CudaAvailable()


class GpuSetupMainTests(unittest.TestCase):
    def test_estimates_fp16_parameter_capacity_in_billions(self):
        self.assertEqual(gpu_main.estimate_fp16_params(24_000_000_000), 12.0)

    def test_collect_gpu_info_handles_missing_torch(self):
        info = gpu_main.collect_gpu_info(torch_module=None)

        self.assertFalse(info["torch_installed"])
        self.assertFalse(info["cuda_available"])
        self.assertIn("pip install torch", info["message"])

    def test_collect_gpu_info_handles_cpu_only_torch(self):
        info = gpu_main.collect_gpu_info(torch_module=CpuOnlyTorch)

        self.assertTrue(info["torch_installed"])
        self.assertEqual(info["torch_version"], "2.4.0")
        self.assertFalse(info["cuda_available"])
        self.assertIn("Google Colab", info["message"])

    def test_collect_gpu_info_reports_cuda_device_details(self):
        info = gpu_main.collect_gpu_info(torch_module=CudaTorch)

        self.assertTrue(info["torch_installed"])
        self.assertTrue(info["cuda_available"])
        self.assertEqual(info["cuda_version"], "12.1")
        self.assertEqual(info["gpu_name"], "NVIDIA Test GPU")
        self.assertEqual(info["memory_gb"], 24.0)
        self.assertEqual(info["compute_capability"], "8.9")
        self.assertEqual(info["estimated_fp16_params_b"], 12.0)

    def test_main_exits_zero_when_torch_is_missing(self):
        with patch.object(gpu_main, "load_torch", return_value=None):
            output = io.StringIO()
            with redirect_stdout(output):
                code = gpu_main.main([])

        self.assertEqual(code, 0)
        self.assertIn("PyTorch not installed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
