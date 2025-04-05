

import platform
import subprocess
import os
from setuptools import setup, find_packages

def check_gpu_provider_for_setup():
    if platform.system() != "Windows":
        print("INFO: Non-Windows system detected. Cannot use WMI for GPU check.")
        return None # WMI is Windows-only

    try:
        command = "wmic path win32_VideoController get name"
        output = subprocess.check_output(
            command, text=True, shell=True,
            stdin=subprocess.DEVNULL, stderr=subprocess.PIPE
        ).strip()

        if "NVIDIA" in output:
            print("INFO: Detected 'NVIDIA' in WMI video controller output.")
            return "NVIDIA"
        elif "AMD" in output or "Radeon" in output:
            print("INFO: Detected 'AMD' or 'Radeon' in WMI video controller output.")
            return "AMD"
        else:
            print("INFO: No NVIDIA or AMD GPU detected via WMI.")
            return None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print(f"WARN: Failed to run WMI check for GPU: {e}")
        return None 

install_requires = [
    "edge-tts", "numpy==1.26.0", "librosa==0.9.1", "pydub>=0.25.1",
    "soundfile>=0.12.1", "ffmpeg-python>=0.2.0", "huggingface_hub",
    "nest_asyncio", "standard-aifc", "pyworld==0.3.2", "setuptools",
]

gpu_vendor = check_gpu_provider_for_setup()
selected_onnx_package = None

if gpu_vendor == "NVIDIA":
    print("--> Selecting onnxruntime-gpu based on WMI check.")
    selected_onnx_package = 'onnxruntime-gpu>=1.15,<2.0'
elif gpu_vendor == "AMD" and platform.system() == "Windows":
    print("--> Selecting onnxruntime-directml based on WMI check.")
    selected_onnx_package = 'onnxruntime-directml>=1.15,<2.0'

if not selected_onnx_package:
    print("--> Selecting standard onnxruntime (CPU default).")
    selected_onnx_package = 'onnxruntime>=1.15,<2.0'

install_requires.append(selected_onnx_package)
print(f"INFO: Final ONNX package added to requirements: {selected_onnx_package}")

setup(
    name='tts_with_rvc',
    version='0.1.10a',
    description='TTS with RVC pipeline (ONNX)',
    author='Atm4x',
    packages=find_packages(),
    install_requires=install_requires
)
