# File: setup.py
# Content:
import platform
import subprocess
import os
from setuptools import setup, find_packages

# Function to detect GPU vendor on Windows using WMI
def check_gpu_provider_for_setup():
    if platform.system() != "Windows":
        return None # WMI is Windows-only
    try:
        command = "wmic path win32_VideoController get name"
        output = subprocess.check_output(
            command, text=True, shell=True,
            stdin=subprocess.DEVNULL, stderr=subprocess.PIPE
        ).strip()
        if "NVIDIA" in output: return "NVIDIA"
        elif "AMD" in output or "Radeon" in output: return "AMD"
        else: return None
    except Exception:
        return None

# Base requirements
install_requires = [
    "edge-tts>=6.1.8",
    "numpy==1.26.0",
    "librosa>=0.9.1",
    "soundfile>=0.12.1",
    "ffmpeg-python>=0.2.0",
    "huggingface_hub>=0.17.0",
    "nest_asyncio>=1.5.0",
    "pyworld>=0.3.2",
    "setuptools",
    "onnxruntime>=1.15,<2.0",
    "faiss-cpu>=1.7.3",
    "torch>=1.13.1",
    "torchaudio>=0.13.1",
    "torchcrepe>=0.0.20",
    "praat-parselmouth>=0.4.3",
    "scipy>=1.10.0",
]

# Determine GPU vendor and select appropriate ONNX Runtime package
gpu_vendor = check_gpu_provider_for_setup()
selected_onnx_package = None

if gpu_vendor == "NVIDIA":
    print("--> Selecting onnxruntime-gpu based on WMI check.")
    selected_onnx_package = 'onnxruntime-gpu>=1.15,<2.0'
elif gpu_vendor == "AMD" and platform.system() == "Windows":
    print("--> Selecting onnxruntime-directml based on WMI check.")
    selected_onnx_package = 'onnxruntime-directml>=1.15,<2.0'

# Replace base onnxruntime if a specific GPU package is selected
if selected_onnx_package:
    install_requires = [req for req in install_requires if not req.startswith('onnxruntime>=')]
    install_requires.append(selected_onnx_package)
    print(f"INFO: Final ONNX package added to requirements: {selected_onnx_package}")
else:
    print("--> Selecting standard onnxruntime (CPU default).")

setup(
    name='tts-with-rvc-onnx', 
    version='0.1.9',
    description='TTS with RVC pipeline (ONNX Version)', 
    author='Atm4x', 
    packages=find_packages(), 
    install_requires=install_requires 
)