from setuptools import setup, find_packages

setup(
    name='tts_with_rvc',
    version='0.1.10a',
    description='TTS with RVC pipeline (ONNX)',
    author='Atm4x',
    packages=find_packages(),
    install_requires=[
        "edge-tts",
        "numpy==1.26.0",
        "librosa==0.9.1",
        "pydub>=0.25.1",
        "soundfile>=0.12.1",
        "ffmpeg-python>=0.2.0",
        "huggingface_hub",
        "nest_asyncio",
        "standard-aifc",
        "onnxruntime",
        "onnxruntime-gpu",
        "onnxruntime-directml",
        "pyworld==0.3.2",
    ],
    scripts=['tts_with_rvc/inference.py']
)
