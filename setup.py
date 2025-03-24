from setuptools import setup, find_packages
import sys

# Базовые зависимости
install_requires = [
    "huggingface_hub",
    "nest_asyncio"
    "torch",
    "torchaudio",
    "edge-tts",
    "joblib>=1.1.0",
    "omegaconf==2.3.0",
    "hydra-core==1.3.2",
    "antlr4-python3-runtime==4.9.3",
    "numba==0.59.0",
    "numpy==1.26.0",
    "librosa==0.9.1",
    "llvmlite==0.42.0",
    "faiss-cpu==1.10.0",
    "gradio==3.14.0",
    "Cython",
    "pydub>=0.25.1",
    "soundfile>=0.12.1",
    "ffmpeg-python>=0.2.0",
    "tensorboardX",
    "Jinja2>=3.1.2",
    "json5",
    "Markdown",
    "matplotlib>=3.7.0",
    "matplotlib-inline>=0.1.3",
    "praat-parselmouth>=0.4.2",
    "Pillow>=9.1.1",
    "resampy>=0.4.2",
    "scikit-learn",
    "tensorboard",
    "tqdm>=4.63.1",
    "tornado>=6.1",
    "Werkzeug>=2.2.3",
    "uc-micro-py>=1.0.1",
    "sympy>=1.11.1",
    "tabulate>=0.8.10",
    "PyYAML>=6.0",
    "pyasn1>=0.4.8",
    "pyasn1-modules>=0.2.8",
    "fsspec>=2022.11.0",
    "absl-py>=1.2.0",
    "audioread",
    "uvicorn>=0.21.1",
    "colorama>=0.4.5",
    "pyworld==0.3.2",
    "httpx>=0.23.0",
    "torchcrepe==0.0.20",
    "fastapi==0.88",
    "ffmpy==0.3.1"
]

# Определяем версию Python и добавляем нужный пакет fairseq
python_version = f"{sys.version_info.major}{sys.version_info.minor}"

fairseq_links = {
    "312": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp312-cp312-win_amd64.whl",
    "311": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp311-cp311-win_amd64.whl",
    "310": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp310-cp310-win_amd64.whl",
}

if python_version in fairseq_links:
    install_requires.append(f"fairseq @ {fairseq_links[python_version]}")
else:
    install_requires.append("fairseq @ git+https://github.com/One-sixth/fairseq.git")

setup(
    name='tts_with_rvc',
    version='0.1.6',
    description='TTS with RVC pipeline',
    author='Atm4x',
    packages=find_packages(),
    install_requires=install_requires,
    scripts=['tts_with_rvc/inference.py']
)
