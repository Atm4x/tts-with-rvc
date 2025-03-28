from setuptools import setup, find_packages
import sys
import platform

install_requires = [
    "huggingface_hub",
    "av",
    "nest_asyncio",
    "torch",   
    "edge-tts",     
    "numpy==1.26.0",
    "librosa==0.9.1",
    "faiss-cpu==1.10.0",
    "soundfile>=0.12.1",
    "ffmpeg-python>=0.2.0",
    "praat-parselmouth>=0.4.2",
    "resampy>=0.4.2",   
    "scikit-learn",  
    "tqdm>=4.63.1",      
    "audioread",  
    "pyworld==0.3.2",
    "torchcrepe==0.0.20",
]

python_version = f"{sys.version_info.major}{sys.version_info.minor}"
system = platform.system()

windows_fairseq_links = {
    "312": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp312-cp312-win_amd64.whl",
    "311": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp311-cp311-win_amd64.whl",
    "310": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp310-cp310-win_amd64.whl",
}

linux_fairseq_links = {
    "312": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp312-cp312-linux_x86_64.whl",
    "311": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp311-cp311-linux_x86_64.whl",
    "310": "https://github.com/Atm4x/fairseq-win-whl-3.12/releases/download/3.12/fairseq-0.12.3-cp310-cp310-linux_x86_64.whl",
}

if system == "Windows" and python_version in windows_fairseq_links:
    install_requires.append(f"fairseq @ {windows_fairseq_links[python_version]}")
elif system == "Linux" and python_version in linux_fairseq_links:
    install_requires.append(f"fairseq @ {linux_fairseq_links[python_version]}")
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